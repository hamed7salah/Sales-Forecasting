# 04_train_with_mlflow.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
import pickle
import warnings
import os
from utils import create_features # Import from utils.py

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_PATH = r'C:\Users\154064\Downloads\Downloads\SalesForecasting\notebooks\cleaned_retail_data_milestone1.csv'
MODEL_ARTIFACT_PATH = "sales_forecaster_rf"
MLFLOW_EXPERIMENT_NAME = "Sales Forecasting Retraining"

# --- Load and Prepare Data ---
print("Loading and preparing data...")
if not os.path.exists(DATA_PATH):
    print(f"Error: Data file not found at {DATA_PATH}. Please place it in the correct directory.")
    exit()

df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
print(f"Loaded data from {DATA_PATH}")

# Aggregate daily features - Ensure ALL base features needed are calculated here
required_aggregations = {
    'TotalUnitsSold': ('Units Sold', 'sum'), # Target variable for lags/rolls
    'AvgPrice': ('Price', 'mean'),
    'AvgDiscount': ('Discount', 'mean'),
    'AvgCompetitorPrice': ('Competitor Pricing', 'mean'),
    'TotalUnitsOrdered': ('Units Ordered', 'sum'),
    'AvgInventoryLevel': ('Inventory Level', 'mean'),
    'PromotionDay': ('Holiday/Promotion', 'max')
}
# Filter aggregation dict based on columns actually present in df
valid_aggregations = {k: v for k, v in required_aggregations.items() if v[0] in df.columns}
missing_base_cols_for_agg = [v[0] for k, v in required_aggregations.items() if v[0] not in df.columns]
if missing_base_cols_for_agg:
    print(f"Warning: Source CSV is missing columns needed for full aggregation: {missing_base_cols_for_agg}")

df_agg = df.groupby('Date').agg(**valid_aggregations).reset_index()
print("Aggregated data daily.")


# --- Feature Engineering ---
print("Engineering features...")
df_ml_features = create_features(df_agg)

# Drop NaNs created by initial lags/rolling windows
df_ml_features = df_ml_features.dropna()
print(f"Data shape after feature engineering and NaN drop: {df_ml_features.shape}")
if df_ml_features.empty:
    print("Error: No data remaining after feature engineering. Check lag/rolling window settings in utils.py or initial data.")
    exit()

# Ensure 'Date' is the index for splitting
if 'Date' in df_ml_features.columns:
    df_ml_features = df_ml_features.set_index('Date')
df_ml_features.sort_index(inplace=True)


# --- Define TARGET and Explicit FEATURES List ---
TARGET = 'TotalUnitsSold'

# ***** USE THE EXPLICITLY REQUESTED FEATURES *****
FEATURES = [
    'AvgPrice',
    'AvgDiscount',
    'AvgCompetitorPrice',
    'TotalUnitsOrdered',
    'AvgInventoryLevel',
    'PromotionDay',
    'DayOfWeek',
    'Month',
    'Year',
    'DayOfYear',
    'DayOfMonth',
    'WeekOfYear',
    'Quarter',
    'TotalUnitsSold_lag_7',
    'TotalUnitsSold_lag_14',
    'TotalUnitsSold_lag_28',
    'TotalUnitsSold_roll_mean_7',
    'TotalUnitsSold_roll_std_7'
]
print(f"Attempting to use {len(FEATURES)} specified features.")

# --- Validate Feature Availability ---
missing_features = [f for f in FEATURES if f not in df_ml_features.columns]
if missing_features:
    print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"Error: The following required features are MISSING from the dataframe after processing:")
    for f in missing_features:
        print(f" - {f}")
    print(f"Present columns are: {df_ml_features.columns.tolist()}")
    print(f"Please check:")
    print(f"  1. The aggregation step calculates all necessary base features ({[f for f in FEATURES if not any(ind in f for ind in ['Day','Month','Year','Quarter','Week','lag','roll'])]}).")
    print(f"  2. The 'create_features' function in utils.py generates all required engineered features.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    exit()
else:
    print("Successfully verified all specified features are present in the dataframe.")

# --- Select Features (X) and Target (y) ---
X = df_ml_features[FEATURES]
y = df_ml_features[TARGET]

print(f"Selected features ({len(FEATURES)}): {FEATURES}")
print(f"Target: {TARGET}")


# --- Time Series Split ---
test_size_days = 90 # Keep consistent split for comparison if desired
split_date = X.index.max() - pd.Timedelta(days=test_size_days)

train_ml = df_ml_features.loc[df_ml_features.index <= split_date]
test_ml = df_ml_features.loc[df_ml_features.index > split_date]

# Use the FEATURES list to select columns for X
X_train, y_train = train_ml[FEATURES], train_ml[TARGET]
X_test, y_test = test_ml[FEATURES], test_ml[TARGET]

print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print(f"Train period: {X_train.index.min()} to {X_train.index.max()}")
print(f"Test period: {X_test.index.min()} to {X_test.index.max()}")

if X_train.empty or X_test.empty:
     print("Error: Train or test set is empty after split. Adjust test_size_days or check data.")
     exit()


# --- MLflow Experiment Setup ---
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# --- Model Training and MLflow Logging ---
# Define model parameters (adjust if you have tuned parameters)
rf_params = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'random_state': 42,
    'n_jobs': -1
}

print(f"\n--- Training Random Forest Model with MLflow using specified features ---")
with mlflow.start_run(run_name="Sales Forecasting RF - Explicit Features") as run:
    print(f"MLflow Run ID: {run.info.run_id}")
    mlflow.log_params(rf_params)
    mlflow.log_param("target_variable", TARGET)
    # ***** LOG THE EXACT FEATURES USED *****
    mlflow.log_param("features_used", str(FEATURES))
    mlflow.log_param("train_start_date", str(X_train.index.min()))
    mlflow.log_param("train_end_date", str(X_train.index.max()))
    mlflow.log_param("test_start_date", str(X_test.index.min()))
    mlflow.log_param("test_end_date", str(X_test.index.max()))
    mlflow.log_param("data_source", DATA_PATH)

    model = RandomForestRegressor(**rf_params)
    # Train ONLY with the specified FEATURES
    model.fit(X_train, y_train)

    # --- Evaluation ---
    # Predict ONLY using the specified FEATURES
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"Evaluation on Test Set:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")

    # --- Log Metrics ---
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_rmse", rmse)

    # --- Log Model ---
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=MODEL_ARTIFACT_PATH,
        input_example=X_train.iloc[:5], # Log example using the specific features
        signature=mlflow.models.infer_signature(X_train, predictions) # Signature based on specific features
    )
    print(f"Model trained with specified features logged to MLflow artifact path: {MODEL_ARTIFACT_PATH}")

    # --- (Optional) Save model locally ---
    local_model_filename = r'notebooks\sales_forecasting_final_model.pkl'
    with open(local_model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model also saved locally as: {local_model_filename}")
    mlflow.log_artifact(local_model_filename) # Optionally log pickle

print("\n--- MLflow Training Run Completed ---")
last_run_id = run.info.run_id
print(f"Run ID: {last_run_id}")
print(f"Features used for this run were logged to MLflow parameter 'features_used'.")