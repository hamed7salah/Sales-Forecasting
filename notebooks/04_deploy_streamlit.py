import streamlit as st
import pandas as pd
import numpy as np
import pickle
import mlflow
import warnings
import os
from io import StringIO
from utils import create_features

warnings.filterwarnings('ignore')

# --- Configuration ---
MLFLOW_EXPERIMENT_NAME = "Sales Forecasting Retraining"
MODEL_ARTIFACT_PATH = "sales_forecaster_rf"
HISTORICAL_DATA_PATH = r'C:\Users\154064\Downloads\Downloads\SalesForecasting\notebooks\cleaned_retail_data_milestone1.csv'

# --- Define placeholders early ---
MLFLOW_MODEL_URI = None
LATEST_RUN_ID = None
FEATURES = [] # Final list of features model expects
base_features_needed = [] # Base features needed in user upload
hist_cols_to_keep = ['Date', 'TotalUnitsSold'] # Always need these for history

# --- MLflow Setup & Model/Feature List Loading ---
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
if experiment:
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if runs:
        LATEST_RUN_ID = runs[0].info.run_id
        MLFLOW_MODEL_URI = f"runs:/{LATEST_RUN_ID}/{MODEL_ARTIFACT_PATH}"
        print(f"Found latest MLflow run ID: {LATEST_RUN_ID}")
        print(f"Attempting to load model from MLflow URI: {MLFLOW_MODEL_URI}")
        run_data = client.get_run(LATEST_RUN_ID).data
        if "features_used" in run_data.params:
             try:
                 FEATURES = eval(run_data.params["features_used"])
                 print(f"Loaded FEATURES list from MLflow run ({len(FEATURES)} features)")
             except Exception as e:
                 st.warning(f"Could not evaluate FEATURES list from MLflow ({e}). Using manual fallback.")
                 FEATURES = [] # Reset on failure
        else:
            st.warning("MLflow parameter 'features_used' not found. Using manual fallback.")
            FEATURES = [] # Reset if not found

        # --- MANUAL FALLBACK FOR FEATURES (if MLflow param fails/missing) ---
        if not FEATURES:
             st.warning("Defining FEATURES list manually as fallback.")
             # IMPORTANT: This *must* match the training FEATURES list
             FEATURES = [
                'AvgPrice', 'AvgDiscount', 'AvgCompetitorPrice', 'TotalUnitsOrdered',
                'AvgInventoryLevel', 'PromotionDay', 'DayOfWeek', 'Month', 'Year',
                'DayOfYear', 'DayOfMonth', 'WeekOfYear', 'Quarter',
                'TotalUnitsSold_lag_7', 'TotalUnitsSold_lag_14', 'TotalUnitsSold_lag_28',
                'TotalUnitsSold_roll_mean_7', 'TotalUnitsSold_roll_std_7',
                'TotalUnitsSold_roll_mean_14', 'TotalUnitsSold_roll_std_14'
             ]
             print(f"Using manually defined FEATURES list ({len(FEATURES)} features)")

        # --- Calculate base_features_needed AFTER FEATURES is loaded/defined ---
        if FEATURES:
             engineered_feature_indicators = ['DayOfWeek', 'Month', 'Year', 'DayOfYear', 'DayOfMonth', 'WeekOfYear', 'Quarter', '_lag_', '_roll_']
             base_features_needed = [
                 f for f in FEATURES
                 if not any(indicator in f for indicator in engineered_feature_indicators)
             ]
             print(f"Base features identified: {base_features_needed}")
             # Update hist_cols_to_keep based on base features actually needed
             hist_cols_to_keep.extend(base_features_needed)
             hist_cols_to_keep = list(set(hist_cols_to_keep)) # Remove duplicates
             print(f"Columns to keep from historical aggregation: {hist_cols_to_keep}")
        else:
             st.error("Could not determine the FEATURES list required by the model.")


    else:
        st.error(f"MLflow Setup Warning: No runs found in experiment '{MLFLOW_EXPERIMENT_NAME}'.")
else:
    st.error(f"MLflow Setup Warning: Experiment '{MLFLOW_EXPERIMENT_NAME}' not found.")


@st.cache_resource
def load_model_from_mlflow(model_uri):
    # (Keep the load_model_from_mlflow function exactly as before)
     if model_uri:
        print(f"Loading model from: {model_uri}")
        loaded_model = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully from MLflow.")
        st.session_state['model_source'] = f"MLflow Run ID: {LATEST_RUN_ID}"
        return loaded_model
     else:
        print("MLflow model URI is not available. Cannot load model.")
        st.session_state['model_source'] = "MLflow loading failed"
        return None

model = load_model_from_mlflow(MLFLOW_MODEL_URI)

# --- Load and Aggregate Historical Data ---
@st.cache_data
def load_and_aggregate_historical_data(file_path):
    # (Keep the load_and_aggregate_historical_data function exactly as before)
    # It aggregates columns needed based on the agg_dict defined inside it.
    df_agg_hist = None
    if os.path.exists(file_path):
        df_hist = pd.read_csv(file_path, parse_dates=['Date'])
        agg_dict = {
            'TotalUnitsSold': ('Units Sold', 'sum'),
            'AvgPrice': ('Price', 'mean'),
            'AvgDiscount': ('Discount', 'mean'),
            'AvgCompetitorPrice': ('Competitor Pricing', 'mean'),
            'TotalUnitsOrdered': ('Units Ordered', 'sum'),
            'AvgInventoryLevel': ('Inventory Level', 'mean'),
            'PromotionDay': ('Holiday/Promotion', 'max')
        }
        valid_agg_dict = {k: v for k, v in agg_dict.items() if v[0] in df_hist.columns}
        df_agg_hist = df_hist.groupby('Date').agg(**valid_agg_dict).reset_index()
        df_agg_hist = df_agg_hist.sort_values('Date')
        print(f"Historical data loaded and aggregated from {file_path}.")
    else:
        st.error(f"Error: Historical data file not found at {HISTORICAL_DATA_PATH}.")
    return df_agg_hist

df_historical_agg = load_and_aggregate_historical_data(HISTORICAL_DATA_PATH)

# --- Define Expected Raw Columns for Upload (Informational) ---
# These are the columns expected in the user's uploaded CSV for aggregation
cols_for_aggregation = ['Date', 'Units Sold', 'Price', 'Discount', 'Competitor Pricing', 'Units Ordered', 'Inventory Level', 'Holiday/Promotion']

# --- Streamlit App UI ---
st.title("Retail Sales Forecasting App (Batch Forecast from Raw CSV)")

if 'model_source' in st.session_state:
    st.caption(f"Using model from: {st.session_state['model_source']}")
else:
     st.caption("Model source status unknown.")

# Instructions use cols_for_aggregation now
st.header("1. Upload CSV File with Future Raw Data")
st.markdown(f"""
Upload a CSV file containing future transaction/product-level data.
The file needs columns for aggregation, including **at least**:
- **Date**: Future dates (YYYY-MM-DD).
- Columns needed: `{', '.join(cols_for_aggregation)}`
    - *Note: Provide expected values for future Price, Discount, Holiday/Promotion etc. Future 'Units Sold' can be 0 or omitted.*

The app aggregates this daily, generates time-series features, and predicts **TotalUnitsSold**.
""")


uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# --- Processing Logic ---
# Check model, historical data, AND that FEATURES and base_features_needed were determined
if model is None:
    st.error("Application cannot proceed: Model failed to load.")
elif df_historical_agg is None:
    st.error("Application cannot proceed: Historical data failed to load.")
elif not FEATURES:
     st.error("Application cannot proceed: Feature list for the model could not be determined (check MLflow run/fallback).")
elif not base_features_needed:
     st.error("Application cannot proceed: Could not determine base features needed (check FEATURES list).")
elif uploaded_file is not None:
    st.header("2. Processing and Forecasting")
    with st.spinner("Reading CSV, aggregating, generating features, and predicting..."):
        df_input_raw = pd.read_csv(uploaded_file)
        st.write("Uploaded Raw Data Sample:")
        st.dataframe(df_input_raw.head())

        # --- Validation ---
        valid = True
        missing_cols = []
        # ... (keep date validation as before) ...
        if 'Date' not in df_input_raw.columns:
            st.error("Validation Error: Uploaded CSV must contain a 'Date' column.")
            valid = False
        else:
            try:
                df_input_raw['Date'] = pd.to_datetime(df_input_raw['Date'])
                min_forecast_date = df_input_raw['Date'].min()
                last_historical_date = df_historical_agg['Date'].max()
                if min_forecast_date <= last_historical_date:
                     st.warning(f"Warning: Uploaded dates start on or before the last historical date ({last_historical_date.strftime('%Y-%m-%d')}).")
            except Exception as e:
                st.error(f"Validation Error: Could not parse 'Date' column. Error: {e}")
                valid = False

        # Check for columns needed for aggregation (allow 'Units Sold' to be missing in upload)
        for col in cols_for_aggregation:
             if col not in df_input_raw.columns and col != 'Units Sold':
                 missing_cols.append(col)
                 valid = False
        if missing_cols:
             st.error(f"Validation Error: Uploaded CSV is missing columns needed for aggregation: {', '.join(missing_cols)}")


        # --- Aggregation, Feature Engineering & Prediction (if valid) ---
        if valid:
            st.write("Aggregating uploaded data daily...")
            agg_dict_future = {
                'AvgPrice': ('Price', 'mean'),
                'AvgDiscount': ('Discount', 'mean'),
                'AvgCompetitorPrice': ('Competitor Pricing', 'mean'),
                'TotalUnitsOrdered': ('Units Ordered', 'sum'),
                'AvgInventoryLevel': ('Inventory Level', 'mean'),
                'PromotionDay': ('Holiday/Promotion', 'max')
            }
            valid_agg_dict_future = {k: v for k, v in agg_dict_future.items() if v[0] in df_input_raw.columns}
            df_input_agg = df_input_raw.groupby('Date').agg(**valid_agg_dict_future).reset_index()
            df_input_agg['TotalUnitsSold'] = np.nan # Add placeholder

            st.write("Aggregated Future Data Sample:")
            st.dataframe(df_input_agg.head())

            # --- Combine Historical Aggregated with Future Aggregated ---
            # Use the globally defined hist_cols_to_keep (which includes base features)
            hist_subset_for_combine = df_historical_agg[[c for c in hist_cols_to_keep if c in df_historical_agg.columns]].copy()

            # Align columns for concat
            cols_to_combine = list(set(hist_subset_for_combine.columns) | set(df_input_agg.columns))
            hist_aligned = hist_subset_for_combine.reindex(columns=cols_to_combine)
            input_aligned = df_input_agg.reindex(columns=cols_to_combine)


            df_for_features = pd.concat([hist_aligned, input_aligned], ignore_index=True)
            df_for_features = df_for_features.sort_values('Date').reset_index(drop=True)

            # --- Apply Feature Engineering ---
            st.write("Running feature engineering...")
            df_engineered_all = create_features(df_for_features)

            # Filter engineered data for the forecast dates
            forecast_dates = df_input_agg['Date'].unique()
            df_engineered_future = df_engineered_all[df_engineered_all['Date'].isin(forecast_dates)].copy()


            st.write("Generated Features for Prediction (Sample):")
            st.dataframe(df_engineered_future.head())

            # --- Select Final Features and Predict ---
            missing_final_features = [f for f in FEATURES if f not in df_engineered_future.columns]
            if missing_final_features:
                st.error(f"Feature mismatch error before prediction. Missing features: {missing_final_features}. Check FEATURES list and aggregation/engineering logic.")
            else:
                X_predict = df_engineered_future[FEATURES] # Select features

                # Final NaN check
                if X_predict.isnull().values.any():
                    st.warning("NaN values detected before prediction. Attempting final fill...")
                    X_predict = X_predict.fillna(method='ffill').fillna(method='bfill').fillna(0)

                st.write("Predicting...")
                predictions = model.predict(X_predict)

                # --- Display Results ---
                df_results = pd.DataFrame({
                    'Date': df_engineered_future['Date'].values,
                    'PredictedTotalUnitsSold': predictions.round(0).astype(int)
                })
                # Merge back the AGGREGATED base features for context display
                cols_to_show_context = ['Date'] + [bf for bf in base_features_needed if bf in df_engineered_future.columns]
                df_results_context = pd.merge(
                    df_engineered_future[list(set(cols_to_show_context))], # Use set to avoid duplicates
                    df_results,
                    on='Date'
                )

                st.header("3. Forecast Results")
                st.dataframe(df_results_context.style.format({'Date': '{:%Y-%m-%d}', **{col: '{:.2f}' for col in df_results_context.columns if col not in ['Date', 'PredictedTotalUnitsSold', 'PromotionDay']}}), height=400) # Format floats


                st.subheader("Predicted Sales Over Time")
                st.line_chart(df_results.rename(columns={'Date': 'index'}).set_index('index')['PredictedTotalUnitsSold'])
        else:
            st.error("Processing halted due to validation errors in the uploaded file.")


# --- Monitoring Placeholder (Keep as before) ---
st.sidebar.header("Model Monitoring (Conceptual)")
st.sidebar.info("(Monitoring details remain the same)")
st.sidebar.header("Model Retraining")
st.sidebar.info("(Retraining details remain the same)")