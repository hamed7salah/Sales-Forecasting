# utils.py
import pandas as pd
import numpy as np

def create_features(df_in):
    """Creates time series features from Date column"""
    df_out = df_in.copy()
    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_out['Date']):
         df_out['Date'] = pd.to_datetime(df_out['Date'])

    # Set index temporarily for feature engineering if not already set
    original_index = df_out.index
    is_indexed = isinstance(df_out.index, pd.DatetimeIndex)
    if not is_indexed:
        df_out = df_out.set_index('Date')

    df_out['DayOfWeek'] = df_out.index.dayofweek
    df_out['Month'] = df_out.index.month
    df_out['Year'] = df_out.index.year
    df_out['DayOfYear'] = df_out.index.dayofyear
    df_out['DayOfMonth'] = df_out.index.day
    df_out['WeekOfYear'] = df_out.index.isocalendar().week.astype(int)
    df_out['Quarter'] = df_out.index.quarter

    # --- Lag features for the target variable 'TotalUnitsSold' ---
    target = 'TotalUnitsSold'
    # Ensure target exists before creating lags
    if target in df_out.columns:
        # Lags (using past actual values for training/evaluation features)
        df_out[f'{target}_lag_7'] = df_out[target].shift(7)
        df_out[f'{target}_lag_14'] = df_out[target].shift(14)
        df_out[f'{target}_lag_28'] = df_out[target].shift(28)

        # --- Rolling window features ---
        # Use shift(1) to prevent data leakage (use data available *before* the day)
        df_out[f'{target}_roll_mean_7'] = df_out[target].shift(1).rolling(window=7, min_periods=1).mean()
        df_out[f'{target}_roll_std_7'] = df_out[target].shift(1).rolling(window=7, min_periods=1).std()
        # Add more rolling features if desired (e.g., 14-day, 28-day)
        df_out[f'{target}_roll_mean_14'] = df_out[target].shift(1).rolling(window=14, min_periods=1).mean()
        df_out[f'{target}_roll_std_14'] = df_out[target].shift(1).rolling(window=14, min_periods=1).std()

    # Fill potential NaNs created by rolling/lagging, common strategy is forward fill then backfill
    # Important: Only fill NaNs in engineered features, not the original target if present
    engineered_cols = [col for col in df_out.columns if '_lag_' in col or '_roll_' in col]
    df_out[engineered_cols] = df_out[engineered_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)

    # Reset index if it was set temporarily
    if not is_indexed:
        df_out = df_out.reset_index()
    else:
        # Keep the original index if it was already a DatetimeIndex
        df_out.index = original_index

    return df_out
