# agents/preprocessing_agent.py
import os
import numpy as np
import pandas as pd
import joblib
from langchain_core.runnables import RunnableConfig

FEATURES = [
    'RH2M', 'WS10M', 'T2M', 'WD10M',
    'ALLSKY_SFC_SW_DWN', 'EVPTRNS', 'PS',
    'QV2M', 'T2M_RANGE', 'TS', 'CLRSKY_SFC_SW_DWN'
]
TARGET_COL = "PRECTOTCORR"

def preprocessing_agent(state: dict, config: dict | None = None) -> dict:
    """
    Preprocess NASA data for model readiness.
    Handles both daily and monthly datasets based on user intent.
    Includes cleaning, transformation, lag, and rolling features.
    """

    # Identify mode from intent (daily or monthly)
    intent = state.get("intent", {})
    mode = intent.get("mode", "daily").lower()
    df = state.get("nasa_parameters")

    if df is None or df.empty:
        return {"error": "No NASA data found in state."}

    # Determine scaler path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if mode == "daily":
        scaler_path = os.path.join(BASE_DIR, "../models/scaler_daily.pkl")
    else:
        scaler_path = os.path.join(BASE_DIR, "../models/scaler_monthly.pkl")

    if not os.path.exists(scaler_path):
        return {"error": f"Scaler file not found: {scaler_path}"}

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Fill missing columns if not present
    for col in FEATURES + [TARGET_COL]:
        if col not in df.columns:
            df[col] = 0

    df = df.replace(-999.0, np.nan).ffill().bfill()  # Fill missing values

    # --- DAILY preprocessing ---
    if mode == "daily":
        for col in FEATURES + [TARGET_COL]:
            df[col] = df[col].clip(lower=0)

        df["log_PRECTOTCORR"] = np.log1p(df[TARGET_COL])
        for col in FEATURES:
            df[f"log_{col}"] = np.log1p(df[col])

        # Lag and rolling features (7-day window)
        for lag in [1, 3, 7]:
            df[f"log_PRECTOTCORR_lag{lag}"] = df["log_PRECTOTCORR"].shift(lag)
        df["rain_rolling_mean"] = df["log_PRECTOTCORR"].rolling(window=7).mean()
        df["rain_rolling_std"] = df["log_PRECTOTCORR"].rolling(window=7).std()
        df = df.dropna().copy()

        if len(df) < 15:
            return {"error": "Not enough data to make daily prediction."}

        final_features = [f"log_{col}" for col in FEATURES] + [
            "log_PRECTOTCORR_lag1", "log_PRECTOTCORR_lag3", "log_PRECTOTCORR_lag7",
            "rain_rolling_mean", "rain_rolling_std"
        ]

        window = df[final_features + ["log_PRECTOTCORR"]].tail(15).copy()
        scaled = scaler.transform(window)
        X_input = np.expand_dims(scaled, axis=0)

        return {
            **state,
            "scaled": scaled,
            "final_features": final_features,
            "preprocessed_data": X_input,
            "preprocessed_window": window,
            "mode": "daily"
        }

    # --- MONTHLY preprocessing ---
    elif mode == "monthly":
        df = df.reset_index().rename(columns={'index': 'DATE'})
        df["log_PRECTOTCORR"] = np.log1p(df[TARGET_COL])
        for col in FEATURES:
            df[f"log_{col}"] = np.log1p(df[col])

        for lag in [1, 3, 7]:
            df[f"log_PRECTOTCORR_lag{lag}"] = df["log_PRECTOTCORR"].shift(lag)

        df["rain_rolling_mean"] = df["log_PRECTOTCORR"].rolling(window=3).mean()
        df["rain_rolling_std"] = df["log_PRECTOTCORR"].rolling(window=3).std()
        df = df.dropna().copy()

        if len(df) < 7:
            return {"error": "Not enough data to compute lag7 for monthly prediction."}

        final_features = [f"log_{col}" for col in FEATURES] + [
            "log_PRECTOTCORR_lag1", "log_PRECTOTCORR_lag3", "log_PRECTOTCORR_lag7",
            "rain_rolling_mean", "rain_rolling_std"
        ]

        window = df[final_features + ["log_PRECTOTCORR"]].tail(7).copy()
        scaled = scaler.transform(window)
        X_input = np.expand_dims(scaled, axis=0)

        return {
            **state,
            "scaled": scaled,
            "final_features": final_features,
            "preprocessed_data": X_input,
            "preprocessed_window": window,
            "mode": "monthly"
        }

    else:
        return {"error": f"Invalid mode '{mode}'. Expected 'daily' or 'monthly'."}
