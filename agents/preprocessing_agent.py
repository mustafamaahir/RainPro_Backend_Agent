import numpy as np
import pandas as pd
from langchain_core.runnables import RunnableConfig 

FEATURES = [
    'RH2M', 'WS10M', 'T2M', 'WD10M',
    'ALLSKY_SFC_SW_DWN', 'EVPTRNS', 'PS',
    'QV2M', 'T2M_RANGE', 'TS', 'CLRSKY_SFC_SW_DWN'
]
TARGET_COL = "PRECTOTCORR"

def preprocessing_agent(state: dict, config: RunnableConfig | None = None):
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

    # Choose scaler and other config depending on mode
    scaler = config.get("models/scaler_daily.pkl") if mode == "daily" else config.get("models/scaler_monthly.pkl")

    # Common cleanup
    # Fill missing columns if not present
    for col in FEATURES + [TARGET_COL]:
        if col not in df.columns:
            df[col] = 0

    # Replace missing flags and fill gaps
    df = df.replace(-999.0, np.nan).ffill().bfill()

    # Daily preprocessing
    if mode == "daily":
        # Fill missing and clip negatives
        for col in FEATURES + [TARGET_COL]:
            df[col] = df[col].fillna(0)
            df[col] = df[col].clip(lower=0)

        # Log-transform safely
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

        # Prepare features
        final_features = [f"log_{col}" for col in FEATURES] + [
            "log_PRECTOTCORR_lag1", "log_PRECTOTCORR_lag3", "log_PRECTOTCORR_lag7",
            "rain_rolling_mean", "rain_rolling_std"
        ]

        # Take last 15 days window
        window = df[final_features + ["log_PRECTOTCORR"]].tail(15).copy()

        # Scale for model input
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

    # Monthly preprocessing
    elif mode == "monthly":
        # Reset index and rename
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'DATE'}, inplace=True)

        # Log-transform features
        df["log_PRECTOTCORR"] = np.log1p(df[TARGET_COL])
        for col in FEATURES:
            df[f"log_{col}"] = np.log1p(df[col])

        # Create lag features (1, 3, 7 months)
        for lag in [1, 3, 7]:
            df[f"log_PRECTOTCORR_lag{lag}"] = df["log_PRECTOTCORR"].shift(lag)

        # Rolling mean/std over 3 months
        df["rain_rolling_mean"] = df["log_PRECTOTCORR"].rolling(window=3).mean()
        df["rain_rolling_std"] = df["log_PRECTOTCORR"].rolling(window=3).std()
        df = df.dropna().copy()

        if len(df) < 7:
            return {"error": "Not enough data to compute lag7 for monthly prediction."}

        # Prepare window and features
        final_features = [f"log_{col}" for col in FEATURES] + [
            "log_PRECTOTCORR_lag1", "log_PRECTOTCORR_lag3", "log_PRECTOTCORR_lag7",
            "rain_rolling_mean", "rain_rolling_std"
        ]
        window = df[final_features + ["log_PRECTOTCORR"]].tail(7).copy()

        # Scale for model input
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
