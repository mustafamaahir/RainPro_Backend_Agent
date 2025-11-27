import logging
import numpy as np
import pandas as pd
import joblib  # ADD THIS IMPORT
from langchain_core.runnables import RunnableConfig 

logger = logging.getLogger(__name__)
FEATURES = [
    'RH2M', 'WS10M', 'T2M', 'WD10M',
    'ALLSKY_SFC_SW_DWN', 'EVPTRNS', 'PS',
    'QV2M', 'T2M_RANGE', 'TS', 'CLRSKY_SFC_SW_DWN'
]
TARGET_COL = "PRECTOTCORR"

def preprocessing_agent(state: dict, config: RunnableConfig | None = None):
    """
    Preprocess NASA data for model readiness.
    """
    logger.info("🚀 preprocessing_agent started")
    
    # Identify mode from intent (daily or monthly)
    intent = state.get("intent", {})
    mode = intent.get("mode", "daily").lower()
    df = state.get("nasa_parameters")

    logger.info(f"📋 Mode: {mode}, NASA data available: {df is not None}")

    if df is None or df.empty:
        logger.error("❌ No NASA data found")
        return {**state, "error": "No NASA data found in state."}

    # FIXED: Load scaler from file path
    try:
        if mode == "daily":
            scaler_path = "models/scaler_daily.pkl"
        else:
            scaler_path = "models/scaler_monthly.pkl"
        
        logger.info(f"📁 Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
        logger.info("✅ Scaler loaded successfully")
    except Exception as e:
        logger.exception(f"❌ Failed to load scaler: {e}")
        return {**state, "error": f"Failed to load scaler: {e}"}

    # Common cleanup
    for col in FEATURES + [TARGET_COL]:
        if col not in df.columns:
            df[col] = 0

    df = df.replace(-999.0, np.nan).ffill().bfill()

    # Daily preprocessing
    if mode == "daily":
        logger.info("🌤️ Processing DAILY data...")
        
        for col in FEATURES + [TARGET_COL]:
            df[col] = df[col].fillna(0)
            df[col] = df[col].clip(lower=0)

        df["log_PRECTOTCORR"] = np.log1p(df[TARGET_COL])
        for col in FEATURES:
            df[f"log_{col}"] = np.log1p(df[col])

        for lag in [1, 3, 7]:
            df[f"log_PRECTOTCORR_lag{lag}"] = df["log_PRECTOTCORR"].shift(lag)
        df["rain_rolling_mean"] = df["log_PRECTOTCORR"].rolling(window=7).mean()
        df["rain_rolling_std"] = df["log_PRECTOTCORR"].rolling(window=7).std()
        df = df.dropna().copy()

        logger.info(f"📊 DataFrame shape after processing: {df.shape}")

        if len(df) < 15:
            logger.error("❌ Not enough data (need 15+ rows)")
            return {**state, "error": "Not enough data to make daily prediction."}

        final_features = [f"log_{col}" for col in FEATURES] + [
            "log_PRECTOTCORR_lag1", "log_PRECTOTCORR_lag3", "log_PRECTOTCORR_lag7",
            "rain_rolling_mean", "rain_rolling_std"
        ]

        window = df[final_features + ["log_PRECTOTCORR"]].tail(15).copy()

        try:
            scaled = scaler.transform(window)
            X_input = np.expand_dims(scaled, axis=0)
            logger.info(f"✅ Preprocessing complete - X_input shape: {X_input.shape}")
        except Exception as e:
            logger.exception(f"❌ Scaling failed: {e}")
            return {**state, "error": f"Scaling failed: {e}"}

        return {
            **state,
            "scaled": scaled,
            "final_features": final_features,
            "preprocessed_data": X_input,
            "preprocessed_window": window
        }

    # Monthly preprocessing
    elif mode == "monthly":
        logger.info("📅 Processing MONTHLY data...")
        
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'DATE'}, inplace=True)

        df["log_PRECTOTCORR"] = np.log1p(df[TARGET_COL])
        for col in FEATURES:
            df[f"log_{col}"] = np.log1p(df[col])

        for lag in [1, 3, 7]:
            df[f"log_PRECTOTCORR_lag{lag}"] = df["log_PRECTOTCORR"].shift(lag)

        df["rain_rolling_mean"] = df["log_PRECTOTCORR"].rolling(window=3).mean()
        df["rain_rolling_std"] = df["log_PRECTOTCORR"].rolling(window=3).std()
        df = df.dropna().copy()

        if len(df) < 7:
            logger.error("❌ Not enough data for monthly (need 7+ rows)")
            return {**state, "error": "Not enough data to compute lag7 for monthly prediction."}

        final_features = [f"log_{col}" for col in FEATURES] + [
            "log_PRECTOTCORR_lag1", "log_PRECTOTCORR_lag3", "log_PRECTOTCORR_lag7",
            "rain_rolling_mean", "rain_rolling_std"
        ]
        window = df[final_features + ["log_PRECTOTCORR"]].tail(7).copy()

        try:
            scaled = scaler.transform(window)
            X_input = np.expand_dims(scaled, axis=0)
            logger.info(f"✅ Preprocessing complete - X_input shape: {X_input.shape}")
        except Exception as e:
            logger.exception(f"❌ Scaling failed: {e}")
            return {**state, "error": f"Scaling failed: {e}"}

        return {
            **state,
            "scaled": scaled,
            "final_features": final_features,
            "preprocessed_data": X_input,
            "preprocessed_window": window
        }

    else:
        return {**state, "error": f"Invalid mode '{mode}'. Expected 'daily' or 'monthly'."}