# agents/model_prediction_agent.py
"""
Predict rainfall using trained tensorflow models and scalers.
Produces either:
 - state["forecasts"] for daily predictions (list of {day, predicted_rainfall_mm})
 - state["monthly_forecasts"] for monthly predictions (list of {month_ahead, predicted_rainfall_mm})

Expects in state:
 - intent: dict with mode ('daily'|'monthly'), optionally 'days' or 'months', latitude/longitude
 - preprocessed_data: numpy array shaped for model input (e.g., (1, n_features))
 - preprocessed_window: pandas.DataFrame containing the recent window used to build features
 - final_features: list of column names used in the window
 - scaled: numpy array (window scaled) used for inverse transform

Config expectations (RunnableConfig or dict-like):
 - configurable: {
       "models/rainfall_daily_predictor.h5": "<path to daily model>",
       "models/rainfall_monthly_predictor.h5": "<path to monthly model>",
       "models/scaler_daily.pkl": "<path to daily scaler>",
       "models/scaler_monthly.pkl": "<path to monthly scaler>"
   }
If those keys are missing, the agent falls back to sensible defaults.
"""

import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Default artifact paths (adjust if your project layout differs)
DEFAULTS = {
    "models/rainfall_daily_predictor.h5": "../models/rainfall_daily_predictor.h5",
    "models/rainfall_monthly_predictor.h5": "../models/rainfall_monthly_predictor.h5",
    "models/scaler_daily.pkl": "../models/scaler_daily.pkl",
    "models/scaler_monthly.pkl": "../models/scaler_monthly.pkl",
}


def _get_config_value(config, key, default=None):
    """
    Safely get a value from config.
    Works whether config is a dict or object.
    """
    if config is None:
        return default

    if isinstance(config, dict):
        return config.get(key, default)

    return getattr(config, key, default)


def inverse_transform_prediction(pred_scaled: float, scaler, last_row_scaled: np.ndarray) -> float:
    """
    Takes a single predicted scaled value (model output) and inverse-transforms it
    to original rainfall mm scale using the provided scaler and a copy of last_row_scaled.
    last_row_scaled must be shaped (1, n_features). Returns positive rainfall in mm.
    """
    try:
        temp = last_row_scaled.copy()
        # Place prediction into last column (assumes target is last in scaler ordering)
        temp[0, -1] = pred_scaled
        temp_inv = scaler.inverse_transform(temp)
        pred_log = temp_inv[0, -1]
        rainfall_mm = float(np.expm1(pred_log))
        return max(0.0, rainfall_mm)
    except Exception as e:
        logger.exception("Failed during inverse_transform_prediction: %s", e)
        # Return 0 on failure (safe fallback)
        return 0.0


def model_prediction_agent(state: Dict[str, Any], config: RunnableConfig | None = None) -> Dict[str, Any]:
    """
    Main agent invoked by LangGraph.
    """
    logger.info("🚀 model_prediction_agent started")
    
    try:
        # Basic validation
        intent = state.get("intent") or {}
        mode = intent.get("mode", "daily").lower()
        logger.info(f"📋 Mode detected: {mode}")
        
        if mode not in ("daily", "monthly"):
            logger.error(f"❌ Invalid mode: {mode}")
            return {**state, "error": f"Invalid intent mode '{mode}'."}

        # Preprocessed inputs
        window: pd.DataFrame = state.get("preprocessed_window")
        X_input = state.get("preprocessed_data")
        final_features = state.get("final_features")
        scaled = state.get("scaled")

        logger.info(f"📊 Data check - window: {window is not None}, X_input: {X_input is not None}, final_features: {final_features is not None}, scaled: {scaled is not None}")

        if window is None or X_input is None or final_features is None or scaled is None:
            logger.error("❌ Missing preprocessed data!")
            return {**state, "error": "Missing required preprocessed data in state"}

        # Resolve model and scaler paths from config
        daily_model_path = _get_config_value(config, "models/rainfall_daily_predictor.h5")
        monthly_model_path = _get_config_value(config, "models/rainfall_monthly_predictor.h5")
        daily_scaler_path = _get_config_value(config, "models/scaler_daily.pkl")
        monthly_scaler_path = _get_config_value(config, "models/scaler_monthly.pkl")

        logger.info(f"📁 Model path: {daily_model_path if mode == 'daily' else monthly_model_path}")

        latitude = intent.get("latitude", 6.585)
        longitude = intent.get("longitude", 3.983)

        if mode == "daily":
            logger.info("🌤️ Starting DAILY prediction...")
            
            # Load artifacts
            try:
                logger.info(f"Loading model from: {daily_model_path}")
                model = load_model(daily_model_path, compile=False)
                logger.info("✅ Model loaded successfully")
            except Exception as e:
                logger.exception("❌ Failed to load daily model from %s: %s", daily_model_path, e)
                return {**state, "error": f"Failed to load daily model: {e}"}

            try:
                logger.info(f"Loading scaler from: {daily_scaler_path}")
                scaler = joblib.load(daily_scaler_path)
                logger.info("✅ Scaler loaded successfully")
            except Exception as e:
                logger.exception("❌ Failed to load daily scaler from %s: %s", daily_scaler_path, e)
                return {**state, "error": f"Failed to load daily scaler: {e}"}

            forecasts = []
            days = int(intent.get("days", 7))
            logger.info(f"🔢 Forecasting for {days} days")

            # Ensure shapes are correct
            last_row_scaled = np.array(scaled[-1:]).reshape(1, -1)

            # We operate on a copy of the window so we can append simulated rows
            window_local = window.copy().reset_index(drop=True)

            for day in range(1, days + 1):
                try:
                    pred_scaled = float(model.predict(X_input, verbose=0)[0][0])
                except Exception as e:
                    logger.exception("Prediction failed for daily model: %s", e)
                    pred_scaled = 0.0

                rainfall_mm = inverse_transform_prediction(pred_scaled, scaler, last_row_scaled)

                forecasts.append({
                    "day": day,
                    "predicted_rainfall_mm": round(float(rainfall_mm), 3)
                })

                # Build next row inputs for iterative forecasting
                new_row = {}
                for col in final_features:
                    # If final_features includes the target column (log_PRECTOTCORR) skip copying it
                    if col in window_local.columns and col != "log_PRECTOTCORR":
                        new_row[col] = window_local[col].iloc[-1]
                    else:
                        # If missing in existing window, set 0
                        new_row[col] = window_local[col].iloc[-1] if col in window_local.columns else 0.0

                # lags
                lag1 = window_local["log_PRECTOTCORR"].iloc[-1]
                lag3 = window_local["log_PRECTOTCORR"].iloc[-3] if len(window_local) >= 3 else lag1
                lag7 = window_local["log_PRECTOTCORR"].iloc[-7] if len(window_local) >= 7 else lag1

                new_row["log_PRECTOTCORR_lag1"] = lag1
                new_row["log_PRECTOTCORR_lag3"] = lag3
                new_row["log_PRECTOTCORR_lag7"] = lag7

                rolling_window = list(window_local["log_PRECTOTCORR"].iloc[-6:]) + [np.log1p(rainfall_mm)]
                new_row["rain_rolling_mean"] = float(np.mean(rolling_window))
                new_row["rain_rolling_std"] = float(np.std(rolling_window))
                new_row["log_PRECTOTCORR"] = float(np.log1p(rainfall_mm))

                # Append and keep window size consistent (tail 15 used in preprocessing)
                window_local = pd.concat([window_local, pd.DataFrame([new_row])], ignore_index=True).tail(15)

                # Recompute scaled and X_input for next iteration
                try:
                    last_row_scaled = scaler.transform(window_local)[-1:].reshape(1, -1)
                    X_input = np.expand_dims(scaler.transform(window_local)[-1:], axis=0)
                except Exception:
                    # If we can't re-scale (e.g., scaler expects a different shape), keep previous X_input
                    pass

            # AFTER the loop completes
            logger.info(f"DAILY prediction complete. Forecasts: {forecasts}")
            return {
                **state,
                "forecasts": forecasts
            }

        # MONTHLY mode
        elif mode == "monthly":
            logger.info("📅 Starting MONTHLY prediction...")
            
            try:
                logger.info(f"Loading model from: {monthly_model_path}")
                model = load_model(monthly_model_path, compile=False)
                logger.info("✅ Model loaded successfully")
            except Exception as e:
                logger.exception("❌ Failed to load monthly model from %s: %s", monthly_model_path, e)
                return {**state, "error": f"Failed to load monthly model: {e}"}

            try:
                logger.info(f"Loading scaler from: {monthly_scaler_path}")
                scaler = joblib.load(monthly_scaler_path)
                logger.info("✅ Scaler loaded successfully")
            except Exception as e:
                logger.exception("❌ Failed to load monthly scaler from %s: %s", monthly_scaler_path, e)
                return {**state, "error": f"Failed to load monthly scaler: {e}"}

            forecasts = []
            months = int(intent.get("months", 6))
            logger.info(f"🔢 Forecasting for {months} months")

            last_row_scaled = np.array(scaled[-1:]).reshape(1, -1)
            window_local = window.copy().reset_index(drop=True)

            for month in range(1, months + 1):
                try:
                    pred_scaled = float(model.predict(X_input, verbose=0)[0][0])
                except Exception as e:
                    logger.exception("Prediction failed for monthly model: %s", e)
                    pred_scaled = 0.0

                rainfall_mm = inverse_transform_prediction(pred_scaled, scaler, last_row_scaled)

                forecasts.append({
                    "month_ahead": month,
                    "predicted_rainfall_mm": round(float(rainfall_mm), 3)
                })

                # Build next row
                new_row = {}
                for col in final_features:
                    if col in window_local.columns and col != "log_PRECTOTCORR":
                        new_row[col] = window_local[col].iloc[-1]
                    else:
                        new_row[col] = window_local[col].iloc[-1] if col in window_local.columns else 0.0

                lag1 = window_local["log_PRECTOTCORR"].iloc[-1]
                lag3 = window_local["log_PRECTOTCORR"].iloc[-3] if len(window_local) >= 3 else lag1
                lag7 = window_local["log_PRECTOTCORR"].iloc[-7] if len(window_local) >= 7 else lag1

                new_row["log_PRECTOTCORR_lag1"] = lag1
                new_row["log_PRECTOTCORR_lag3"] = lag3
                new_row["log_PRECTOTCORR_lag7"] = lag7

                roll_vals = list(window_local["log_PRECTOTCORR"].iloc[-2:]) + [np.log1p(rainfall_mm)]
                new_row["rain_rolling_mean"] = float(np.mean(roll_vals))
                new_row["rain_rolling_std"] = float(np.std(roll_vals))
                new_row["log_PRECTOTCORR"] = float(np.log1p(rainfall_mm))

                window_local = pd.concat([window_local, pd.DataFrame([new_row])], ignore_index=True).tail(7)

                # Try update scaled + X_input similarly to daily
                try:
                    last_row_scaled = scaler.transform(window_local)[-1:].reshape(1, -1)
                    X_input = np.expand_dims(scaler.transform(window_local)[-1:], axis=0)
                except Exception:
                    pass

            # AFTER the loop completes
            logger.info(f"MONTHLY prediction complete. Forecasts: {forecasts}")
            return {
                **state,
                "monthly_forecasts": forecasts
            }

    except Exception as exc:
        logger.exception("Unexpected error in model_prediction_agent: %s", exc)
        return {**state, "error": f"model_prediction_agent unexpected error: {exc}"}