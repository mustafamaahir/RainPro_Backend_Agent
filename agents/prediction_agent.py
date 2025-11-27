import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from langchain_core.runnables import RunnableConfig


FEATURES = [
    'RH2M', 'WS10M', 'T2M', 'WD10M',
    'ALLSKY_SFC_SW_DWN', 'EVPTRNS', 'PS',
    'QV2M', 'T2M_RANGE', 'TS', 'CLRSKY_SFC_SW_DWN'
]

TARGET_COL = "PRECTOTCORR"


# ===============================
# Helper: Inverse Transform
# ===============================
def inverse_transform_prediction(pred_scaled, scaler, last_row_scaled):
    temp = last_row_scaled.copy()
    temp[0, -1] = pred_scaled
    temp_inv = scaler.inverse_transform(temp)
    pred_log = temp_inv[0, -1]
    rainfall_mm = np.expm1(pred_log)
    return max(0, rainfall_mm)


# ===============================
# Main Prediction Agent
# ===============================
def model_prediction_agent(state: dict, config: RunnableConfig | None = None) -> dict:
    """
    Predict rainfall using trained models.
    Returns dict response.
    """

    # -------------------------------
    # Safety checks
    # -------------------------------
    if "intent" not in state:
        return {"error": "Missing intent in state"}

    intent = state.get("intent", {})
    mode = intent.get("mode", "daily").lower()

    window = state.get("preprocessed_window")
    X_input = state.get("preprocessed_data")
    final_features = state.get("final_features")
    scaled = state.get("scaled")

    if window is None or X_input is None or final_features is None or scaled is None:
        return {"error": "Missing required preprocessed data in state"}

    latitude = intent.get("latitude", 6.585)
    longitude = intent.get("longitude", 3.983)

    # ===============================
    # DAILY MODE
    # ===============================
    if mode == "daily":
        model = load_model("models/rainfall_daily_predictor.h5", compile=False)
        scaler = joblib.load("models/scaler_daily.pkl")

        forecasts = []
        days = int(intent.get("days", 7))

        for day in range(1, days + 1):

            # Model prediction
            pred_scaled = model.predict(X_input, verbose=0)[0][0]
            rainfall_mm = inverse_transform_prediction(
                pred_scaled,
                scaler,
                scaled[-1:].reshape(1, -1)
            )

            # Save forecast
            forecasts.append({
                "day": day,
                "predicted_rainfall_mm": round(float(rainfall_mm), 3)
            })

            # Create new row
            new_row = {}
            for col in final_features:
                if col != "log_PRECTOTCORR":
                    new_row[col] = window[col].iloc[-1]

            lag1 = window["log_PRECTOTCORR"].iloc[-1]
            lag3 = window["log_PRECTOTCORR"].iloc[-3] if len(window) >= 3 else lag1
            lag7 = window["log_PRECTOTCORR"].iloc[-7] if len(window) >= 7 else lag1

            new_row["log_PRECTOTCORR_lag1"] = lag1
            new_row["log_PRECTOTCORR_lag3"] = lag3
            new_row["log_PRECTOTCORR_lag7"] = lag7

            rolling_window = list(window["log_PRECTOTCORR"].iloc[-6:]) + [np.log1p(rainfall_mm)]
            new_row["rain_rolling_mean"] = np.mean(rolling_window)
            new_row["rain_rolling_std"] = np.std(rolling_window)
            new_row["log_PRECTOTCORR"] = np.log1p(rainfall_mm)

            # Append row and trim window
            window = pd.concat([window, pd.DataFrame([new_row])], ignore_index=True).tail(15)

        return {
            "status": "success",
            "mode": "daily",
            "location": {"latitude": latitude, "longitude": longitude},
            "forecasts": forecasts
        }

    # ===============================
    # MONTHLY MODE
    # ===============================
    elif mode == "monthly":
        model = load_model("models/rainfall_monthly_predictor.h5", compile=False)
        scaler = joblib.load("models/scaler_monthly.pkl")

        forecasts = []
        months = int(intent.get("months", 6))

        for month in range(1, months + 1):

            pred_scaled = model.predict(X_input, verbose=0)[0][0]
            rainfall_mm = inverse_transform_prediction(
                pred_scaled,
                scaler,
                scaled[-1:].reshape(1, -1)
            )

            forecasts.append({
                "month_ahead": month,
                "predicted_rainfall_mm": round(float(rainfall_mm), 3)
            })

            # Create new row
            new_row = {}
            for col in final_features:
                if col != "log_PRECTOTCORR":
                    new_row[col] = window[col].iloc[-1]

            lag1 = window["log_PRECTOTCORR"].iloc[-1]
            lag3 = window["log_PRECTOTCORR"].iloc[-3] if len(window) >= 3 else lag1
            lag7 = window["log_PRECTOTCORR"].iloc[-7] if len(window) >= 7 else lag1

            new_row["log_PRECTOTCORR_lag1"] = lag1
            new_row["log_PRECTOTCORR_lag3"] = lag3
            new_row["log_PRECTOTCORR_lag7"] = lag7

            roll_vals = list(window["log_PRECTOTCORR"].iloc[-2:]) + [np.log1p(rainfall_mm)]
            new_row["rain_rolling_mean"] = np.mean(roll_vals)
            new_row["rain_rolling_std"] = np.std(roll_vals)
            new_row["log_PRECTOTCORR"] = np.log1p(rainfall_mm)

            window = pd.concat([window, pd.DataFrame([new_row])], ignore_index=True).tail(7)

        return {
            "status": "success",
            "mode": "monthly",
            "location": {"latitude": latitude, "longitude": longitude},
            "monthly_forecasts": forecasts
        }

    # ===============================
    # INVALID MODE
    # ===============================
    else:
        return {
            "status": "error",
            "message": f"Invalid mode '{mode}'. Use 'daily' or 'monthly'."
        }
