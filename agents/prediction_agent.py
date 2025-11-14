# graph/agents/model_prediction_agent.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

FEATURES = [
    'RH2M', 'WS10M', 'T2M', 'WD10M',
    'ALLSKY_SFC_SW_DWN', 'EVPTRNS', 'PS',
    'QV2M', 'T2M_RANGE', 'TS', 'CLRSKY_SFC_SW_DWN'
]

TARGET_COL = "PRECTOTCORR"


# Helper function for inverse
def inverse_transform_prediction(pred_scaled, scaler, last_row_scaled):
    """
    Reverse MinMax scaling and log1p transformation for the target.
    pred_scaled: output of model in scaled space
    scaler: fitted MinMaxScaler used on the features (including log_PRECTOTCORR)
    last_row_scaled: the last row of input scaled data (needed to preserve scaling shape)
    """
    temp = last_row_scaled.copy()
    temp[0, -1] = pred_scaled  # target is last column
    temp_inv = scaler.inverse_transform(temp)
    pred_log = temp_inv[0, -1]
    rainfall_mm = np.expm1(pred_log)
    return max(0, rainfall_mm)  # clip negatives


# Main prediction agent
def model_prediction_agent(state: dict, config: dict):
    """
    Predict rainfall using pre-trained Keras model.
    Supports both daily and monthly prediction.
    """
    intent = state.get("intent", {})
    mode = intent.get("mode", "daily").lower()
    window = state.get("preprocessed_window")
    X_input = state.get("preprocessed_data")
    final_features = state.get("final_features")
    scaled = state.get("scaled")

    latitude = intent.get("latitude", 6.585)
    longitude = intent.get("longitude", 3.983)

    if mode == "daily":
        model = load_model(r"models\rainfall_daily_predictor.h5", compile=False)
        scaler = config.get(r"models\scaler_daily.pkl")

        forecasts = []
        days = intent.get("days", 7)

        for day in range(1, days + 1):
            
            # Predict
            pred_scaled = model.predict(X_input)
            rainfall_mm = inverse_transform_prediction(pred_scaled, scaler, scaled[-1:].reshape(1, -1))

            # Append forecast
            forecasts.append({
                "day": day,
                "predicted_rainfall_mm": round(float(rainfall_mm), 3)
            })

            # Prepare new row for iterative prediction
            new_row = {}
            for col in final_features:
                if col != "log_PRECTOTCORR":
                    new_row[col] = window[col].iloc[-1]

            lag_values = [window["log_PRECTOTCORR"].iloc[-lag] for lag in [1, 3, 7]]
            new_row["log_PRECTOTCORR_lag1"] = lag_values[0]
            new_row["log_PRECTOTCORR_lag3"] = lag_values[1]
            new_row["log_PRECTOTCORR_lag7"] = lag_values[2]

            rolling_window = list(window["log_PRECTOTCORR"].iloc[-6:]) + [np.log1p(rainfall_mm)]
            new_row["rain_rolling_mean"] = np.mean(rolling_window)
            new_row["rain_rolling_std"] = np.std(rolling_window)
            new_row["log_PRECTOTCORR"] = np.log1p(rainfall_mm)

            # Append and trim
            window = pd.concat([window, pd.DataFrame([new_row])], ignore_index=True).tail(15)

        return {
            "forecasts": forecasts,
            "location": {"latitude": latitude, "longitude": longitude},
            "mode": "daily"
        }

    elif mode == "monthly":
        model = load_model("model/rainfall_monthly_predictor.h5", compile=False)
        scaler = config.get(r"models\scaler_monthly.pkl")

        forecasts = []
        months = intent.get("months", 6)

        for month in range(1, months + 1):

            # Predict
            pred_scaled = model.predict(X_input)
            rainfall_mm = inverse_transform_prediction(pred_scaled, scaler, scaled[-1:].reshape(1, -1))

            # Append forecast
            forecasts.append({
                "month_ahead": month,
                "predicted_rainfall_mm": round(float(rainfall_mm), 3)
            })

            # Prepare new row
            new_row = {col: window[col].iloc[-1] for col in final_features if col != "log_PRECTOTCORR"}
            lag_values = [window["log_PRECTOTCORR"].iloc[-lag] for lag in [1, 3, 7]]
            new_row.update({
                "log_PRECTOTCORR_lag1": lag_values[0],
                "log_PRECTOTCORR_lag3": lag_values[1],
                "log_PRECTOTCORR_lag7": lag_values[2],
                "rain_rolling_mean": np.mean(list(window["log_PRECTOTCORR"].iloc[-2:]) + [np.log1p(rainfall_mm)]),
                "rain_rolling_std": np.std(list(window["log_PRECTOTCORR"].iloc[-2:]) + [np.log1p(rainfall_mm)]),
                "log_PRECTOTCORR": np.log1p(rainfall_mm)
            })

            window = pd.concat([window, pd.DataFrame([new_row])], ignore_index=True).tail(7)

        return {
            "monthly_forecasts": forecasts,
            "location": {"latitude": latitude, "longitude": longitude},
            "mode": "monthly"
        }

    else:
        return {"error": f"Invalid mode '{mode}' for prediction."}
