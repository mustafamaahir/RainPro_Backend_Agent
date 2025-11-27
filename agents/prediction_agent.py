import numpy as np
import joblib
from tensorflow.keras.models import load_model

def inverse_transform_prediction(pred_scaled, scaler, last_row_scaled):
    temp = last_row_scaled.copy()
    temp[0,-1] = pred_scaled
    temp_inv = scaler.inverse_transform(temp)
    pred_log = temp_inv[0,-1]
    rainfall_mm = np.expm1(pred_log)
    return max(0, rainfall_mm)

def model_prediction_agent(state: dict, config=None):
    """
    Predict daily or monthly rainfall using pre-trained models and preprocessed data.
    Returns updated state with 'forecasts' or 'monthly_forecasts'.
    """
    intent = state.get("intent", {})
    mode = intent.get("mode", "daily").lower()
    X_input = state.get("preprocessed_data")
    window = state.get("preprocessed_window")
    final_features = state.get("final_features")
    scaled = state.get("scaled")

    if None in [X_input, window, final_features, scaled]:
        state["error"] = "Preprocessed data missing."
        return state

    # Load model & scaler
    if config is None or "configurable" not in config:
        state["error"] = "Model config missing."
        return state

    model_path = config.configurable.get("models/rainfall_daily_predictor.h5") if mode=="daily" else config.configurable.get("models/rainfall_monthly_predictor.h5")
    scaler_path = config.configurable.get("models/scaler_daily.pkl") if mode=="daily" else config.configurable.get("models/scaler_monthly.pkl")

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)

    forecasts = []
    steps = intent.get("days",7) if mode=="daily" else intent.get("months",6)

    for step in range(1, steps+1):
        pred_scaled = model.predict(X_input, verbose=0)[0][0]
        rainfall_mm = inverse_transform_prediction(pred_scaled, scaler, scaled[-1:].reshape(1,-1))
        forecasts.append({
            "day" if mode=="daily" else "month_ahead": step,
            "predicted_rainfall_mm": round(float(rainfall_mm),3)
        })

        # update window for next step
        new_row = window.iloc[-1:].copy()
        new_row["log_PRECTOTCORR"] = np.log1p(rainfall_mm)
        window = window.append(new_row).tail(len(window))

    state["forecasts" if mode=="daily" else "monthly_forecasts"] = forecasts
    return state
