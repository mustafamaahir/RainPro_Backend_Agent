import numpy as np
from langchain_core.runnables import RunnableConfig

FEATURES = [
    'RH2M', 'WS10M', 'T2M', 'WD10M',
    'ALLSKY_SFC_SW_DWN', 'EVPTRNS', 'PS',
    'QV2M', 'T2M_RANGE', 'TS', 'CLRSKY_SFC_SW_DWN'
]
TARGET_COL = "PRECTOTCORR"

def preprocessing_agent(state: dict, config: RunnableConfig | None = None):
    """
    Preprocess NASA data for daily/monthly prediction.
    Includes cleaning, log-transform, lag features, rolling mean/std.
    """

    df = state.get("nasa_parameters")
    if df is None or df.empty:
        return {"error": "No NASA data in state."}

    mode = state.get("intent", {}).get("mode", "daily").lower()

    # Load scaler from config
    if config is None or "configurable" not in config:
        return {"error": "Scaler/model config missing."}
    scaler_path = config.configurable.get("models/scaler_daily.pkl") if mode=="daily" else config.configurable.get("models/scaler_monthly.pkl")
    import joblib
    scaler = joblib.load(scaler_path)

    # Fill missing columns
    for col in FEATURES + [TARGET_COL]:
        if col not in df.columns:
            df[col] = 0
    df = df.replace(-999.0, np.nan).ffill().bfill()

    # Log-transform
    df["log_PRECTOTCORR"] = np.log1p(df[TARGET_COL])
    for col in FEATURES:
        df[f"log_{col}"] = np.log1p(df[col])

    # Lag features
    for lag in [1,3,7]:
        df[f"log_PRECTOTCORR_lag{lag}"] = df["log_PRECTOTCORR"].shift(lag)

    # Rolling
    window_size = 7 if mode=="daily" else 3
    df["rain_rolling_mean"] = df["log_PRECTOTCORR"].rolling(window=window_size).mean()
    df["rain_rolling_std"] = df["log_PRECTOTCORR"].rolling(window=window_size).std()
    df = df.dropna().copy()

    if len(df) < window_size:
        return {"error": f"Not enough data to compute rolling window for {mode} mode."}

    # Prepare features for model
    final_features = [f"log_{col}" for col in FEATURES] + [
        "log_PRECTOTCORR_lag1","log_PRECTOTCORR_lag3","log_PRECTOTCORR_lag7",
        "rain_rolling_mean","rain_rolling_std"
    ]
    window_df = df[final_features + ["log_PRECTOTCORR"]].tail(window_size*2)  # keep extra rows for rolling

    # Scale
    scaled = scaler.transform(window_df)
    X_input = np.expand_dims(scaled, axis=0)

    state.update({
        "scaled": scaled,
        "final_features": final_features,
        "preprocessed_data": X_input,
        "preprocessed_window": window_df,
        "mode": mode
    })
    return state
