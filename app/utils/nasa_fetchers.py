import requests
import pandas as pd
from datetime import datetime
import numpy as np

REQUIRED_FEATURES = [
        'T2M','RH2M','WS10M','WD10M','ALLSKY_SFC_SW_DWN',
        'EVPTRNS','PS','QV2M','T2M_RANGE','TS','CLRSKY_SFC_SW_DWN','PRECTOTCORR'
    ]

def nasa_monthly(latitude=6.585, longitude=3.983, start_year=2022, end_year=2025):
    # Get current year-month to avoid future dates
    current_year_month = datetime.utcnow().strftime("%Y%m")

    params = {
        "parameters": "T2M,RH2M,WS10M,WD10M,ALLSKY_SFC_SW_DWN,EVPTRNS,PS,QV2M,T2M_RANGE,TS,CLRSKY_SFC_SW_DWN,PRECTOTCORR",
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": f"{start_year}",      # just the year
        "end": f"{end_year}",          # just the year
        "format": "JSON"
    }

    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Convert NASA JSON to DataFrame
    df = pd.DataFrame({k: pd.Series(v) for k, v in data["properties"]["parameter"].items()})

    # Keep only valid YYYYMM keys
    valid_keys = [k for k in df.index if len(k) == 6 and k.isdigit()]
    df = df.loc[valid_keys]

    # Convert to datetime safely
    df.index = pd.to_datetime(df.index, format="%Y%m", errors="coerce")
    df = df[~df.index.isna()]

    # Loop through required features
    for col in REQUIRED_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # Handle missing/future values
    df = df[REQUIRED_FEATURES]
    df = df.replace(-999.0, np.nan)   # Replace NASA missing flag with NaN
    df = df.ffill().bfill()           # Fills missing values forward/backward

    return df


def nasa_daily(latitude=6.585, longitude=3.983, days=1):
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)
    
    params = {
        "parameters": "T2M,RH2M,WS10M,WD10M,ALLSKY_SFC_SW_DWN,EVPTRNS,PS,QV2M,T2M_RANGE,TS,CLRSKY_SFC_SW_DWN,PRECTOTCORR",
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON"
    }

    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data["properties"]["parameter"])
    df = pd.DataFrame({k: pd.Series(v) for k, v in df.items()})
    df.index = pd.to_datetime(df.index)

    # Loop through required features for confirmation
    for col in REQUIRED_FEATURES:
        if col not in df.columns:
            df[col] = 0

    df = df[REQUIRED_FEATURES].ffill().bfill()
    return df
