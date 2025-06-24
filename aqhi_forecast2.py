import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor
from io import StringIO
import pytz

# Load AQHI data
def load_aqhi_data():
    url = "https://raw.githubusercontent.com/DKevinM/AB_datapull/main/data/last6h.csv"
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    
    df["ParameterName"] = df["ParameterName"].fillna("AQHI")
    df["ReadingDate"] = pd.to_datetime(df["ReadingDate"], utc=True)
    df["ReadingDate"] = df["ReadingDate"].dt.tz_convert("America/Edmonton")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    
    # Filter AQHI
    aqhi_now = df[df["ParameterName"] == "AQHI"].copy()
    aqhi_now = aqhi_now.dropna(subset=["Value", "Latitude", "Longitude"])
    aqhi_now = aqhi_now.sort_values("ReadingDate").groupby("StationName").tail(1)
    
    return aqhi_now

# Fetch forecast data for a station
def get_forecast_weather(station_name, lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m",
        "forecast_days": 1,
        "timezone": "America/Edmonton"
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        times = pd.to_datetime(data['hourly']['time']).tz_localize('UTC').tz_convert("America/Edmonton")
        now = pd.Timestamp.now(tz="America/Edmonton")

        future_idx = np.where(times > now)[0][:3]  # next 3 hours
        if len(future_idx) < 3:
            return None
        
        df = pd.DataFrame({
            "StationName": [station_name] * 3,
            "ForecastTime": times[future_idx].values,
            "temperature": np.array(data["hourly"]["temperature_2m"])[future_idx],
            "humidity": np.array(data["hourly"]["relative_humidity_2m"])[future_idx],
            "windspeed": np.array(data["hourly"]["windspeed_10m"])[future_idx],
            "Latitude": lat,
            "Longitude": lon
        })

        return df
    except Exception as e:
        print(f"Forecast failed for {station_name}: {e}")
        return None

# Prepare dataset
def build_model_and_predict_with_lags(aqhi_now, historical_df):
    df = historical_df.copy()
    df["ReadingDate"] = pd.to_datetime(df["ReadingDate"], utc=True).dt.tz_convert("America/Edmonton")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["ParameterName"] = df["ParameterName"].fillna("AQHI")

    # Filter relevant parameters
    df = df[df["ParameterName"].isin(["AQHI", "Outdoor Temperature", "Relative Humidity", "Wind Speed"])]
    
    # Pivot to wide format
    df_wide = df.pivot_table(
        index=["StationName", "ReadingDate", "Latitude", "Longitude"],
        columns="ParameterName",
        values="Value"
    ).reset_index()

    # Sort and create lag features
    df_wide = df_wide.sort_values(by=["StationName", "ReadingDate"])
    df_wide["AQHI_lag1"] = df_wide.groupby("StationName")["AQHI"].shift(1)
    df_wide["AQHI_lag2"] = df_wide.groupby("StationName")["AQHI"].shift(2)
    df_wide["AQHI_lag3"] = df_wide.groupby("StationName")["AQHI"].shift(3)

    # Drop rows with NA values in predictors or target
    model_df = df_wide.dropna(subset=[
        "Outdoor Temperature", "Relative Humidity", "Wind Speed", "AQHI", 
        "AQHI_lag1", "AQHI_lag2", "AQHI_lag3"
    ])

    # Predictors and target
    X = model_df[["Outdoor Temperature", "Relative Humidity", "Wind Speed", 
                  "AQHI_lag1", "AQHI_lag2", "AQHI_lag3"]]
    y = model_df["AQHI"]

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

# Main
if __name__ == "__main__":
    print("Loading AQHI data...")
    aqhi_now = load_aqhi_data()

    print("Fetching weather forecasts...")
    forecast_list = []
    for _, row in aqhi_now.iterrows():
        out = get_forecast_weather(row["StationName"], row["Latitude"], row["Longitude"])
        if out is not None:
            forecast_list.append(out)

    forecast_df = pd.concat(forecast_list, ignore_index=True)
    print("Forecast data collected.")

    # Reload the full CSV for historical training data
    print("Reloading full dataset for model training...")
    full_df = requests.get("https://raw.githubusercontent.com/DKevinM/AB_datapull/main/data/last6h.csv")
    historical_df = pd.read_csv(StringIO(full_df.text))

    print("Training model...")
    model = build_model_and_predict(aqhi_now, historical_df)

    print("Predicting AQHI for next 3 hours...")
    forecast_df["predicted_AQHI"] = model.predict(
        forecast_df[["temperature", "humidity", "windspeed"]]
    )

    print(forecast_df[["StationName", "ForecastTime", "predicted_AQHI"]].head())

    # You can now plug `forecast_df` into your IDW grid logic



