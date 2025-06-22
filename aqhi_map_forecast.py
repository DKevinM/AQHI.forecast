import geopandas as gpd
import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import requests
from io import StringIO


def get_aqhi_color(val):
    try:
        if isinstance(val, str) and val.strip() == "10+":
            return "#640100"
        v = int(np.ceil(float(val)))  # round up and ensure integer
        if v < 1:
            return "#D3D3D3"
        elif v == 1:
            return "#01cbff"
        elif v == 2:
            return "#0099cb"
        elif v == 3:
            return "#016797"
        elif v == 4:
            return "#fffe03"
        elif v == 5:
            return "#ffcb00"
        elif v == 6:
            return "#ff9835"
        elif v == 7:
            return "#fd6866"
        elif v == 8:
            return "#fe0002"
        elif v == 9:
            return "#cc0001"
        elif v == 10:
            return "#9a0100"
        else:
            return "#640100"  # >10
    except:
        return "#D3D3D3"



# 1. Load CSV data from GitHub (make sure it's raw URL)
url = 'https://raw.github.com/DKevinM/AB_datapull/main/data/last6h.csv'
response = requests.get(url)
df = pd.read_csv(StringIO(response.text))


df["ParameterName"] = df["ParameterName"].apply(lambda x: "AQHI" if pd.isna(x) or str(x).strip() == "" else x)
df["ReadingDate"] = pd.to_datetime(df["ReadingDate"])
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df.dropna(subset=["Value", "Latitude", "Longitude"])


def prepare_forecast_features(df, lags=3):
    df = df[df["ParameterName"].isin(["AQHI", "Outdoor Temperature", "Relative Humidity", "Wind Speed"])].copy()

    # Parse and convert timestamps to Alberta local time
    df["ReadingDate"] = pd.to_datetime(df["ReadingDate"], utc=True).dt.tz_convert("America/Edmonton")
    df["ReadingDate"] = df["ReadingDate"].dt.tz_localize(None)

    aqhi_now_df = (
        df[df["ParameterName"] == "AQHI"]
        .sort_values("ReadingDate")
        .groupby("StationName", as_index=False)
        .tail(1)
        .dropna(subset=["Value", "Latitude", "Longitude"])
        .rename(columns={"Value": "AQHI_now"})
    )

    import requests


    station_meta = df[["StationName", "Latitude", "Longitude"]].drop_duplicates()
    pivoted = df.pivot_table(index=["ReadingDate", "StationName"], columns="ParameterName", values="Value").reset_index()
    pivoted = pivoted.merge(station_meta, on="StationName", how="left")

    station_dfs = []
    for station in pivoted['StationName'].unique():
        station_data = pivoted[pivoted['StationName'] == station].sort_values("ReadingDate").copy()
        
        for lag in range(1, lags + 1):
            for col in ["AQHI", "Outdoor Temperature", "Relative Humidity", "Wind Speed"]:
                if col in station_data.columns:
                    station_data[f"{col}_lag{lag}"] = station_data[col].shift(lag)

        if "AQHI" in station_data.columns:
            station_data["AQHI_target"] = station_data["AQHI"]

        station_dfs.append(station_data.dropna(subset=["AQHI_target"] + [f"AQHI_lag{i}" for i in range(1, lags+1)]))

    result = pd.concat(station_dfs, ignore_index=True)
    keep_cols = ["StationName", "ReadingDate", "Latitude", "Longitude", "AQHI_target"] + \
                [col for col in result.columns if "lag" in col]
    return result[keep_cols]

    
    def get_forecast_weather(lat, lon):
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m,windspeed_10m",
                "forecast_days": 1,
                "timezone": "America/Edmonton"
            }
            response = requests.get(url, params=params)
            data = response.json()
    
            df = pd.DataFrame({
                "time": pd.to_datetime(data["hourly"]["time"]),
                "temperature": data["hourly"]["temperature_2m"],
                "humidity": data["hourly"]["relative_humidity_2m"],
                "windspeed": data["hourly"]["windspeed_10m"]
            })
            return df


from sklearn.ensemble import RandomForestRegressor

def forecast_next_3_hours(data):
    station_predictions = []

    for station in data['StationName'].unique():
        station_data = data[data['StationName'] == station].sort_values("ReadingDate")
        train_data = station_data.iloc[:-1]
        test_row = station_data.iloc[-1:].copy()

        if len(train_data) < 5:
            continue

        # Weather forecast fetch
        lat = test_row["Latitude"].values[0]
        lon = test_row["Longitude"].values[0]
        weather_forecast = get_forecast_weather(lat, lon)

        feature_cols = [
            col for col in train_data.columns
            if any(col.startswith(v + "_lag") for v in ["AQHI", "Outdoor Temperature", "Wind Speed", "Relative Humidity"])
        ]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        X_train = train_data[feature_cols]
        y_train = train_data["AQHI_target"]
        model.fit(X_train, y_train)

        preds = []

        for step in range(3):
            # Update weather lags
            t_step = test_row["ReadingDate"].values[0] + pd.Timedelta(hours=step + 1)
            row_weather = weather_forecast[weather_forecast["time"] == pd.Timestamp(t_step).round("h")]

            if not row_weather.empty:
                test_row["Outdoor Temperature_lag1"] = row_weather["temperature"].values[0]
                test_row["Relative Humidity_lag1"] = row_weather["humidity"].values[0]
                test_row["Wind Speed_lag1"] = row_weather["windspeed"].values[0]

            pred = model.predict(test_row[feature_cols])[0]
            preds.append(pred)

            # Shift AQHI lags forward
            for lag in range(3, 0, -1):
                from_col = f"AQHI_lag{lag - 1}" if lag > 1 else None
                to_col = f"AQHI_lag{lag}"
                test_row[to_col] = pred if lag == 1 else test_row[from_col].values[0]

        station_predictions.append({
            "StationName": station,
            "Latitude": test_row["Latitude"].values[0],
            "Longitude": test_row["Longitude"].values[0],
            "ForecastBaseTime": test_row["ReadingDate"].values[0],
            "AQHI_forecast_t1": preds[0],
            "AQHI_forecast_t2": preds[1],
            "AQHI_forecast_t3": preds[2]
        })

    return pd.DataFrame(station_predictions)


def generate_current_grid(df, shapefile_path, output_dir="output", cellsize=0.005):
    latest_aqhi = df[df["ParameterName"] == "AQHI"].sort_values("ReadingDate").groupby("StationName").tail(1)
    latest_aqhi = latest_aqhi.dropna(subset=["Value", "Latitude", "Longitude"])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    region = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    xmin, ymin, xmax, ymax = region.total_bounds
    x = np.arange(xmin, xmax, cellsize)
    y = np.arange(ymin, ymax, cellsize)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    point_geom = gpd.GeoSeries([Point(xy) for xy in grid_points], crs="EPSG:4326")
    inside_mask = point_geom.within(region.unary_union)
    grid_inside = grid_points[inside_mask.values]

    def idw(xy, values, grid_xy, power=2):
        dist = np.sqrt(((grid_xy[:, None, :] - xy[None, :, :])**2).sum(axis=2))
        with np.errstate(divide='ignore'):
            weights = 1 / dist**power
        weights[dist == 0] = 1e10
        weights_sum = weights.sum(axis=1)
        interp_values = (weights @ values) / weights_sum
        return interp_values

    xy = latest_aqhi[["Longitude", "Latitude"]].values
    values = latest_aqhi["Value"].values
    grid_values = idw(xy, values, grid_inside)

    polygons, aqhi_vals, colors = [], [], []

    for i, (x0, y0) in enumerate(grid_inside):
        poly = Polygon([
            (x0, y0),
            (x0 + cellsize, y0),
            (x0 + cellsize, y0 + cellsize),
            (x0, y0 + cellsize)
        ])

        val = grid_values[i]
        if np.isnan(val):
            val_ceiled = np.nan
            color = "#D3D3D3"
        else:
            val_ceiled = int(np.ceil(val))
            color = get_aqhi_color(val_ceiled)

        polygons.append(poly)
        aqhi_vals.append(val_ceiled)
        colors.append(color)

    gdf = gpd.GeoDataFrame({
        "value": aqhi_vals,
        "color": colors,
        "geometry": polygons
    }, crs="EPSG:4326")

    out_path = os.path.join(output_dir, "AQHI_now.geojson")
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"Saved: {out_path}")


def generate_and_save_forecast_grids(forecast_df, shapefile_path, output_dir="output", cellsize=0.005):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    region = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    xmin, ymin, xmax, ymax = region.total_bounds
    x = np.arange(xmin, xmax, cellsize)
    y = np.arange(ymin, ymax, cellsize)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    point_geom = gpd.GeoSeries([Point(xy) for xy in grid_points], crs="EPSG:4326")
    inside_mask = point_geom.within(region.unary_union)
    grid_inside = grid_points[inside_mask.values]


    def idw(xy, values, grid_xy, power=2):
        dist = np.sqrt(((grid_xy[:, None, :] - xy[None, :, :])**2).sum(axis=2))
        with np.errstate(divide='ignore'):
            weights = 1 / dist**power
        weights[dist == 0] = 1e10
        weights_sum = weights.sum(axis=1)
        interp_values = (weights @ values) / weights_sum
        return interp_values

    time_labels = {
        "AQHI_forecast_t1": "interpolated_grid_t+1.geojson",
        "AQHI_forecast_t2": "interpolated_grid_t+2.geojson",
        "AQHI_forecast_t3": "interpolated_grid_t+3.geojson"
    }

    xy = forecast_df[["Longitude", "Latitude"]].values

    for col, filename in time_labels.items():
        if col not in forecast_df.columns:
            continue

        values = forecast_df[col].values
        grid_values = idw(xy, values, grid_inside)

        polygons = []
        colors = []
        aqhi_vals = []

        for i, (x0, y0) in enumerate(grid_inside):
            poly = Polygon([
                (x0, y0),
                (x0 + cellsize, y0),
                (x0 + cellsize, y0 + cellsize),
                (x0, y0 + cellsize)
            ])

            val = grid_values[i]
            if np.isnan(val):
                val_ceiled = np.nan
                color = "#D3D3D3"
            else:
                val_ceiled = int(np.ceil(val))
                color = get_aqhi_color(val_ceiled)

            polygons.append(poly)
            aqhi_vals.append(val_ceiled)
            colors.append(color)

        gdf = gpd.GeoDataFrame({
            "value": aqhi_vals,
            "color": colors,
            "geometry": polygons
        }, crs="EPSG:4326")

        out_path = os.path.join(output_dir, filename)
        gdf.to_file(out_path, driver="GeoJSON")
        print(f"Saved: {out_path}")



        # Run this as your final process:
        generate_current_grid(df, shapefile_path="data/Edm.shp", output_dir="output")
        
        forecast_df = forecast_next_3_hours(prepare_forecast_features(df))
        generate_and_save_forecast_grids(forecast_df, shapefile_path="data/Edm.shp", output_dir="output")
        
