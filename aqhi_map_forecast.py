import geopandas as gpd
import pandas as pd
import numpy as np
import os
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import requests
from io import StringIO


def get_aqhi_color(val):
    if isinstance(val, str) and val.strip() == "10+":
        return "#640100"
    try:
        v = float(val)
        if np.isnan(v) or v < 1:
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
        
def validate_aqhi(val):
    if np.isnan(val) or val < 1:
        return "NA"
    elif val > 10:
        return "10+"
    else:
        return str(int(round(val)))



# 1. Load CSV data from GitHub (make sure it's raw URL)
url = 'https://raw.github.com/DKevinM/AB_datapull/main/data/last6h.csv'
response = requests.get(url)
df = pd.read_csv(StringIO(response.text))
df["ParameterName"] = df["ParameterName"].apply(lambda x: "AQHI" if pd.isna(x) or str(x).strip() == "" else x)
df["ReadingDate"] = pd.to_datetime(df["ReadingDate"])

# Get latest reading per station
latest_df = df.sort_values("ReadingDate").groupby("StationName").tail(1)
# Drop rows with missing info
latest_df = latest_df.dropna(subset=["Value", "Latitude", "Longitude"])


def prepare_forecast_features(df, lags=3):
    # Ensure proper naming for AQHI
    df["ParameterName"] = df["ParameterName"].apply(
        lambda x: "AQHI" if pd.isna(x) or str(x).strip() == "" else x
    )
    df["ReadingDate"] = pd.to_datetime(df["ReadingDate"])

    # Pull lat/lon info
    station_meta = df[["StationName", "Latitude", "Longitude"]].drop_duplicates()

    # Pivot to wide format
    pivoted = df.pivot_table(index=["ReadingDate", "StationName"],
                             columns="ParameterName", values="Value").reset_index()

    # Merge lat/lon back in
    pivoted = pivoted.merge(station_meta, on="StationName", how="left")

    station_dfs = []

    for station in pivoted['StationName'].unique():
        station_data = pivoted[pivoted['StationName'] == station].sort_values("ReadingDate").copy()

        # Add lag features
        for lag in range(1, lags + 1):
            for col in [
                "AQHI",
                "Ozone",
                "Fine Particulate Matter",
                "Nitrogen Dioxide",
                "Outdoor Temperature",
                "Wind Speed",
                "Relative Humidity"
            ]:
                if col in station_data.columns:
                    station_data[f"{col}_lag{lag}"] = station_data[col].shift(lag)

        # Define prediction target
        if "AQHI" in station_data.columns:
            station_data["AQHI_target"] = station_data["AQHI"]

        # Keep only complete cases
        required_cols = [f"AQHI_lag{lag}" for lag in range(1, lags + 1)] + ["AQHI_target"]
        station_dfs.append(station_data.dropna(subset=required_cols))

    # Final output
    result = pd.concat(station_dfs, ignore_index=True)

    # Reorder for convenience
    cols = ["StationName", "ReadingDate", "Latitude", "Longitude"] + \
           [col for col in result.columns if col not in ["StationName", "ReadingDate", "Latitude", "Longitude"]]
    return result[cols]



from sklearn.ensemble import RandomForestRegressor

def forecast_next_3_hours(data):
    station_predictions = []

    for station in data['StationName'].unique():
        station_data = data[data['StationName'] == station].sort_values("ReadingDate")

        # Use all rows except most recent for training
        train_data = station_data.iloc[:-1]
        test_row = station_data.iloc[-1:].copy()

        feature_cols = [col for col in station_data.columns if 'lag' in col]
        
        if len(train_data) < 5 or test_row.empty:
            continue  # skip if not enough history
        
        # Train model
        X_train = train_data[feature_cols]
        y_train = train_data["AQHI_target"]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = []

        for step in range(3):
            # Predict next AQHI
            pred = model.predict(test_row[feature_cols])[0]
            preds.append(pred)

            # Shift lag values forward for AQHI only
            for lag in range(len(feature_cols), 0, -1):
                colname = f"AQHI_lag{lag}"
                if lag == 1:
                    test_row[colname] = pred
                else:
                    test_row[f"AQHI_lag{lag}"] = test_row.get(f"AQHI_lag{lag-1}", test_row[colname]).values[0]

        station_predictions.append({
            "StationName": station,
            "Latitude": test_row["Latitude"].values[0],
            "Longitude": test_row["Longitude"].values[0],
            "AQHI_forecast_t1": preds[0],
            "AQHI_forecast_t2": preds[1],
            "AQHI_forecast_t3": preds[2]
        })

    return pd.DataFrame(station_predictions)


def generate_and_save_forecast_grids(forecast_df, shapefile_path, output_dir="output", cellsize=0.005):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load region shapefile and create grid
    region = gpd.read_file("data/Strathcona.shp").to_crs("EPSG:4326")
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
        "AQHI_now": "interpolated_grid_now.geojson",
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

        # Convert grid values to polygons
        polygons = []
        vals = []
        for i, (x0, y0) in enumerate(grid_inside):
            poly = Polygon([
                (x0, y0),
                (x0 + cellsize, y0),
                (x0 + cellsize, y0 + cellsize),
                (x0, y0 + cellsize)
            ])
            polygons.append(poly)
            vals.append(grid_values[i])

        print(f"✅ Forecast DataFrame shape: {forecast_df.shape}")
        print("Columns:", forecast_df.columns.tolist())
        print("Sample:", forecast_df.head())

        grid_gdf = gpd.GeoDataFrame({'value': vals}, geometry=polygons, crs="EPSG:4326")
        grid_gdf = gpd.overlay(grid_gdf, region, how="intersection")
        grid_gdf['aqhi_str'] = grid_gdf['value'].apply(validate_aqhi)
        grid_gdf['hex_color'] = grid_gdf['aqhi_str'].apply(get_aqhi_color)

        out_file = os.path.join(output_dir, filename)
        grid_gdf[['aqhi_str', 'hex_color', 'geometry']].to_file(out_file, driver="GeoJSON")
