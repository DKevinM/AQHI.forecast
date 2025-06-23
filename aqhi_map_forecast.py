import requests
import geopandas as gpd
import pandas as pd
import numpy as np
import os
from datetime import datetime
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
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


def generate_current_grid(df, shapefile_path, output_dir="output", cellsize=0.005):
    latest_aqhi = df[df["ParameterName"] == "AQHI"].sort_values("ReadingDate").groupby("StationName").tail(1)
    latest_aqhi = latest_aqhi.dropna(subset=["Value", "Latitude", "Longitude"])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    else:
        print(f"Output directory already exists: {output_dir}")

    # Debug: show how many points we’re interpolating from
    print(f"Generating current grid from {len(latest_aqhi)} stations")

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

    print("Done generating GeoJSONs")

    
    out_path = os.path.join(output_dir, "AQHI_now.geojson")
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"Saved: {out_path}")


# Run this as your final process:
generate_current_grid(df, shapefile_path="data/Edm.shp", output_dir="output")
