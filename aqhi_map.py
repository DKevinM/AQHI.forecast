import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import requests
from io import StringIO


# 1. Load CSV data from GitHub (make sure it's raw URL)
url = 'https://raw.github.com/DKevinM/AB_datapull/main/data/last6h.csv'
response = requests.get(url)
df = pd.read_csv(StringIO(response.text))
df = df[df["ParameterName"].isna() | (df["ParameterName"] == "")]
df["ReadingDate"] = pd.to_datetime(df["ReadingDate"])

# Get latest reading per station
latest_df = df.sort_values("ReadingDate").groupby("StationName").tail(1)
# Drop rows with missing info
latest_df = latest_df.dropna(subset=["Value", "Latitude", "Longitude"])

# 6. Load Alberta shapefile
alberta = gpd.read_file("data/Strathcona.shp").to_crs("EPSG:4326")


# Filter points inside Alberta
gdf = gpd.GeoDataFrame(latest_df, geometry=gpd.points_from_xy(latest_df.Longitude, latest_df.Latitude), crs='EPSG:4326')
gdf = gpd.overlay(gdf, alberta, how="intersection")  # <-- THIS


# 3. Generate grid
# Use Alberta shape for grid bounds
xmin, ymin, xmax, ymax = alberta.total_bounds
cellsize = 0.005  # degrees (~0.5 km)
x = np.arange(xmin, xmax, cellsize)
y = np.arange(ymin, ymax, cellsize)
xx, yy = np.meshgrid(x, y)
grid_points = np.c_[xx.ravel(), yy.ravel()]

# 4. Perform IDW
def idw(xy, values, grid_xy, power=2):
    dist = np.sqrt(((grid_xy[:, None, :] - xy[None, :, :])**2).sum(axis=2))
    with np.errstate(divide='ignore'):
        weights = 1 / dist**power
    weights[dist == 0] = 1e10  # handle divide by zero
    weights[dist == 0] = weights.max()
    weights_sum = weights.sum(axis=1)
    interp_values = (weights @ values) / weights_sum
    return interp_values

xy = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
values = gdf['Value'].values  # replace with actual column
grid_values = idw(xy, values, grid_points)

# 5. Convert grid points to polygons
polygons = []
vals = []
for i in range(len(grid_points)):
    x0, y0 = grid_points[i]
    poly = Polygon([
        (x0, y0),
        (x0 + cellsize, y0),
        (x0 + cellsize, y0 + cellsize),
        (x0, y0 + cellsize)
    ])
    polygons.append(poly)
    vals.append(grid_values[i])
    
grid_gdf = gpd.GeoDataFrame({'value': vals}, geometry=polygons, crs="EPSG:4326")

grid_gdf = gpd.overlay(grid_gdf, alberta, how="intersection")


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
        
grid_gdf['aqhi_str'] = grid_gdf['value'].apply(validate_aqhi)
grid_gdf['hex_color'] = grid_gdf['aqhi_str'].apply(get_aqhi_color)

grid_gdf[['aqhi_str', 'hex_color', 'geometry']].to_file("interpolated_grid.geojson", driver="GeoJSON")
