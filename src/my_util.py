import osmnx as ox
import os
# import joblib
import numpy as np
import pyproj
import pathlib
from math import radians, cos, sin, asin, sqrt

data_dir = pathlib.Path(os.environ["DATA_DIR"])
graph_dir = data_dir / "graph"
graph_dir.mkdir(parents=True, exist_ok=True)
spd_dir = data_dir / "shortest_path_distance_dict"
spd_dir.mkdir(parents=True, exist_ok=True)
ed_dir = data_dir / "euclidean_distance_dict"
ed_dir.mkdir(parents=True, exist_ok=True)
prior_dir = data_dir / "prior_distribution"
prior_dir.mkdir(parents=True, exist_ok=True)
result_dir = pathlib.Path("results")
result_dir.mkdir(parents=True, exist_ok=True)
img_dir = pathlib.Path("imgs")
img_dir.mkdir(parents=True, exist_ok=True)


class LatlonRange():
    def __init__(self, min_lat, max_lat, min_lon, max_lon):
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        

def convert_key_to_str(dic):
    b = {}
    for key, dic_ in dic.items():
        b[str(key)] = {} 
    for key, dic_ in dic.items():
        for key_, value_ in dic_.items():
            b[str(key)][str(key_)] = value_
    return b

EPSG4612 = pyproj.Proj("+init=EPSG:4612")
EPSG2451 = pyproj.Proj("+init=EPSG:2451")

def convert_to_cart(lon, lat):
    x,y = pyproj.transform(EPSG4612, EPSG2451, lon,lat)
    return x,y


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    m = 6371* c * 1000
    return m