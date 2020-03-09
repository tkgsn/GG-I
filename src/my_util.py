import osmnx as ox
import os
import joblib
import numpy as np
import pyproj

"""
def load_data(location_name):
    graph_dir = "data/graph/"
    data_dir = "data/graph_data"
    
    G = ox.load_graphml(f'{location_name}.ml',folder=graph_dir)
    H = ox.load_graphml(f'sub_{location_name}.ml',folder=graph_dir)
    dict_sd = joblib.load(os.path.join(data_dir, f"dict_sd_{location_name}.jbl"))
    
    return G, H, dict_sd
"""

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