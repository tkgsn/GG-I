import osmnx as ox
import numpy as np
import os
import networkx as nx
from src.my_util import LatlonRange, graph_dir, spd_dir, prior_dir, ed_dir
from src.graph_maker import MapGraphMaker, KyotoMapGraphMaker, TruncatedMapGraphMaker
import json

from logging import getLogger, config
with open('/GG-I/data/log_config.json', 'r') as f:
    log_conf = json.load(f)
config.dictConfig(log_conf)
logger = getLogger("sub_module")

BUS = "bus"
UNIFORM = "uniform"

with open('/GG-I/data/location_data.json', 'r') as f:
    location_data = json.load(f)

class DataLoader():
    
    def __init__(self, location_name, **args):
        if location_name == "Kyoto":
            graph, sub_graph, spd, ed, prior = self.load_kyoto(args["latlon_range"], args["prior_type"])
        else:
            lat = location_data[location_name]["lat"]
            lon = location_data[location_name]["lon"]
            graph, sub_graph, spd, ed, prior = self.load(lat, lon, args["distance"], args["prior_type"], args["simplify"], args["n_graph_nodes"])
            
        self.G = graph
        self.H = sub_graph
        self.g_index = self._make_g_index(graph)
        nodes = list(graph.nodes)
        self.index_g = {i:nodes[i] for i in range(len(nodes))}
        self.h_index = self._make_h_index(graph, sub_graph)
        
        self.np_spd, self.np_sub_spd, self.np_sub_sub_spd = self._convert_to_np_spd(graph, sub_graph, spd)
        self.np_ed, self.np_sub_ed, self.np_sub_sub_ed = self._convert_to_np_spd(graph, sub_graph, ed)
        self.np_pr, self.np_sub_pr = self._cp_np_pr(graph, sub_graph, prior)
        
    def load(self, lat, lon, distance, prior_type, simplify=False, n_graph_nodes=0):
        
        gm = TruncatedMapGraphMaker(lat, lon, distance, prior_type, simplify, n_chosen=n_graph_nodes)
        
        if gm.check_existence():
            graph, sub_graph, spd, ed, prior = gm.load()
                
        else:
            logger.info(f"constructing graph and computing auxiliary information take some time")
            logger.info(f"constructing graph")
            graph, sub_graph = gm.make_graph(lat, lon, distance, simplify, n_chosen=n_graph_nodes)
            graph = nx.relabel_nodes(graph, str)
            sub_graph = nx.relabel_nodes(sub_graph, str)
            logger.info(f"computing shortest path distances")
            spd = gm.compute_shortest_path_distance_dict(graph)
            logger.info(f"computing euclidean distances")
            ed = gm.compute_euclidean_distance_dict(graph)
            if prior_type == UNIFORM:
                prior = gm.make_uniform_prior_distribution(graph, sub_graph)
            
            gm.save(graph, sub_graph, spd, ed, prior)
            
        logger.info(f"the number of nodes all: {len(graph)}, sub: {len(sub_graph)}")
        
        return graph, sub_graph, spd, ed, prior
        
    
    def _cp_np_pr(self, G, H, prior):
        np_pr = np.zeros((len(G.nodes), 1))
        np_sub_pr = np.zeros((len(H.nodes), 1))
        
        for i, node in enumerate(H.nodes):
            np_pr[self.g_index[node], 0] = prior[node]
            np_sub_pr[self.h_index[node], 0] = prior[node]
        return np_pr, np_sub_pr
    
            
    def load_kyoto(self, latlon_range, prior_type):
        data_name = f"Kyoto_{latlon_range.min_lat}_{latlon_range.max_lat}_{latlon_range.min_lon}_{latlon_range.max_lon}"
        self.data_name = data_name
        
        graph_data_dir = graph_dir / f"{data_name}.ml"
        spd_data_dir = spd_dir / f"{data_name}.json"
        prior_data_dir = prior_dir /  f"{data_name}_{prior_type}.json"
        
        if graph_data_dir.exists() and spd_data_dir.exists() and prior_data_dir.exists():
            logger.info(f"loading graph from {graph_data_dir}")
            logger.info(f"loading spd from {spd_data_dir}")
            logger.info(f"loading prior from {prior_data_dir}")
            
            graph = ox.load_graphml(graph_data_dir)
            graph = nx.relabel_nodes(graph, str)
            with open(spd_data_dir, "r") as f:
                spd = json.load(f)
            with open(prior_data_dir, "r") as f:
                prior = json.load(f)
            
        else:
            logger.info(f"constructing graph and computing auxiliary information take some time")
            gm = KyotoMapGraphMaker()
            graph, sub_graph = gm.make_graph(latlon_range)
            spd = gm.compute_shortest_path_distance_dict(graph)
            graph = nx.relabel_nodes(graph, str)
            ox.save_graphml(graph, graph_data_dir)
            with open(spd_data_dir, "w") as f:
                json.dump(spd, f)
            
            if prior_type == BUS:
                prior = gm.make_prior_distribution(graph, spd)
            with open(prior_data_dir, "w") as f:
                json.dump(prior, f)
            logger.info(f"saved graph to {graph_data_dir}")
            logger.info(f"saved spd to {spd_data_dir}")
            logger.info(f"saved prior to {prior_data_dir}")
        
        return graph, graph, spd, None, prior
    
    def _convert_to_np_spd(self, G, H, spd):
            
        np_spd = np.array([[spd[dic_node][node] for node in G.nodes] for dic_node in G.nodes])
        np_sub_spd = np.array([[spd[dic_node][node] for node in G.nodes] for dic_node in G.nodes if dic_node in H.nodes])
        np_sub_sub_spd = np.array([[spd[dic_node][node] for node in G.nodes if node in H.nodes] for dic_node in G.nodes if dic_node in H.nodes])   
        return np_spd, np_sub_spd, np_sub_sub_spd
    
    def _compute_Euclidean_distance(self, coords, ids):
        euclidean_distances = {id:[np.linalg.norm(coords[id]-coords[id_in]) for id_in in ids] for id in ids}
        return euclidean_distances
    
    def _make_g_index(self, G):
        return {node:list(self.G.nodes()).index(node) for node in self.G}
    
    def _make_h_index(self, G, H):
        h_node = [node for node in G.nodes if node in H.nodes]
        return {node:h_node.index(node) for node in H.nodes}