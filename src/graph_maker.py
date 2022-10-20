from pyproj import Geod
# import joblib
import pickle
import copy
import os
import networkx as nx
import osmnx as ox
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import src.my_util as util
import os
import pathlib
import json
import functools
from src.my_util import LatlonRange, graph_dir, spd_dir, prior_dir, ed_dir
from logging import getLogger, config
import random
from tqdm import tqdm

with open('/GG-I/data/log_config.json', 'r') as f:
    log_conf = json.load(f)
config.dictConfig(log_conf)
logger = getLogger("sub_module")

with open('/GG-I/data/location_data.json', 'r') as f:
    location_data = json.load(f)

class GraphMaker():
    def __init__(self):
        pass
    
    def make_graph(self):
        pass
    
    def compute_shortest_path_distance_dict(self, graph, weight="length"):
        # shortest_path_distance_dict = dict(nx.all_pairs_dijkstra_path_length(graph, weight=weight, cutoff=cutoff))
        shortest_path_distance_dict = dict(nx.all_pairs_dijkstra_path_length(graph, weight=weight))
        _shortest_path_distance_dict = {node:{} for node in graph.nodes}
        for node in graph.nodes:
            for node_ in graph.nodes:
                _shortest_path_distance_dict[node][node_] = shortest_path_distance_dict[node][node_]
        return _shortest_path_distance_dict

    def make_uniform_prior_distribution(self, graph, sub_graph):
        n_sub_graph_nodes = len(sub_graph)
        prior = {node:0 for node in graph.nodes()}
        for node in sub_graph.nodes():
            prior[node] = 1/n_sub_graph_nodes
        return prior

class MapGraphMaker(GraphMaker):
    def __init__(self):
        pass
    
    def register_dir(self, data_name, prior_type):
        self.graph_data_dir = graph_dir / f"{data_name}.ml"
        self.sub_graph_data_dir = graph_dir / f"sub_{data_name}.ml"
        self.spd_data_dir = spd_dir / f"{data_name}.json"
        self.ed_data_dir = ed_dir / f"{data_name}.json"
        self.prior_data_dir = prior_dir /  f"{data_name}_{prior_type}.json"
        
    def check_existence(self):
        return self.graph_data_dir.exists() and self.sub_graph_data_dir.exists() and self.spd_data_dir.exists() and self.prior_data_dir.exists() and self.ed_data_dir.exists()
    
    def compute_euclidean_distance_dict(self, graph):
        euclidean_distance_dict = {node:{node_in:util.haversine(float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"]), float(graph.nodes[node_in]["x"]), float(graph.nodes[node_in]["y"])) for node_in in graph.nodes()} for node in graph.nodes()}
        return euclidean_distance_dict
    
    def make_graph(self, lat, lon, distance, simplify, **kwargs):
        name = f"{lat}_{lon}_{distance}_symplify{simplify}.ml"
        sub_name = f"sub_{lat}_{lon}_{distance}_symplify{simplify}.ml"
        graph_data_dir = graph_dir / name
        sub_graph_data_dir = graph_dir / sub_name
        
        logger.info(f"downloading from OpenStreetMap with keyword ({lat},{lon}), distance={distance}, simplyfy={simplify} takes some time")
        graph = ox.graph_from_point((lat, lon),dist_type="network", dist=distance, network_type="walk", simplify=simplify)
        sub_graph = ox.graph_from_point((lat, lon),dist_type="network", dist=distance/2, network_type="walk", simplify=simplify)
        logger.info(f"n nodes\t main: {len(graph)}, sub {len(sub_graph)}")
            
        ox.plot_graph(graph, save=True, filepath=f"imgs/{name}.png")
        return graph, sub_graph
    
    def compute_range(self, graph):
        lats = [i[1]["y"] for i in graph.nodes(data=True)]
        lons = [i[1]["x"] for i in graph.nodes(data=True)]

        min_lat = min(lats)
        max_lat = max(lats)
        min_lon = min(lons)
        max_lon = max(lons)
        return min_lat, max_lat, min_lon, max_lon
    
    def load(self):
        
        logger.info(f"loading cached file from {self.graph_data_dir}")
        graph = ox.load_graphml(self.graph_data_dir)
        graph = nx.relabel_nodes(graph, str)
        sub_graph = ox.load_graphml(self.sub_graph_data_dir)
        sub_graph = nx.relabel_nodes(sub_graph, str)
        with open(self.spd_data_dir, "r") as f:
            spd = json.load(f)
        with open(self.prior_data_dir, "r") as f:
            prior = json.load(f)
        with open(self.ed_data_dir, "r") as f:
            ed = json.load(f)
            
        return graph, sub_graph, spd, ed, prior
    
    def save(self, graph, sub_graph, spd, ed, prior):

        ox.write_graphml(graph, self.graph_data_dir)
        ox.write_graphml(sub_graph, self.sub_graph_data_dir)
        with open(self.spd_data_dir, "w") as f:
            json.dump(spd, f)
        with open(self.ed_data_dir, "w") as f:
            json.dump(ed, f)
        with open(self.prior_data_dir, "w") as f:
            json.dump(prior, f)
        logger.info(f"saved graph to {self.graph_data_dir}")
        logger.info(f"saved spd to {self.spd_data_dir}")
        logger.info(f"saved ed to {self.ed_data_dir}")
        logger.info(f"saved prior to {self.prior_data_dir}")

class TruncatedMapGraphMaker(MapGraphMaker):

    def __init__(self, **kwargs):
        location_name = kwargs["location"]
        lat = location_data[location_name]["lat"]
        lon = location_data[location_name]["lon"]
        distance = kwargs["distance"]
        prior_type = kwargs["prior_type"]
        simplify = kwargs["simplify"]
        n_graph_nodes = kwargs["n_graph_nodes"]
        
        self.original_data_name = f"{location_name}_{distance}_simplify{simplify}"
        self.original_graph_data_dir = graph_dir / f"{self.original_data_name}.ml"
        
        self.data_name = f"{location_name}_{distance}_simplify{simplify}_ngraphnodes{n_graph_nodes}"
        self.register_dir(self.data_name, prior_type)
        
        if self.original_graph_data_dir.exists():
            logger.info(f"loading original graph from {self.original_graph_data_dir}")
            graph = ox.load_graphml(self.original_graph_data_dir)
            self.original_graph = nx.relabel_nodes(graph, str)
        else:
            logger.info(f"constructing original graph")
            graph, sub_graph = super().make_graph(lat, lon, distance, simplify)
            self.original_graph = nx.relabel_nodes(graph, str)
            logger.info(f"saving original graph to {self.original_graph_data_dir}")
            ox.save_graphml(graph, self.original_graph_data_dir)
            
        
    # def make_graph(self, lat, lon, distance, simplify, **kwargs):
    def make_graph(self, **kwargs):
        distance = kwargs["distance"]
        simplify = kwargs["simplify"]
        n_chosen = kwargs["n_graph_nodes"]
        
        all_nodes = list(self.original_graph.nodes(data=True))
        
        if n_chosen == 0:
            nodes = all_nodes
        else:
            random.seed(0)
            nodes = []
            while (len(nodes) < n_chosen) and (len(nodes) != len(all_nodes)):
                choiced = random.choice(all_nodes)
                if choiced not in nodes:
                    nodes.append(choiced)
        
        logger.info(f"truncated by {len(nodes)} from {len(all_nodes)}")

        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.graph["crs"] = self.original_graph.graph["crs"]

        return graph, graph

    def compute_shortest_path_distance_dict(self, graph, weight="length"):

        shortest_path_distance_dict = {node:{} for node in graph.nodes()}
        for node in tqdm(graph.nodes()):
            shortest_path_distances = nx.single_source_dijkstra_path_length(self.original_graph, node, weight=weight)
            shortest_path_distance_dict[node] = {node:shortest_path_distances[node] for node in graph.nodes()}
        return shortest_path_distance_dict

    def save(self, graph, sub_graph, spd, ed, prior):

        nx.write_graphml(graph, self.graph_data_dir)
        nx.write_graphml(sub_graph, self.sub_graph_data_dir)
        with open(self.spd_data_dir, "w") as f:
            json.dump(spd, f)
        with open(self.ed_data_dir, "w") as f:
            json.dump(ed, f)
        with open(self.prior_data_dir, "w") as f:
            json.dump(prior, f)
        logger.info(f"saved graph to {self.graph_data_dir}")
        logger.info(f"saved spd to {self.spd_data_dir}")
        logger.info(f"saved ed to {self.ed_data_dir}")
        logger.info(f"saved prior to {self.prior_data_dir}")

k = 0.001
class KyotoMapGraphMaker(MapGraphMaker):
    
    MIN_LAT = 34.8955054
    MAX_LAT = 35.301761
    MIN_LON = 135.5617132
    MAX_LON = 135.8579904
    ALTITUDE = 0
    g = Geod(ellps='WGS84')
    
    @classmethod
    def max_height(self):
        return KyotoGraphMaker.g.inv(KyotoGraphMaker.MIN_LON, KyotoGraphMaker.MIN_LAT, KyotoGraphMaker.MIN_LON, KyotoGraphMaker.MAX_LAT)[2]
    @classmethod
    def max_width(self):
        return KyotoGraphMaker.g.inv(KyotoGraphMaker.MIN_LON, KyotoGraphMaker.MIN_LAT, KyotoGraphMaker.MAX_LON, KyotoGraphMaker.MIN_LAT)[2]

    def graph_width(self):
        return KyotoGraphMaker.g.inv(self.min_lon, self.min_lat, self.max_lon, self.min_lat)[2]
    
    def graph_height(self):
        return KyotoGraphMaker.g.inv(self.min_lon, self.min_lat, self.min_lon, self.max_lat)[2]
    
    def __init__(self, **kwargs):
        self.kyoto_graph = self._load_kyoto_graph()
        
        self.data_name = f"{kwargs['min_lat']}_{kwargs['max_lat']}_{kwargs['min_lon']}_{kwargs['max_lon']}"
        self.register_dir(self.data_name, kwargs["prior_type"])
        
    def _load_kyoto_graph(self):
        kyoto_graph_path = graph_dir / "kyotoshi.ml"
        if kyoto_graph_path.exists():
            logger.info(f"loading cached Kyoto graph from {kyoto_graph_path}")
            kyoto_graph = ox.load_graphml(kyoto_graph_path)
            kyoto_graph = nx.relabel_nodes(kyoto_graph, str)
        else:
            logger.info("downloading from OpenStreetMap with keyword \"Kyoto\" takes some time")
            kyoto_graph = ox.graph_from_place("Kyoto", simplify=True)
            kyoto_graph = nx.relabel_nodes(kyoto_graph, str)
            ox.save_graphml(kyoto_graph, kyoto_graph_path)
        return kyoto_graph

    def make_graph(self, **kwargs):

        truncated_graph = copy.deepcopy(self.kyoto_graph)
        for node in self.kyoto_graph.nodes(data=True):
            lat = node[1]["y"]
            lon = node[1]["x"]
            if (kwargs["min_lat"] > lat) or (kwargs["max_lat"] < lat) or (kwargs["min_lon"] > lon) or (kwargs["max_lon"] < lon):
                truncated_graph.remove_node(node[0])
        truncated_graph = truncated_graph.to_undirected()      
    
        return truncated_graph, truncated_graph
        
    def _load_shibus_data(self, graph):
        min_lat, max_lat, min_lon, max_lon = self.compute_range(graph)
        with open("/GG-I/data/shibus_data.json", "r") as f:
            shibus_data = json.load(f)
            
        shibus_data = {name:v for name, v in shibus_data.items() if not ((min_lat > v["lat"]) or (max_lat < v["lat"]) or (min_lon > v["lon"]) or (max_lon < v["lon"]))}
        
        for name, v in shibus_data.items():
            v["node"] = ox.nearest_nodes(graph, v["lon"], v["lat"])
        
        return shibus_data
    
    def make_prior_distribution(self, graph, shortest_path_distance_dict):
        shibus_data = self._load_shibus_data(graph)
        
        prior_distribution = {node:0 for node in graph.nodes}
        sum_ = 0
        for name, v in shibus_data.items():
            for node, value in shortest_path_distance_dict[v["node"]].items():
                score = np.exp(-k * value) * v["sum"]
                prior_distribution[node] += score
                sum_ += score
        prior_distribution = {node:value/sum_ for node, value in prior_distribution.items()}
        
        return prior_distribution
    

    def plot_bus_stop(self, graph):
        shibus_data = self._load_shibus_data(graph)
        x_cord = [v["lon"] for name, v in shibus_data.items()]
        y_cord = [v["lat"] for name, v in shibus_data.items()]
        data = [v["sum"] for name, v in shibus_data.items()]
        name = list(shibus_data.keys())
            
        fig, ax = ox.plot_graph(graph,show=False,close=False,node_color="black",node_size=0)
        im = ax.scatter(x_cord,y_cord, c=data, s=30, cmap=cm.Spectral)
        plt.axis("equal")
        fig.colorbar(im)
        plt.savefig("prior.eps")
        plt.show() 
        

class LatticeGraphMaker(GraphMaker):
    def __init__(self, side_length=100, side_lattice_number=10):
        self.side_lattice_number = side_lattice_number
        self.side_length = side_length
    
    def make_graph(self):
        if(self.side_lattice_number <= 1):
            print("side lattice number is more than 1")
            return
        
        self.name = f"lattice_side_{self.side_length}_nlattice_{self.side_lattice_number}"
        
        self.graph = nx.grid_graph(dim=[self.side_lattice_number, self.side_lattice_number])
        for u, d in self.graph.nodes(data=True):
            d["x"] = u[0]
            d["y"] = u[1]
        for u, v, d in self.graph.edges(data=True):
            d["length"] = self.side_length/(self.side_lattice_number-1)

        self.cp_dict_sd()
        
    def make_road_pr(self):
        self._make_sub_graph()
        nodes_on_road = [node for node in self.sub_gp.nodes if not node[0] == 0 and (self.side_lattice_number-1) / node[0] == 2]
        self.sub_gp = self.gp.subgraph(nodes_on_road)
        self.pr = {node: 0 for node in self.gp.nodes}
        for node in nodes_on_road:
            self.pr[node] =  1/len(nodes_on_road)
    
    def _make_sub_graph(self):
        temp = (self.side_lattice_number-1)/4
        nodes = []
        for node in self.gp.nodes:
            x,y = node
            if x >= temp and x <= self.side_lattice_number -1 - temp and y >= temp and y <= self.side_lattice_number -1 - temp:
                nodes.append((x,y))
        self.sub_gp = self.gp.subgraph(nodes)
        
    def plot_graph(self):
        nx.draw(self.gp, pos=dict((n, n) for n in self.gp.nodes()) , node_size=0)
        plt.axis('equal')
        plt.show()
    
    
    def rt_graph(self):
        return self.gp
    
    def make_unbalanced_point(self, ratio=1/3):
        high_point = int(ratio * self.side_lattice_number)
        
        self.high_point1, self.high_point2, self.high_point3, self.high_point4 = (high_point-1, high_point-1), (high_point-1, self.side_lattice_number -high_point), (self.side_lattice_number - high_point, high_point-1), (self.side_lattice_number - high_point, self.side_lattice_number - high_point)
        
        


def apply_spanner(data_loader, delta, n_nodes, distance_type="spd"):

    spanner = nx.Graph()
    
    if n_nodes == 0:
        n_nodes = len(data_loader.G)
        
    logger.info(f"chosen node is fixed by seed {0}")

    candidate_nodes = list(data_loader.G.nodes(data=True))
    nodes = []
    random.seed(0)
    while len(nodes) < n_nodes:
        choiced = random.choice(candidate_nodes)
        if choiced not in nodes:
            nodes.append(choiced)
        
    spanner.add_nodes_from(nodes)
    spanner.graph["crs"] = data_loader.G.graph["crs"]
    
    node_indice = [data_loader.g_index[node] for node in spanner.nodes]
    truncated_np_ed = data_loader.np_ed[node_indice,:][:,node_indice]
    truncated_index_g = [data_loader.index_g[node_index] for node_index in np.sort(np.array(node_indice))]
    truncated_g_index = {node: truncated_index_g.index(node) for node in spanner.nodes}

    sorted_indice = np.vstack(np.unravel_index(np.argsort(-truncated_np_ed, axis=None), truncated_np_ed.shape)).T
    
    logger.info(f"the number of candidate edges: {len(sorted_indice)}")
    for index in sorted_indice:
        node_from = truncated_index_g[index[0]]
        node_to = truncated_index_g[index[1]]
        index_in_g_node_from = data_loader.g_index[node_from]
        index_in_g_node_to = data_loader.g_index[node_to]
        
        if distance_type == "spd":
            distance = data_loader.np_spd[index_in_g_node_from][index_in_g_node_to]
        else:
            distance = data_loader.np_ed[index_in_g_node_from][index_in_g_node_to]
        
        if nx.has_path(spanner, node_from, node_to):
            shortest_path_distance = nx.dijkstra_path_length(spanner, node_from, node_to, weight="length")
        else:
            shortest_path_distance = float("inf")
            
        if shortest_path_distance >= delta * distance:
            spanner.add_edge(node_from, node_to, length=distance)
                  
    node_mapping = {node:"" for node in data_loader.G.nodes()}
    truncated_np_pr = data_loader.np_pr[node_indice,:]
    for node in data_loader.G.nodes(data=True):
        if not node[0] in spanner.nodes():
            nearest_node = ox.nearest_nodes(spanner, node[1]["x"], node[1]["y"])
            node_mapping[node[0]] = nearest_node
            index_in_spanner = truncated_g_index[nearest_node]
            index_in_g = data_loader.g_index[node[0]]
            truncated_np_pr[index_in_spanner,0] += data_loader.np_pr[index_in_g,0]
        else:
            node_mapping[node[0]] = node[0]
    
    gm = GraphMaker()
    spanner_spd = gm.compute_shortest_path_distance_dict(spanner)
    spanner_np_spd = np.zeros((n_nodes, n_nodes))
    for node1 in spanner.nodes():
        index1 = truncated_g_index[node1]
        for node2 in spanner.nodes():
            index2 = truncated_g_index[node2]
            spanner_np_spd[index1][index2] = spanner_spd[node1][node2]

    return spanner, spanner_np_spd, truncated_np_pr, truncated_index_g, truncated_g_index, node_mapping