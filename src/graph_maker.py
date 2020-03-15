from pyproj import Geod
import joblib
import copy
import os
import networkx as nx
import osmnx as ox
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import src.my_util as util


class GraphMaker():
    def __init__(self):
        pass
    
    def make_graph(self):
        pass
    
    def cp_dict_sd(self):
        dict_sd = dict(nx.all_pairs_dijkstra_path_length(self.G, weight='length'))
        self.dict_sd = {str(node):{} for node in self.G.nodes}
        for node in self.G.nodes:
            for node_ in self.G.nodes:
                self.dict_sd[str(node)][str(node_)] = dict_sd[node][node_] if node_ in list(dict_sd[node].keys()) else 0
    
    def save(self):
        graph_path = os.path.join("data", "graph")
        os.makedirs(graph_path, exist_ok=True)
        nx.write_graphml(self.G, os.path.join(graph_path, self.name + ".ml"))
        nx.write_graphml(self.G, os.path.join(graph_path, "sub_" + self.name + ".ml"))
        
        data_path = os.path.join("data", "graph_data")
        os.makedirs(data_path, exist_ok=True)
        joblib.dump(filename=os.path.join(data_path, "dict_sd_" + self.name + ".jbl"), value=self.dict_sd)
        

class MapGraphMaker(GraphMaker):
    def __init__(self, lat, lon, distance):
        self.lat = lat
        self.lon = lon
        self.distance = distance
    
    def make_graph(self, name):
        self.G = ox.graph_from_point((self.lat, self.lon),distance_type="network", distance=self.distance, network_type="walk")
        self.name = name
        self.cp_dict_sd()
        
    def plot_graph(self):
        ox.plot_graph(self.G, save=True, file_format="png")
        
    def save(self):
        graph_path = os.path.join("/", "data", "takagi", "GG-I", "data", "graph")
        ox.save_graphml(self.G, os.path.join(graph_path, self.name + ".ml"))
        ox.save_graphml(self.G, os.path.join(graph_path, "sub_" + self.name + ".ml"))
        
        data_path = os.path.join("/", "data", "takagi", "GG-I", "data", "graph_data")
        joblib.dump(filename=os.path.join(data_path, "dict_sd_" + self.name + ".jbl"), value=self.dict_sd)
        
k = 0.001
class KyotoMapGraphMaker(GraphMaker):
    
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
    
    def save(self):
        graph_path = os.path.join("/", "data", "takagi", "GG-I", "data", "graph")
        ox.save_graphml(self.G, os.path.join(graph_path, self.name + ".ml"))
        ox.save_graphml(self.G, os.path.join(graph_path, "sub_" + self.name + ".ml"))
        
        data_path = os.path.join("/", "data", "takagi", "GG-I", "data", "graph_data")
        joblib.dump(filename=os.path.join(data_path, "dict_sd_" + self.name + ".jbl"), value=self.dict_sd)

    def compute_shortest_distances(self):
        self.shortest_dists = dict(nx.all_pairs_dijkstra_path_length(self.G,weight='length'))    

    def graph_width(self):
        return KyotoGraphMaker.g.inv(self.min_lon, self.min_lat, self.max_lon, self.min_lat)[2]
    
    def graph_height(self):
        return KyotoGraphMaker.g.inv(self.min_lon, self.min_lat, self.min_lon, self.max_lat)[2]

    def __init__(self):
        self.name = "Kyoto"
        self.G = None
        self.made_graph = None
        self.shibus_data = None
        self.latlons = {}
        self.load_shibus_data()
        self.load_graph()
        
    def load_graph(self):
        self.G = joblib.load(os.path.join(os.getcwd(),"data/graph/kyoto.jbl"))
        
    def make_graph(self, min_lat = 34.8955054, max_lat = 35.301761, min_lon = 135.5617132, max_lon = 135.8579904):
        self.made_graph  = copy.deepcopy(self.G)
        for node in self.G.nodes(data=True):
            lat = node[1]["y"]
            lon = node[1]["x"]
            if (min_lat > lat) or (max_lat < lat) or (min_lon > lon) or (max_lon < lon):
                self.made_graph.remove_node(node[0])
        self.made_graph = self.made_graph.to_undirected()
       
        max_component = self.max_component()
        for node in copy.deepcopy(self.made_graph):
            if not node in max_component:
                self.made_graph.remove_node(node)            
        
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        
        self.G = self.made_graph
        
        self.compute_shortest_distances()
        self.make_latlons()
        self.make_bus_stop_node_dict()
        #self.make_nearest_bus_stop_list_s()
        self.make_prior_distribution_exponential()
        self._cp_np_pr()
        
       
    def max_component(self):
        max_size = 0
        for component in nx.connected_components(self.made_graph):
            if len(component) > max_size:
                max_cluster = component
                max_size = len(max_cluster)
        return max_cluster
        
    def load_shibus_data(self):
        self.shibus_data = joblib.load(os.path.join(os.getcwd(),"data/shibus_data.jbl"))
        
    def make_latlons(self):
        for name, data in self.shibus_data.items():
            lat = float(data["lat"])
            lon = float(data["lon"])
            get_off = int(data["get_off"])
            ride_on = int(data["ride_on"])
            if not  ((self.min_lat > lat) or (self.max_lat < lat) or (self.min_lon > lon) or (self.max_lon < lon)):
                self.latlons[name] = {}
                self.latlons[name]["lon"] = lon
                self.latlons[name]["lat"] = lat
                self.latlons[name]["get_off"] = get_off
                self.latlons[name]["ride_on"] = ride_on
                self.latlons[name]["sum"] = get_off + ride_on
                
    def make_bus_stop_node_dict(self):
        self.bus_stop_node_dict = {}
        for name, data in self.latlons.items():
            node = ox.get_nearest_node(G=self.made_graph, point=(data["lat"],data["lon"]))
            self.bus_stop_node_dict[name] = node
            
    def make_nearest_bus_stop_list_s(self):
        self.bus_stop_counter_s = {}
        self.sum_bus_user_s = 0
        self.distance_to_bus_stop = {}
        for key,node in self.bus_stop_node_dict.items():
            self.bus_stop_counter_s[key] = 0
            self.sum_bus_user_s += self.latlons[key]["sum"]
        self.nearest_bus_stop_s = {}
        for key, node in self.made_graph.nodes(data=True):
            nearest_bus_stop = self.find_nearest_bus_stop(key)
            self.nearest_bus_stop_s[key] = nearest_bus_stop
            distance = self.shortest_dists[key][self.bus_stop_node_dict[nearest_bus_stop]]
            self.distance_to_bus_stop[key] = distance
            self.bus_stop_counter_s[nearest_bus_stop] += np.exp(-0.001 * np.power(distance,2))
                
    def find_nearest_bus_stop(self, node):
        shortest_distances = {a: self.shortest_dists[node][a] for a in self.bus_stop_node_dict.values()}
        return list(self.bus_stop_node_dict.keys())[np.argmin(list(shortest_distances.values()))]
    
    def make_nearest_bus_stop_list(self):
        H = nx.Graph()
        self.bus_stop_counter = {}
        self.sum_bus_user = 0
        for key,node in self.latlons.items():
            H.add_node(key)
            H.node[key]["y"] = node["lat"]
            H.node[key]["x"] = node["lon"]
            self.bus_stop_counter[key] = 0
            self.sum_bus_user += node["sum"]
        self.nearest_bus_stop = {}
        for key, node in self.made_graph.nodes(data=True):
            nearest_bus_stop = ox.get_nearest_node(G=H,point=(node["y"],node["x"]))
            self.nearest_bus_stop[key] = nearest_bus_stop
            self.bus_stop_counter[nearest_bus_stop] += 1
    
    def make_prior_distribution_exponential(self):
        self.prior_distribution_s = {node:0 for node in self.made_graph.nodes}
        suma = 0
        for name, bus_stop_node in self.bus_stop_node_dict.items():
            for node, value in self.shortest_dists[bus_stop_node].items():
                score = np.exp(-k * value) * self.latlons[name]["sum"]
                self.prior_distribution_s[node] += score
                suma += score
        self.prior_distribution_s = {node:value/suma for node, value in self.prior_distribution_s.items()}
            
            
    def make_prior_distribution(self):
        self.prior_distribution = {}
        for node, bus_stop in self.nearest_bus_stop.items():
            distance = self.distance_to_bus_stop[node]
            self.prior_distribution[node] = np.exp(-0.001 * distance)*self.latlons[bus_stop]["sum"]/(self.sum_bus_user*self.bus_stop_counter[bus_stop])

    def make_prior_distribution_s(self):
        self.prior_distribution_s = {}
        for node, bus_stop in self.nearest_bus_stop_s.items():
            distance = self.distance_to_bus_stop[node]
            #self.prior_distribution_s[node] = self.latlons[bus_stop]["sum"]/(self.sum_bus_user_s*self.bus_stop_counter_s[bus_stop])
            self.prior_distribution_s[node] = np.exp(-0.001 * distance)*self.latlons[bus_stop]["sum"]/(self.sum_bus_user_s*self.bus_stop_counter_s[bus_stop])
            
    def plot_bus_stop(self):
        x_cord = []
        y_cord = []
        data = []
        name = []
        for bus_stop, value in self.latlons.items():
            x_cord.append(value["lon"])
            y_cord.append(value["lat"])
            data.append(value["sum"])
            name.append(bus_stop)
        fig, ax = ox.plot_graph(self.made_graph,show=False,close=False,node_color="black",node_size=0)
        im = ax.scatter(x_cord,y_cord,  c=data, s=30, cmap=cm.Spectral)
        #for i,(x,y) in enumerate(zip(x_cord,y_cord)):
        #    ax.annotate(name[i],(x,y))
        plt.axis("equal")
        fig.colorbar(im)
        #ax.set_title(f"prior")
        plt.savefig("prior.eps")
        plt.show() 
        
    def _cp_np_pr(self):
        self.np_pr = np.zeros((len(self.G.nodes), 1))
        self.np_sub_pr = np.zeros((len(self.G.nodes), 1))
        
        for i, node in enumerate(self.G.nodes):
            self.np_pr[i, 0] = self.prior_distribution_s[node]
            self.np_sub_pr[i, 0] = self.prior_distribution_s[node]
        
        
class LatticeGraphMaker(GraphMaker):
    def __init__(self, side_length=100, side_lattice_number=10):
        self.side_lattice_number = side_lattice_number
        self.side_length = side_length
    
    def make_graph(self):
        if(self.side_lattice_number <= 1):
            print("side lattice number is more than 1")
            return
        
        self.name = f"lattice_side_{self.side_length}_nlattice_{self.side_lattice_number}"
        
        self.G = nx.grid_graph(dim=[self.side_lattice_number, self.side_lattice_number])
        for u, d in self.G.nodes(data=True):
            d["x"] = u[0]
            d["y"] = u[1]
        for u, v, d in self.G.edges(data=True):
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