import osmnx as ox
import joblib
import numpy as np
import os
import networkx as nx

class DataLoader():
    def __init__(self, location_name, prior="uniform", graph_maker=None):
        self.G, self.H, self.dict_sd = self.load(location_name)
        self.np_sd, self.np_sub_sd, self.np_sub_sub_sd = self._convert_to_np_sd(self.G, self.dict_sd, self.H)
        self.g_index = self._make_g_index(self.G)
        self.h_index = self._make_h_index(self.G, self.H)
        
        if prior == "uniform":
            self._make_uniform_prior()
        elif prior == "unbalance":
            if graph_maker is None:
                raise
            self._make_unbpr(graph_maker)
        elif prior == "bus":
            if graph_maker is None:
                raise
            self.np_pr = graph_maker.np_pr
            self.np_sub_pr = graph_maker.np_sub_pr
    
    def load(self, location_name):
        graph_dir = os.path.join("data", "graph")
        data_dir = os.path.join("data", "graph_data")
        
        #G = ox.load_graphml(f'{location_name}.ml',folder=graph_dir)
        #H = ox.load_graphml(f'sub_{location_name}.ml',folder=graph_dir)
        G = nx.read_graphml(os.path.join(graph_dir, f'{location_name}.ml'))
        H = nx.read_graphml(os.path.join(graph_dir, f'sub_{location_name}.ml'))
        dict_sd = joblib.load(os.path.join(data_dir, f"dict_sd_{location_name}.jbl"))

        return G, H, dict_sd
    
    def _convert_to_np_sd(self, G, dict_sd, H=None):
        if H is None:
            H = G
            
        np_sd = np.array([[dict_sd[str(dic_node)][str(node)] for node in G.nodes] for dic_node in G.nodes])
        np_sub_sd = np.array([[dict_sd[str(dic_node)][str(node)] for node in G.nodes] for dic_node in G.nodes if dic_node in H.nodes])
        np_sub_sub_sd = np.array([[dict_sd[str(dic_node)][str(node)] for node in G.nodes if node in H.nodes] for dic_node in G.nodes if dic_node in H.nodes])   
        return np_sd, np_sub_sd, np_sub_sub_sd
    
    def make_dict_sd(self, G):
        gen = nx.all_pairs_dijkstra_path_length(G, weight='length')
        dict_sd = {node: {node_:0 for node_ in G} for node in G}
        
        for node, dic in gen:
            for node_, leng in dic.items():
                dict_sd[node][node_] = leng
                    
        return dict_sd
    
    def _make_g_index(self, G):
        return {node:list(self.G.nodes()).index(node) for node in self.G}
    
    def _make_h_index(self, G, H):
        h_node = [node for node in G.nodes if node in H.nodes]
        return {node:h_node.index(node) for node in H.nodes}
    
    
    def _make_uniform_prior(self):
        n_graph_nodes = len(self.G)
        n_sub_graph_nodes = len(self.H)
        
        np_pr = np.zeros((n_graph_nodes,1))
        np_sub_pr = np.zeros((n_sub_graph_nodes,1))
        for node in self.H.nodes():
            node_ind = self.g_index[node]
            sub_node_ind = self.h_index[node]
            np_pr[node_ind][0] = 1/n_sub_graph_nodes
            np_sub_pr[sub_node_ind][0] = 1/n_sub_graph_nodes
            
        self.np_pr = np_pr
        self.np_sub_pr = np_sub_pr
        
    def _make_unbpr(self, graph_maker):
        k = 0.00003
        pr = np.zeros((len(self.G), 1))
        high_points = [graph_maker.high_point1, graph_maker.high_point2, graph_maker.high_point3, graph_maker.high_point4]
        sums = 0
        for node in high_points:
            node_ind = self.g_index[str(node)]
            scores = np.exp(-k * np.power(self.np_sd[node_ind],2)).reshape((len(self.G), 1))
            pr += scores
            sums += scores.sum()
        pr /= sums
        self.np_pr = pr
        self.np_sub_pr = pr