import os
import sys
sys.path.append(os.path.dirname(__file__))

import my_util as util
import numpy as np
import pulp
import copy
import src.make_plg as plg_maker
import json
from src.graph_maker import apply_spanner
import time


from logging import getLogger, config
with open('/GG-I/data/log_config.json', 'r') as f:
    log_conf = json.load(f)
config.dictConfig(log_conf)
logger = getLogger("sub_module")

def score(mec):
    score = {}
    score["SQL"] = mec.compute_SQL()
    score["optimal_AE"] = mec.compute_AE()
    score["bayes_AE"] = mec.compute_AE(attack="bayes")
    score["optimal_PC"] = score["optimal_AE"]/score["SQL"]
    score["bayes_PC"] = score["bayes_AE"]/score["SQL"]
    score["time"] = mec.time
    return score

class Mechanism():
    def __init__(self, data_loader, **kwargs):
        self.data_loader = data_loader
        self.build_distribution(**kwargs)
        logger.info(f"initialized {self} with {kwargs}")
        
    def __str__(self):
        return "SUPER"
    
    def build_distribution(self, **kwargs):
        pass
        
    def compute_posterior_dist(self):
        dist = self.dist * self.data_loader.np_pr
        pr_v = np.sum(dist,axis=0)
        inverse_pr_v = np.array([1/v if not v == 0 else 0 for v in pr_v]).reshape(1,-1)
        pos = (dist * inverse_pr_v).T
        self.posterior_dist = pos
        self.sub_posterior_dist = pos.T[np.any(pos, axis=0)].T
        return self.sub_posterior_dist
        
    def compute_SQL(self, euclid=False):
        if euclid:
            self.sql = np.sum(self.dist * self.data_loader.pr * self.est)
        else:
            pass
        logger.debug(self.data_loader.np_pr)
        self.sql = np.sum(self.dist * self.data_loader.np_pr * self.data_loader.np_spd)
        logger.info(f"SQL: {self.sql}")
        return self.sql
    
    def compute_AE(self, euclid=False, attack="optimal"):
        if attack == "optimal":
            remap = self._compute_inference_function(euclid)
        elif attack == "bayes":
            remap = self.compute_posterior_dist()
        else:
            print("attack is optimal or bayes")
            return
        
        dist = self.dist * self.data_loader.np_pr
        indice = np.array([self.data_loader.g_index[node] for node in self.data_loader.G if node in self.data_loader.H.nodes])
        dist = dist[indice]
        
        if euclid:
            self.lp = np.sum(self.rem * np.dot(temp.T, self.sub_est))
        else:
            self.ae = np.sum(remap * np.dot(dist.T, self.data_loader.np_sub_sub_spd))
            logger.info(f"AE: {self.ae}")
            return self.ae
            
    def _compute_inference_function(self, euclid=False):
        logger.info("setting variables, constraints, the objective function for computing optimal inference function")
        remapping = [[pulp.LpVariable(perturbed_node+","+inf_node,0,1,'Continuous') for inf_node in self.data_loader.H.nodes] for perturbed_node in self.data_loader.G.nodes]
        problem = pulp.LpProblem('AE', pulp.LpMinimize)
        #目的関数
        # logger.info("setting the objective function.")

        dist = self.dist * self.data_loader.np_pr
        indice = np.array([self.data_loader.g_index[node] for node in self.data_loader.G if node in self.data_loader.H.nodes])
        dist = dist[indice]
        
        if euclid:
            problem += pulp.lpSum(remapping * np.dot(temp.T, self.sub_est))
        else:
            problem += pulp.lpSum(remapping * np.dot(dist.T, self.data_loader.np_sub_sub_spd))
        #制約 Σk_xz = 1
        # logger.info("setting costraints.")
        for i in range(len(self.data_loader.G)):
            problem += pulp.lpSum(remapping[i]) == 1.0
        #計算
        logger.info("solving the optimization problem")
        status = problem.solve(pulp.PULP_CBC_CMD(msg=0))
        if(status):
            # print(f"the optimal value: {pulp.value(problem.objective)}")
            self.obj_value = pulp.value(problem.objective)
            f = np.frompyfunc(lambda x:x.value(), 1, 1)
            return f(remapping)
        else:
            logger.info("not solved")
    
class PlanarLaplaceMechanismOnGraph(Mechanism):
    
    def __str__(self):
        return f"PLMG_{self.epsilon}"
    
    def build_distribution(self, **kwargs):
        start_time = time.time()
        
        epsilon = kwargs["epsilon"]
        self.epsilon = epsilon
        self.dist = np.zeros((len(self.data_loader.G),len(self.data_loader.G)))
        logger.info("constructing PLMG")
        dist = plg_maker.MakePLG(self.data_loader.G, self.data_loader.H, epsilon, "ox").dist
        for node, dist_ in dist.items():
            node_ind = self.data_loader.g_index[node]
            for node_, prob in dist_.items():
                node_ind_ = self.data_loader.g_index[node_]
                self.dist[node_ind][node_ind_] = prob
        
        end_time = time.time()
        self.time = end_time - start_time

class GraphExponentialMechanism(Mechanism):
        
    def __str__(self):
        return f"GEM_{self.epsilon}"
    
    def build_distribution(self, **kwargs):
        start_time = time.time()
        
        epsilon = kwargs["epsilon"]
        self.epsilon = epsilon
        
        M = np.exp((-epsilon/2) * self.data_loader.np_spd)
        self.dist = self.normalize(M)

        end_time = time.time()
        self.time = end_time - start_time
    
    def normalize(self, dist):
        sum_dist = np.sum(dist, axis=1).reshape(-1,1)
        m = dist / sum_dist
        return m
    

        
        
class OptimalGraphExponentialMechanism(GraphExponentialMechanism):
        
    def __str__(self):
        return f"OptGEM_{self.epsilon}"
    
    def _remove_from_dist(self, rm_list):
        remove_mat = self._make_remove_mat(rm_list)
        removed_dist = np.dot(self.dist, remove_mat)
        return self.normalize(removed_dist)
        
    def _make_remove_mat(self, node_list):
        remove_mat = np.array([[int(i == j) for j in range(len(self.data_loader.G))] for i in range(len(self.data_loader.G))])
        for node in node_list:
            index = self.data_loader.g_index[node]
            remove_mat[index][index] = 0
        return remove_mat
    
    def build_distribution(self, **kwargs):
        start_time = time.time()
        
        epsilon = kwargs["epsilon"]
        self.epsilon = epsilon
        
        super().build_distribution(**kwargs)
        
        thre = self.compute_SQL()
        
        logger.debug(f"threshold (SQL) is {thre}")
        
        rm_list = self._pre_optimization()
        remain_nodes = [node for node in self.data_loader.G.nodes if node not in rm_list]
        
        n_targets = len(remain_nodes)
        n_nodes = len(self.data_loader.G.nodes)
        n_sub_nodes = len(self.data_loader.H.nodes)
        
        np_removed_sub_spd = np.zeros((n_sub_nodes, n_targets))
        for v in self.data_loader.H.nodes:
            for i, v_prime in enumerate(remain_nodes):
                v_ind = self.data_loader.h_index[v]
                v_prime_ind = self.data_loader.g_index[v_prime]
                np_removed_sub_spd[v_ind,i] = self.data_loader.np_sub_spd[v_ind, v_prime_ind]
        
        alpha_sub_mat = np.exp(- (self.epsilon/2) * np_removed_sub_spd)
        prior_mat = self.data_loader.np_sub_pr * alpha_sub_mat
        d_prior_mat = np_removed_sub_spd * prior_mat
        
        def alpha_sub_value(node_ind):
            return alpha_sub_mat[:,node_ind].reshape(-1,1)
        
        def d_prior_value(node_ind):
            return d_prior_mat[:,node_ind].reshape(-1,1)
        
        sum_alpha_sub_mat = np.sum(alpha_sub_mat, axis=1).reshape(-1,1)
        sum_d_prior_mat = np.sum(d_prior_mat, axis=1).reshape(-1,1)
        
        sql = np.sum(sum_d_prior_mat / sum_alpha_sub_mat)
        ae = self.compute_AE(attack="bayes")
        pc = ae / sql
        
        logger.debug(f"initial sql {sql}")
        
        n_iter = 0
        
        logger.info("start optimization")
        while True:
            dif_list = []
            n_rm_list = len(rm_list)
            n_targets = n_nodes - n_rm_list
            counter = 0
            d = 0
            
            for counter, node in enumerate(remain_nodes):
                
                if node in rm_list:
                    d += 1
                    continue
                print(f"\riter{n_iter:3d}: {counter+1-d:4d} / {n_targets:4d} , n_removed_nodes: {len(dif_list):4d}", end="")
                
                sum_alpha_sub_mat_ = sum_alpha_sub_mat - alpha_sub_value(counter)
                sum_d_prior_mat_ = sum_d_prior_mat - d_prior_value(counter)
                
                sql_ = np.sum(sum_d_prior_mat_ / sum_alpha_sub_mat_)
                
                logger.debug(f"new sql {sql_}")
                if sql_ > thre:
                    logger.debug("skip")
                    continue
                
                alpha_sub_mat_ = copy.deepcopy(alpha_sub_mat)
                alpha_sub_mat_[:,counter] = 0
                prior_mat_ = alpha_sub_mat_ / sum_alpha_sub_mat_
                
                pr_v = np.sum(prior_mat_,axis=0)
                inverse_pr_v = np.array([1/v if not v == 0 else 0 for v in pr_v]).reshape(1,-1)
                pos = (prior_mat_ * inverse_pr_v).T
                remap = pos.T[np.any(pos, axis=0)].T
                
                dist = prior_mat_ * self.data_loader.np_sub_pr
                ae_ = np.sum(remap * np.dot(dist.T, self.data_loader.np_sub_sub_spd))
                
                
                if pc < ae_/sql_:
                    pc = ae_/sql_
                    alpha_sub_mat[:,counter] = 0
                    sum_alpha_sub_mat = sum_alpha_sub_mat_
                    sum_d_prior_mat = sum_d_prior_mat_
                    rm_list.append(node)
                    dif_list.append(node)
                    
            n_iter += 1
            print("")
            if not dif_list:
                break    
            
        self.dist = self._remove_from_dist(rm_list)
        # self.rm_list = rm_list
        end_time = time.time()
        self.time = end_time - start_time
        
    def _make_sql_function(self):
        
        alpha_sub_mat = np.exp(- (self.epsilon/2) * self.data_loader.np_sub_spd)
        d_prior_mat = self.data_loader.np_sub_spd * self.data_loader.np_sub_pr * alpha_sub_mat
        self.sum_alpha_sub_mat = np.sum(alpha_sub_mat, axis=1).reshape(-1,1)
        self.sum_d_prior_mat = np.sum(d_prior_mat, axis=1).reshape(-1,1)
        
        def _cp_sql_from_dif(index=None):

            def alpha_sub_value(node_ind):
                if node_ind == None:
                    return np.zeros((alpha_sub_mat.shape[0],1))
                return alpha_sub_mat[:,node_ind].reshape(-1,1)

            def d_prior_value(node_ind):
                if node_ind == None:
                    return np.zeros((alpha_sub_mat.shape[0],1))
                return d_prior_mat[:,node_ind].reshape(-1,1)
            
            sum_alpha_sub_mat_ = self.sum_alpha_sub_mat - alpha_sub_value(index)
            sum_d_prior_mat_ = self.sum_d_prior_mat - d_prior_value(index)

            return sum_alpha_sub_mat_, sum_d_prior_mat_, np.sum(sum_d_prior_mat_ / sum_alpha_sub_mat_)
        
        return _cp_sql_from_dif
    
    def _pre_optimization(self):

        _cp_sql_from_dif = self._make_sql_function()
        _, _, sql = _cp_sql_from_dif()
        
        n_nodes = len(self.data_loader.G.nodes)
        n_sub_nodes = len(self.data_loader.H.nodes)
        n_iter = 0
        rm_list = []

        logger.info("start preprocess optimization")
        while True:
            dif_list = []
            
            n_rm_list = len(rm_list)
            n_targets = n_nodes - n_rm_list
            counter = 0
            
            for node in self.data_loader.G.nodes:
                if node in rm_list:
                    continue
                counter += 1
                print(f"\riter{n_iter:3d}: {counter:4d} / {n_targets:4d} , n_removed_nodes: {len(dif_list):4d}", end="")

                index = self.data_loader.g_index[node]
                
                sum_alpha_sub_mat_, sum_d_prior_mat_, sql_ = _cp_sql_from_dif(index)
                     
                logger.debug(f"{sql} -> {sql_}")
                if sql > sql_:
                    self.sum_alpha_sub_mat = sum_alpha_sub_mat_
                    self.sum_d_prior_mat = sum_d_prior_mat_
                    rm_list.append(node)
                    dif_list.append(node)
                    sql = sql_
            
            n_iter += 1
            print("")
                    
            if not dif_list:
                break
        logger.debug(rm_list)
                
        return rm_list
    
    
class OptGeoIMechanism(Mechanism):

    def __str__(self):
        return f"OptGeoIM_{self.epsilon}_delta{self.delta}_nnodes{self.n_optgeoi_nodes}"
    
    def apply_spanner(self, **kwargs):
        return apply_spanner(self.data_loader, kwargs["delta"], kwargs["n_optgeoi_nodes"], "ed")
        
    def build_distribution(self, **kwargs):
        start_time = time.time()
        
        epsilon = kwargs["epsilon"]
        self.epsilon = epsilon
        self.delta = kwargs["delta"]
        self.n_optgeoi_nodes = kwargs["n_optgeoi_nodes"]
        
        logger.info(f"applying spanner with delta={kwargs['delta']}, n_optgeoi_nodes={kwargs['n_optgeoi_nodes']}")
        graph, spanner_np_spd, truncated_np_pr, truncated_index_g, truncated_g_index, node_mapping = self.apply_spanner(delta=kwargs["delta"], n_optgeoi_nodes=kwargs["n_optgeoi_nodes"])
        
        logger.info(f"the number of edges decreases to {len(graph.edges())} from {len(graph.nodes)*(len(graph.nodes)+1)/2}")
        logger.info("setting variables and the optimization problem for constructing the optimal distribution")
        variables = [[pulp.LpVariable(str(node1)+"_"+str(node2),0,1,'Continuous') for node2 in graph.nodes()] for node1 in graph.nodes()]
        problem = pulp.LpProblem('optGeoI', pulp.LpMinimize)
        problem += pulp.lpSum(variables * truncated_np_pr * spanner_np_spd)

        for line_variables in variables:
            problem += pulp.lpSum(line_variables) == 1.0

        for variables_ in variables:
            for variable in variables_:
                problem += variable >= 0

        for edge in graph.edges():
            node_from_index_1 = truncated_g_index[edge[0]]
            node_from_index_2 = truncated_g_index[edge[1]]
            for node in graph.nodes():
                node_to_index = truncated_g_index[node]
                
                if (epsilon * spanner_np_spd[node_from_index_1][node_from_index_2] / kwargs["delta"]) > 40:
                    continue
                else:
                    problem += variables[node_from_index_1][node_to_index] <= np.exp(epsilon * spanner_np_spd[node_from_index_1][node_from_index_2] / kwargs["delta"]) * variables[node_from_index_2][node_to_index]
                    problem += variables[node_from_index_2][node_to_index] <= np.exp(epsilon * spanner_np_spd[node_from_index_1][node_from_index_2] / kwargs["delta"]) * variables[node_from_index_1][node_to_index]
        
        logger.info("solving the optimization problem")
        status = problem.solve(pulp.PULP_CBC_CMD(msg=0))
        logger.info(f"status: {status}")
        f = np.frompyfunc(lambda x:x.value(), 1, 1)
        
        sum_ = 0
        for variable in variables:
            for variable_ in variable:
                sum_ += (variable_.value())
        
        print(sum_)
        truncated_dist = f(variables).astype(np.float64)
        
        
        self.dist = np.zeros((len(self.data_loader.G.nodes()), len(self.data_loader.G.nodes())))
        for node in self.data_loader.G.nodes():
            index = self.data_loader.g_index[node]
            nearest_node = node_mapping[node]
            index_in_spanner = truncated_g_index[nearest_node]
            distribution = truncated_dist[index_in_spanner]
            for i, v in enumerate(distribution):
                node_to = truncated_index_g[i]
                index_in_g = self.data_loader.g_index[node_to]
                self.dist[index][index_in_g] += v
                
        end_time = time.time()
        self.time = end_time - start_time
        
class OptGeoGIMechanism(OptGeoIMechanism):
    
    def __str__(self):
        return f"OptGeoGIM_{self.epsilon}_delta{self.delta}_nnodes{self.n_optgeoi_nodes}"
    
    def apply_spanner(self, **kwargs):
        return apply_spanner(self.data_loader, kwargs["delta"], kwargs["n_optgeoi_nodes"], "spd")