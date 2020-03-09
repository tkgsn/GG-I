import os
import sys
sys.path.append(os.path.dirname(__file__))

import my_util as util
import numpy as np
import pulp
import copy
import src.make_plg as plg_maker

class Mechanism():
    def __init__(self, data_loader):
        self._load_data(data_loader)
    
    def _load_data(self, data_loader):
        self.data_loader = data_loader
        
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
        self.sql = np.sum(self.dist * self.data_loader.np_pr * self.data_loader.np_sd)
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
            self.ae = np.sum(remap * np.dot(dist.T, self.data_loader.np_sub_sub_sd))
            return self.ae
            
    def _compute_inference_function(self, euclid=False):
        print("setting variables", end="\r")
        remapping = [[pulp.LpVariable((perturbed_node,inf_node),0,1,'Continuous') for inf_node in self.data_loader.H.nodes] for perturbed_node in self.data_loader.G.nodes]
        problem = pulp.LpProblem('AE', pulp.LpMinimize)
        #目的関数
        print("setting the objective function.", end="\r")

        dist = self.dist * self.data_loader.np_pr
        indice = np.array([self.data_loader.g_index[node] for node in self.data_loader.G if node in self.data_loader.H.nodes])
        dist = dist[indice]
        
        if euclid:
            problem += pulp.lpSum(remapping * np.dot(temp.T, self.sub_est))
        else:
            problem += pulp.lpSum(remapping * np.dot(dist.T, self.data_loader.np_sub_sub_sd))
        #制約 Σk_xz = 1
        print("setting costraints.", end="\r")
        for i in range(len(self.data_loader.G)):
            problem += pulp.lpSum(remapping[i]) == 1.0
        #計算
        print("solving..", end="\r")
        status = problem.solve(pulp.PULP_CBC_CMD(msg=1))
        if(status):
            print(f"the optimal value: {pulp.value(problem.objective)}")
            self.obj_value = pulp.value(problem.objective)
            f = np.frompyfunc(lambda x:x.value(), 1, 1)
            return f(remapping)
        else:
            print("not solved")
    
class PlanarLaplaceMechanismOnGraph(Mechanism):
    
    def build_distribution(self, epsilon, is_pr=False):
        self.epsilon = epsilon
        self.dist = np.zeros((len(self.data_loader.G),len(self.data_loader.G)))
        dist = plg_maker.MakePLG(self.data_loader.G, self.data_loader.H, epsilon, "ox").dist
        for node, dist_ in dist.items():
            node_ind = self.g_index(node)
            for node_, prob in dist_.items():
                node_ind_ = self.g_index(node_)
                self.dist[node_ind][node_ind_] = prob

class GraphExponentialMechanism(Mechanism):
    
    def build_distribution(self, epsilon, rm_list=[]):
        self.epsilon = epsilon
        
        M = np.exp((-epsilon/2) * self.data_loader.np_sd)
        
        remove_mat = self._make_remove_mat(rm_list)
        removed_dist = np.dot(M, remove_mat)
        self.dist = self._normalize(removed_dist)
        
    def _make_remove_mat(self, node_list):
        remove_mat = np.array([[int(i == j) for j in range(len(self.data_loader.G))] for i in range(len(self.data_loader.G))])
        for node in node_list:
            index = self.data_loader.g_index[node]
            remove_mat[index][index] = 0
        return remove_mat
    
    def _normalize(self, dist):
        sum_dist = np.sum(dist, axis=1).reshape(-1,1)
        m = dist / sum_dist
        return m
    
    def _make_sql_function(self):
        
        alpha_sub_mat = np.exp(- (self.epsilon/2) * self.data_loader.np_sub_sd)
        d_prior_mat = self.data_loader.np_sub_sd * self.data_loader.np_sub_pr * alpha_sub_mat
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

        print("\nstart preprocess optimization")
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
                
        return rm_list
    
    def build_optimal_distribution(self):
        
        thre = self.compute_SQL()        
        rm_list = self._pre_optimization()
        
        remain_nodes = [node for node in self.data_loader.G.nodes if node not in rm_list]
        
        n_targets = len(remain_nodes)
        n_nodes = len(self.data_loader.G.nodes)
        n_sub_nodes = len(self.data_loader.H.nodes)
        
        np_removed_sub_sd = np.zeros((n_sub_nodes, n_targets))
        for v in self.data_loader.H.nodes:
            for i, v_prime in enumerate(remain_nodes):
                v_ind = self.data_loader.h_index[v]
                v_prime_ind = self.data_loader.g_index[v_prime]
                np_removed_sub_sd[v_ind,i] = self.data_loader.np_sub_sd[v_ind, v_prime_ind]
        
        alpha_sub_mat = np.exp(- (self.epsilon/2) * np_removed_sub_sd)
        prior_mat = self.data_loader.np_sub_pr * alpha_sub_mat
        d_prior_mat = np_removed_sub_sd * prior_mat
        
        def alpha_sub_value(node_ind):
            return alpha_sub_mat[:,node_ind].reshape(-1,1)
        
        def d_prior_value(node_ind):
            return d_prior_mat[:,node_ind].reshape(-1,1)
        
        sum_alpha_sub_mat = np.sum(alpha_sub_mat, axis=1).reshape(-1,1)
        sum_d_prior_mat = np.sum(d_prior_mat, axis=1).reshape(-1,1)
        
        sql = np.sum(sum_d_prior_mat / sum_alpha_sub_mat)
        ae = self.compute_AE(attack="bayes")
        pc = ae / sql
        
        n_iter = 0
        
        print("\nstart optimization")
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
                
                if sql_ > thre:
                    continue
                
                alpha_sub_mat_ = copy.deepcopy(alpha_sub_mat)
                alpha_sub_mat_[:,counter] = 0
                prior_mat_ = alpha_sub_mat_ / sum_alpha_sub_mat_
                
                pr_v = np.sum(prior_mat_,axis=0)
                inverse_pr_v = np.array([1/v if not v == 0 else 0 for v in pr_v]).reshape(1,-1)
                pos = (prior_mat_ * inverse_pr_v).T
                remap = pos.T[np.any(pos, axis=0)].T
                
                dist = prior_mat_ * self.data_loader.np_sub_pr
                ae_ = np.sum(remap * np.dot(dist.T, self.data_loader.np_sub_sub_sd))
                
                
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
                
            
        self.build_distribution(self.epsilon, rm_list)
        self.rm_list = rm_list