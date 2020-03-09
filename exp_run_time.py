import src.Mechanisms as Mec
import src.data_loader as DataLoader
import src.graph_maker as gm
import joblib
import time
import os

def make_graph(n_nodes):

    lgm = gm.LatticeGraphMaker(1000, n_nodes)
    lgm.make_unbalanced_point()
    lgm.make_graph()
    lgm.save()
    
    return lgm
    
    
def score(mec):
    score = {}
    score["SQL"] = mec.compute_SQL()
    score["optimal_AE"] = mec.compute_AE()
    score["bayes_AE"] = mec.compute_AE(attack="bayes")
    score["optimal_PC"] = score["optimal_AE"]/score["SQL"]
    score["bayes_PC"] = score["bayes_AE"]/score["SQL"]
    return score


if __name__ == '__main__':
    epsilon = 0.01
    n_nodes = [5, 10, 15, 25, 50]

    os.makedirs(f"results/optimize_synthetic/", exist_ok="True")
    for n_node in n_nodes:
        lgm = make_graph(n_node)
        data_loader = DataLoader.DataLoader(lgm.name, prior="unbalance", graph_maker=lgm)
        
        mec = Mec.GraphExponentialMechanism(data_loader)

        mec.build_distribution(epsilon)

        pre_score = score(mec)

        start = time.time()
        mec.build_optimal_distribution()
        elapsed_time = time.time() - start

        optimal_score = score(mec)
        joblib.dump(filename=f"results/optimize_synthetic/optimize_score_n_nodes_{n_node**2}.jbl", value=[elapsed_time, pre_score, optimal_score])