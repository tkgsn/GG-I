import argparse
from src.mechanism import score, GraphExponentialMechanism, OptimalGraphExponentialMechanism, OptGeoIMechanism, OptGeoGIMechanism, PlanarLaplaceMechanismOnGraph
from src.data_loader import DataLoader
import json
from src.my_util import result_dir

parser = argparse.ArgumentParser()
parser.add_argument("--mechanism", default="GEM", type=str)
parser.add_argument("--location", default="Akita", type=str)
parser.add_argument("--distance", default=2000, type=int)
parser.add_argument("--n_optgeoi_nodes", default=100, type=int)
parser.add_argument("--n_graph_nodes", default=0, type=int)
parser.add_argument("--epsilon", default=0.001, type=float)
parser.add_argument("--delta", default=1.3, type=float)
parser.add_argument("--prior", default="uniform", type=str)
parser.add_argument("--simplify", action="store_true")
args = parser.parse_args()

strtmec = {"GEM": GraphExponentialMechanism,
          "OPT_GEM": OptimalGraphExponentialMechanism,
          "OPT_GEOI": OptGeoIMechanism,
          "OPT_GEOGI": OptGeoGIMechanism,
          "PLMG": PlanarLaplaceMechanismOnGraph}

if __name__ == "__main__":

    data_loader = DataLoader(args.location, distance=args.distance, prior_type=args.prior, simplify=args.simplify, n_graph_nodes=args.n_graph_nodes)
    mechanism = strtmec[args.mechanism](data_loader, **vars(args))
        
    save_dir = result_dir / f"{args.location}_{args.distance}_simplify{args.simplify}_ngraphnodes{args.n_graph_nodes}"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"{mechanism}.json", "w") as f:
        json.dump(score(mechanism), f)