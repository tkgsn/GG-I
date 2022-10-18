import argparse
from src.mechanism import score, GraphExponentialMechanism, OptimalGraphExponentialMechanism, OptGeoIMechanism
from src.data_loader import DataLoader
import json
from src.my_util import result_dir

parser = argparse.ArgumentParser()
parser.add_argument("--mechanism", default="GEM", type=str)
parser.add_argument("--location", default="Akita", type=str)
parser.add_argument("--distance", default=2000, type=int)
parser.add_argument("--n_nodes", default=30, type=int)
parser.add_argument("--epsilon", default=0.01, type=float)
parser.add_argument("--delta", default=1.3, type=float)
parser.add_argument("--prior", default="uniform", type=str)
parser.add_argument("--simplify", action="store_true")
args = parser.parse_args()

strtmec = {"GEM": GraphExponentialMechanism,
          "OPT_GEM": OptimalGraphExponentialMechanism,
          "OPT_GEOI": OptGeoIMechanism}

if __name__ == "__main__":

    data_loader = DataLoader(args.location, distance=args.distance, prior_type=args.prior, simplify=args.simplify)
    mechanism = strtmec[args.mechanism](data_loader, **vars(args))
        
    save_dir = result_dir / f"{args.location}_{args.distance}_simplify{args.simplify}"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"{args.mechanism}_{args.epsilon}.json", "w") as f:
        json.dump(score(mechanism), f)