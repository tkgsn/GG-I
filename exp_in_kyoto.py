from src.data_loader import DataLoader, BUS
from src.my_util import LatlonRange, result_dir
import src.Mechanisms as Mec
import json

if __name__ == "__main__":
    latlon_range = LatlonRange(min_lat=35.02, max_lat= 35.04, min_lon=135.76, max_lon = 135.78)
    data_loader = DataLoader("Kyoto", latlon_range=latlon_range, prior_type=BUS)
    
    epsilons = [0.01, 0.02, 0.03, 0.04, 0.05]
    results = {}
    
    for epsilon in epsilons:
        mec = Mec.GraphExponentialMechanism(data_loader)
        mec.build_distribution(epsilon)
        baseline_score = Mec.score(mec)
        mec.build_optimal_distribution()
        opt_score = Mec.score(mec)
        
        results[epsilon] = {"baseline": baseline_score, "opt": opt_score}
    print(results)
    
    with open(result_dir / f"{data_loader.data_name}.json", "w") as f:
        json.dump(results, f)