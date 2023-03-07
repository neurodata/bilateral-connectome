from pathlib import Path
import pickle

out_path = Path("bilateral-connectome/results/outputs/perturbations_unmatched")

with open(out_path / "unmatched_power_full.pickle", "rb") as f:
    results = pickle.load(f)

simple_results = results[
    [
        "stat",
        "pvalue",
        "test",
        "perturbation",
        "effect_size",
        "sim",
        "perturb_elapsed",
        "test_elapsed",
        "target",
        "perturbation_type",
        "model_base",
        "model_postfix",
    ]
]
simple_results.to_csv(out_path / "unmatched_power_simple.csv")
