"""Compare held-out recommendations with feasible low-error baselines."""

import argparse
import json
from pathlib import Path

import pandas as pd

if __package__:
    from .recommend import load_model
else:
    from recommend import load_model

BASE_DIR = Path(__file__).resolve().parent.parent


def evaluate(artifacts, predictions, output_dir):
    if predictions["circuit"].duplicated().any():
        raise ValueError("Duplicate circuits in predictions")
    if set(predictions["circuit"]) != set(artifacts["test_circuits"]):
        raise ValueError("Predictions do not match the model's held-out circuits")
    cache = artifacts["backend_features_cache"]
    measured = artifacts["test_df"].set_index(["Circuit", "Backend"])["FOM"]
    comparisons = []
    for label, feature in [
        ("lowest_readout_error", "ro_error_mean"),
        ("lowest_single_qubit_error", "s1_error_mean"),
        ("lowest_two_qubit_error", "cnot_error_mean"),
    ]:
        diffs, percentages = [], []
        for row in predictions.itertuples():
            width = artifacts["circuit_features_cache"][row.circuit]["width"]
            eligible = [
                name
                for name in artifacts["all_backends"]
                if cache[name]["backend_num_qubits"] >= width
            ]
            if row.pred_best_backend not in eligible:
                raise ValueError(f"Infeasible recommendation for {row.circuit}")
            baseline = min(eligible, key=lambda name: (cache[name][feature], name))
            actual = float(measured.loc[(row.circuit, row.pred_best_backend)])
            if abs(actual - row.actual_fom_selected) > 1e-6:
                raise ValueError(f"Prediction measurements do not match model for {row.circuit}")
            baseline_fom = float(measured.loc[(row.circuit, baseline)])
            diffs.append(actual - baseline_fom)
            if baseline_fom > 0:
                percentages.append((actual - baseline_fom) / baseline_fom * 100)
        delta = pd.Series(diffs)
        comparisons.append(
            {
                "baseline": label,
                "circuits": len(delta),
                "better_pct": float((delta > 1e-6).mean() * 100),
                "worse_pct": float((delta < -1e-6).mean() * 100),
                "equal_count": int((delta.abs() <= 1e-6).sum()),
                "mean_fom_difference": float(delta.mean()),
                "mean_improvement_pct": sum(percentages) / len(percentages)
                if percentages
                else None,
                "relative_comparison_circuits": len(percentages),
            }
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(comparisons)
    frame.to_csv(output_dir / "baseline_comparison_optimized.csv", index=False)
    (output_dir / "baseline_analysis_optimized.json").write_text(
        json.dumps(comparisons, indent=2) + "\n",
        encoding="utf-8",
    )
    print(frame.to_string(index=False))
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=BASE_DIR / "results/trained_model_optimized.pkl"
    )
    parser.add_argument(
        "--predictions", type=Path, default=BASE_DIR / "results/predictions_optimized.csv"
    )
    parser.add_argument("--output", type=Path, default=BASE_DIR / "results")
    args = parser.parse_args(argv)
    evaluate(load_model(args.model), pd.read_csv(args.predictions), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
