"""Evaluate backend rankings on the held-out circuits."""

import argparse
import json
from pathlib import Path

import pandas as pd

if __package__:
    from .features import rank_backends
    from .recommend import load_model
else:
    from features import rank_backends
    from recommend import load_model

BASE_DIR = Path(__file__).resolve().parent.parent


def predict_test_set(artifacts, output_dir, algorithm=None):
    rows = []
    test_df = artifacts["test_df"]
    for name in artifacts["test_circuits"]:
        cf = artifacts["circuit_features_cache"][name]
        rankings = rank_backends(cf, artifacts, algorithm)
        measured = test_df[test_df["Circuit"] == name].set_index("Backend")["FOM"]
        eligible = {row["backend"] for row in rankings}
        measured = measured[measured.index.isin(eligible)].sort_values(ascending=False)
        if measured.empty or rankings[0]["backend"] not in measured.index:
            raise ValueError(f"Missing held-out measurements for {name}")
        best, selected = str(measured.index[0]), rankings[0]
        top3 = [row["backend"] for row in rankings[:3]]
        rows.append(
            {
                "circuit": name,
                "true_best_backend": best,
                "true_fom": float(measured.iloc[0]),
                "pred_best_backend": selected["backend"],
                "pred_fom": selected["predicted_fom"],
                "actual_fom_selected": float(measured[selected["backend"]]),
                "correct": selected["backend"] == best,
                "pred_rank": int(measured.index.get_loc(selected["backend"])) + 1,
                "top3_backends": json.dumps(top3),
                "top3_preds": json.dumps([row["predicted_fom"] for row in rankings[:3]]),
                "top3_correct": best in top3,
                "algorithm": algorithm,
            }
        )
    if not rows:
        raise ValueError("Model has no held-out circuits")
    frame = pd.DataFrame(rows)
    summary = {
        "total_test_circuits": len(frame),
        "algorithm": algorithm,
        "top1_accuracy": float(frame["correct"].mean()),
        "top3_accuracy": float(frame["top3_correct"].mean()),
        "mean_fom_regret": float((frame["true_fom"] - frame["actual_fom_selected"]).mean()),
        "dataset_sha256": artifacts.get("metadata", {}).get("dataset_sha256"),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "predictions_optimized.csv", index=False)
    (output_dir / "prediction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=BASE_DIR / "results/trained_model_optimized.pkl"
    )
    parser.add_argument("--output", type=Path, default=BASE_DIR / "results")
    parser.add_argument("--algorithm", help="Apply preferences for this algorithm to every circuit")
    args = parser.parse_args(argv)
    predict_test_set(load_model(args.model), args.output, args.algorithm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
