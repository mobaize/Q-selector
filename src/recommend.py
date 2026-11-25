"""Rank processor profiles for OpenQASM 2 circuits."""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

if __package__:
    from .features import interaction_feats as interaction_feats
else:
    from features import interaction_feats as interaction_feats

if __package__:
    from .features import (
        coupling_edges,
        extract_circuit_features,
        load_json,
        rank_backends,
    )
else:
    from features import (
        coupling_edges,
        extract_circuit_features,
        load_json,
        rank_backends,
    )

BASE_DIR = Path(__file__).resolve().parent.parent
QASM_EXTENSIONS = {".qasm", ".QASM"}


def get_algorithm_type(circuit_name):
    """Parse legacy dataset names; inference uses an explicit algorithm argument."""
    return Path(circuit_name).stem.split("_")[0]


def collect_qasm_files(path):
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix.lower() == ".qasm" else []
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".qasm")
    return []


def load_coupling_maps(backend_json_folder, all_backends):
    maps = {}
    folder = Path(backend_json_folder)
    for name in all_backends:
        key = name.lower().replace("fake", "").replace("v2", "")
        for path in [folder / f"conf_{name.lower()}.json", folder / f"conf_{key}.json"]:
            if path.exists():
                conf = load_json(path)
                if conf is None or "coupling_map" not in conf:
                    raise ValueError(f"Invalid backend configuration: {path}")
                maps[name] = coupling_edges(conf)
                break
    return maps


def load_model(model_path, backend_json_folder=None):
    """Load locally trained pickle artifacts and optional coupling maps."""
    with Path(model_path).open("rb") as stream:
        artifacts = pickle.load(stream)
    required = {"model", "scaler", "selected_features", "all_backends", "backend_features_cache"}
    missing = required.difference(artifacts)
    if missing:
        raise ValueError(f"Model artifacts missing: {', '.join(sorted(missing))}")
    if backend_json_folder is not None:
        maps = load_coupling_maps(backend_json_folder, artifacts["all_backends"])
        for name, edges in maps.items():
            artifacts["backend_features_cache"][name]["_coupling_set"] = edges
    return artifacts


def predict_circuit(qasm_path, artifacts, algorithm=None):
    path = Path(qasm_path)
    features = extract_circuit_features(path)
    rankings = rank_backends(features, artifacts, algorithm=algorithm)
    return {
        "circuit_name": path.stem,
        "qasm_filename": path.name,
        "qasm_path": str(path),
        "algorithm": algorithm,
        "features": {
            name: features[name]
            for name in [
                "width",
                "depth",
                "total_gates",
                "single_qubit_gates",
                "two_qubit_gates",
            ]
        },
        "rankings": rankings,
        "best_backend": rankings[0]["backend"],
        "best_fom": rankings[0]["predicted_fom"],
    }


def save_results(results, output_dir, failures=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    best_rows, rank_rows = [], []
    for result in results:
        base = {
            "circuit": result["circuit_name"],
            "qasm_file": result["qasm_filename"],
            "qasm_path": result["qasm_path"],
        }
        best = result["rankings"][0]
        best_rows.append(
            {
                **base,
                **result["features"],
                "recommended_backend": best["backend"],
                "predicted_fom": best["predicted_fom"],
                "num_qubits": best["num_qubits"],
            }
        )
        for rank, backend in enumerate(result["rankings"], 1):
            rank_rows.append({**base, "rank": rank, **backend})
    pd.DataFrame(best_rows).to_csv(output_dir / "best_recommendation.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(output_dir / "full_ranking.csv", index=False)
    payload = {"total_circuits": len(results), "failures": failures or [], "results": results}
    (output_dir / "full_ranking.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="QASM file or directory (searched recursively)")
    parser.add_argument(
        "--model", type=Path, default=BASE_DIR / "results/trained_model_optimized.pkl"
    )
    parser.add_argument(
        "--backend-json",
        type=Path,
        default=None,
        help="Optional directory of coupling maps; defaults to the model cache",
    )
    parser.add_argument("--algorithm", help="Apply training-set preferences for this algorithm")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/your_work"),
        help="Parent directory for timestamped results",
    )
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    if args.top < 1:
        parser.error("--top must be positive")
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}. Run python src/train.py first.")
    if args.backend_json is not None and not args.backend_json.is_dir():
        parser.error(f"Backend directory not found: {args.backend_json}")
    files = collect_qasm_files(args.input)
    if not files:
        parser.error(f"No QASM files found: {args.input}")
    try:
        artifacts = load_model(args.model, args.backend_json)
    except (OSError, ValueError, pickle.UnpicklingError) as exc:
        parser.error(str(exc))
    results, failures = [], []
    for path in files:
        try:
            result = predict_circuit(path, artifacts, args.algorithm)
        except ValueError as exc:
            failures.append({"file": str(path), "error": str(exc)})
            print(f"{path}: {exc}", file=sys.stderr)
            continue
        results.append(result)
        print(f"{path}: {result['best_backend']} (FOM {result['best_fom']:.4f})")
        for rank, row in enumerate(result["rankings"][: args.top], 1):
            print(f"  {rank:2d}  {row['backend']:<24} {row['predicted_fom']:.4f}")
    if results and not args.no_save:
        output_dir = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_results(results, output_dir, failures)
        print(f"Saved to {output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
