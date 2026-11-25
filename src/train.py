"""Train a random forest on the bundled circuit/backend measurements."""

import argparse
import hashlib
import json
import pickle
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from .features import extract_circuit_features, feature_row, load_backend_features
else:
    from features import extract_circuit_features, feature_row, load_backend_features

BASE_DIR = Path(__file__).resolve().parent.parent
RANDOM_SEED = 42
THRESHOLD = 0.005


def split_circuits(df):
    best = df.loc[df.groupby("Circuit")["FOM"].idxmax()].set_index("Circuit")["Backend"]
    circuits = sorted(best.index)
    counts = Counter(best)
    rare = [c for c in circuits if counts[best[c]] == 1]
    remaining = [c for c in circuits if c not in rare]
    train_size = int(len(circuits) * 0.7) - len(rare)
    split = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=RANDOM_SEED)
    train_idx, test_idx = next(split.split(remaining, [best[c] for c in remaining]))
    return rare + [remaining[i] for i in train_idx], [remaining[i] for i in test_idx]


def train(data_csv, qasm_folder, backend_folder, output_dir, n_jobs=-1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_csv, usecols=["Circuit", "Backend", "Hellinger_Fidelity"])
    if df.isna().any().any() or not df["Hellinger_Fidelity"].between(0, 1).all():
        raise ValueError("Dataset contains missing values or fidelity outside [0, 1]")
    if df.duplicated(["Circuit", "Backend"]).any():
        raise ValueError("Dataset contains duplicate circuit/backend pairs")
    df["FOM"] = df["Hellinger_Fidelity"] * 10
    backends = load_backend_features(backend_folder)
    unknown = set(df["Backend"]) - backends.keys()
    if unknown:
        raise ValueError(f"Missing backend profiles: {sorted(unknown)}")
    circuits = {
        name: extract_circuit_features(Path(qasm_folder) / f"{name}.qasm")
        for name in sorted(df["Circuit"].unique())
    }
    eligible = df.apply(
        lambda row: (
            circuits[row["Circuit"]]["width"] <= backends[row["Backend"]]["backend_num_qubits"]
        ),
        axis=1,
    )
    excluded = int((~eligible).sum())
    df = df[eligible].copy()
    train_names, test_names = split_circuits(df)
    train_df = df[df["Circuit"].isin(train_names)].copy()
    test_df = df[df["Circuit"].isin(test_names)].copy()

    def matrix(frame):
        return pd.DataFrame(
            [
                feature_row(circuits[row.Circuit], backends[row.Backend])
                for row in frame.itertuples()
            ]
        )

    x_train, x_test = matrix(train_df), matrix(test_df)
    y_train, y_test = train_df["FOM"].to_numpy(), test_df["FOM"].to_numpy()
    selector = RandomForestRegressor(n_estimators=80, random_state=RANDOM_SEED, n_jobs=n_jobs)
    selector.fit(x_train, y_train)
    importance = pd.Series(selector.feature_importances_, index=x_train.columns)
    selected = importance[importance > THRESHOLD].index.tolist()
    if not selected:
        raise ValueError("No features passed the importance threshold")
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=n_jobs),
            ),
        ]
    )
    pipeline.fit(x_train[selected], y_train)
    train_pred, test_pred = pipeline.predict(x_train[selected]), pipeline.predict(x_test[selected])
    metrics = {
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "train_mae": float(mean_absolute_error(y_train, train_pred)),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
    }
    preference_df = train_df.assign(algorithm=train_df["Circuit"].str.split("_").str[0])
    preferences = {
        algorithm: group.groupby("Backend")["FOM"].mean().to_dict()
        for algorithm, group in preference_df.groupby("algorithm")
    }
    baselines = {
        key: min(backends, key=lambda name: (backends[name][feature], name))
        for key, feature in [
            ("lowest_ro_backend", "ro_error_mean"),
            ("lowest_s1_backend", "s1_error_mean"),
            ("lowest_cnot_backend", "cnot_error_mean"),
        ]
    }
    metadata = {
        "random_seed": RANDOM_SEED,
        "feature_threshold": THRESHOLD,
        "propensity_weight": 0.2,
        "default_algorithm": None,
        "versions": {name: version(name) for name in ["qiskit", "scikit-learn", "numpy", "pandas"]},
        "dataset_sha256": hashlib.sha256(Path(data_csv).read_bytes()).hexdigest(),
        "training_circuits": len(train_names),
        "test_circuits": len(test_names),
        "excluded_infeasible_pairs": excluded,
        "selected_features": selected,
        **metrics,
    }
    artifacts = {
        "pipeline": pipeline,
        "model": pipeline.named_steps["model"],
        "scaler": pipeline.named_steps["scaler"],
        "selected_features": selected,
        "all_backends": sorted(backends),
        "backend_features_cache": backends,
        "circuit_features_cache": circuits,
        "train_circuits": train_names,
        "test_circuits": test_names,
        "algorithm_preferences": preferences,
        "global_avg_fom": float(train_df["FOM"].mean()),
        "propensity_weight": 0.2,
        "train_df": train_df,
        "test_df": test_df,
        "metadata": metadata,
        **baselines,
        **metrics,
    }
    with (output_dir / "trained_model_optimized.pkl").open("wb") as stream:
        pickle.dump(artifacts, stream, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(
        {
            "feature": importance.index,
            "importance": importance.values,
            "retained": importance.index.isin(selected),
        }
    ).to_csv(output_dir / "feature_list_threshold_005.csv", index=False)
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [f"{key}: {value:.6f}" for key, value in metrics.items()]
    (output_dir / "optimization_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Train: {len(train_names)} circuits; test: {len(test_names)}; features: {len(selected)}")
    print("\n".join(report))
    return artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=BASE_DIR / "data/MQTBench_578_analysis_results.csv"
    )
    parser.add_argument("--qasm", type=Path, default=BASE_DIR / "data/QASM")
    parser.add_argument("--backend-json", type=Path, default=BASE_DIR / "data/backend_json")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "results")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args(argv)
    train(args.data, args.qasm, args.backend_json, args.output, args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
