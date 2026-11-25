import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import extract_circuit_features, rank_backends
from src.predict import predict_test_set
from src.recommend import main, predict_circuit


@pytest.fixture
def circuit_path(tmp_path):
    path = tmp_path / "ae_sample.qasm"
    path.write_text(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[10]; h q[0]; cx q[0],q[1];',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def artifacts():
    frame = pd.DataFrame({"width": [2, 10, 20]})
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", DummyRegressor())])
    pipeline.fit(frame, np.array([5.0, 5.0, 5.0]))
    return {
        "pipeline": pipeline,
        "model": pipeline.named_steps["model"],
        "scaler": pipeline.named_steps["scaler"],
        "selected_features": ["width"],
        "all_backends": ["a_small", "b_large", "c_large"],
        "backend_features_cache": {
            "a_small": {"backend_num_qubits": 5},
            "b_large": {"backend_num_qubits": 20},
            "c_large": {"backend_num_qubits": 20},
        },
        "algorithm_preferences": {"ae": {"b_large": 1.0, "c_large": 9.0}},
        "global_avg_fom": 5.0,
        "propensity_weight": 0.2,
    }


def test_capacity_filters_small_backend(circuit_path, artifacts):
    result = predict_circuit(circuit_path, artifacts)
    assert {row["backend"] for row in result["rankings"]} == {"b_large", "c_large"}
    assert all(row["num_qubits"] >= 10 for row in result["rankings"])


def test_no_feasible_backend(circuit_path, artifacts):
    artifacts["all_backends"] = ["a_small"]
    with pytest.raises(ValueError, match="No backend can accommodate"):
        predict_circuit(circuit_path, artifacts)


def test_renaming_does_not_change_default_ranking(circuit_path, artifacts):
    first = predict_circuit(circuit_path, artifacts)
    renamed = circuit_path.with_name("renamed.qasm")
    renamed.write_bytes(circuit_path.read_bytes())
    second = predict_circuit(renamed, artifacts)
    assert first["rankings"] == second["rankings"]
    assert first["algorithm"] is None


def test_explicit_preference_changes_ranking(circuit_path, artifacts):
    assert predict_circuit(circuit_path, artifacts)["best_backend"] == "b_large"
    assert predict_circuit(circuit_path, artifacts, algorithm="ae")["best_backend"] == "c_large"
    with pytest.raises(ValueError, match="Unknown algorithm"):
        predict_circuit(circuit_path, artifacts, algorithm="unrecognized")


def test_invalid_model_score_is_rejected(circuit_path, artifacts, monkeypatch):
    monkeypatch.setattr(artifacts["pipeline"], "predict", lambda _: np.array([np.nan, 1.0]))
    with pytest.raises(ValueError, match="Non-finite"):
        predict_circuit(circuit_path, artifacts)


def test_held_out_and_cli_rankings_agree(circuit_path, artifacts, tmp_path):
    name = circuit_path.stem
    artifacts["test_circuits"] = [name]
    artifacts["circuit_features_cache"] = {name: extract_circuit_features(circuit_path)}
    artifacts["test_df"] = pd.DataFrame(
        {
            "Circuit": [name, name],
            "Backend": ["b_large", "c_large"],
            "FOM": [4.0, 6.0],
        }
    )
    frame = predict_test_set(artifacts, tmp_path / "predictions")
    assert (
        frame.iloc[0]["pred_best_backend"]
        == predict_circuit(circuit_path, artifacts)["best_backend"]
    )
    assert frame.iloc[0]["actual_fom_selected"] == 4.0
    assert (
        json.loads((tmp_path / "predictions/prediction_summary.json").read_text())["algorithm"]
        is None
    )


def test_partial_batch_failure_is_recorded(circuit_path, artifacts, tmp_path):
    broken = circuit_path.with_name("broken.qasm")
    broken.write_text("invalid QASM", encoding="utf-8")
    model = tmp_path / "model.pkl"
    model.write_bytes(pickle.dumps(artifacts))
    output = tmp_path / "output"
    assert main([str(tmp_path), "--model", str(model), "--output", str(output)]) == 1
    report = next(output.glob("*/full_ranking.json"))
    payload = json.loads(report.read_text())
    assert payload["total_circuits"] == 1
    assert payload["failures"][0]["file"] == str(broken)


def test_model_cache_suffices_without_external_backend_files(circuit_path, artifacts, tmp_path):
    model = tmp_path / "model.pkl"
    model.write_bytes(pickle.dumps(artifacts))
    assert main([str(circuit_path), "--model", str(model), "--no-save"]) == 0


def test_legacy_artifacts_share_capacity_checks(circuit_path, artifacts):
    del artifacts["pipeline"]
    rankings = rank_backends(extract_circuit_features(circuit_path), artifacts)
    assert "a_small" not in {row["backend"] for row in rankings}
