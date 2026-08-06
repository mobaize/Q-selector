# Copyright (c) 2024 Q-Selector Contributors
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Q-Selector recommendation logic."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest


@pytest.fixture
def trained_model() -> dict | None:
    """Load the trained model from the results directory."""
    model_path = Path("results/trained_model_optimized.pkl")
    if not model_path.exists():
        pytest.skip("Trained model not found. Run src/train.py first.")
    with open(model_path, "rb") as f:
        return pickle.load(f)


class TestRecommend:
    """Test suite for the recommendation logic."""

    def test_model_artifacts_exist(self, trained_model: dict) -> None:
        """Verify that the trained model contains all required artifacts."""
        assert trained_model is not None
        assert "model" in trained_model
        assert "scaler" in trained_model
        assert "selected_features" in trained_model
        assert "all_backends" in trained_model
        assert "backend_features_cache" in trained_model

    def test_all_backends_available(self, trained_model: dict) -> None:
        """Verify that the model has multiple backends to choose from."""
        backends = trained_model["all_backends"]
        assert len(backends) >= 2, "Model should support at least 2 backends"

    def test_selected_features(self, trained_model: dict) -> None:
        """Verify that feature selection produced a reasonable feature set."""
        features = trained_model["selected_features"]
        assert len(features) >= 5, "Should have at least 5 selected features"

    def test_recommend_function_importable(self) -> None:
        """Verify that the recommend function can be imported."""
        from src.recommend import predict_circuit, load_model  # type: ignore[import-untyped]

        assert predict_circuit is not None
        assert load_model is not None

    def test_collect_qasm_files_single(self) -> None:
        """Test that file collection works for a single file."""
        from src.recommend import collect_qasm_files  # type: ignore[import-untyped]

        qasm_path = Path("data/QASM/ae_indep_qiskit_10.qasm")
        if not qasm_path.exists():
            pytest.skip("QASM sample file not found")

        files = collect_qasm_files(str(qasm_path))
        assert len(files) == 1
        assert files[0].name == "ae_indep_qiskit_10.qasm"

    def test_collect_qasm_files_directory(self) -> None:
        """Test that file collection works for a directory."""
        from src.recommend import collect_qasm_files  # type: ignore[import-untyped]

        qasm_dir = Path("data/QASM")
        if not qasm_dir.exists():
            pytest.skip("QASM directory not found")

        files = collect_qasm_files(str(qasm_dir))
        assert len(files) >= 10, "Should find at least 10 QASM files in the directory"
