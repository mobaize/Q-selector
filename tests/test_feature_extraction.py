# Copyright (c) 2024 Q-Selector Contributors
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Q-Selector circuit feature extraction."""

from __future__ import annotations

from pathlib import Path

import pytest
from qiskit import QuantumCircuit


@pytest.fixture
def sample_circuit() -> QuantumCircuit:
    """Load a sample QASM circuit for testing features."""
    qasm_path = Path("data/QASM/ae_indep_qiskit_10.qasm")
    if not qasm_path.exists():
        pytest.skip("QASM sample file not found")
    return QuantumCircuit.from_qasm_file(str(qasm_path))


class TestFeatureExtraction:
    """Test suite for circuit feature extraction."""

    def test_circuit_loading(self, sample_circuit: QuantumCircuit) -> None:
        """Verify that a QASM circuit can be loaded."""
        assert sample_circuit is not None
        assert sample_circuit.num_qubits > 0

    def test_circuit_has_gates(self, sample_circuit: QuantumCircuit) -> None:
        """Verify that the circuit contains gate operations."""
        assert len(sample_circuit.data) > 0

    def test_extract_circuit_features_importable(self) -> None:
        """Verify that the extract function can be imported."""
        from src.recommend import extract_circuit_features  # type: ignore[import-untyped]

        assert extract_circuit_features is not None

    def test_extract_circuit_features_runs(self) -> None:
        """Test that feature extraction runs successfully on a valid QASM file."""
        from src.recommend import extract_circuit_features  # type: ignore[import-untyped]

        qasm_path = Path("data/QASM/ae_indep_qiskit_10.qasm")
        if not qasm_path.exists():
            pytest.skip("QASM sample file not found")

        feats = extract_circuit_features(qasm_path)
        assert feats is not None
        assert feats["width"] > 0
        assert feats["depth"] > 0
        assert feats["total_gates"] > 0
        assert "cx_gates" in feats
        assert "rz_gates" in feats
        assert "_circuit_edges" in feats
