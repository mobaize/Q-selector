# Copyright (c) 2024 Q-Selector Contributors
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Q-Selector recommendation logic."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestRecommend:
    """Test suite for the recommendation logic."""

    def test_recommend_function_importable(self) -> None:
        """Verify that the recommend function can be imported."""
        from src.recommend import load_model, predict_circuit  # type: ignore[import-untyped]

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
