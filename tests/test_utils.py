# Copyright (c) 2024 Q-Selector Contributors
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Q-Selector utils and helper functions."""

from __future__ import annotations

from pathlib import Path


class TestUtils:
    """Test suite for utility functions."""

    def test_load_json_exists(self) -> None:
        """Test loading a valid backend JSON file."""
        from src.recommend import load_json  # type: ignore[import-untyped]

        json_path = Path("data/backend_json/conf_fakealmadenv2.json")
        if not json_path.exists():
            return  # Skip if file not available

        data = load_json(str(json_path))
        assert data is not None
        assert "name" in data
        assert data["name"] == "FakeAlmadenV2"

    def test_load_json_not_exists(self) -> None:
        """Test loading a non-existent JSON file returns None."""
        from src.recommend import load_json  # type: ignore[import-untyped]

        data = load_json("nonexistent_file.json")
        assert data is None

    def test_get_algorithm_type(self) -> None:
        """Test algorithm type extraction from circuit name."""
        from src.recommend import get_algorithm_type  # type: ignore[import-untyped]

        assert get_algorithm_type("ae_indep_qiskit_10") == "ae"
        assert get_algorithm_type("dj_indep_qiskit_5") == "dj"
        assert get_algorithm_type("ghz_indep_qiskit_3") == "ghz"
        assert get_algorithm_type("unknown") == "unknown"
