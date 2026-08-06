# Copyright (c) 2024 Q-Selector Contributors
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Q-Selector interaction and backend features."""

from __future__ import annotations

from pathlib import Path

import pytest


class TestInteractionFeatures:
    """Test suite for circuit-backend interaction features."""

    def test_interaction_feats_importable(self) -> None:
        """Verify that the interaction features function can be imported."""
        from src.recommend import interaction_feats  # type: ignore[import-untyped]

        assert interaction_feats is not None

    def test_load_coupling_maps(self) -> None:
        """Test that coupling maps can be loaded from backend JSON files."""
        from src.recommend import load_coupling_maps  # type: ignore[import-untyped]

        bk_json_dir = Path("data/backend_json")
        if not bk_json_dir.exists():
            pytest.skip("Backend JSON directory not found")

        # Use a few known backend names
        test_backends = ["FakeAlmadenV2", "FakeKolkataV2"]
        maps = load_coupling_maps(str(bk_json_dir), test_backends)
        assert "FakeAlmadenV2" in maps
        assert len(maps["FakeAlmadenV2"]) > 0, "Coupling map should not be empty"
