# Copyright (c) 2024 Q-Selector Contributors
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Pytest configuration and fixtures for Q-Selector tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass


@pytest.fixture(scope="session")
def qasm_sample_path() -> str:
    """Return the path to a sample QASM file for testing."""
    return "data/QASM/ae_indep_qiskit_10.qasm"


@pytest.fixture(scope="session")
def trained_model_path() -> str:
    """Return the path to the trained model for testing."""
    return "results/trained_model_optimized.pkl"
