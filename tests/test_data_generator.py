from unittest.mock import Mock

import pandas as pd
import pytest
from qiskit import QuantumCircuit

from src.data_generator import DataGeneratorConfig, NoiseModel, QuantumCircuitDataGenerator


@pytest.fixture
def generator(tmp_path):
    return QuantumCircuitDataGenerator(
        DataGeneratorConfig(
            qasm_dir=str(tmp_path),
            output_dir=str(tmp_path / "output"),
            backends=["FakeYorktownV2"],
            shots=512,
            seed=7,
        )
    )


def test_unknown_backend_is_rejected(tmp_path):
    config = DataGeneratorConfig(qasm_dir=str(tmp_path), backends=["not_a_backend"])
    with pytest.raises(ValueError, match="Unknown or unavailable"):
        QuantumCircuitDataGenerator(config)


def test_failed_noise_model_raises(generator, monkeypatch):
    monkeypatch.setattr(generator, "get_backend", lambda _: object())
    monkeypatch.setattr(
        NoiseModel, "from_backend", Mock(side_effect=ValueError("calibration failure"))
    )
    with pytest.raises(RuntimeError, match="Cannot create noise model"):
        generator.create_noise_model("FakeYorktownV2")


def test_missing_noise_cannot_produce_perfect_label(generator, monkeypatch):
    monkeypatch.setattr(generator, "create_noise_model", lambda _: None)
    with pytest.raises(RuntimeError, match="Missing noise model"):
        generator.run_noisy_simulation(QuantumCircuit(1), "FakeYorktownV2")


def test_empty_noise_model_is_rejected(generator, monkeypatch):
    monkeypatch.setattr(generator, "get_backend", lambda _: object())
    monkeypatch.setattr(NoiseModel, "from_backend", Mock(return_value=NoiseModel()))
    with pytest.raises(RuntimeError, match="Empty noise model"):
        generator.create_noise_model("FakeYorktownV2")


def test_capacity_checked_before_simulation(generator, monkeypatch):
    noise = Mock(side_effect=AssertionError("Noise construction must not run"))
    monkeypatch.setattr(generator, "create_noise_model", noise)
    with pytest.raises(ValueError, match="exceeds.*capacity"):
        generator.run_noisy_simulation(QuantumCircuit(6), "FakeYorktownV2")
    noise.assert_not_called()


def test_failed_simulation_is_not_written(generator, tmp_path, monkeypatch):
    path = tmp_path / "bell.qasm"
    path.write_text(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; '
        "h q[0]; cx q[0],q[1]; measure q -> c;",
        encoding="utf-8",
    )
    monkeypatch.setattr(generator, "create_noise_model", Mock(side_effect=RuntimeError("failed")))
    result = generator.run()
    assert result["analysis_results"]["FakeYorktownV2"] == {"success": 0, "error": 1}
    assert not generator.unified_csv_path.exists()


def test_local_simulation_is_seeded_and_respects_target(generator, monkeypatch):
    import src.data_generator as module

    actual_transpile = module.transpile
    transpile = Mock(wraps=actual_transpile)
    monkeypatch.setattr(module, "transpile", transpile)
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    first = generator.run_noisy_simulation(circuit, "FakeYorktownV2")
    second = generator.run_noisy_simulation(circuit, "FakeYorktownV2")
    assert first[:4] == second[:4]
    assert 0 <= first[2] <= 1
    backend = generator.get_backend("FakeYorktownV2")
    assert transpile.call_args_list[1].args[1] is backend
    assert first[-1].num_qubits <= backend.num_qubits
    single, two, readout = generator.calculate_error_rates("FakeYorktownV2")
    assert all(0 < value < 1 for value in [single, two, readout])


def test_ideal_mode_is_explicit(generator):
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.measure_all()
    ideal, noisy, fidelity, _, _ = generator.run_noisy_simulation(circuit, "ideal")
    assert ideal == noisy
    assert fidelity == pytest.approx(1.0)


def test_resume_rejects_changed_run_settings(generator):
    pd.DataFrame(
        {
            "Circuit": ["bell"],
            "Backend": ["FakeYorktownV2"],
            "Shots": [1],
            "Optimization_Level": [1],
            "Seed": [7],
        }
    ).to_csv(generator.unified_csv_path, index=False)
    with pytest.raises(ValueError, match="different run settings"):
        QuantumCircuitDataGenerator(generator.config)
