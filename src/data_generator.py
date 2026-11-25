"""Generate labeled circuit/backend measurements with local Aer simulations."""

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import fake_provider

BACKEND_NAMES = [
    "FakeAlmadenV2",
    "FakeBoeblingenV2",
    "FakeBrisbane",
    "FakeBrooklynV2",
    "FakeCairoV2",
    "FakeCasablancaV2",
    "FakeGeneva",
    "FakeHanoiV2",
    "FakeJohannesburgV2",
    "FakeKolkataV2",
    "FakeKyoto",
    "FakeManhattanV2",
    "FakeMontrealV2",
    "FakeNairobiV2",
    "FakeParisV2",
    "FakeQuebec",
    "FakeRochesterV2",
    "FakeSherbrooke",
    "FakeSingaporeV2",
    "FakeSydneyV2",
    "FakeTorontoV2",
    "FakeValenciaV2",
    "FakeWashingtonV2",
    "FakeYorktownV2",
]


def calculate_polarization_multiple_targets(counts, correct_states=None):
    if not counts or not sum(counts.values()):
        return 0.0
    width = len(next(iter(counts)).replace(" ", ""))
    if width == 0:
        return 0.0
    if correct_states is None:
        correct_states = ["0" * width]
    elif isinstance(correct_states, str):
        correct_states = [correct_states]
    states = {state.ljust(width, "0")[:width] for state in correct_states}
    success = sum(counts.get(state, 0) for state in states) / sum(counts.values())
    uniform = 1 / 2**width
    return max(-1.0, min(1.0, (success - uniform) / (1 - uniform)))


@dataclass
class DataGeneratorConfig:
    qasm_dir: str
    output_dir: str = "results"
    unified_csv: str | None = None
    shots: int = 2000
    backends: list[str] = field(default_factory=lambda: ["FakeWashingtonV2"])
    incremental_update: bool = True
    optimization_level: int = 1
    seed: int = 42

    def __post_init__(self):
        if self.shots < 1:
            raise ValueError("shots must be positive")
        if self.optimization_level not in range(4):
            raise ValueError("optimization_level must be 0, 1, 2 or 3")
        if self.unified_csv is None:
            self.unified_csv = (
                f"{Path(self.qasm_dir).name}_analysis_results_O{self.optimization_level}.csv"
            )


class QuantumCircuitDataGenerator:
    BACKEND_MAP = {
        name: getattr(fake_provider, name) for name in BACKEND_NAMES if hasattr(fake_provider, name)
    }

    def __init__(self, config):
        self.config = config
        unknown = set(config.backends) - self.BACKEND_MAP.keys() - {"ideal"}
        if unknown:
            raise ValueError(f"Unknown or unavailable backends: {sorted(unknown)}")
        self._backends = {}
        self.unified_csv_path = Path(config.unified_csv)
        if not self.unified_csv_path.is_absolute():
            self.unified_csv_path = Path(config.output_dir) / self.unified_csv_path
        self.unified_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.processed_circuits = self.load_processed_circuits()

    def get_backend(self, name):
        if name not in self.BACKEND_MAP:
            raise ValueError(f"Unknown or unavailable backend: {name}")
        if name not in self._backends:
            self._backends[name] = self.BACKEND_MAP[name]()
        return self._backends[name]

    def load_processed_circuits(self):
        if not self.unified_csv_path.exists():
            return set()
        df = pd.read_csv(self.unified_csv_path)
        required = {"Circuit", "Backend", "Shots", "Optimization_Level", "Seed"}
        if not required.issubset(df.columns):
            raise ValueError("Existing CSV lacks run settings; use a new output file")
        if not (
            df["Shots"].eq(self.config.shots).all()
            and df["Optimization_Level"].eq(self.config.optimization_level).all()
            and df["Seed"].eq(self.config.seed).all()
        ):
            raise ValueError("Existing CSV has different run settings; use a new output file")
        return set(zip(df["Circuit"], df["Backend"]))

    def is_processed(self, circuit_name, backend_name):
        return (circuit_name, backend_name) in self.processed_circuits

    def calculate_error_rates(self, backend_name):
        if backend_name == "ideal":
            return 0.0, 0.0, 0.0
        target = self.get_backend(backend_name).target
        single, two, readout = [], [], []
        for name in target.operation_names:
            if name in {"reset", "delay"}:
                continue
            for qubits, properties in target[name].items():
                if properties is None or properties.error is None or qubits is None:
                    continue
                error = properties.error
                if not 0 <= error <= 1:
                    raise ValueError(f"Invalid error rate for {backend_name}: {name}{qubits}")
                if name == "measure":
                    readout.append(error)
                elif len(qubits) == 1:
                    single.append(error)
                elif len(qubits) == 2:
                    two.append(error)
        if not single or not two or not readout:
            raise ValueError(f"Missing error rates for {backend_name}")
        return tuple(float(np.mean(values)) for values in [single, two, readout])

    def create_noise_model(self, backend_name):
        if backend_name == "ideal":
            return None
        backend = self.get_backend(backend_name)
        try:
            noise_model = NoiseModel.from_backend(backend)
        except Exception as exc:
            raise RuntimeError(f"Cannot create noise model for {backend_name}: {exc}") from exc
        if noise_model is None or noise_model.is_ideal():
            raise RuntimeError(f"Empty noise model for {backend_name}")
        return noise_model

    def run_noisy_simulation(self, qc, backend_name):
        backend = None if backend_name == "ideal" else self.get_backend(backend_name)
        if backend is not None and qc.num_qubits > backend.num_qubits:
            raise ValueError(f"{qc.num_qubits}-qubit circuit exceeds {backend_name} capacity")
        noise_model = self.create_noise_model(backend_name)
        if backend_name != "ideal" and noise_model is None:
            raise RuntimeError(f"Missing noise model for {backend_name}")
        ideal_sim = AerSimulator(seed_simulator=self.config.seed)
        ideal_circuit = transpile(
            qc,
            ideal_sim,
            optimization_level=self.config.optimization_level,
            seed_transpiler=self.config.seed,
        )
        ideal_counts = ideal_sim.run(ideal_circuit, shots=self.config.shots).result().get_counts()
        if backend_name == "ideal":
            counts, compiled = ideal_counts, ideal_circuit
        else:
            compiled = transpile(
                qc,
                backend,
                optimization_level=self.config.optimization_level,
                seed_transpiler=self.config.seed,
            )
            simulator = AerSimulator.from_backend(
                backend,
                noise_model=noise_model,
                seed_simulator=self.config.seed,
            )
            counts = simulator.run(compiled, shots=self.config.shots).result().get_counts()
        fidelity = hellinger_fidelity(ideal_counts, counts)
        polarization = calculate_polarization_multiple_targets(counts, correct_states=ideal_counts)
        return ideal_counts, counts, fidelity, polarization, compiled

    @staticmethod
    def extract_algorithm_name(circuit_name):
        return circuit_name.split("_")[0]

    def analyze_single_circuit(self, qasm_file, backend_name):
        try:
            qc = QuantumCircuit.from_qasm_file(str(qasm_file))
            if not any(inst.operation.name == "measure" for inst in qc.data):
                raise ValueError("No measurement gates")
            ideal, noisy, fidelity, polarization, compiled = self.run_noisy_simulation(
                qc, backend_name
            )
            return {
                "circuit_name": Path(qasm_file).stem,
                "algorithm_name": self.extract_algorithm_name(Path(qasm_file).stem),
                "backend": backend_name,
                "width": compiled.num_qubits,
                "depth": compiled.depth(),
                "total_gates": len(compiled.data),
                "single_qubit_gates": sum(
                    len(i.qubits) == 1 and i.operation.name != "measure" for i in compiled.data
                ),
                "two_qubit_gates": sum(len(i.qubits) == 2 for i in compiled.data),
                "measure_gates": compiled.count_ops().get("measure", 0),
                "hellinger_fidelity": fidelity,
                "polarization": polarization,
                "ideal_counts": ideal,
                "noisy_counts": noisy,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def save_to_unified_csv(self, result):
        single, two, readout = self.calculate_error_rates(result["backend"])
        row = {
            "Circuit": result["circuit_name"],
            "Algorithm": result["algorithm_name"],
            "Backend": result["backend"],
            "Width": result["width"],
            "Depth": result["depth"],
            "Total_Gates": result["total_gates"],
            "Single_Qubit_Gates": result["single_qubit_gates"],
            "Two_Qubit_Gates": result["two_qubit_gates"],
            "Measure_Gates": result["measure_gates"],
            "Single_Qubit_Error_Rate": single,
            "Two_Qubit_Error_Rate": two,
            "Readout_Error_Rate": readout,
            "Hellinger_Fidelity": result["hellinger_fidelity"],
            "Polarization": result["polarization"],
            "Ideal_Counts_Full": json.dumps(result["ideal_counts"]),
            "Noisy_Counts_Full": json.dumps(result["noisy_counts"]),
            "Shots": self.config.shots,
            "Optimization_Level": self.config.optimization_level,
            "Seed": self.config.seed,
            "Timestamp": result["timestamp"],
        }
        frame = pd.DataFrame([row])
        if self.unified_csv_path.exists():
            existing = pd.read_csv(self.unified_csv_path)
            duplicate = (existing["Circuit"] == row["Circuit"]) & (
                existing["Backend"] == row["Backend"]
            )
            frame = pd.concat([existing[~duplicate], frame], ignore_index=True)
        temporary = self.unified_csv_path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(self.unified_csv_path)

    def analyze_single_backend(self, backend_name):
        success = errors = 0
        files = sorted(Path(self.config.qasm_dir).glob("*.qasm"))
        for path in files:
            if self.config.incremental_update and self.is_processed(path.stem, backend_name):
                continue
            result = self.analyze_single_circuit(path, backend_name)
            if "error" in result:
                errors += 1
                print(f"{path.name} @ {backend_name}: {result['error']}")
                continue
            self.save_to_unified_csv(result)
            self.processed_circuits.add((path.stem, backend_name))
            success += 1
            print(f"{path.name} @ {backend_name}: fidelity {result['hellinger_fidelity']:.6f}")
        return success, errors

    def run(self):
        if not Path(self.config.qasm_dir).is_dir():
            raise ValueError(f"QASM directory not found: {self.config.qasm_dir}")
        if not list(Path(self.config.qasm_dir).glob("*.qasm")):
            raise ValueError(f"No QASM files found: {self.config.qasm_dir}")
        results = {}
        for backend in self.config.backends:
            success, errors = self.analyze_single_backend(backend)
            results[backend] = {"success": success, "error": errors}
        return {"unified_csv": str(self.unified_csv_path), "analysis_results": results}


def quick_generate_data(
    qasm_dir, backends=None, shots=2000, incremental_update=True, optimization_level=1
):
    config = DataGeneratorConfig(
        qasm_dir=qasm_dir,
        backends=backends or ["FakeWashingtonV2"],
        shots=shots,
        incremental_update=incremental_update,
        optimization_level=optimization_level,
    )
    return QuantumCircuitDataGenerator(config).run()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Directory of QASM files")
    parser.add_argument("--backend", action="append", dest="backends", required=True)
    parser.add_argument("--output", type=Path, default=Path("results/generated"))
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimization-level", type=int, choices=range(4), default=1)
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing pairs")
    args = parser.parse_args(argv)
    config = DataGeneratorConfig(
        qasm_dir=str(args.input),
        output_dir=str(args.output),
        backends=args.backends,
        shots=args.shots,
        seed=args.seed,
        optimization_level=args.optimization_level,
        incremental_update=not args.overwrite,
    )
    try:
        results = QuantumCircuitDataGenerator(config).run()
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return int(any(row["error"] for row in results["analysis_results"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
