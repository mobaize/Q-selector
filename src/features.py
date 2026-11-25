"""Circuit, backend and interaction features shared by training and inference."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

CIRCUIT_FEATURES = [
    "width",
    "depth",
    "total_gates",
    "single_qubit_gates",
    "two_qubit_gates",
    "measure_gates",
    "entanglement_ratio",
    "program_communication",
    "parallelism",
    "critical_depth",
    "liveness",
    "h_gates",
    "x_gates",
    "cx_gates",
    "rz_gates",
    "gate_density",
    "gate_diversity",
    "non_clifford_ratio",
    "coupling_density_circuit",
    "twoq_nonmeasure_ratio",
    "parallel_capacity",
    "aspect_ratio",
    "entanglement_strength",
]
BACKEND_FEATURES = [
    "backend_num_qubits",
    "num_edges",
    "connectivity_density",
    "avg_degree",
    "s1_error_mean",
    "s1_error_std",
    "s1_error_max",
    "s1_error_min",
    "cnot_error_mean",
    "cnot_error_std",
    "cnot_error_max",
    "cnot_error_min",
    "ro_error_mean",
    "ro_error_std",
    "ro_error_max",
    "ro_error_min",
    "t1_mean",
    "t1_std",
    "t2_mean",
    "t2_std",
]


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def coupling_edges(conf):
    return {tuple(sorted(edge)) for edge in conf["coupling_map"]}


def load_backend_features(folder):
    cache = {}
    for path in sorted(Path(folder).glob("conf_*.json")):
        conf = load_json(path)
        props_path = path.with_name(path.name.replace("conf_", "props_", 1))
        props = load_json(props_path)
        if conf is None or props is None:
            raise ValueError(f"Missing or invalid backend configuration: {path}, {props_path}")
        edges = coupling_edges(conf)
        nq = conf["num_qubits"]
        if nq <= 0:
            raise ValueError(f"Invalid qubit count in {path}")
        feats = {
            "backend_num_qubits": nq,
            "num_edges": len(edges),
            "connectivity_density": len(edges) / (nq * (nq - 1) / 2) if nq > 1 else 0,
            "avg_degree": 2 * len(edges) / nq,
            "_coupling_set": edges,
        }
        for source, prefix in [("single_qubit_errors", "s1"), ("two_qubit_errors", "cnot")]:
            values = []
            for value in props[source].values():
                values.extend(value.values() if isinstance(value, dict) else [value])
            values = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
            if not values:
                raise ValueError(f"Missing {source} in {props_path}")
            for suffix, fn in zip(["mean", "std", "max", "min"], [np.mean, np.std, np.max, np.min]):
                feats[f"{prefix}_error_{suffix}"] = float(fn(values))
        readout = props["readout_errors"]
        readout = list(readout.values()) if isinstance(readout, dict) else readout
        if not readout:
            raise ValueError(f"Missing readout errors in {props_path}")
        for suffix, fn in zip(["mean", "std", "max", "min"], [np.mean, np.std, np.max, np.min]):
            feats[f"ro_error_{suffix}"] = float(fn(readout))
        times = np.asarray(props["t1_t2"], dtype=float)
        if times.shape != (nq, 2) or not np.isfinite(times).all() or (times <= 0).any():
            raise ValueError(f"Invalid coherence times in {props_path}")
        for idx, prefix in enumerate(["t1", "t2"]):
            feats[f"{prefix}_mean"] = float(times[:, idx].mean())
            feats[f"{prefix}_std"] = float(times[:, idx].std())
        cache[conf["name"]] = feats
    if not cache:
        raise ValueError(f"No backend configurations found in {folder}")
    return cache


def extract_circuit_features(qasm_path):
    try:
        circ = QuantumCircuit.from_qasm_file(str(qasm_path))
    except Exception as exc:
        raise ValueError(f"Cannot load QASM file {qasm_path}: {exc}") from exc
    w, d, tg = circ.num_qubits, circ.depth(), len(circ.data)
    if w == 0:
        raise ValueError(f"Circuit has no qubits: {qasm_path}")
    gc = dict(circ.count_ops())
    s1 = sum(len(i.qubits) == 1 for i in circ.data)
    s2 = sum(len(i.qubits) == 2 for i in circ.data)
    meas = gc.get("measure", 0)
    edges = {
        tuple(sorted(circ.find_bit(q).index for q in inst.qubits))
        for inst in circ.data
        if len(inst.qubits) == 2
    }
    active = {circ.find_bit(q).index for inst in circ.data for q in inst.qubits}
    dag = circuit_to_dag(circ)
    widths = [len(layer["graph"].op_nodes()) for layer in dag.layers()]
    max_edges = len(active) * (len(active) - 1) / 2
    return {
        "width": w,
        "depth": d,
        "total_gates": tg,
        "single_qubit_gates": s1,
        "two_qubit_gates": s2,
        "measure_gates": meas,
        "entanglement_ratio": s2 / tg if tg else 0,
        "program_communication": 2 * len(edges) / (w * (w - 1)) if w > 1 else 0,
        "parallelism": 1 - d / tg if tg else 0,
        "critical_depth": s2 / d if d else 0,
        "liveness": len(active) / w,
        "h_gates": gc.get("h", 0),
        "x_gates": gc.get("x", 0),
        "cx_gates": gc.get("cx", 0),
        "rz_gates": gc.get("rz", 0),
        "gate_density": tg / (w * d) if d else 0,
        "gate_diversity": len(gc) / 15,
        "non_clifford_ratio": gc.get("rz", 0) / tg if tg else 0,
        "coupling_density_circuit": len(edges) / max_edges if max_edges else 0,
        "twoq_nonmeasure_ratio": s2 / (tg - meas) if tg > meas else 0,
        "parallel_capacity": max(widths, default=1) / w,
        "aspect_ratio": np.log1p(d / w),
        "entanglement_strength": (gc.get("cx", 0) + gc.get("swap", 0)) / tg if tg else 0,
        "_circuit_edges": edges,
    }


def interaction_feats(cf, bf):
    s1, cnot, ro = (bf.get(k, 0) for k in ["s1_error_mean", "cnot_error_mean", "ro_error_mean"])
    ce, bc = cf.get("_circuit_edges", set()), bf.get("_coupling_set", set())
    t1, t2 = bf.get("t1_mean", 100e-6), bf.get("t2_mean", 50e-6)
    return {
        "cum_cnot_error": cf["two_qubit_gates"] * cnot,
        "cum_s1_error": cf["single_qubit_gates"] * s1,
        "cum_ro_error": cf["width"] * ro,
        "depth_weighted_error": cf["depth"] * (s1 + cnot),
        "cnot_sensitivity": cf["entanglement_ratio"] * cnot,
        "size_mismatch": max(0, cf["width"] / bf["backend_num_qubits"] - 1),
        "coupling_match_ratio": len(ce & bc) / len(ce) if ce and bc else 0,
        "parallel_connect_need": cf["parallelism"] * bf.get("connectivity_density", 0),
        "comm_degree": cf["program_communication"] * bf.get("avg_degree", 0),
        "qubit_utilization": cf["liveness"] * bf["backend_num_qubits"],
        "t1_risk": cf["depth"] * 100e-9 / t1 if t1 > 0 else 0,
        "t2_risk": cf["depth"] * 100e-9 / t2 if t2 > 0 else 0,
        "noise_budget": cf["single_qubit_gates"] * s1
        + cf["two_qubit_gates"] * cnot
        + cf["width"] * ro,
    }


def feature_row(cf, bf):
    row = {name: cf[name] for name in CIRCUIT_FEATURES}
    row.update({name: bf.get(name, 0) for name in BACKEND_FEATURES})
    row.update(interaction_feats(cf, bf))
    return row


def rank_backends(cf, artifacts, algorithm=None):
    cache = artifacts["backend_features_cache"]
    eligible = [
        name
        for name in artifacts["all_backends"]
        if name in cache and cache[name].get("backend_num_qubits", 0) >= cf["width"]
    ]
    if not eligible:
        raise ValueError(f"No backend can accommodate a {cf['width']}-qubit circuit")
    preferences = artifacts.get("algorithm_preferences", {})
    if algorithm is not None and algorithm not in preferences:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. Available: {', '.join(sorted(preferences))}"
        )
    frame = pd.DataFrame([feature_row(cf, cache[name]) for name in eligible])
    frame = frame[artifacts["selected_features"]]
    if "pipeline" in artifacts:
        scores = artifacts["pipeline"].predict(frame)
    else:
        scores = artifacts["model"].predict(artifacts["scaler"].transform(frame))
    result = []
    for name, score in zip(eligible, scores):
        if algorithm is not None:
            preference = preferences[algorithm].get(name, artifacts.get("global_avg_fom", 0))
            score += (preference - artifacts.get("global_avg_fom", 0)) * artifacts.get(
                "propensity_weight", 0.2
            )
        if not np.isfinite(score):
            raise ValueError(f"Non-finite model prediction for {name}")
        bf = cache[name]
        result.append(
            {
                "backend": name,
                "predicted_fom": float(score),
                "num_qubits": bf["backend_num_qubits"],
                "cnot_error": bf.get("cnot_error_mean", 0),
                "ro_error": bf.get("ro_error_mean", 0),
            }
        )
    return sorted(result, key=lambda row: (-row["predicted_fom"], row["backend"]))
