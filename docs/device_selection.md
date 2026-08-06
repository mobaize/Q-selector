# Device Selection

Q-Selector's device selection component is built on a **Random Forest regressor** that predicts the expected fidelity (FOM) for each circuit–backend pair without running any actual compilation or noisy simulation.

## Feature Engineering

### Circuit Features (23 dimensions)

**15 base features:**
- `width` — number of qubits
- `depth` — circuit depth
- `total_gates`, `single_qubit_gates`, `two_qubit_gates`, `measure_gates`
- `entanglement_ratio` = two_qubit_gates / total_gates
- `program_communication` — average connectivity per qubit
- `parallelism` — proportion of gates that can execute in parallel
- `critical_depth` — two-qubit gate critical path density
- `liveness` — fraction of qubits actively used
- Per-gate counts: `h_gates`, `x_gates`, `cx_gates`, `rz_gates`

**8 advanced features:**
- `gate_density` — gates per (qubit × depth)
- `gate_diversity` — unique gate types normalized
- `non_clifford_ratio` — Rz gate fraction
- `coupling_density_circuit` — edge utilization vs complete graph
- `twoq_nonmeasure_ratio` — two-qubit fraction among non-measurement gates
- `parallel_capacity` — max width of any DAG layer normalized
- `aspect_ratio` — log depth/width ratio
- `entanglement_strength` — (CX + SWAP) / total_gates

### Backend Features (20 dimensions)

- `backend_num_qubits`, `num_edges`, `connectivity_density`, `avg_degree`
- Single-qubit error statistics: `s1_error_mean`, `s1_error_std`, `s1_error_max`, `s1_error_min`
- Two-qubit error statistics: `cnot_error_mean`, `cnot_error_std`, `cnot_error_max`, `cnot_error_min`
- Readout error statistics: `ro_error_mean`, `ro_error_std`, `ro_error_max`, `ro_error_min`
- Coherence times: `t1_mean`, `t1_std`, `t2_mean`, `t2_std`

### Interaction Features (13 dimensions)

- Cumulative error terms: `cum_cnot_error`, `cum_s1_error`, `cum_ro_error`
- `depth_weighted_error`, `cnot_sensitivity`, `size_mismatch`
- `coupling_match_ratio` — fraction of circuit edges supported by backend topology
- `parallel_connect_need`, `comm_degree`, `qubit_utilization`
- `t1_risk`, `t2_risk`, `noise_budget`

## Model Training

The model is trained via `src/train.py` with:

- **Algorithm**: Random Forest (n_estimators=100)
- **Feature selection**: importance threshold of 0.005
- **Stratified split**: 70/30 train/test stratified by best backend
- **Propensity weighting**: 0.20 weight for algorithm-level backend preferences
- **Output**: `results/trained_model_optimized.pkl` (model, scaler, feature list, backend cache)

## Prediction

The `src/recommend.py` script reloads the trained artifacts and, for each input QASM circuit:

1. Extracts the 23 circuit features
2. For each backend, builds the 20 backend features + 13 interaction features
3. Selects the 32 most important features (post-threshold)
4. Scales using the trained StandardScaler
5. Predicts FOM via the Random Forest model
6. Ranks backends by predicted FOM descending
