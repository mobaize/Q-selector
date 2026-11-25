# Model and features

## Training

- Random forest: 100 trees, seed 42.
- Target: Hellinger fidelity × 10.
- Split: 70% training, 30% test, stratified by the best measured backend.
- Circuits whose best-backend class occurs once are assigned to training.
- Feature selection: a separate 80-tree forest fitted on training data; retain
  importance above 0.005.
- StandardScaler and the final forest are fitted in a pipeline on training data.

The split groups rows by circuit name. It does not hold out entire algorithm
families or equivalent circuits represented at different compilation stages.
Reported accuracy describes this split, not transfer to unseen hardware.

## Features

`src/features.py` supplies all training and inference features:

| Group | Count | Content |
|---|---:|---|
| Circuit | 23 | Width, depth, gate counts, interaction graph, and derived ratios |
| Backend | 20 | Qubit count, connectivity, error statistics, T1/T2 statistics |
| Interaction | 13 | Error accumulation, size, connectivity, and coherence proxies |

Several names are retained from the original model. `critical_depth` is the
total two-qubit operation count divided by circuit depth; it is not a longest-path
calculation. `non_clifford_ratio` counts Rz operations without inspecting angles.
`single_qubit_gates` includes one-qubit instructions such as measurements.
`coupling_match_ratio` compares qubit indices directly before hardware layout.
T1/T2 risk assumes 100 ns per depth layer. These are model features, not compiled
execution estimates.

## Ranking

Backends with fewer qubits than the input circuit are excluded. Remaining
backends are sorted by predicted FOM. Equal scores are ordered by backend name.
Capacity filtering alone does not establish hardware executability.

Algorithm preferences are optional and require `--algorithm`. The adjustment is
`0.2 × (algorithm/backend mean training FOM − global mean training FOM)`.
Adjusted scores can fall outside the physical fidelity range. Training and
inference use the same feature and ranking functions.

The supplied backend profiles are calibration snapshots. Recommendations do
not query current devices or run the input circuit.

## Data generation

Noisy simulation compiles against the selected fake backend target. Aer uses
that backend's noise model and connectivity. Compilation and sampling seeds are
fixed by `--seed`. Failed or empty noise models produce errors rather than
ideal-simulation labels. `--backend ideal` selects ideal simulation explicitly.

The bundled labels come from the upstream dataset and have not been regenerated
with the corrected generator. Provenance is recorded in `data/provenance.json`.
The compressed full dataset retains ideal and noisy measurement counts.
`python src/verify_fidelity.py` recomputes the labels from those counts. Training
uses the resulting scalar labels; it does not learn the full output distributions.
