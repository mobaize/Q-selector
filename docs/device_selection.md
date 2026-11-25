# Model and features

## Training

- Random forest: 100 trees, seed 42.
- Target: Hellinger fidelity × 10.
- Split: 70% training, 30% test, stratified by the best measured backend.
- Circuits whose best-backend class occurs once are assigned to training.
- Feature selection: a separate 80-tree forest fitted on training data; retain
  importance above 0.005.
- StandardScaler and the final forest are fitted in a pipeline on training data.

Rows are grouped by circuit name. Algorithm families and compilation variants
may appear in both sets. Transfer to new hardware has not been evaluated.

## Features

`src/features.py` supplies all training and inference features:

| Group | Count | Content |
|---|---:|---|
| Circuit | 23 | Width, depth, gate counts, interaction graph, and derived ratios |
| Backend | 20 | Qubit count, connectivity, error statistics, T1/T2 statistics |
| Interaction | 13 | Error accumulation, size, connectivity, and coherence proxies |

Feature definitions:

| Feature | Definition |
|---|---|
| `critical_depth` | Two-qubit operation count / circuit depth |
| `non_clifford_ratio` | Rz operation ratio, regardless of rotation angle |
| `single_qubit_gates` | All one-qubit instructions, including measurements |
| `coupling_match_ratio` | Qubit-index comparison before hardware layout |
| T1/T2 risk | Assumes 100 ns per depth layer |

## Ranking

Backends with fewer qubits than the input circuit are excluded. Remaining
backends are sorted by predicted FOM. Equal scores are ordered by backend name.

Algorithm preferences are optional and require `--algorithm`. The adjustment is
`0.2 × (algorithm/backend mean training FOM − global mean training FOM)`.
Adjusted FOM can fall outside [0, 10].

The supplied backend profiles are calibration snapshots. Recommendations do
not query current devices or run the input circuit.

## Data generation

Noisy simulation compiles against the selected fake backend target. Aer uses
that backend's noise model and connectivity. Compilation and sampling seeds are
fixed by `--seed`. Failed or empty noise models raise errors.
Use `--backend ideal` for ideal simulation.

Bundled labels predate the generator fixes. Provenance: `data/provenance.json`.
Run `python src/verify_fidelity.py` to check labels against the archived counts.
