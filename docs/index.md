# Welcome to Q-Selector's documentation!

**Q-Selector** is a machine learning tool that predicts the optimal quantum backend for a given quantum circuit.
Given an unknown QASM file (or an entire folder of circuits), it loads a pre-trained Random Forest model and recommends the best-suited quantum processor — no ground-truth comparison or manual benchmarking needed.

```{toctree}
:caption: User Guide
:glob:
:hidden:
:maxdepth: 1

installation
quickstart
device_selection
```

```{toctree}
:caption: Development
:glob:
:hidden:
:maxdepth: 1

contributing
references
```

## Key Features

- **Direct processor recommendation** — Load a trained model and recommend the best backend for any QASM circuit without running any compilation or fidelity estimation.
- **Batch processing** — Process an entire folder of QASM files in one command.
- **Comprehensive feature engineering** — 23 circuit features (width, depth, gate mix, entanglement, parallelism, etc.) + 20 backend features (gate errors, readout errors, coherence times, connectivity) + 13 interaction features.
- **Automatic archiving** — Each run creates a timestamped output folder with `best_recommendation.csv`, `full_ranking.csv`, and `full_ranking.json`.

## How It Works

Q-Selector uses a Random Forest regressor trained on a labeled dataset of circuit–backend pairs evaluated with Hellinger fidelity as the figure of merit. At inference time, it extracts the same set of features from the input QASM circuit, predicts the expected FOM for each available backend, and ranks them by predicted performance.
