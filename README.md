# Q-Selector

[![Checks](https://github.com/mobaize/Q-selector/actions/workflows/checks.yml/badge.svg)](https://github.com/mobaize/Q-selector/actions/workflows/checks.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

Q-Selector is a Python tool for quantum backend selection. Given an OpenQASM 2
circuit, it combines circuit structure with backend connectivity and calibration
features to predict measurement fidelity and rank candidate backends.

The random forest model provides a shortlist of backends before running further
simulations or hardware experiments. The repository includes 578 benchmark
circuits, 20 simulated backend profiles, and 11,560 circuit/backend measurements
for local training and evaluation. Recommendations use the supplied calibration
snapshots.

[Quickstart](#quickstart) · [Usage guide](docs/quickstart.md) ·
[Model and features](docs/device_selection.md) · [Dataset](data/README.md)

## Features

- Rank backends by predicted Hellinger fidelity, excluding those with too few qubits.
- Process individual QASM files or directories and export rankings as CSV and JSON.
- Train locally and compare recommendations with error-rate baselines.
- Recompute fidelity labels from the included ideal and noisy measurement counts.
- Generate measurements with Qiskit Aer and fake backend noise models.

## Quickstart

Requires Python 3.10 or later.

```bash
git clone https://github.com/mobaize/Q-selector.git
cd Q-selector
python -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate  # Linux/macOS
```

```powershell
.venv\Scripts\Activate.ps1  # Windows PowerShell
```

```bash
python -m pip install -e .
```

Train the model and rank backends for an example circuit:

```bash
python src/train.py
q-selector data/QASM/ae_indep_qiskit_10.qasm --no-save
```

Run training first to create `results/trained_model_optimized.pkl` and
`training_metadata.json`. No pretrained model is included.

Scores are reported as FOM: predicted Hellinger fidelity × 10.

## Usage

Rank all circuits in a directory:

```bash
q-selector data/QASM/ --top 3
```

Enable algorithm preferences with `--algorithm` (off by default):

```bash
q-selector circuit.qasm --algorithm ae
```

Other options:

```bash
q-selector circuit.qasm --model path/to/model.pkl --output results/your_work
q-selector circuit.qasm --backend-json data/backend_json
q-selector --help
```

Only load trusted pickle files with `--model`.

Each saved run contains `best_recommendation.csv`, `full_ranking.csv`, and
`full_ranking.json` under `results/your_work/<timestamp>/`. The JSON records failed
inputs. A run with failed inputs returns a nonzero exit code.

## Evaluate

```bash
python src/predict.py
python src/evaluate.py
python src/verify_fidelity.py
```

- `predict.py`: rank backends for held-out circuits.
- `evaluate.py`: compare with lowest readout, single-qubit, and two-qubit error
  baselines, subject to qubit capacity.
- `verify_fidelity.py`: recompute labels from the full measurement counts.

Example outputs: [`results/example_results/`](results/example_results/).

## Generate data

```bash
python -m pip install -e ".[data]"
python src/data_generator.py path/to/qasm --backend FakeYorktownV2 --shots 2000 --seed 42
```

Repeat `--backend` for additional profiles. Results go to `results/generated/`.
Failed noise-model construction or simulation produces no labeled row.

## Development

```bash
python -m pip install -e ".[dev,data,docs]"
python -m pytest
python -m ruff check src tests docs/conf.py
python -m sphinx -W -b html docs docs/_build/html
```

## Data and source

- [`data/QASM/`](data/QASM/): 578 circuits.
- [`data/backend_json/`](data/backend_json/): 20 backend snapshots.
- [`data/MQTBench_578_analysis_results.csv`](data/MQTBench_578_analysis_results.csv):
  11,560 circuit/backend measurements used for training.
- [`data/MQTBench_578_analysis_results_full.csv.gz`](data/MQTBench_578_analysis_results_full.csv.gz):
  full measurements with ideal and noisy counts for recomputing fidelity.
  See [`data/README.md`](data/README.md) and [`data/provenance.json`](data/provenance.json).
- [`docs/`](docs/): installation, model details, and contribution notes.

Maintained alongside [QCFlow/qcflow-q-selector](https://github.com/QCFlow/qcflow-q-selector).
License: [MIT](LICENSE.md). Citations: [`CITATION.bib`](CITATION.bib).
