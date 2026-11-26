<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="fig/q-selector-dark.svg">
    <img src="fig/q-selector-light.svg" alt="Q-Selector" width="420">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/mobaize/Q-selector/actions/workflows/checks.yml"><img src="https://github.com/mobaize/Q-selector/actions/workflows/checks.yml/badge.svg" alt="Checks"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-3776ab" alt="Python 3.10+"></a>
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/License-MIT-64748b" alt="MIT License"></a>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> &middot;
  <a href="docs/quickstart.md">Usage guide</a> &middot;
  <a href="docs/device_selection.md">Model and features</a> &middot;
  <a href="data/README.md">Dataset</a>
</p>

Q-Selector is a Python tool for choosing quantum processors and exploring how
well different processors suit a given circuit. Given an OpenQASM 2 circuit, it
combines circuit structure with backend connectivity and calibration features to
predict measurement fidelity and rank candidate processors.

You can generate your own datasets and train models for the circuits and
processors you want to study.

## Features

- Rank backends by predicted Hellinger fidelity, excluding those with too few qubits.
- Process individual QASM files or directories and export rankings as CSV and JSON.
- Train on your own measurement datasets and compare with error-rate baselines.
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

Train on your own measurements, circuits, and backend profiles:

```bash
python src/train.py --data path/to/measurements.csv --qasm path/to/qasm --backend-json path/to/backend_json --output results/custom
q-selector circuit.qasm --model results/custom/trained_model_optimized.pkl
```

CSV fields and count formats are described in [`data/README.md`](data/README.md).

## Development

```bash
python -m pip install -e ".[dev,data,docs]"
python -m pytest
python -m ruff check src tests docs/conf.py
python -m sphinx -W -b html docs docs/_build/html
```

## Example data and source

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
