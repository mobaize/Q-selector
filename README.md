# Q-Selector

Rank quantum processor profiles for OpenQASM 2 circuits using a random forest.
The bundled dataset covers 578 circuits and 20 simulated backend profiles.

## Install

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

## Train and recommend

```bash
python src/train.py
q-selector data/QASM/ae_indep_qiskit_10.qasm --no-save
q-selector data/QASM/ --top 3
```

Run training first to create `results/trained_model_optimized.pkl` and
`training_metadata.json`. No pretrained model is included.

Backends with enough qubits are ranked by predicted FOM (Hellinger fidelity × 10).
Predictions use the bundled calibration snapshots.

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
Only load trusted pickle files.
