# Q-Selector

Q-Selector is a machine-learning tool for selecting a suitable quantum
processor for an OpenQASM circuit. It extracts circuit, processor, and
circuit--processor interaction features, predicts a figure of merit (FOM) for
each candidate processor, and returns a ranked recommendation.

The primary recommendation workflow does not execute the input circuit or
require ground-truth results. A labeled dataset is needed only when training or
evaluating a model.

## Features

- Recommend a processor for one OpenQASM 2 circuit.
- Batch-process all `.qasm` files in a directory.
- Rank 20 simulated processor profiles using a pre-trained random forest.
- Extract 23 circuit features, 20 processor features, and 13 interaction
  features.
- Export the best recommendation and complete ranking as CSV and JSON.
- Train and evaluate a model from the included labeled dataset.

## Requirements

- Python 3.10 or later
- A supported operating system: Linux, macOS, or Windows

The tested core stack uses Qiskit 2.3 and scikit-learn 1.x. Create a virtual
environment before installation to avoid conflicts with other Qiskit
installations.

## Installation

From the project root, create a virtual environment and install Q-Selector:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional dependency groups are available for development, documentation, and
data generation:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[docs]"
python -m pip install -e ".[data]"
```

`requirements.txt` contains the same core runtime dependencies for users who
prefer requirements-based installation.

## Included assets

The repository contains:

- `data/MQTBench_578_analysis_results.csv`: labeled training and evaluation
  data.
- `data/QASM/`: OpenQASM circuits used by the dataset.
- `data/backend_json/`: configuration and calibration data for 20 simulated
  processors.
- `results/trained_model_optimized.pkl`: pre-trained model and preprocessing
  artifacts.

The pre-trained model is loaded with Python `pickle`. Only load model files
from a trusted source.

## Quick start

Run a recommendation without writing output files:

```bash
q-selector data/QASM/ae_indep_qiskit_10.qasm --no-save
```

The source-tree command is also supported:

```bash
python src/recommend.py data/QASM/ae_indep_qiskit_10.qasm --no-save
```

Process a directory and save results:

```bash
q-selector data/QASM/
```

Show more ranked candidates:

```bash
q-selector my_circuit.qasm --top 10
```

Use alternative model and processor-profile locations:

```bash
q-selector my_circuit.qasm \
  --model path/to/trained_model.pkl \
  --backend-json path/to/backend_json
```

Run `q-selector --help` for the complete command-line reference.

## Training and evaluation

Training is optional when the included pre-trained model is available. To
rebuild it from the labeled dataset:

```bash
python src/train.py
```

The training command writes the model, selected-feature list, and training
report to `results/`.

To generate predictions for the held-out test split and compare them with the
configured baselines:

```bash
python src/predict.py
python src/evaluate.py
```

These commands write their reports, tables, and plots to `results/`.

## Output files

Unless `--no-save` is used, each recommendation run creates
`results/your_work/<timestamp>/` containing:

| File | Purpose |
|---|---|
| `best_recommendation.csv` | Top-ranked processor for each input circuit |
| `full_ranking.csv` | Complete circuit--processor ranking in tabular form |
| `full_ranking.json` | Structured features, best choice, and all rankings |

The main fields include the circuit name, width, depth, gate count,
recommended processor, predicted FOM, processor size, and representative error
rates.

## Data generation

Generating a new execution dataset requires the optional `data` dependencies:

```bash
python -m pip install -e ".[data]"
```

The implementation is provided in `src/data_generator.py`. Data generation
runs noisy simulations and can be substantially more expensive than processor
recommendation.

## Project structure

```text
Q-Selector/
|-- data/
|   |-- MQTBench_578_analysis_results.csv
|   |-- QASM/
|   `-- backend_json/
|-- docs/
|-- results/
|   |-- trained_model_optimized.pkl
|   |-- example_results/
|   `-- your_work/
|-- src/
|   |-- data_generator.py
|   |-- evaluate.py
|   |-- predict.py
|   |-- recommend.py
|   `-- train.py
|-- tests/
|-- CITATION.bib
|-- LICENSE.md
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Testing

Install the development dependencies and run the test suite from the project
root:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Documentation

Additional local guides are available in `docs/`, including installation,
quick-start, feature-engineering, contribution, and citation information.

## Citation

Citation metadata is provided in `CITATION.bib`.

## License

Q-Selector is distributed under the MIT License. See `LICENSE.md` for details.
