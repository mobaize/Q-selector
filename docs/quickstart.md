# Usage

## Train

```bash
python src/train.py
```

Outputs in `results/`: model PKL, selected features, regression metrics, and
training metadata with dependency versions and dataset checksum.

## Recommend

```bash
q-selector data/QASM/ae_indep_qiskit_10.qasm --no-save
q-selector data/QASM/ --top 3
q-selector circuit.qasm --algorithm ae
```

Directories are searched recursively. `--algorithm` adds a preference learned
from the training set; default rankings are independent of the filename.

| Option | Purpose |
|---|---|
| `--model PATH` | Load a model from another location |
| `--backend-json DIR` | Override cached coupling maps |
| `--output DIR` | Set the parent directory for saved runs |
| `--top N` | Print the first N backends |
| `--no-save` | Print results without saving files |

Saved runs contain:

| File | Content |
|---|---|
| `best_recommendation.csv` | First-ranked backend per input |
| `full_ranking.csv` | All eligible backend scores |
| `full_ranking.json` | Rankings, circuit features, and failed inputs |

Backends with too few qubits are excluded. Inputs with no eligible backend or
invalid QASM are reported as failures. Batch runs retain successful results and
return a nonzero exit code when any input fails.

## Evaluate

```bash
python src/predict.py
python src/evaluate.py
```

The prediction report records Top-1/Top-3 accuracy and mean FOM regret on held-out
circuits. Baseline comparisons use measured FOM, with capacity filtering applied
to each circuit.

## Generate measurements

```bash
python -m pip install -e ".[data]"
python src/data_generator.py path/to/qasm --backend FakeYorktownV2 --shots 2000 --seed 42
```

Use `--optimization-level 0`, `1`, `2`, or `3` to select the compilation setting.
Existing pairs are skipped when run settings match; `--overwrite` recomputes them.
Use a new output directory when changing shots, optimization level, or seed.
