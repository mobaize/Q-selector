# Quickstart

This guide walks you through recommending a processor for an unknown quantum circuit and optionally rebuilding the model.

## Prerequisites

Ensure the package is installed and dependencies are available (see {doc}`installation`).

## Step 1: Recommend a Backend

The repository includes a pre-trained model, so recommendation can run directly.

### Single Circuit

```console
$ q-selector data/QASM/ae_indep_qiskit_10.qasm
```

Output is saved to `results/your_work/<timestamp>/`.

```console
$ q-selector data/QASM/ae_indep_qiskit_10.qasm --no-save
```

Terminal output only, no file written.

### Batch Folder

```console
$ q-selector data/QASM/
```

Processes all `.qasm` files in the folder and produces a single consolidated output archive.

### Output Files

Each run creates a timestamped archive under `results/your_work/<timestamp>/`:

| File | Description |
|------|-------------|
| `best_recommendation.csv` | One row per circuit with Top-1 backend and predicted FOM |
| `full_ranking.csv` | Long table: every backend per circuit with rank, FOM, error rates |
| `full_ranking.json` | Per-circuit grouped JSON with full rankings and circuit features |

## Step 2: Interpret the Results

The model assigns a **predicted FOM** (Figure of Merit, scaled Hellinger fidelity × 10) to each circuit–backend pair. Higher values indicate better expected performance. The recommended backend is the one with the highest predicted FOM.

For a quick sanity check, compare the recommended backend's characteristics (qubit count, gate errors, connectivity) against the circuit's requirements (qubit width, two-qubit gate density, entanglement ratio).

## Step 3: Optionally Rebuild the Model

To train a new model from the included labeled dataset:

```console
$ python src/train.py
```

This writes the trained model and preprocessing artifacts to `results/`.
