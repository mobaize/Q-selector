# Quickstart

This guide walks you through training Q-Selector and recommending a processor for an unknown quantum circuit.

## Prerequisites

Ensure the package is installed and dependencies are available (see {doc}`installation`).

## Step 1: Train the Model

The generated PKL model is not stored in the repository. Create it locally
before the first recommendation:

```console
$ python src/train.py
```

This writes the trained model and preprocessing artifacts to `results/`.

## Step 2: Recommend a Backend

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

## Step 3: Interpret the Results

The model assigns a **predicted FOM** (Figure of Merit, scaled Hellinger fidelity × 10) to each circuit–backend pair. Higher values indicate better expected performance. The recommended backend is the one with the highest predicted FOM.

For a quick sanity check, compare the recommended backend's characteristics (qubit count, gate errors, connectivity) against the circuit's requirements (qubit width, two-qubit gate density, entanglement ratio).
