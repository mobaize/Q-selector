# Dataset

Both datasets contain 11,560 rows: 578 circuits × 20 backends.

- `MQTBench_578_analysis_results.csv`: training table. Training reads `Circuit`,
  `Backend`, and `Hellinger_Fidelity`.
- `MQTBench_578_analysis_results_full.csv.gz`: complete upstream CSV, including
  `Ideal_Counts_Full` and `Noisy_Counts_Full`. Decompression restores the original
  file byte for byte. Pandas reads this file directly with `pd.read_csv(path)`.

The full dataset stores measurement counts. Normalize each count dictionary to
obtain a probability distribution. The label is Hellinger fidelity:
`F(P, Q) = (sum(sqrt(P(x) * Q(x)))) ** 2`.
This compares measurement distributions; it does not determine arbitrary quantum
state fidelity. See [Qiskit's definition](https://quantum.cloud.ibm.com/docs/en/api/qiskit/quantum_info#hellinger_fidelity).

Recompute the labels from the stored counts:

```bash
python src/verify_fidelity.py
```

The command accepts a full CSV or `.csv.gz` path. It reports label differences
above `1e-6`, allowing for six-decimal rounding, and exits nonzero on mismatches.
Both the upstream `state:count; state:count` format and the generator's JSON count
format are supported.

Training learns circuit/backend features → fidelity × 10. It uses precomputed
labels and does not reconstruct distributions. Recomputing labels requires the
full counts. Prediction requires the circuit and backend features.

Labels have not been regenerated with the corrected data generator.

Original files at upstream commit `238cca87805a58193f8d00e483cf2ea75343aa1a`:

- [Full measurement CSV](https://github.com/mobaize/Q-selector/blob/238cca87805a58193f8d00e483cf2ea75343aa1a/data/MQTBench_578_analysis_results.csv)
- [Legacy model ZIP](https://github.com/mobaize/Q-selector/blob/238cca87805a58193f8d00e483cf2ea75343aa1a/results/trained_model_optimized.zip)

Checksums and file sizes are recorded in `provenance.json`.

`QASM/` contains the circuits; `backend_json/` contains the calibration snapshots
used for feature extraction.
