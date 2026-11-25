# Dataset

Both datasets contain 11,560 rows: 578 circuits × 20 backends.

- `MQTBench_578_analysis_results.csv`: training table. Training reads `Circuit`,
  `Backend`, and `Hellinger_Fidelity`.
- `MQTBench_578_analysis_results_full.csv.gz`: original CSV with
  `Ideal_Counts_Full` and `Noisy_Counts_Full`. Read directly with `pd.read_csv(path)`.

Labels use [Hellinger fidelity](https://quantum.cloud.ibm.com/docs/en/api/qiskit/quantum_info#hellinger_fidelity)
between normalized measurement distributions:
`F(P, Q) = (sum(sqrt(P(x) * Q(x)))) ** 2`.

Recompute the labels from the stored counts:

```bash
python src/verify_fidelity.py
```

The command accepts a full CSV or `.csv.gz` path. It reports label differences
above `1e-6`, allowing for six-decimal rounding, and exits nonzero on mismatches.
Both the upstream `state:count; state:count` format and the generator's JSON count
format are supported.

Training target: `Hellinger_Fidelity × 10`. Labels predate the data-generator fixes.

Original measurement file at source commit `238cca87805a58193f8d00e483cf2ea75343aa1a`:

- [Full measurement CSV](https://github.com/mobaize/Q-selector/blob/238cca87805a58193f8d00e483cf2ea75343aa1a/data/MQTBench_578_analysis_results.csv)

Checksums and file sizes are recorded in `provenance.json`.

`QASM/` contains the circuits; `backend_json/` contains the calibration snapshots
used for feature extraction.
