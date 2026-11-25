# Dataset

`MQTBench_578_analysis_results.csv` contains 11,560 rows: 578 circuits × 20 backends.
Training reads `Circuit`, `Backend`, and `Hellinger_Fidelity`.

The current CSV omits `Ideal_Counts_Full` and `Noisy_Counts_Full`. All other field
values are retained. Labels have not been regenerated with the corrected data
generator.

Original files at upstream commit `238cca87805a58193f8d00e483cf2ea75343aa1a`:

- [Full measurement CSV](https://github.com/mobaize/Q-selector/blob/238cca87805a58193f8d00e483cf2ea75343aa1a/data/MQTBench_578_analysis_results.csv)
- [Legacy model ZIP](https://github.com/mobaize/Q-selector/blob/238cca87805a58193f8d00e483cf2ea75343aa1a/results/trained_model_optimized.zip)

Checksums and file sizes are recorded in `provenance.json`. The original files
also remain in this repository's imported Git history.

`QASM/` contains the circuits; `backend_json/` contains the calibration snapshots
used for feature extraction.
