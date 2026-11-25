# Evaluation example

Generated from the bundled dataset with:

```bash
python src/train.py --jobs 2
python src/predict.py
python src/evaluate.py
```

Algorithm preferences are disabled. The split contains 404 training circuits and
174 test circuits; 652 circuit/backend pairs are excluded because the backend
has too few qubits. Dataset checksum and dependency versions are in
`training_metadata.json`.

| File | Content |
|---|---|
| `training_metadata.json` | Training settings, versions, and regression metrics |
| `feature_list_threshold_005.csv` | Feature importances and selection flags |
| `optimization_report.txt` | Regression metrics |
| `predictions_optimized.csv` | Held-out circuit rankings and measured outcomes |
| `prediction_summary.json` | Top-1/Top-3 accuracy and mean regret |
| `baseline_comparison_optimized.csv` | Comparisons with feasible low-error baselines |
| `baseline_analysis_optimized.json` | The same comparisons in JSON |

These results use the upstream measurement labels. The full dataset has not
been regenerated with the corrected simulator.
