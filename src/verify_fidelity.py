"""Recompute stored Hellinger fidelities from measurement counts."""

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

from qiskit.quantum_info import hellinger_fidelity

DEFAULT_DATA = (
    Path(__file__).resolve().parents[1] / "data" / "MQTBench_578_analysis_results_full.csv.gz"
)


def parse_counts(value: str) -> dict[str, int]:
    """Read JSON counts or the upstream ``state:count; state:count`` format."""
    value = value.strip()
    if value.startswith("{"):
        pairs = json.loads(value, object_pairs_hook=list)
    else:
        pairs = [part.strip().split(":") for part in value.split(";") if part.strip()]
    counts = {}
    for state, count in pairs:
        state = state.strip()
        if not state or state in counts:
            raise ValueError("Empty or duplicate measurement state")
        if isinstance(count, str):
            count = int(count)
        if type(count) is not int or count < 0:
            raise ValueError("Measurement counts must be nonnegative integers")
        counts[state] = count
    if not counts or sum(counts.values()) == 0:
        raise ValueError("Measurement counts must have a positive total")
    return counts


def verify_dataset(path: Path, tolerance: float = 1e-6) -> dict:
    """Stream a full CSV and count labels that differ beyond rounding tolerance."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    rows = mismatches = 0
    max_error = 0.0
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Hellinger_Fidelity", "Ideal_Counts_Full", "Noisy_Counts_Full"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            rows += 1
            try:
                stored = float(row["Hellinger_Fidelity"])
                if not math.isfinite(stored) or not 0 <= stored <= 1:
                    raise ValueError("Stored fidelity must lie in [0, 1]")
                actual = hellinger_fidelity(
                    parse_counts(row["Ideal_Counts_Full"]),
                    parse_counts(row["Noisy_Counts_Full"]),
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(f"CSV record {rows}: {exc}") from exc
            error = abs(actual - stored)
            max_error = max(max_error, error)
            mismatches += error > tolerance
    if not rows:
        raise ValueError("Dataset contains no measurements")
    return {
        "rows": rows,
        "mismatches": mismatches,
        "max_absolute_error": max_error,
        "tolerance": tolerance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    try:
        summary = verify_dataset(args.csv, args.tolerance)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(summary, indent=2))
    return int(summary["mismatches"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
