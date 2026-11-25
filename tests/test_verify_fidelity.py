import csv
import gzip

import pytest

from src.verify_fidelity import parse_counts, verify_dataset


@pytest.mark.parametrize("value", ['{"0": 3, "1": 1}', "0:3; 1:1"])
def test_count_formats(value):
    assert parse_counts(value) == {"0": 3, "1": 1}


@pytest.mark.parametrize("value", ["", "0:0", "0:-1", "0:1; 0:2", '{"0":1,"0":2}', '{"0":1.5}'])
def test_invalid_counts(value):
    with pytest.raises(ValueError):
        parse_counts(value)


@pytest.mark.parametrize("compressed", [True, False])
def test_recompute_labels(tmp_path, compressed):
    path = tmp_path / ("counts.csv.gz" if compressed else "counts.csv")
    opener = gzip.open if compressed else open
    with opener(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Ideal_Counts_Full", "Noisy_Counts_Full", "Hellinger_Fidelity"])
        writer.writerow(["0:100", '{"0":50,"1":50}', "0.5"])
        writer.writerow(["0:100", "1:100", "0.0"])
        writer.writerow(["0:100", "0:200", "1.0"])
        writer.writerow(["0:100", "1:100", "1.0"])
    summary = verify_dataset(path)
    assert summary["rows"] == 4
    assert summary["mismatches"] == 1
    assert summary["max_absolute_error"] == pytest.approx(1)


def test_training_csv_cannot_verify_labels(tmp_path):
    path = tmp_path / "training.csv"
    path.write_text("Circuit,Backend,Hellinger_Fidelity\nbell,backend,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing columns"):
        verify_dataset(path)
