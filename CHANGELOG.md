# Changelog

## Unreleased

- Exclude backends with insufficient qubits and report inputs with no eligible backend.
- Make algorithm preferences explicit with `--algorithm`.
- Share feature extraction and ranking across training, prediction, and recommendation.
- Reject failed noise models and compile simulations against the selected backend target.
- Record training versions and dataset checksum; refresh held-out evaluation examples.
- Remove raw count columns, the legacy model ZIP, and personal run outputs from the current tree.
- Update repository links, shorten documentation, and add regression tests and CI checks.
