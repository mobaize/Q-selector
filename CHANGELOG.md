# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- ✨ New `src/recommend.py` — dedicated script for recommending processors for unknown QASM circuits without ground-truth comparison ([#1])
- ✨ Support for **folder input** — batch recommend an entire directory of QASM files ([#1])
- ✨ Automatic timestamped output archive under `results/your_work/<timestamp>/` with `best_recommendation.csv`, `full_ranking.csv`, and `full_ranking.json` ([#1])
- 📝 Professional documentation structure with `docs/` folder, Sphinx configuration, and comprehensive README ([#1])
- 🧪 Test infrastructure with `tests/` folder and `conftest.py` fixtures ([#1])
- 📦 Python packaging via `pyproject.toml` ([#1])
- 📄 MIT License, `CITATION.bib`, `CHANGELOG.md`, and `.gitignore` ([#1])

### Changed

- 🚚 Validation results moved from `results/` to `results/example_results/` to separate user-facing output from validation artifacts ([#1])
- 🔧 Output file naming changed from per-circuit names (`{circuit}_best_recommendation.csv`) to fixed names (`best_recommendation.csv`) since the parent folder is already timestamped ([#1])

[#1]: https://github.com/your-org/q-selector/pull/1
