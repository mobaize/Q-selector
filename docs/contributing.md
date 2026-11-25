# Development

```bash
python -m pip install -e ".[dev,data,docs]"
python -m pytest
python -m ruff check src tests docs/conf.py
python -m ruff format --check src tests docs/conf.py
python -m sphinx -W -b html docs docs/_build/html
```

Keep shared feature and ranking logic in `src/features.py`. Add regression tests
for behavior changes and update the affected usage documentation.

Generated models, measurements, and recommendation runs belong under `results/`
and are ignored by Git. Committed examples include training metadata so their
data and dependency versions can be identified.
