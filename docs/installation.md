# Installation

Requires Python 3.10 or later.

```bash
git clone https://github.com/mobaize/Q-selector.git
cd Q-selector
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

```bash
python -m pip install -e .
python src/train.py
q-selector data/QASM/ae_indep_qiskit_10.qasm --no-save
```

Optional dependency groups:

| Group | Command |
|---|---|
| Tests and lint | `python -m pip install -e ".[dev,data]"` |
| Data generation | `python -m pip install -e ".[data]"` |
| Documentation | `python -m pip install -e ".[docs]"` |

Model files are not included. Train locally before the first recommendation.
Generated models and run outputs are ignored by Git.
