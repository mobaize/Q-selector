# Installation

Q-Selector requires Python 3.10 or later and can be installed on all major operating systems.

## Quick Install

```console
$ python -m pip install -e .
```

## Development Setup

1.  Open a terminal in the Q-Selector project directory.

2.  Create and activate a virtual environment:

    ```console
    $ python -m venv .venv
    $ source .venv/bin/activate  # Linux/macOS
    $ .venv\Scripts\activate     # Windows
    ```

3.  Install Q-Selector and the development dependencies:

    ```console
    (.venv) $ python -m pip install -e ".[dev]"
    ```

4.  Optionally rebuild the included pre-trained model:

    ```console
    (.venv) $ python src/train.py
    ```

## Verify Installation

```console
(.venv) $ python src/recommend.py data/QASM/ae_indep_qiskit_10.qasm --no-save
```

If the installation is successful, you should see a ranked list of recommended backends.
