# mtspbc

ALNS-based solver for the **multiple Travelling Salesman Problem with Backup Coverage (mTSP-BC)**.

Supplemental material for:

> **[Paper title]** — *[Authors]*  
> DOI: `[DOI placeholder]`

## Installation

Requires a C++17 compiler, Python ≥ 3.9, and pybind11.

```bash
pip install pybind11 numpy
pip install -e .
```

## Usage

```python
import mtspbc
mtspbc.solve("path/to/instance.BC")
```

Or from the command line:

```bash
mtspbc path/to/instance.BC
```

### Instance format

```
m  n  d
x0 y0
x1 y1
...
```

First line: number of vehicles `m`, number of clients `n`, maximum coverage distance `d`.  
Followed by `n+1` coordinate pairs (depot first).

## Acknowledgements

The ALNS framework used here is adapted from the open-source
[`alns`](https://alns.readthedocs.io/en/latest/) Python package by
Wouter Kool, Niels Wouda, and Leon Lan.
