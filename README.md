# access-pricing

Dynamical model and figure-reproduction code for the access-pricing paper
*[full paper title and citation here once finalized]*.

## What's in here

```
access_pricing/         importable package
├── model.py            building blocks (mu, p, f, alpha) + ODE RHS + derivations
├── parameters.py       named parameter sets, one per paper figure
└── plot_style.py       shared matplotlib settings

notebooks/              thin notebooks, one per paper figure
└── fig2_phase_portrait.ipynb     Section 3 phase portrait
```

## Setup

```bash
git clone <repo-url>
cd access-pricing

# Optional but recommended: use a virtual environment
python3 -m venv .venv
source .venv/bin/activate            # macOS/Linux
# .venv\Scripts\activate              # Windows

# Install the package (editable) and dependencies
pip install -e .
pip install jupyterlab               # if you want to open the notebooks

# Run a notebook
jupyter lab notebooks/fig2_phase_portrait.ipynb
```

After `pip install -e .` you can also run individual notebook cells from any
Python process; the `access_pricing` package is importable from anywhere.

## Reproducing the paper figures

Each notebook generates one figure and writes a PDF to the same directory.
The notebooks are deliberately short (~5 cells) — all the math lives in
`access_pricing/`.

| Figure | Notebook | Section |
|---|---|---|
| Fig. 2 | `fig2_phase_portrait.ipynb` | Section 3 |
| ... | *more to come as the rest of the paper figures are ported* | |

## Computational notes

- The ODE is integrated with `scipy.integrate.odeint` (LSODA, adaptive step).
- Each evaluation of the dynamics is $O(n)$ in the number of user classes;
  a typical simulation (e.g. $T = 200$ time units, $n \leq 5$) runs in well
  under one second on a standard laptop.
- Tested with Python 3.10–3.12, NumPy 1.24+, SciPy 1.10+, Matplotlib 3.7+.

## License

*[to be added]*
