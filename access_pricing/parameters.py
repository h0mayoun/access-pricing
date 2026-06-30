"""Named parameter sets for paper figures.

Each parameter set is a frozen dataclass storing the *input* choices for one
figure: arrival rates, queue caps, price-function shape parameters, and the
locations of the equilibria from which derived quantities (alpha slope, intcp)
are computed. Notebooks call the corresponding derivation helpers from
`access_pricing.model` explicitly, so the connection from chosen equilibria
to derived alpha parameters is visible at the point of use.
"""

from dataclasses import dataclass


# ============================================================
# 2D figures (Section 3 phase portraits)
# ============================================================

@dataclass(frozen=True)
class FigParams2D:
    """Inputs for a 2D phase-portrait figure.

    Attributes
    ----------
    K_R, mu_max, q_c, q_max : float
        Service / arrival parameters.
    q_m, beta : float
        Price-function shape parameters (triangle: peak at q_m, zero at 2*q_m).
    q1, q2 : float
        Chosen locations of the two equilibria. The alpha slope and intercept
        are derived from these via `derive_alpha_2d_from_two_equilibria` in
        the notebook.
    description : str
        Human-readable description.
    """
    K_R: float
    mu_max: float
    q_c: float
    q_max: float
    q_m: float
    beta: float
    q1: float
    q2: float
    description: str = ""


FIG2 = FigParams2D(
    K_R=10.0,
    mu_max=5.0,
    q_c=55.0,
    q_max=100.0,
    q_m=45.0,
    beta=0.001,
    q1=35.0,
    q2=85.0,
    description="Section 3 phase portrait (Fig. 2 in the paper).",
)
