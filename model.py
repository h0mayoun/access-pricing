"""Access-pricing 3D model: building blocks + ODE RHS.

Mirrors the functions used in the original 3DModel.ipynb, with cleaned-up
names and vectorization. Imported by the notebook via `from model import *`.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit


# ---- Service rate ---------------------------------------------------------

def mu(q, q_c, q_max, mu_max):
    """Trapezoidal service rate: ramps to mu_max at q_c, then flat to q_max."""
    q = np.asarray(q, dtype=float)
    if np.any(q > q_max):
        raise ValueError("Queue overflow in mu (q > q_max).")
    return np.where(q < q_c, mu_max * q / q_c, mu_max)


# ---- Price functions ------------------------------------------------------

def f_surge(q, beta):
    """Monotonic surge: f(q) = beta * q. Baseline (unfair) pricing."""
    return beta * np.asarray(q, dtype=float)


def f_saturated(q, q_m, q_n, beta):
    """Triangle with a positive floor (paper's non-vanishing variant).

        f(q) = beta*q                       for q < q_m
             = beta*(2*q_m - q)             for q_m <= q <= q_n
             = beta*(2*q_m - q_n)           for q > q_n

    Recovers the vanishing triangle when q_n = 2*q_m (floor becomes 0).
    """
    q = np.asarray(q, dtype=float)
    return np.where(
        q < q_m,
        beta * q,
        np.where(q <= q_n, beta * (2 * q_m - q), beta * (2 * q_m - q_n)),
    )


# ---- Admission rate -------------------------------------------------------

def alpha(q, params):
    """Cubic admission rate: alpha(q) = a*q^3 + b*q^2 + c*q + d.

    params = (a, b, c, d); fit via `fit_alpha_cubic`.
    """
    a, b, c, d = params
    q = np.asarray(q, dtype=float)
    return a * q ** 3 + b * q ** 2 + c * q + d


def fit_alpha_cubic(x_points, y_points, p0=(0.001, 0.001, 0.0, 0.0)):
    """Fit a*x^3 + b*x^2 + c*x + d through (x_points, y_points). Returns (a, b, c, d)."""
    def cubic(x, a, b, c, d):
        return a * x ** 3 + b * x ** 2 + c * x + d
    (a, b, c, d), _ = curve_fit(cubic, np.asarray(x_points),
                                np.asarray(y_points), p0=p0)
    return (a, b, c, d)


# ---- ODE RHS --------------------------------------------------------------

def dX_dt_surge(X, t, K_R, K_U, q_c, q_max, mu_max, q_m, q_n, beta, alpha_params):
    """3D system under surge pricing. State X = [R, U, q].

    q_m, q_n accepted for API symmetry with dX_dt_fair (unused for surge).
    """
    R, U, q = X
    a_q = alpha(q, alpha_params)
    Rdot = K_R - f_surge(q, beta) * R - a_q * R
    Udot = K_U - a_q * U
    qdot = a_q * (R + U) - mu(q, q_c, q_max, mu_max)
    return np.array([Rdot, Udot, qdot])


def dX_dt_fair(X, t, K_R, K_U, q_c, q_max, mu_max, q_m, q_n, beta, alpha_params):
    """3D system under fair (non-monotonic) pricing. State X = [R, U, q].

    Pass q_n = 2*q_m for the vanishing variant; q_n < 2*q_m for the saturated variant.
    """
    R, U, q = X
    a_q = alpha(q, alpha_params)
    Rdot = K_R - f_saturated(q, q_m, q_n, beta) * R - a_q * R
    Udot = K_U - a_q * U
    qdot = a_q * (R + U) - mu(q, q_c, q_max, mu_max)
    return np.array([Rdot, Udot, qdot])
