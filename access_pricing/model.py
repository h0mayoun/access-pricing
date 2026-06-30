"""Core dynamical model for the access-pricing system.

Implements the four building-block functions of the model used in the paper:

    mu(q)       service rate
    p(q)        price function (multiple variants)
    f(p)        class sensitivity (dropout rate as a function of price)
    alpha(q)    admission rate

and the right-hand side dX/dt of the closed-loop ODE.

Notation matches the paper. Each building block has multiple shape variants
because different figures in the paper use different shapes. Notebooks pick
the variants they need and pass them into dX_dt.
"""

from __future__ import annotations
import numpy as np


# ============================================================
# SERVICE RATE  mu(q)
# ============================================================

def mu_trapezoidal(q, q_c, q_max, mu_max):
    """Service rate: linear ramp to mu_max at q_c, then flat to q_max.

    Used for: 2D phase-portrait figures.

    Raises ValueError for q > q_max (queue overflow).
    """
    q = np.asarray(q, dtype=float)
    if np.any(q > q_max):
        raise ValueError("Queue overflow in mu_trapezoidal (q > q_max).")
    return np.where(q < q_c, mu_max * q / q_c, mu_max)


def mu_triangular(q, q_c, q_max, mu_max):
    """Service rate: linear ramp to mu_max at q_c, linear drop to 0 at q_max.

    Used for: 3D timeseries figures.

    Raises ValueError for q > q_max (queue overflow).
    """
    q = np.asarray(q, dtype=float)
    if np.any(q > q_max):
        raise ValueError("Queue overflow in mu_triangular (q > q_max).")
    return np.where(
        q < q_c,
        mu_max * q / q_c,
        mu_max * (q_max - q) / (q_max - q_c),
    )


# ============================================================
# PRICE FUNCTION  p(q)
# ============================================================

def p_surge(q, q_max, beta):
    """Monotonic surge price: p(q) = beta * q on [0, q_max].

    The baseline (unfair) pricing scheme.
    """
    q = np.asarray(q, dtype=float)
    if np.any(q > q_max):
        raise ValueError("Queue overflow in p_surge (q > q_max).")
    return beta * q


def p_triangle(q, q_m, beta):
    """Symmetric triangle non-monotonic price.

    Rises linearly to beta * q_m at q = q_m, falls linearly to 0 at q = 2*q_m,
    zero beyond.

    Used for: 2D phase-portrait figures.
    """
    q = np.asarray(q, dtype=float)
    return np.where(
        q < q_m,
        beta * q,
        np.where(q <= 2 * q_m, beta * (2 * q_m - q), 0.0),
    )


def p_trapezoid(q, q_m, q_max, beta):
    """Asymmetric trapezoid non-monotonic price.

    Rises linearly to beta * q_m at q = q_m, falls linearly to 0 at q = q_max,
    zero beyond.

    Used for: 3D timeseries figures.
    """
    q = np.asarray(q, dtype=float)
    return np.where(
        q < q_m,
        beta * q,
        np.where(
            q <= q_max,
            beta * q_m * (q_max - q) / (q_max - q_m),
            0.0,
        ),
    )


def p_saturated(q, q_m, q_max, beta, p_min):
    """Saturated trapezoid: same as p_trapezoid but with floor p_min instead of 0.

    Used for: Section 5 (non-vanishing prices).
    """
    return np.maximum(p_trapezoid(q, q_m, q_max, beta), p_min)


# ============================================================
# CLASS SENSITIVITY  f(p)
# ============================================================

def f_identity(p):
    """Linear/identity sensitivity: f(p) = p.

    With p_triangle, composes to recover the old `f_fair1(q)` shape used in
    the 2D phase-portrait figures.
    """
    return np.asarray(p, dtype=float)


def f_quadratic(p, c=100.0):
    """Quadratic sensitivity: f(p) = c * p**2.

    Used for: 3D timeseries figures (default c=100 matches the notebooks).
    """
    p = np.asarray(p, dtype=float)
    return c * p ** 2


# ============================================================
# ADMISSION RATE  alpha(q)
# ============================================================

def alpha_linear(q, slope, intcp, q_max):
    """Linear admission rate with intercept.

    alpha(q) = max(intcp + slope*q, 0) for q in [0, q_max), zero outside.

    Used for: 2D figures (slope and intcp are derived from equilibrium choice).
    """
    q = np.asarray(q, dtype=float)
    raw = intcp + slope * q
    return np.where((0 <= q) & (q < q_max), np.maximum(raw, 0.0), 0.0)


def alpha_linear_no_intcp(q, slope, q_max, cutoff_frac=0.7):
    """Linear admission rate, zero at q = cutoff_frac * q_max.

    alpha(q) = (cutoff_frac*q_max - q) * slope for q in [0, cutoff_frac*q_max],
    zero outside.

    Used for: 3D figures (default cutoff_frac=0.7 matches the notebooks).
    """
    q = np.asarray(q, dtype=float)
    cutoff = cutoff_frac * q_max
    return np.where((0 <= q) & (q <= cutoff), (cutoff - q) * slope, 0.0)


# ============================================================
# ODE RHS
# ============================================================

def dX_dt(X, t, *, K_R, mu_fn, p_fn, f_fn, alpha_fn, K_U=0.0):
    """Right-hand side of the closed-loop ODE.

    State X is interpreted by length:
        len(X) == 2  ->  X = [R, q]    (2D case; K_U defaults to 0 and is unused)
        len(X) == 3  ->  X = [R, U, q] (3D case; pass K_U > 0)

    The four shape functions must already be parameterized closures of q only
    (e.g. via functools.partial) so that mu_fn(q), p_fn(q), f_fn(p_fn(q)),
    and alpha_fn(q) all evaluate scalars at scalar q.
    """
    if len(X) == 2:
        R, q = X
        a = alpha_fn(q)
        Rdot = K_R - f_fn(p_fn(q)) * R - a * R
        qdot = a * R - mu_fn(q)
        return np.array([Rdot, qdot])

    elif len(X) == 3:
        R, U, q = X
        a = alpha_fn(q)
        Rdot = K_R - f_fn(p_fn(q)) * R - a * R
        Udot = K_U - a * U
        qdot = a * (R + U) - mu_fn(q)
        return np.array([Rdot, Udot, qdot])

    else:
        raise ValueError(
            f"State X has unsupported length {len(X)}; expected 2 or 3."
        )


# ============================================================
# DERIVATION HELPERS
# Closed-form derivation of alpha parameters from the choice of
# equilibrium location(s). Notebooks call these so the connection
# between chosen equilibria and the resulting alpha shape is explicit.
# ============================================================

def derive_alpha_2d_from_two_equilibria(*, q1, q2, K_R, mu_max, q_c, q_max, q_m, beta):
    """Derive (alpha_slope, alpha_intcp) so that the 2D system has equilibria at q = q1 and q = q2.

    At an equilibrium (R*, q*) with f(p) = p (the 2D identity sensitivity), the
    conditions dR/dt = 0 and dq/dt = 0 give:

        alpha(q*) = mu(q*) * p(q*) / (K_R - mu(q*))

    A linear alpha is then fit through (q1, alpha*(q1)) and (q2, alpha*(q2)).
    Returns (slope, intercept) for `alpha_linear`.
    """
    mu_star_1 = float(mu_trapezoidal(q1, q_c, q_max, mu_max))
    mu_star_2 = float(mu_trapezoidal(q2, q_c, q_max, mu_max))
    p_star_1 = float(p_triangle(q1, q_m, beta))
    p_star_2 = float(p_triangle(q2, q_m, beta))

    alpha_star_1 = (p_star_1 * mu_star_1) / (K_R - mu_star_1)
    alpha_star_2 = (p_star_2 * mu_star_2) / (K_R - mu_star_2)

    slope = (alpha_star_2 - alpha_star_1) / (q2 - q1)
    intcp = alpha_star_1 - q1 * slope
    return slope, intcp


def derive_alpha_3d_slope_from_equilibrium(
    *, q_star, K_R, K_U, mu_max, q_c, q_max, q_m, beta,
    c=100.0, anchor_frac=0.8,
):
    """Derive alpha_slope so that the 3D system has an equilibrium at q = q_star.

    With f(p) = c*p^2 (the 3D quadratic sensitivity), at equilibrium (R*, U*, q*):

        alpha(q*) = (mu(q*) - K_U) * f(p(q*)) / (K_R + K_U - mu(q*))

    The 3D alpha has the form alpha(q) = (cutoff_frac * q_max - q) * slope; the
    notebooks use cutoff_frac=0.7 in the alpha function itself but anchor the
    derivation against `anchor_frac` * q_max (default 0.8), so alpha(q_star) is
    not exactly the derived alpha_star. This (deliberate) inconsistency is
    preserved here so figures match the published values.
    """
    mu_star = float(mu_triangular(q_star, q_c, q_max, mu_max))
    p_star = float(p_trapezoid(q_star, q_m, q_max, beta))
    f_star = c * p_star ** 2

    alpha_star = ((mu_star - K_U) * f_star) / (K_R + K_U - mu_star)
    slope = alpha_star / (anchor_frac * q_max - q_star)
    return slope
