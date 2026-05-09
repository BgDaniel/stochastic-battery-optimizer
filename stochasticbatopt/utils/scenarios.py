from __future__ import annotations
import numpy as np
import pandas as pd


def generate_price_scenarios(
    time_index   : pd.DatetimeIndex,
    sigma        : float,
    kappa        : float,
    n_scenarios  : int,
    random_state : int = 42,
) -> np.ndarray:
    """
    Generate synthetic DA price scenarios.

    A deterministic hourly shape (typical German double-peak profile) is
    combined with a mean-reverting Ornstein-Uhlenbeck stochastic component.

        p_t = shape_t + X_t
        dX_t = −κ · X_t · dt + σ · √dt · dW_t

    Parameters
    ----------
    time_index   : hourly DatetimeIndex
    sigma        : OU volatility  [EUR/MWh / √h]
    kappa        : mean-reversion speed  [1/h]
    n_scenarios  : number of Monte Carlo paths
    random_state : numpy RNG seed

    Returns
    -------
    scenarios : np.ndarray of shape (n_scenarios, T)
    """
    rng = np.random.default_rng(random_state)
    T   = len(time_index)
    dt  = 1.0 / (24.0 * 365.0)

    _shape_24 = np.array([
        32, 30, 29, 28, 29, 34,
        48, 62, 67, 57, 46, 36,
        35, 32, 36, 42, 54, 72,
        82, 76, 66, 56, 46, 38,
    ], dtype=float)

    n_days     = int(np.ceil(T / 24))
    shape_full = np.tile(_shape_24, n_days)[:T]

    X        = np.zeros((n_scenarios, T))
    for t in range(1, T):
        dW      = rng.standard_normal(n_scenarios)
        X[:, t] = (
            X[:, t - 1]
            - kappa * X[:, t - 1] * dt
            + sigma * np.sqrt(dt) * dW
        )

    return shape_full[np.newaxis, :] + X
