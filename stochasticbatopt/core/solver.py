from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from stochasticbatopt.core.params import StorageParams


def solve_dispatch_lp(
    prices    : np.ndarray,
    params    : StorageParams,
    soc_start : float,
    soc_end   : float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solve the intra-day dispatch LP for a fixed SoC start and end.

    Decision variables
    ------------------
    x = [e_in(0..H-1), e_out(0..H-1)]
      e_in_h  : energy charged   in hour h  [MWh]
      e_out_h : energy discharged in hour h [MWh]

    Objective (minimise):
        Σ_h  p_h / η_c · e_in_h  −  p_h · η_d · e_out_h

    Constraints:
        0 ≤ e_in_h  ≤ P_c · Δt
        0 ≤ e_out_h ≤ P_d · Δt
        Σ e_in − Σ e_out = soc_end − soc_start       [energy balance]
        Intra-day SoC path ∈ [min_energy, energy_capacity]

    Returns
    -------
    e_in, e_out : hourly energy arrays [MWh]
    revenue     : daily revenue [EUR]  (positive = profit)
                  Returns -inf if LP is infeasible.
    """
    p  = params
    dt = p.time_step
    H  = len(prices)

    c = np.concatenate([
        prices / p.charge_efficiency,
        -prices * p.discharge_efficiency,
    ])
    bounds = (
        [(0.0, p.charge_power    * dt)] * H +
        [(0.0, p.discharge_power * dt)] * H
    )

    L    = np.tril(np.ones((H, H)))
    A_ub = np.block([[L, -L], [-L, L]])
    b_ub = np.concatenate([
        np.full(H, p.energy_capacity - soc_start),
        np.full(H, soc_start         - p.min_energy),
    ])

    A_eq = np.concatenate([np.ones(H), -np.ones(H)])[np.newaxis, :]
    b_eq = np.array([soc_end - soc_start])

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if not res.success:
        return np.zeros(H), np.zeros(H), -np.inf

    e_in  = res.x[:H]
    e_out = res.x[H:]
    revenue = float(np.sum(
        prices * e_out * p.discharge_efficiency
        - prices * e_in  / p.charge_efficiency
    ))
    return e_in, e_out, revenue
