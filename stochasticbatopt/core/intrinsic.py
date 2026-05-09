from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from stochasticbatopt.core.optimizer import StochasticBatteryOptimizer


class IntrinsicOptimizer(StochasticBatteryOptimizer):
    """
    Deterministic special case of StochasticBatteryOptimizer.

    Runs on a single price path (N=1) with no regression — the
    continuation value is read directly from V[:, 0, day+1].
    Used internally to compute the intrinsic (deterministic) value
    as a baseline for the extrinsic value decomposition.
    """

    def _backward_induction(self) -> None:
        N, T = self.price_scenarios.shape
        assert N == 1, "IntrinsicOptimizer requires exactly one price path (N=1)."

        n_days   = T // 24
        self.n_days = n_days
        grid     = self._energy_grid()
        self.grid = grid
        n_lv     = self.n_levels

        prices_daily = self.price_scenarios.reshape(1, n_days, 24)

        self.V             = np.zeros((n_lv, 1, n_days),     dtype=np.float32)
        self._reg_weights  = np.zeros((n_lv, n_days, self.n_basis), dtype=np.float32)
        self.fwd_e_in      = np.zeros((n_lv, 1, n_days, 24), dtype=np.float32)
        self.fwd_e_out     = np.zeros((n_lv, 1, n_days, 24), dtype=np.float32)
        self.fwd_level_end = np.zeros((n_lv, 1, n_days),     dtype=np.int32)

        resume_from = self._restore_latest(n_days)
        if resume_from < 0:
            return

        for day in tqdm(range(resume_from, -1, -1),
                        total=resume_from + 1,
                        desc="Intrinsic backward induction"):

            for lv in range(n_lv):
                cont_vec = (
                    self.V[:, 0, day + 1].astype(np.float64)
                    if day < n_days - 1 else np.zeros(n_lv)
                )
                _, _, e_in, e_out, end_lv, val = self._dispatch_job(
                    0, lv, prices_daily[0, day], cont_vec, grid,
                )
                self.fwd_e_in[lv, 0, day]      = e_in
                self.fwd_e_out[lv, 0, day]     = e_out
                self.fwd_level_end[lv, 0, day] = end_lv
                self.V[lv, 0, day]             = val

            self._save(day, self._pack_state())

    def fit(
        self,
        price_scenarios : np.ndarray,
        time_index      : pd.DatetimeIndex,
        start_energy    : float = 0.0,
        run_id          : Optional[str] = None,
    ) -> dict:
        if run_id is not None:
            self.run_id = run_id
        self.price_scenarios = price_scenarios
        self._time_index     = time_index
        self._run_key        = self._make_run_key(price_scenarios)
        self._backward_induction()
        fwd = self._forward_simulation(start_energy)
        return {
            "expected_revenue"     : float(fwd["revenue_per_scenario"].mean()),
            "revenue_per_scenario" : fwd["revenue_per_scenario"],
            "energy_path"          : fwd["energy_path"],
            "e_in"                 : fwd["e_in"],
            "e_out"                : fwd["e_out"],
            "time_index"           : time_index,
        }
