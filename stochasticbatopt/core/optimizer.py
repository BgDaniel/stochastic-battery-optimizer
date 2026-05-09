from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from stochasticbatopt.core.params import StorageParams
from stochasticbatopt.core.solver import solve_dispatch_lp
from stochasticbatopt.core.checkpoint import CheckpointMixin


class StochasticBatteryOptimizer(CheckpointMixin):
    """
    LSMC valuation of a grid-scale storage asset with explicit SoC grid.

    The Bellman state is (energy_level, mean_daily_price).
    For each (scenario, energy_level) pair the inner LP is solved for every
    candidate end-of-day energy level, and the best is selected by adding
    the regressed continuation value.

    Parameters
    ----------
    params      : StorageParams
    n_levels    : Number of discrete energy levels  (SoC grid)
    n_basis     : Degree of polynomial regression basis
    cache_dir   : Directory for day-level checkpoint files
    run_id      : Identifier prefix for checkpoint filenames
    random_state: RNG seed
    """

    def __init__(
        self,
        params       : StorageParams,
        n_levels     : int = 10,
        n_basis      : int = 4,
        cache_dir    : str | Path = "sbo_cache",
        run_id       : str = "run",
        random_state : Optional[int] = None,
    ) -> None:
        self.params      = params
        self.n_levels    = n_levels
        self.n_basis     = n_basis
        self.cache_dir   = Path(cache_dir)
        self.run_id      = run_id
        self.rng         = np.random.default_rng(random_state)

        self._reg_weights  : Optional[np.ndarray] = None
        self._run_key      : Optional[str]         = None
        self._reference    : Optional[dict]        = None
        self._value_decomp : Optional[dict]        = None

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── energy level grid ─────────────────────────────────────────────────

    def _energy_grid(self) -> np.ndarray:
        p = self.params
        return np.linspace(p.min_energy, p.energy_capacity, self.n_levels)

    # ── regression basis ──────────────────────────────────────────────────

    def _basis(self, price: np.ndarray) -> np.ndarray:
        price  = np.atleast_1d(price).astype(float)
        mu, sg = price.mean(), price.std()
        if sg < 1e-8:
            sg = 1.0
        z     = (price - mu) / sg
        cols  = np.column_stack([z ** j for j in range(self.n_basis)])
        norms = np.linalg.norm(cols, axis=0)
        norms[norms < 1e-8] = 1.0
        return cols / norms

    # ── single dispatch job ───────────────────────────────────────────────

    def _dispatch_job(
        self,
        scenario_idx : int,
        level_idx    : int,
        hourly_prices: np.ndarray,
        cont_vec     : np.ndarray,
        grid         : np.ndarray,
    ) -> tuple:
        soc_start = grid[level_idx]
        best_val  = -np.inf
        best_ein  = np.zeros(len(hourly_prices))
        best_eout = np.zeros(len(hourly_prices))
        best_end  = level_idx

        for end_idx, soc_end in enumerate(grid):
            e_in, e_out, rev = solve_dispatch_lp(
                hourly_prices, self.params, soc_start, soc_end
            )
            if rev == -np.inf:
                continue
            total = rev + cont_vec[end_idx]
            if total > best_val:
                best_val  = total
                best_ein  = e_in
                best_eout = e_out
                best_end  = end_idx

        return scenario_idx, level_idx, best_ein, best_eout, best_end, best_val

    # ── backward induction ────────────────────────────────────────────────

    def _backward_induction(self) -> None:
        N, T     = self.price_scenarios.shape
        n_days   = T // 24
        self.n_days = n_days
        grid     = self._energy_grid()
        self.grid = grid
        n_lv     = self.n_levels

        prices_daily = self.price_scenarios.reshape(N, n_days, 24)
        mean_daily   = prices_daily.mean(axis=2)

        # storage
        self.V             = np.zeros((n_lv, N, n_days),    dtype=np.float32)
        self._reg_weights  = np.zeros((n_lv, n_days, self.n_basis), dtype=np.float32)
        self.fwd_e_in      = np.zeros((n_lv, N, n_days, 24), dtype=np.float32)
        self.fwd_e_out     = np.zeros((n_lv, N, n_days, 24), dtype=np.float32)
        self.fwd_level_end = np.zeros((n_lv, N, n_days),    dtype=np.int32)

        resume_from = self._restore_latest(n_days)
        if resume_from < 0:
            print("  [checkpoint] Backward induction already complete.")
            return

        for day in tqdm(range(resume_from, -1, -1),
                        total=resume_from + 1,
                        desc="Backward induction"):

            price_t = mean_daily[:, day]
            cont    = np.zeros((n_lv, N), dtype=np.float64)

            if day < n_days - 1:
                for lv in range(n_lv):
                    Phi  = self._basis(price_t)
                    w, _, _, _ = np.linalg.lstsq(
                        Phi, self.V[lv, :, day + 1].astype(np.float64),
                        rcond=None,
                    )
                    self._reg_weights[lv, day] = w
                    cont[lv] = Phi @ w

            jobs = (
                delayed(self._dispatch_job)(
                    s, lv,
                    prices_daily[s, day],
                    cont[:, s],
                    grid,
                )
                for s in range(N)
                for lv in range(n_lv)
            )
            results = list(
                tqdm(
                    Parallel(n_jobs=-1, backend="loky", return_as="generator")(jobs),
                    total=N * n_lv,
                    desc=f"  day {day:>3d}  N={N} × lv={n_lv}",
                    leave=False,
                )
            )

            for s, lv, e_in, e_out, end_lv, val in results:
                self.fwd_e_in[lv, s, day]      = e_in
                self.fwd_e_out[lv, s, day]     = e_out
                self.fwd_level_end[lv, s, day] = end_lv
                self.V[lv, s, day]             = val

            self._save(day, self._pack_state())

    # ── forward simulation ────────────────────────────────────────────────

    def _forward_simulation(self, start_energy: float) -> dict:
        N, T   = self.price_scenarios.shape
        p      = self.params
        dt     = p.time_step
        grid   = self.grid

        E_in   = np.zeros((N, T))
        E_out  = np.zeros((N, T))
        SoC    = np.zeros((N, T))
        revenue = np.zeros(N)

        lv_idx = np.full(N, np.argmin(np.abs(grid - start_energy)), dtype=int)

        for day in range(self.n_days):
            h0, h1  = day * 24, day * 24 + 24
            s_idx   = np.arange(N)

            ein_d   = self.fwd_e_in[lv_idx, s_idx, day]
            eout_d  = self.fwd_e_out[lv_idx, s_idx, day]

            E_in[:, h0:h1]  = ein_d
            E_out[:, h0:h1] = eout_d

            net = ein_d - eout_d
            soc_start_d = grid[lv_idx]
            SoC[:, h0:h1] = np.clip(
                soc_start_d[:, None] + np.hstack([
                    np.zeros((N, 1)),
                    np.cumsum(net, axis=1)[:, :-1],
                ]),
                p.min_energy, p.energy_capacity,
            )

            prices_d  = self.price_scenarios[:, h0:h1]
            power_out = eout_d * p.discharge_efficiency / dt
            power_in  = ein_d  / (p.charge_efficiency   * dt)
            revenue  += np.sum(prices_d * power_out - prices_d * power_in, axis=1) * dt

            lv_idx = self.fwd_level_end[lv_idx, s_idx, day]

        return {"e_in": E_in, "e_out": E_out, "energy_path": SoC,
                "revenue_per_scenario": revenue}

    # ── public API ────────────────────────────────────────────────────────

    def fit(
        self,
        price_scenarios : np.ndarray,
        time_index      : pd.DatetimeIndex,
        start_energy    : float = 0.0,
        run_id          : Optional[str] = None,
    ) -> dict:
        """
        Run backward induction + forward simulation.

        Parameters
        ----------
        price_scenarios : (N, T) array of hourly DA prices [EUR/MWh]
        time_index      : hourly DatetimeIndex aligned to axis-1 of scenarios
        start_energy    : initial energy level [MWh]
        run_id          : override instance run_id for cache namespacing

        Returns
        -------
        result dict with keys:
          expected_revenue, revenue_per_scenario, energy_path,
          e_in, e_out, time_index
        """
        if run_id is not None:
            self.run_id = run_id

        self.price_scenarios = price_scenarios
        self._time_index     = time_index
        self._run_key        = self._make_run_key(price_scenarios)
        print(f"Run key: {self._run_key}")

        self._backward_induction()
        fwd = self._forward_simulation(start_energy)

        # ── reference (intrinsic) valuation ──────────────────────────────
        from stochasticbatopt.core.intrinsic import IntrinsicOptimizer
        mean_prices = price_scenarios.mean(axis=0)
        ref_opt = IntrinsicOptimizer(
            params    = self.params,
            n_levels  = self.n_levels,
            n_basis   = self.n_basis,
            cache_dir = self.cache_dir,
            run_id    = self.run_id + "_ref",
        )
        ref_result = ref_opt.fit(
            price_scenarios = mean_prices[np.newaxis, :],
            time_index      = time_index,
            start_energy    = start_energy,
        )

        p      = self.params
        dt     = p.time_step
        ref_ein  = ref_result["e_in"][0]
        ref_eout = ref_result["e_out"][0]
        ref_cum  = np.cumsum(
            mean_prices * ref_eout * p.discharge_efficiency
            - mean_prices * ref_ein  / p.charge_efficiency
        ) * dt

        start_lv  = int(np.argmin(np.abs(self.grid - start_energy)))
        total_val = float(self.V[start_lv, :, 0].mean())
        ref_val   = float(ref_result["expected_revenue"])
        extr_val  = total_val - ref_val

        self._reference = {
            "price_path" : mean_prices,
            "e_in"       : ref_ein,
            "e_out"      : ref_eout,
            "energy_path": ref_result["energy_path"][0],
            "cum_revenue": ref_cum,
            "total"      : ref_val,
        }
        self._value_decomp = {
            "total"     : total_val,
            "intrinsic" : ref_val,
            "extrinsic" : extr_val,
            "intr_pct"  : 100 * ref_val  / total_val if abs(total_val) > 1e-9 else 0.0,
            "extr_pct"  : 100 * extr_val / total_val if abs(total_val) > 1e-9 else 0.0,
        }

        return {
            "expected_revenue"      : float(fwd["revenue_per_scenario"].mean()),
            "revenue_per_scenario"  : fwd["revenue_per_scenario"],
            "energy_path"           : fwd["energy_path"],
            "e_in"                  : fwd["e_in"],
            "e_out"                 : fwd["e_out"],
            "time_index"            : time_index,
        }
