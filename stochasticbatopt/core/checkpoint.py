from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from stochasticbatopt.core.params import StorageParams


class CheckpointMixin:
    """
    Mixin providing pickle-based per-day checkpointing for backward induction.

    Attributes expected on the host class
    --------------------------------------
    params   : StorageParams
    n_levels : int   (SoC grid size)
    n_basis  : int
    run_id   : str
    cache_dir: Path
    """

    def _make_run_key(self, scenarios: np.ndarray) -> str:
        p  = self.params
        md = hashlib.md5(scenarios.tobytes()).hexdigest()[:12]
        return (
            f"{self.run_id}"
            f"_lv{self.n_levels}"
            f"_bs{self.n_basis}"
            f"_dp{p.discharge_power}"
            f"_cp{p.charge_power}"
            f"_ce{p.charge_efficiency:.2f}"
            f"_{md}"
        )

    def _ckpt_path(self, day: int) -> Path:
        return self.cache_dir / f"{self._run_key}_d{day:04d}.pkl"

    def _save(self, day: int, state: dict) -> None:
        with open(self._ckpt_path(day), "wb") as fh:
            pickle.dump(state, fh)

    def _load(self, day: int) -> Optional[dict]:
        path = self._ckpt_path(day)
        if path.exists():
            with open(path, "rb") as fh:
                return pickle.load(fh)
        return None

    def _restore_latest(self, n_days: int) -> int:
        """
        Scan from day n_days-1 downward for the latest saved checkpoint.
        Returns the day index to resume from (i.e. one below the restored day),
        or n_days-1 if nothing found (start fresh).
        """
        for d in range(n_days - 1, -1, -1):
            state = self._load(d)
            if state is not None:
                print(f"  [checkpoint] Restoring day {d} — resuming from day {d - 1}.")
                self.V                = state["V"]
                self._reg_weights     = state["reg_weights"]
                self.fwd_e_in         = state["fwd_e_in"]
                self.fwd_e_out        = state["fwd_e_out"]
                self.fwd_level_end    = state["fwd_level_end"]
                return d - 1

        print("  [checkpoint] No checkpoints found — starting from scratch.")
        return n_days - 1

    def _pack_state(self) -> dict:
        return {
            "V"            : self.V,
            "reg_weights"  : self._reg_weights,
            "fwd_e_in"     : self.fwd_e_in,
            "fwd_e_out"    : self.fwd_e_out,
            "fwd_level_end": self.fwd_level_end,
        }
