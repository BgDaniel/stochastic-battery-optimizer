from __future__ import annotations
from dataclasses import dataclass


@dataclass
class StorageParams:
    """
    Physical parameters of a grid-scale electrical storage asset.

    Attributes
    ----------
    energy_capacity      : Maximum energy content  [MWh]  (= max SoC)
    min_energy           : Minimum energy content  [MWh]  (= min SoC)
    charge_power         : Maximum charging power   [MW]
    discharge_power      : Maximum discharging power [MW]
    charge_efficiency    : Round-trip charge efficiency   ∈ (0, 1]
    discharge_efficiency : Round-trip discharge efficiency ∈ (0, 1]
    time_step            : Dispatch interval [h]  (default 1.0)
    """

    energy_capacity      : float
    min_energy           : float
    charge_power         : float
    discharge_power      : float
    charge_efficiency    : float
    discharge_efficiency : float
    time_step            : float = 1.0

    # ── derived convenience properties ───────────────────────────────────

    @property
    def max_soc(self) -> float:
        return self.energy_capacity

    @property
    def min_soc(self) -> float:
        return self.min_energy
