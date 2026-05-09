from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stochasticbatopt.core.optimizer import StochasticBatteryOptimizer


def plot_overview(
    model       : "StochasticBatteryOptimizer",
    result      : dict,
    percentiles : tuple[float, ...] = (5, 10),
    n_plot_days : int | None = None,
) -> None:
    """Four-panel overview: price, energy path, dispatch, cumulative revenue."""
    time_index  = result["time_index"]
    T           = len(time_index)
    T_plot      = T if n_plot_days is None else min(n_plot_days * 24, T)
    hours       = np.arange(T_plot)
    percentiles = sorted(percentiles)
    p           = model.params
    ref         = model._reference
    vd          = model._value_decomp

    energy_path = result["energy_path"][:, :T_plot]
    e_in        = result["e_in"][:, :T_plot]
    e_out       = result["e_out"][:, :T_plot]

    cum_rev = np.cumsum(
        model.price_scenarios[:, :T_plot] * (
            result["e_out"][:, :T_plot] * p.discharge_efficiency
            - result["e_in"][:, :T_plot] / p.charge_efficiency
        ),
        axis=1,
    ) * p.time_step

    def _bands(arr):
        m = arr.mean(axis=0)
        b = {q: (np.percentile(arr, q, axis=0),
                 np.percentile(arr, 100 - q, axis=0))
             for q in percentiles}
        return m, b

    def _fill(ax, x, mean, bands, color, label):
        ax.plot(x, mean, color=color, lw=2.0, label=label)
        for i, (q, (lo, hi)) in enumerate(bands.items()):
            ax.fill_between(x, lo, hi, color=color,
                            alpha=0.15 + 0.12 * i,
                            label=f"P{q}–P{100-q}")

    pm, pb  = _bands(model.price_scenarios[:, :T_plot])
    em, eb  = _bands(energy_path)
    im, ib  = _bands(e_in)
    om, ob  = _bands(e_out)
    cm, cb  = _bands(cum_rev)

    tick_h  = np.arange(0, T_plot, 12)
    xlabels = [time_index[h].strftime("%d %b\n%H:%M") for h in tick_h]

    fig, axes = plt.subplots(4, 1, figsize=(14, 22))
    fig.suptitle(
        f"Stochastic Battery Optimizer — first {T_plot // 24} days\n"
        f"Total: {vd['total']:.2f} EUR  |  "
        f"Intrinsic: {vd['intrinsic']:.2f} EUR ({vd['intr_pct']:.1f}%)  |  "
        f"Extrinsic: {vd['extrinsic']:.2f} EUR ({vd['extr_pct']:.1f}%)",
        fontsize=11, fontweight="bold",
    )
    fig.subplots_adjust(top=0.88, hspace=0.80)

    _fill(axes[0], hours, pm, pb, "steelblue", "Mean price")
    axes[0].plot(hours, ref["price_path"][:T_plot], color="navy",
                 lw=1.0, linestyle="-.", label="Reference path")
    for sep in range(24, T_plot, 24):
        axes[0].axvline(sep, color="grey", lw=0.5, ls=":")
    axes[0].axhline(0, color="grey", lw=0.5, ls=":")
    axes[0].set_ylabel("EUR/MWh"); axes[0].set_title("DA Price")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[0].set_xticks(tick_h); axes[0].set_xticklabels(xlabels, fontsize=7)

    _fill(axes[1], hours, em, eb, "darkorange", "Mean energy level")
    axes[1].plot(hours, ref["energy_path"][:T_plot], color="navy",
                 lw=1.0, linestyle="-.", label="Reference energy path")
    axes[1].axhline(p.energy_capacity, color="red",   lw=0.8, ls="--",
                    label=f"Max={p.energy_capacity} MWh")
    axes[1].axhline(p.min_energy,      color="green", lw=0.8, ls="--",
                    label=f"Min={p.min_energy} MWh")
    for sep in range(24, T_plot, 24):
        axes[1].axvline(sep, color="grey", lw=0.5, ls=":")
    axes[1].set_ylabel("MWh"); axes[1].set_title("Energy Level")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    axes[1].set_xticks(tick_h); axes[1].set_xticklabels(xlabels, fontsize=7)

    axes[2].plot(hours,  om, color="seagreen", lw=2.0, label="Discharge mean")
    axes[2].plot(hours, -im, color="crimson",  lw=2.0, label="−Charge mean")
    for i, (q, (lo, hi)) in enumerate(ob.items()):
        axes[2].fill_between(hours, lo, hi, color="seagreen",
                             alpha=0.15 + 0.12 * i, label=f"P{q}–P{100-q}")
    for i, (q, (lo, hi)) in enumerate(ib.items()):
        axes[2].fill_between(hours, -hi, -lo, color="crimson",
                             alpha=0.15 + 0.12 * i)
    axes[2].plot(hours,  ref["e_out"][:T_plot], color="navy",   lw=1.0, ls="-.")
    axes[2].plot(hours, -ref["e_in"][:T_plot],  color="purple", lw=1.0, ls="-.")
    axes[2].axhline(0, color="black", lw=0.7)
    for sep in range(24, T_plot, 24):
        axes[2].axvline(sep, color="grey", lw=0.5, ls=":")
    axes[2].set_ylabel("MWh"); axes[2].set_title("Dispatch")
    axes[2].legend(fontsize=8, ncol=2); axes[2].grid(alpha=0.3)
    axes[2].set_xticks(tick_h); axes[2].set_xticklabels(xlabels, fontsize=7)

    _fill(axes[3], hours, cm, cb, "steelblue", "Mean cumulative revenue")
    axes[3].plot(hours, ref["cum_revenue"][:T_plot], color="navy",
                 lw=1.0, ls="-.", label="Reference cum. revenue")
    for sep in range(24, T_plot, 24):
        axes[3].axvline(sep, color="grey", lw=0.5, ls=":")
    axes[3].axhline(0, color="grey", lw=0.5, ls=":")
    axes[3].set_ylabel("EUR"); axes[3].set_title("Cumulative Revenue")
    axes[3].legend(fontsize=8); axes[3].grid(alpha=0.3)
    axes[3].set_xticks(tick_h); axes[3].set_xticklabels(xlabels, fontsize=7)

    plt.show(block=True)


def plot_value_distribution(
    model : "StochasticBatteryOptimizer",
    day   : int = 0,
) -> None:
    """Mean V(energy_level) with P5-P95 / P10-P90 bands + intrinsic reference."""
    D = model.n_days
    if day >= D:
        raise ValueError(f"day={day} out of range")

    grid  = model.grid
    V_day = model.V[:, :, day]
    ref   = model._reference
    h0    = day * 24
    cum_so_far = float(ref["cum_revenue"][h0 - 1]) if h0 > 0 else 0.0
    ref_vtg    = ref["total"] - cum_so_far

    mean_V = V_day.mean(axis=1)
    p5  = np.percentile(V_day, 5,  axis=1)
    p10 = np.percentile(V_day, 10, axis=1)
    p90 = np.percentile(V_day, 90, axis=1)
    p95 = np.percentile(V_day, 95, axis=1)

    label = (
        model._time_index[h0].strftime("%Y-%m-%d")
        if model._time_index is not None else f"day {day}"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"Value Function V(energy, day) — Day {day}  ({label})\n"
        f"Grid levels={model.n_levels}  |  N={V_day.shape[1]} scenarios",
        fontsize=11, fontweight="bold",
    )
    fig.subplots_adjust(top=0.85)

    ax.fill_between(grid, p5,  p95, color="steelblue", alpha=0.15, label="P5–P95")
    ax.fill_between(grid, p10, p90, color="steelblue", alpha=0.25, label="P10–P90")
    ax.plot(grid, mean_V, color="steelblue", lw=2.0, label="Mean V(energy, day)")
    ax.axhline(ref_vtg, color="tomato", lw=1.5, linestyle="--",
               label=(
                   f"Intrinsic V-to-go = {ref_vtg:.2f} EUR  "
                   f"(value from day {day} onward on mean price path)"
               ))
    ax.set_xlabel("Energy level  [MWh]")
    ax.set_ylabel("V(energy, day)  [EUR]")
    ax.set_title("Mean V(energy) with quantile bands — slope = marginal value of 1 MWh stored")
    ax.set_xticks(grid)
    ax.set_xticklabels([f"{s:.2f}" for s in grid],
                       rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.show(block=True)


def plot_convexity_cross_section(
    model    : "StochasticBatteryOptimizer",
    day      : int   = 0,
    n_points : int   = 50,
    n_std    : float = 3.0,
) -> None:
    """
    Cross-section of F(p) = max_{x in C} <p,x> along v = x*(p_bar)/||x*(p_bar)||.
    Two panels (shared x-axis):
      Row 0 — F curve, tangent, convexity gap (shaded), gap on twin axis,
               vertex switch points as orange dashed vertical lines.
      Row 1 — Histogram of scenario projections t_i = <p_i - p_bar, v>.
    """
    from stochasticbatopt.core.solver import solve_dispatch_lp

    N, T = model.price_scenarios.shape
    D    = model.n_days

    if day >= D:
        raise ValueError(f"day={day} out of range")

    h0         = day * 24
    h1         = h0 + 24
    prices_day = model.price_scenarios[:, h0:h1]
    p_bar      = prices_day.mean(axis=0)
    label      = (model._time_index[h0].strftime("%Y-%m-%d")
                  if model._time_index is not None else f"day {day}")

    ref    = model._reference
    params = model.params
    x_out  = ref["e_out"][h0:h1] * params.discharge_efficiency
    x_in   = ref["e_in"][h0:h1]  / params.charge_efficiency
    x_star = x_out - x_in
    norm   = float(np.linalg.norm(x_star))
    if norm < 1e-10:
        raise ValueError("Reference dispatch is zero — no cross-section defined.")
    v = x_star / norm

    t_scen = (prices_day - p_bar) @ v
    sig    = float(t_scen.std()) or 1.0

    t_grid   = np.linspace(-n_std * sig, n_std * sig, n_points)
    F_vals   = np.zeros(n_points)
    dispatch = np.zeros((n_points, 48))

    for j, t in enumerate(t_grid):
        e_in_j, e_out_j, rev = solve_dispatch_lp(
            p_bar + t * v, params,
            params.min_energy, params.min_energy,
        )
        F_vals[j]      = max(rev, 0.0)
        dispatch[j, :] = np.concatenate([e_in_j, e_out_j])

    idx0      = np.argmin(np.abs(t_grid))
    F0        = F_vals[idx0]
    tangent   = F0 + t_grid * norm
    gap_curve = F_vals - tangent

    tol     = 1e-4
    rounded = np.round(dispatch / tol).astype(np.int64)
    ids, reg, nxt = [], {}, [0]
    for j in range(n_points):
        key = tuple(rounded[j])
        if key not in reg:
            reg[key] = nxt[0]; nxt[0] += 1
        ids.append(reg[key])
    switch_t = [t_grid[j] for j in range(1, n_points) if ids[j] != ids[j - 1]]

    fig = plt.figure(figsize=(11, 7))
    fig.suptitle(
        f"F cross-section along v = x_ref(p̄)/‖x_ref(p̄)‖ — Day {day}  ({label})\n"
        f"σ(t_i) = {sig:.2f} EUR/MWh  |  Grid levels={model.n_levels}  |  "
        f"{nxt[0]} distinct vertices",
        fontsize=10, fontweight="bold",
    )
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 1.5],
                   hspace=0.35, top=0.88, bottom=0.09)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    ax0.plot(t_grid, F_vals,  color="steelblue", lw=2.5, label=r"$F(\bar{p}+tv)$")
    ax0.plot(t_grid, tangent, color="navy",       lw=1.5, linestyle="--",
             label="Tangent at t=0")
    ax0.fill_between(t_grid, tangent, F_vals,
                     color="tomato", alpha=0.25, label="Convexity gap")
    ax0.axvline(0, color="black", lw=0.8, ls=":", label="t=0  (p̄)")
    for i, ts in enumerate(switch_t):
        ax0.axvline(ts, color="darkorange", lw=1.0, ls="--",
                    label="Vertex switch" if i == 0 else None)
    ax0.set_ylabel("F(p)  [EUR]")
    ax0.set_xlabel("t  (EUR/MWh along v)")
    ax0.set_title("F(p̄+tv) — shaded = convexity gap  |  orange dashed = vertex switches")
    ax0.legend(fontsize=8); ax0.grid(alpha=0.3)

    ax0r = ax0.twinx()
    ax0r.plot(t_grid, gap_curve, color="tomato", lw=1.0, linestyle="-.", alpha=0.8,
              label="Gap F−tangent")
    ax0r.set_ylabel("Gap  [EUR]", fontsize=8, color="tomato")
    ax0r.tick_params(axis="y", colors="tomato")

    ax1.hist(t_scen, bins=40, color="steelblue", alpha=0.7, edgecolor="white",
             label="t_i = <p_i − p̄, v>")
    ax1.axvline(0,                    color="navy",  lw=1.5, ls="--", label="t=0")
    ax1.axvline(float(t_scen.mean()), color="tomato", lw=1.5, ls="-.",
                label=f"E[t_i]={t_scen.mean():.3f}")
    for ts in switch_t:
        ax1.axvline(ts, color="darkorange", lw=1.0, ls="--", alpha=0.7)
    ax1.set_xlabel("t  (EUR/MWh along v)")
    ax1.set_ylabel("Count")
    ax1.set_title("Scenario projections onto v — orange = vertex switches")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    plt.show(block=True)
