"""Figures for docs/results.md, drawn from results/<config>/ outputs.

    python -m mltrading.experiments.plots        (needs the `viz` extra: pip install -e .[viz])

Encoding: hue = model (blue ridge, orange HGB; the first two slots of the
validated categorical palette), line style = portfolio type, gray =
buy-and-hold benchmark. Every series is also direct-labeled and listed
in a legend, so identity never depends on color alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mltrading.backtest.engine import MarketData
from mltrading.config import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    load_config,
)
from mltrading.features.build import FEATURE_TABLE_PATH

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
MODEL_COLOR = {"ridge": BLUE, "hgb": ORANGE}
DOCS = PROJECT_ROOT / "docs"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)


def equity_curves(results_dir: Path, out_path: Path, holdout_start: str) -> Path:
    nav = pd.read_parquet(results_dir / "nav.parquet").dropna()
    data = MarketData.from_features(pd.read_parquet(FEATURE_TABLE_PATH))
    bench = (1 + data.returns.mean(axis=1).loc[nav.index[0]:]).cumprod()
    bench = 100 * bench / bench.loc[nav.index[0]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, facecolor=SURFACE)
    for ax, kind, title in zip(axes, ("baseline", "risk_aware"), ("Baseline: long top decile / short bottom decile", "Risk-aware: constrained optimizer"), strict=True):
        _style(ax)
        ax.axvspan(pd.Timestamp(holdout_start), nav.index[-1], color=GRID, alpha=0.55, linewidth=0)
        ax.plot(bench.index, bench, color=MUTED, linewidth=1.4, label="Buy & hold equal-weight (long-only)")
        ax.plot(nav.index, 100 * nav[f"ridge|{kind}"] / nav[f"ridge|{kind}"].iloc[0], color=BLUE, linewidth=2,
                label="Ridge")
        ax.plot(nav.index, 100 * nav[f"hgb|{kind}"] / nav[f"hgb|{kind}"].iloc[0], color=ORANGE, linewidth=2,
                label="HGB")
        ax.axhline(100, color=INK2, linewidth=0.6)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
        ax.text(pd.Timestamp(holdout_start) + pd.Timedelta(days=12), ax.get_ylim()[1] * 0.97, "holdout", fontsize=8.5, color=INK2, va="top")
        end = nav.index[-1]
        for name, series, color in (("Ridge", nav[f"ridge|{kind}"], BLUE), ("HGB", nav[f"hgb|{kind}"], ORANGE), ("Benchmark", bench, MUTED)):
            val = 100 * series.iloc[-1] / series.iloc[0] if name != "Benchmark" else series.iloc[-1]
            ax.annotate(f"{name} {val:.0f}", (end, val), xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=INK2, va="center", annotation_clip=False)
        ax.set_xlim(nav.index[0], end + pd.Timedelta(days=160))
    axes[0].set_ylabel("Value of 100 invested (strategies net of costs)", color=INK2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Equity curves over the walk-forward test period (55-stock universe)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def cost_sensitivity(results_dir: Path, out_path: Path) -> Path:
    c = pd.read_csv(results_dir / "cost_sensitivity.csv")
    c = c[c["period"] == "all"]
    fig, ax = plt.subplots(figsize=(6.6, 4.2), facecolor=SURFACE)
    _style(ax)
    ax.axhline(0, color=INK2, linewidth=0.8)
    xs = sorted(c["cost_multiplier"].unique())
    pos = {m: i for i, m in enumerate(xs)}
    for model in ("ridge", "hgb"):
        for kind, ls, mk in (("baseline", "--", "o"), ("risk_aware", "-", "s")):
            s = c[(c.model == model) & (c.portfolio == kind)].sort_values("cost_multiplier")
            jitter = -0.03 if model == "ridge" else 0.03   # keep near-identical points from hiding each other
            ax.plot([pos[m] + jitter for m in s.cost_multiplier], s["sharpe_net"], color=MODEL_COLOR[model], linestyle=ls, marker=mk,
                    markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, linewidth=2,
                    label=f"{model.upper() if model == 'hgb' else model.capitalize()} · {'baseline' if kind == 'baseline' else 'risk-aware'}")
            ax.annotate(f"{s['sharpe_net'].iloc[-1]:.2f}", (len(xs) - 1 + jitter, s["sharpe_net"].iloc[-1]), xytext=(9 if model == "hgb" else -9, 0), textcoords="offset points",
                        fontsize=8.5, color=INK2, va="center", ha="left" if model == "hgb" else "right")
    ax.set_xticks(range(len(xs)), [f"{m:g}×" for m in xs])
    ax.set_xlabel("Transaction-cost multiplier (1× = 5 bps/side + 50 bps/yr borrow)", color=INK2, fontsize=9)
    ax.set_ylabel("Net Sharpe ratio, full test period", color=INK2, fontsize=9)
    ax.set_title("Costs erase the baseline's edge much faster than the risk-aware book's", loc="left", fontsize=10.5, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower left")
    ax.set_xlim(-0.25, len(xs) - 0.55)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = PROJECT_ROOT / "results" / cfg.name
    print(equity_curves(results, DOCS / "equity_curves.png", cfg.walk_forward.holdout_start))
    print(cost_sensitivity(results, DOCS / "cost_sensitivity.png"))


if __name__ == "__main__":
    main()
