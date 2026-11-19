# -*- coding: utf-8 -*-
"""Final thesis calibration plots from CTMC calibration outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FINAL_ALPHAS = [100.0, 300.0, 1000.0]


def _load_sweep(project_root: Path) -> pd.DataFrame:
    outputs = project_root / "outputs"
    files = [
        outputs / "calibration_results.csv",
        outputs / "calibration_results_extended.csv",
        outputs / "calibration_results_extended2.csv",
    ]
    parts: list[pd.DataFrame] = []
    for p in files:
        if p.is_file():
            df = pd.read_csv(p)
            if "alpha" in df.columns and "P_block" in df.columns:
                parts.append(df)
    if not parts:
        raise FileNotFoundError("No calibration results files found in outputs/.")

    x = pd.concat(parts, ignore_index=True)
    x["alpha"] = pd.to_numeric(x["alpha"], errors="coerce")
    x = x.dropna(subset=["alpha"]).copy()
    x = x.sort_values("alpha").drop_duplicates(subset=["alpha"], keep="last").reset_index(drop=True)
    return x


def _scenario_rows(df: pd.DataFrame) -> pd.DataFrame:
    wanted = pd.DataFrame({"alpha": FINAL_ALPHAS, "scenario_label": ["recommended_low", "recommended_medium", "recommended_high"]})
    m = wanted.merge(df, on="alpha", how="left")
    if m[["P_block", "E[n]", "E[m]", "util_LEO", "util_MEO"]].isna().any().any():
        raise ValueError("Missing required alpha rows (100, 300, 1000) in combined calibration table.")
    return m


def _style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 130,
            "savefig.dpi": 320,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def _mark_final(ax: plt.Axes, df: pd.DataFrame, y_col: str) -> None:
    sub = df[df["alpha"].isin(FINAL_ALPHAS)].copy().sort_values("alpha")
    ax.scatter(sub["alpha"], sub[y_col], color="crimson", s=38, zorder=5, label="final scenarios")
    for _, r in sub.iterrows():
        ax.annotate(
            f"α={int(r['alpha'])}",
            (r["alpha"], r[y_col]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_final_figures(project_root: Path | str) -> list[Path]:
    root = Path(project_root).resolve()
    out_dir = root / "outputs" / "final_figures"
    df = _load_sweep(root)
    scen = _scenario_rows(df)
    _style()
    saved: list[Path] = []

    # 1) Blocking vs alpha
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogx(df["alpha"], df["P_block"], "o-", color="#1f77b4", linewidth=1.8)
    _mark_final(ax, df, "P_block")
    ax.set_xlabel("alpha")
    ax.set_ylabel("P(n=N)")
    ax.set_title("Blocking probability vs alpha")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    saved.append(_save(fig, out_dir / "blocking_vs_alpha_final.png"))

    # 2) E[n] vs alpha
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogx(df["alpha"], df["E[n]"], "o-", color="#2ca02c", linewidth=1.8)
    _mark_final(ax, df, "E[n]")
    ax.set_xlabel("alpha")
    ax.set_ylabel("E[n]")
    ax.set_title("Mean active sessions vs alpha")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    saved.append(_save(fig, out_dir / "En_vs_alpha_final.png"))

    # 3) E[m] vs alpha
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogx(df["alpha"], df["E[m]"], "o-", color="#9467bd", linewidth=1.8)
    _mark_final(ax, df, "E[m]")
    ax.set_xlabel("alpha")
    ax.set_ylabel("E[m]")
    ax.set_title("Mean retrial orbit size vs alpha")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    saved.append(_save(fig, out_dir / "Em_vs_alpha_final.png"))

    # 4) P(m=M) vs alpha
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogx(df["alpha"], df["P(m=M)"], "o-", color="#ff7f0e", linewidth=1.8)
    _mark_final(ax, df, "P(m=M)")
    ax.set_xlabel("alpha")
    ax.set_ylabel("P(m=M)")
    ax.set_title("Orbit overflow probability vs alpha")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    saved.append(_save(fig, out_dir / "PmM_vs_alpha_final.png"))

    # 5) util_LEO and util_MEO vs alpha
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.semilogx(df["alpha"], df["util_LEO"], "o-", color="#1f77b4", linewidth=1.8, label="util_LEO")
    ax.semilogx(df["alpha"], df["util_MEO"], "s-", color="#d62728", linewidth=1.8, label="util_MEO")
    sub = df[df["alpha"].isin(FINAL_ALPHAS)].copy()
    ax.scatter(sub["alpha"], sub["util_LEO"], color="crimson", s=34, zorder=5)
    ax.scatter(sub["alpha"], sub["util_MEO"], color="crimson", s=34, zorder=5)
    for _, r in sub.iterrows():
        ax.annotate(f"α={int(r['alpha'])}", (r["alpha"], r["util_LEO"]), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("alpha")
    ax.set_ylabel("Utilization")
    ax.set_title("LEO/MEO utilization vs alpha")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    saved.append(_save(fig, out_dir / "utilization_vs_alpha_final.png"))

    # 6) trade-off plot: x=P(n=N), y=util_LEO
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.scatter(df["P_block"], df["util_LEO"], c=df["alpha"], cmap="viridis", s=36, alpha=0.85)
    for _, r in scen.iterrows():
        ax.scatter(r["P_block"], r["util_LEO"], color="crimson", s=56, zorder=6)
        ax.annotate(
            r["scenario_label"].replace("recommended_", ""),
            (r["P_block"], r["util_LEO"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=10,
            weight="bold",
        )
    ax.set_xlabel("P(n=N)")
    ax.set_ylabel("util_LEO")
    ax.set_title("Trade-off: blocking vs LEO utilization")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    saved.append(_save(fig, out_dir / "tradeoff_final.png"))

    # 7) Scenario comparison bar chart
    metrics = ["P_block", "E[n]", "E[m]", "util_LEO", "util_MEO"]
    display = {
        "P_block": "P(n=N)",
        "E[n]": "E[n]",
        "E[m]": "E[m]",
        "util_LEO": "util_LEO",
        "util_MEO": "util_MEO",
    }
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4))
    axes = axes.ravel()
    xlab = ["low", "medium", "high"]
    for i, m in enumerate(metrics):
        ax = axes[i]
        vals = scen[m].to_numpy()
        ax.bar(xlab, vals, color=["#6baed6", "#31a354", "#fd8d3c"])
        ax.set_title(display[m])
        if m.startswith("util"):
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)
    axes[-1].axis("off")
    fig.suptitle("Scenario comparison: recommended_low / medium / high", y=1.02, fontsize=12)
    saved.append(_save(fig, out_dir / "scenario_comparison_final.png"))

    return saved
