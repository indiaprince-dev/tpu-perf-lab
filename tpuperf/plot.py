"""Consistent figure style for measured results.

Figures are written as SVG so they scale cleanly and stay small in git.
Colours are chosen to read on both light and dark documentation themes;
axes and text use a mid grey rather than pure black or white.
"""

from __future__ import annotations

import pathlib
from typing import Iterable, Sequence

INK = "#7b8794"       # axes, ticks, labels — legible on light and dark
GRID = "#7b8794"
ACCENT = "#e08a1e"    # rooflines, ceilings
SERIES = ["#2f9e8f", "#5b7fd4", "#c2569a", "#8a9a3c"]


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 4.4),
            "figure.dpi": 120,
            "savefig.transparent": True,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.frameon": False,
            "legend.labelcolor": INK,
        }
    )
    return plt


def roofline(
    peak_tflops: float,
    bandwidth_gb_s: float,
    points: Sequence[tuple[str, float, float]] = (),
    out: str | pathlib.Path = "results/roofline.svg",
    title: str | None = None,
) -> pathlib.Path:
    """Plot a roofline with optional measured points.

    Args:
        peak_tflops: Chip peak throughput, TFLOP/s.
        bandwidth_gb_s: HBM bandwidth, GB/s.
        points: `(label, arithmetic_intensity, achieved_tflops)` triples.
        out: Output path; parent directories are created.
    """
    import numpy as np

    plt = _pyplot()

    balance = (peak_tflops * 1e12) / (bandwidth_gb_s * 1e9)
    intensity = np.logspace(-1, 4, 400)
    attainable = np.minimum(peak_tflops, bandwidth_gb_s * intensity / 1000.0)

    fig, ax = plt.subplots()
    ax.loglog(intensity, attainable, color=ACCENT, lw=2.4, zorder=3)
    ax.axvline(balance, color=INK, lw=1.1, ls="--", alpha=0.5, zorder=2)
    ax.annotate(
        f"machine balance ≈ {balance:.0f}",
        xy=(balance, peak_tflops / 60),
        xytext=(4, 0),
        textcoords="offset points",
        color=INK,
        fontsize=8.5,
        alpha=0.85,
    )

    for i, (label, ai, achieved) in enumerate(points):
        ax.scatter(
            [ai], [achieved], s=44, color=SERIES[i % len(SERIES)],
            zorder=4, label=label,
        )

    ax.set_xlabel("Arithmetic intensity — FLOP / byte")
    ax.set_ylabel("TFLOP/s")
    if title:
        ax.set_title(title, loc="left", pad=12)
    if points:
        ax.legend(loc="lower right", fontsize=9)

    return _save(fig, out)


def sweep(
    x: Iterable[float],
    series: dict[str, Sequence[float]],
    xlabel: str,
    ylabel: str,
    out: str | pathlib.Path,
    title: str | None = None,
    logx: bool = True,
) -> pathlib.Path:
    """Line plot of one or more series against a shared x axis."""
    plt = _pyplot()
    fig, ax = plt.subplots()

    xs = list(x)
    for i, (label, ys) in enumerate(series.items()):
        colour = SERIES[i % len(SERIES)]
        ax.plot(xs, list(ys), color=colour, lw=2, marker="o", ms=4, label=label)

    if logx:
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", pad=12)
    if len(series) > 1:
        ax.legend(fontsize=9)

    return _save(fig, out)


def _save(fig, out: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.clf()
    return path
