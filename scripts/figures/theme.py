"""Minimal vector primitives for a two-column paper architecture figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.path import Path

INK = "#23262B"
MUTED = "#62676F"
BLUE = "#416B94"
PALE_BLUE = "#F2F6FA"

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 11,
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "cashflow-credit-risk-architecture",
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def canvas():
    fig, ax = plt.subplots(figsize=(13.8, 4.25))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 145), ylim=(44.5, 0), aspect="equal")
    ax.axis("off")
    return fig, ax


def text(ax, x, y, label, size=11, color=INK, ha="center", va="center"):
    return ax.text(
        x, y, label, fontsize=size, color=color, ha=ha, va=va, linespacing=1.35, zorder=5
    )


def box(ax, x, y, width, height, label, color=INK, fill="white", size=11):
    patch = Rectangle(
        (x, y), width, height, facecolor=fill, edgecolor=color, linewidth=0.85, zorder=2
    )
    ax.add_patch(patch)
    text(ax, x + width / 2, y + height / 2, label, size=size, color=color)
    return patch


def line(ax, points, color=INK, width=0.9):
    return ax.plot(*zip(*points), color=color, lw=width, solid_capstyle="butt", zorder=1)[0]


def arrow(ax, points, color=INK, width=1.05):
    path = Path(points, [Path.MOVETO] + [Path.LINETO] * (len(points) - 1))
    patch = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=width,
        color=color,
        zorder=3,
        joinstyle="miter",
        capstyle="butt",
    )
    ax.add_patch(patch)
    return patch


def junction(ax, x, y, radius=0.24):
    ax.add_patch(Circle((x, y), radius, color=INK, linewidth=0, zorder=4))


def summation(ax, x, y, radius=3):
    ax.add_patch(Circle((x, y), radius, facecolor="white", edgecolor=INK, lw=0.95, zorder=2))
    text(ax, x, y - 0.15, r"$\Sigma$", size=21)
