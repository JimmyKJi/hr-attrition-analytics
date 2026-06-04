"""Shared figure helpers. Forces a non-interactive backend so the pipeline
runs headless (CI, `make all`)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import FIGURES  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})


def save(fig, name: str, dpi: int = 150):
    """Save a figure into figures/ and close it. Returns the path."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
