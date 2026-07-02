"""Diagnostic plots (requires matplotlib)."""
from __future__ import annotations

import numpy as np

__all__ = ["reliability_diagram"]


def reliability_diagram(y_true, proba, class_index: int, n_bins: int = 10, ax=None):
    """Reliability diagram for one class: mean predicted probability vs
    empirical frequency over equal-frequency bins. Returns the axis."""
    import matplotlib.pyplot as plt

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)[:, class_index]
    hit = (y == class_index).astype(float)
    order = np.argsort(p)
    p, hit = p[order], hit[order]
    xs, ys = [], []
    for b in np.array_split(np.arange(len(p)), n_bins):
        if len(b):
            xs.append(p[b].mean())
            ys.append(hit[b].mean())
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    ax.plot(xs, ys, "o-", label=f"class {class_index}")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("empirical frequency")
    ax.legend()
    return ax
