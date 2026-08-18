"""Thesis-Abbildungen aus Trajektorien.

Alle Funktionen nehmen eine oder mehrere `TrajectoryState` und geben eine
Matplotlib-Figure zurueck. Es wird nichts erfunden: liegt keine Kristallpose
vor, bleiben die fehlerbezogenen Kurven leer statt geraten zu werden.

Die wissenschaftlich wichtigste ist `plot_transport_asymmetry` (Plot F): sie
traegt Translations- und Rotationsfehler gegen dieselbe Zeitachse auf und
testet damit direkt die Hypothese

    "SigmaFlow transportiert die Fragmentpositionen erfolgreich, korrigiert
     die Orientierung aber zu wenig."

Wenn die Hypothese stimmt, faellt die Translationskurve deutlich und die
Rotationskurve bleibt flach nahe der Haar-Referenz.
"""

from __future__ import annotations

import numpy as np

from .trajectory import TrajectoryState
from .writers import compute_metrics

HAAR_MEDIAN_DEG = 132.3   # Median-Rotationswinkel unter Haar, empirisch
HAAR_MEAN_DEG = 126.5     # (pi^2/2 + 2)/pi, analytisch


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _ligand_series(traj: TrajectoryState):
    _, lig, _ = compute_metrics(traj)
    t = np.array([r["time"] for r in lig])

    def col(name):
        v = [r[name] for r in lig]
        return None if any(x == "" for x in v) else np.array(v, dtype=float)

    return t, lig, col


def plot_rotation_error(trajs: dict[str, TrajectoryState], path=None):
    """Plot A -- Median-Fragment-Rotationsfehler gegen Integrationsschritt."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, tr in trajs.items():
        t, _, col = _ligand_series(tr)
        y = col("median_fragment_rotation_error_deg")
        if y is not None:
            ax.plot(t, y, marker="o", ms=3, label=label)
    ax.axhline(HAAR_MEDIAN_DEG, ls="--", c="grey",
               label=f"Haar-Median {HAAR_MEDIAN_DEG}$^\\circ$")
    ax.set_xlabel("Integrationszeit $t$")
    ax.set_ylabel(r"Median-Rotationsfehler [$^\circ$]")
    ax.set_ylim(0, 180)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


def plot_translation_error(trajs: dict[str, TrajectoryState], path=None):
    """Plot B -- Schwerpunktfehler gegen Integrationsschritt."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, tr in trajs.items():
        t, _, col = _ligand_series(tr)
        y = col("centroid_error")
        if y is not None:
            ax.plot(t, y, marker="o", ms=3, label=label)
    ax.set_xlabel("Integrationszeit $t$")
    ax.set_ylabel(r"Schwerpunktfehler [$\AA$]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


def plot_rmsd(trajs: dict[str, TrajectoryState], path=None):
    """Plot C -- Ligand-RMSD gegen Integrationsschritt."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, tr in trajs.items():
        t, _, col = _ligand_series(tr)
        y = col("rmsd_to_crystal")
        if y is not None:
            ax.plot(t, y, marker="o", ms=3, label=label)
    ax.axhline(2.0, ls=":", c="k", lw=1, label=r"2 $\AA$")
    ax.set_xlabel("Integrationszeit $t$")
    ax.set_ylabel(r"RMSD zur Kristallpose [$\AA$]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


def _per_fragment(traj: TrajectoryState, key: str):
    frag, _, _ = compute_metrics(traj)
    F = traj.n_fragments
    T = traj.n_steps
    M = np.zeros((T, F))
    for r in frag:
        M[r["step"], r["fragment"]] = float(r[key])
    return np.array(sorted({r["time"] for r in frag})), M


def plot_cumulative_rotation(traj: TrajectoryState, path=None):
    """Plot D -- kumulierte Drehung je Fragment."""
    plt = _mpl()
    t, M = _per_fragment(traj, "cumulative_rotation_deg")
    fig, ax = plt.subplots(figsize=(6, 4))
    for f in range(M.shape[1]):
        ax.plot(t, M[:, f], lw=1, alpha=0.8, label=f"Frag {f}")
    ax.set_xlabel("Integrationszeit $t$")
    ax.set_ylabel(r"kumulierte Drehung [$^\circ$]")
    if M.shape[1] <= 12:
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


def plot_cumulative_translation(traj: TrajectoryState, path=None):
    """Plot E -- kumulierter Weg je Fragment."""
    plt = _mpl()
    t, M = _per_fragment(traj, "cumulative_translation")
    fig, ax = plt.subplots(figsize=(6, 4))
    for f in range(M.shape[1]):
        ax.plot(t, M[:, f], lw=1, alpha=0.8, label=f"Frag {f}")
    ax.set_xlabel("Integrationszeit $t$")
    ax.set_ylabel(r"kumulierter Weg [$\AA$]")
    if M.shape[1] <= 12:
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


def plot_transport_asymmetry(traj: TrajectoryState, path=None, label=""):
    """Plot F -- die eigentliche Kernabbildung.

    Zwei y-Achsen auf einer Zeitachse: Translationsfehler links,
    Rotationsfehler rechts, mit der Haar-Referenz als Grundlinie. Damit ist
    auf einen Blick zu sehen, ob der Transport nur einen der beiden Kanaele
    bedient.
    """
    plt = _mpl()
    t, _, col = _ligand_series(traj)
    tr = col("centroid_error")
    ro = col("median_fragment_rotation_error_deg")
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    if tr is not None:
        ax1.plot(t, tr, "o-", ms=3, c="tab:blue", label="Schwerpunktfehler")
    ax1.set_xlabel("Integrationszeit $t$")
    ax1.set_ylabel(r"Schwerpunktfehler [$\AA$]", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    if ro is not None:
        ax2.plot(t, ro, "s-", ms=3, c="tab:red", label="Rotationsfehler")
    ax2.axhline(HAAR_MEDIAN_DEG, ls="--", c="grey", lw=1)
    # rechtsbuendig am Rand, damit die Beschriftung nie auf der Kurve liegt
    ax2.text(t[-1], HAAR_MEDIAN_DEG + 4, f"Haar {HAAR_MEDIAN_DEG}$^\\circ$",
             color="grey", fontsize=7, ha="right", va="bottom")
    ax2.set_ylabel(r"Median-Rotationsfehler [$^\circ$]", color="tab:red")
    ax2.set_ylim(0, 180)
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title(label or traj.meta.get("complex_id", ""), fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig
