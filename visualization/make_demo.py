"""Erzeugt ein vollstaendiges Visualisierungspaket aus SYNTHETISCHEN Daten.

Zweck: die PyMOL-Kette laesst sich damit vollstaendig durchspielen, bevor ARC
zurueck ist. Es sind KEINE Modellergebnisse -- die Bewegung ist konstruiert,
nicht generiert. Jede erzeugte Datei traegt das im Kopf, und `complex_id`
beginnt mit `SYNTH_`, damit ein synthetischer Fall nie versehentlich in eine
Abbildung der Thesis geraet.

    python -m visualization.make_demo --out visualisations/SYNTH_DEMO
    pymol visualisations/SYNTH_DEMO/view_trajectory_sigmaflow.pml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .build_case import build
from .trajectory import TrajectoryState


def _rot(axis: np.ndarray, ang: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def synthetic_case(n_frag: int = 8, T: int = 26, seed: int = 0,
                   rotation_fraction: float = 1.0, model: str = "sigmaflow"):
    """Ein Ligand aus `n_frag` starren Fragmenten, der in die Tasche laeuft.

    `rotation_fraction` steuert, WIE VIEL der noetigen Drehung ausgefuehrt
    wird. Mit 1.0 landet das Fragment korrekt, mit 0.15 bleibt es fast in der
    Startorientierung -- das ist das qualitative Bild, das SigmaFlows
    gemessene Rotationsschwaeche erzeugen wuerde, und eignet sich, um die
    Abbildung zu pruefen, bevor echte Daten vorliegen.
    """
    rng = np.random.default_rng(seed)
    elements, frag_ids, local, target_c, target_R = [], [], [], [], []
    palette = ["C", "C", "N", "O", "C", "S"]
    for f in range(n_frag):
        n = int(rng.integers(4, 7))
        X = rng.normal(size=(n, 3)) * np.array([1.5, 0.9, 0.6])
        local.append(X - X.mean(0))
        elements += [palette[i % len(palette)] for i in range(n)]
        frag_ids += [f] * n
        # Zielpose: Fragmente entlang einer gekruemmten Kette in der Tasche
        s = f / max(n_frag - 1, 1)
        target_c.append(np.array([8 * s - 4, 2.2 * np.sin(3 * s), 1.4 * np.cos(3 * s)]))
        target_R.append(_rot(rng.normal(size=3), rng.uniform(0, np.pi)))
    frag_ids = np.array(frag_ids)
    elements = np.array(elements)

    # Die noetige Korrektur wird KONSTRUIERT statt aus zwei Zufallsrotationen
    # zurueckgerechnet: Achse und Winkel sind dann exakt bekannt, und die
    # Interpolation braucht keinen Logarithmus (dessen Zweigwahl bei 180 Grad
    # mehrdeutig ist). start_R = target_R @ Exp(-xi), also ist Exp(xi) genau
    # die fehlende Drehung.
    corr_axis, corr_ang, start_R, start_c = [], [], [], []
    for f in range(n_frag):
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        ang = rng.uniform(0.5 * np.pi, 0.95 * np.pi)   # deutlich, aber nicht am Cut Locus
        corr_axis.append(ax)
        corr_ang.append(ang)
        start_R.append(target_R[f] @ _rot(ax, -ang))
        start_c.append(target_c[f] + rng.normal(size=3) * 4.0)

    T_steps = np.linspace(0.0, 1.0, T)
    positions = np.zeros((T, len(frag_ids), 3))
    crystal = np.zeros((len(frag_ids), 3))
    for f in range(n_frag):
        idx = np.flatnonzero(frag_ids == f)
        crystal[idx] = local[f] @ target_R[f].T + target_c[f]
        for t, tv in enumerate(T_steps):
            # Translation vollstaendig, Rotation nur zum Bruchteil
            R = start_R[f] @ _rot(corr_axis[f], rotation_fraction * tv * corr_ang[f])
            c = (1 - tv) * start_c[f] + tv * target_c[f]
            positions[t, idx] = local[f] @ R.T + c

    return TrajectoryState(
        positions=positions, times=T_steps, elements=elements,
        frag_ids=frag_ids, crystal=crystal,
        meta={"complex_id": "SYNTH_DEMO", "model": model, "time_kind": "ode_t",
              "checkpoint": "SYNTHETIC -- keine Modellergebnisse",
              "synthetic": True, "rotation_fraction": rotation_fraction})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("visualisations/SYNTH_DEMO"))
    ap.add_argument("--fragments", type=int, default=8)
    ap.add_argument("--steps", type=int, default=26)
    a = ap.parse_args()

    trajs = {
        # "gut": fuehrt die Drehung aus; "schwach": fast keine Drehung.
        "sigmaflow": synthetic_case(a.fragments, a.steps, seed=1,
                                    rotation_fraction=0.15, model="sigmaflow"),
        "sigmadock": synthetic_case(a.fragments, a.steps, seed=1,
                                    rotation_fraction=0.85, model="sigmadock"),
    }
    trajs["sigmadock"].meta["time_kind"] = "diffusion_t"

    summary = build("SYNTH_DEMO", trajs, a.out, receptor=None)
    print("\nSYNTHETISCH -- keine Modellergebnisse.")
    for lab, d in summary.items():
        print(f"  {lab}: {d['n_steps']} Schritte, {d['n_fragments']} Fragmente, "
              f"{d['n_atoms']} Atome")
    print(f"\n  pymol {a.out / 'view_trajectory_sigmaflow.pml'}")
    print(f"  pymol {a.out / 'view_static.pml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
