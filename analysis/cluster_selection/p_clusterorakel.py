"""
Cluster-Orakel: einmal den Cluster wechseln duerfen, darin normal ranken.

Vier Stufen, je Zelle und Clusterschwelle S:

  BASIS         hoechster Mixed Score ueber ALLE Posen        (ranker_richtig)
  ORAKEL-MAX    ein Orakel waehlt den guenstigsten Cluster,   any(best_pose_correct)
                darin entscheidet der Mixed Score
  ORAKEL-REAL   unter den Clustern, die eine richtige Pose    realistischere
                enthalten, wird der mit dem hoechsten         Zwischenstufe
                Mixed-Score-Maximum genommen (bzw. der
                groesste), darin der Mixed Score
  VOLL-ORAKEL   irgendeine richtige Pose im Pool              (complex_hat_correct)

Die Luecke ORAKEL-MAX -> VOLL-ORAKEL ist der Teil, den auch ein perfekter
Clusterwaehler NICHT holen kann: dort gewinnt der Mixed Score selbst
innerhalb des richtigen Clusters gegen die richtige Pose.

Unsicherheit: gepaarter Bootstrap ueber Komplexe (dieselben Komplexe in
allen Stufen), 10.000 Ziehungen.
"""
import argparse
import os

import numpy as np
import pandas as pd

HIER = os.path.dirname(os.path.abspath(__file__))
ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]
SPALTEN = ["arm", "nfe", "S", "complex", "cluster", "K",
           "contains_correct", "best_pose_correct", "correct_fraction",
           "selected_by_ranker", "complex_hat_correct", "ranker_richtig",
           "g_gr", "s_h_max"]

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=["pb308", "astex"], default="pb308")
p.add_argument("--B", type=int, default=10000)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

df = pd.read_csv(os.path.join(HIER, f"cluster_{a.satz}.csv"), usecols=SPALTEN)
rng = np.random.default_rng(a.seed)


def ki(x, y, B):
    """Gepaarter Bootstrap ueber Komplexe fuer die Differenz mean(y)-mean(x)."""
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    d = (y[idx].mean(axis=1) - x[idx].mean(axis=1)) * 100
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


for S in sorted(df.S.unique()):
    dS = df[np.isclose(df.S, S)]
    print(f"\n{'='*88}\nSatz {a.satz}, Clusterschwelle {S:.1f} A\n{'='*88}")
    zeilen = []
    for arm, nfe in ZELLEN:
        z = dS[(dS.arm == arm) & (dS.nfe == nfe)]
        if z.empty:
            continue
        g = z.groupby("complex")
        basis = g["ranker_richtig"].first()
        voll = g["complex_hat_correct"].first()
        o_max = g["best_pose_correct"].any()

        # ORAKEL-REAL: unter den Clustern mit einer richtigen Pose den mit
        # dem hoechsten Mixed-Score-Maximum, darin der Mixed Score.
        def real(gr):
            k = gr[gr.contains_correct]
            if k.empty:
                return False
            return bool(k.loc[k.s_h_max.idxmax(), "best_pose_correct"])

        def real_gr(gr):
            k = gr[gr.contains_correct]
            if k.empty:
                return False
            return bool(k.loc[k.g_gr.idxmax(), "best_pose_correct"])

        o_real = z.groupby("complex")[
            ["contains_correct", "best_pose_correct", "s_h_max"]
        ].apply(real)
        o_grs = z.groupby("complex")[
            ["contains_correct", "best_pose_correct", "g_gr"]
        ].apply(real_gr)

        ix = basis.index
        b = basis.loc[ix].to_numpy(bool)
        om = o_max.loc[ix].to_numpy(bool)
        orl = o_real.loc[ix].to_numpy(bool)
        ogr = o_grs.loc[ix].to_numpy(bool)
        vo = voll.loc[ix].to_numpy(bool)
        n_f = int((~b).sum())

        lo, hi = ki(b.astype(float), om.astype(float), a.B)
        zeilen.append({
            "Zelle": f"{arm} {nfe}",
            "Basis %": 100 * b.mean(),
            "Or-real %": 100 * orl.mean(),
            "Or-groesst %": 100 * ogr.mean(),
            "Or-MAX %": 100 * om.mean(),
            "Voll-Or %": 100 * vo.mean(),
            "Gewinn Or-MAX": 100 * (om.mean() - b.mean()),
            "95%-KI": f"[{lo:+.2f},{hi:+.2f}]",
            "n Fehl": n_f,
            "davon gerettet %": 100 * (om & ~b).sum() / n_f if n_f else np.nan,
            "Rest-Luecke": 100 * (vo.mean() - om.mean()),
            "Cluster/Kx": g.size().mean(),
        })
    t = pd.DataFrame(zeilen)
    print(t.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
