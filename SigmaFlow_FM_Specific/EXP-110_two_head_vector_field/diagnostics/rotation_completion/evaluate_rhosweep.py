"""Auswertung des rho-Sweeps: hilft eine nicht-lineare Verteilung der
ODE-Schritte gegen den Kopf-Schwanz-Flip?

HYPOTHESE (aus der Kosinus-ueber-t-Diagnostik, Job 8561684)
Das Modell weiss bei kleinem t nicht, wohin es drehen muss (Kosinus 0.025 bei
t<0.1, 47.7% negativ) und wird erst ab t~0.5 zuverlaessig (0.73 bei t~0.75).
Das lineare Zeitgitter legt aber 10 von 25 Schritten in den blinden Bereich.
Ein Gitter, das die Schritte nach hinten verschiebt, sollte den
Orientierungsfehler senken.

Schritte mit t<0.4 bei N=25:
    linear     10/25   (Ausgangszustand)
    edm rho=1  10/25   (KONTROLLE - muss linear entsprechen)
    edm rho=0.5  4/25
    edm rho=0.333 2/25

ERFOLGSKRITERIUM (vorab): Orientierungsfehler-Median sinkt UND der Anteil
unter 20 Grad steigt ueber die 3.8% des Ausgangszustands, OHNE dass der
Schwerpunktabstand steigt (dort ist SigmaFlow mit 1.16 A besser als SigmaDock).
FALSIFIKATION: keine Bewegung ueber die Seed-Streuung hinaus.

Usage (auf ARC, aus d_frame_fix/):
    python diagnostics/rotation_completion/evaluate_rhosweep.py
    python diagnostics/rotation_completion/evaluate_rhosweep.py <sweep_dir> <data_dir>
"""

import glob
import os
import sys

import numpy as np
from rdkit import Chem

SWEEP = sys.argv[1] if len(sys.argv) > 1 else "sampling_output_pb_rhosweep"
DATA = sys.argv[2] if len(sys.argv) > 2 else "/data/stat-cadd/shug8458/data"
ARMS = ["linear", "edm_rho1", "edm_rho05", "edm_rho033"]
STEPS_BELOW_04 = {"linear": 10, "edm_rho1": 10, "edm_rho05": 4, "edm_rho033": 2}
SEEDS = [0, 1, 2]


def first_mol(path):
    for m in Chem.SDMolSupplier(path, removeHs=False, sanitize=False):
        if m is not None:
            return m
    return None


def find_truth(cid):
    """Die kanonische Ligandendatei - dieselbe, die der Sampler gelesen hat."""
    for pat in (
        f"{DATA}/posebusters_paper/posebusters_benchmark_set/{cid}/{cid}_ligand.sdf",
        f"{DATA}/posebusters_paper/posebusters_benchmark_set/{cid}/{cid}_ligands.sdf",
        f"{DATA}/posebusters/{cid}/{cid}_ligand.sdf",
        f"true_ligands/{cid}_ligands.sdf",
    ):
        if os.path.exists(pat):
            return pat
    return None


def kabsch(P, Q):
    cP, cQ = P.mean(0), Q.mean(0)
    Pc, Qc = P - cP, Q - cQ
    U, _, Vt = np.linalg.svd(Qc.T @ Pc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    ali = float(np.sqrt(((Pc - Qc @ R.T) ** 2).sum(1).mean()))
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return ali, ang, float(np.linalg.norm(cP - cQ))


def pred(arm, seed, cid):
    h = glob.glob(f"{SWEEP}/{arm}/results/posebusters/*/seed_{seed}/{cid}__*.sdf")
    if not h:
        return None
    m = first_mol(h[0])
    return None if m is None else m.GetConformer().GetPositions()


cids = sorted({os.path.basename(p).split("__")[0]
               for p in glob.glob(f"{SWEEP}/linear/results/posebusters/*/seed_*/*.sdf")})
print(f"Komplexe: {len(cids)}   Sweep: {SWEEP}\n")

truth = {}
missing = 0
for c in cids:
    t = find_truth(c)
    if t is None:
        missing += 1
        continue
    m = first_mol(t)
    if m is not None:
        truth[c] = m.GetConformer().GetPositions()
if missing:
    print(f"[WARN] {missing} Referenzliganden nicht gefunden - DATA={DATA} pruefen\n")

# [arm][seed] -> arrays
data = {}
for arm in ARMS:
    for s in SEEDS:
        rows = []
        for c in cids:
            P = pred(arm, s, c)
            if P is None or c not in truth or P.shape != truth[c].shape:
                continue
            Q = truth[c]
            raw = float(np.sqrt(((P - Q) ** 2).sum(1).mean()))
            ali, ang, cen = kabsch(P, Q)
            rows.append((c, raw, ali, ang, cen))
        if rows:
            data[(arm, s)] = rows

have = sorted({s for (_, s) in data})
print(f"Vorhandene Seeds: {have}" + ("  [unvollstaendig - Job laeuft noch]"
                                     if len(have) < len(SEEDS) else ""))
print()
print("=" * 104)
print("ERGEBNIS JE ARM   (pro Seed berechnet, dann Mittel +- SD ueber die vorhandenen Seeds)")
print("=" * 104)
print(f"{'Arm':<12}{'t<0.4':>7}{'Orient.Median':>16}{'<20 Grad':>12}{'>160 Grad':>12}"
      f"{'RMSD Median':>14}{'<=2A':>9}{'Schwerpunkt':>14}")
print("-" * 104)

agg = {}
for arm in ARMS:
    per = {k: [] for k in ("ang_med", "ang_lt20", "ang_gt160", "raw_med", "raw_lt2", "cen_med")}
    for s in have:
        r = data.get((arm, s))
        if not r:
            continue
        a = np.array([[x[1], x[2], x[3], x[4]] for x in r])
        per["ang_med"].append(np.median(a[:, 2]))
        per["ang_lt20"].append(100 * (a[:, 2] < 20).mean())
        per["ang_gt160"].append(100 * (a[:, 2] > 160).mean())
        per["raw_med"].append(np.median(a[:, 0]))
        per["raw_lt2"].append(100 * (a[:, 0] <= 2).mean())
        per["cen_med"].append(np.median(a[:, 3]))
    if not per["ang_med"]:
        continue
    agg[arm] = {k: np.array(v) for k, v in per.items()}
    f = agg[arm]

    def pm(k, d=1):
        v = f[k]
        return f"{v.mean():.{d}f}" + (f" +-{v.std(ddof=1):.{d}f}" if len(v) > 1 else "")

    print(f"{arm:<12}{STEPS_BELOW_04[arm]:>4}/25{pm('ang_med'):>16}{pm('ang_lt20'):>12}"
          f"{pm('ang_gt160'):>12}{pm('raw_med',2):>14}{pm('raw_lt2'):>9}{pm('cen_med',2):>14}")

print()
print("Zufallsbaseline Orientierung: Median 132.3 Grad")
print("Ausgangszustand (10-Seed-Lauf, linear): Orient.Median 145.9, <20 Grad 3.8%,")
print("  >160 Grad 36.5%, RMSD-Median 4.95, <=2A 4.4%, Schwerpunkt 1.16 A")

# ---- gepaart gegen linear ------------------------------------------------
print()
print("=" * 104)
print("GEPAART GEGEN 'linear'  (pro Komplex ueber die Seeds gemittelt; negativ = BESSER als linear)")
print("=" * 104)
print(f"{'Arm':<12}{'d Orient.':>14}{'95%-CI':>24}{'d RMSD':>12}{'95%-CI':>22}{'d Schwerp.':>13}")
print("-" * 104)
rng = np.random.default_rng(0)


def mean_over_seeds(arm, col):
    out = {}
    for c in cids:
        vals = [dict((x[0], x) for x in data[(arm, s)]).get(c) for s in have if (arm, s) in data]
        vals = [v[col] for v in vals if v is not None]
        if vals:
            out[c] = float(np.mean(vals))
    return out


base = {col: mean_over_seeds("linear", col) for col in (1, 3, 4)}
for arm in ARMS:
    if arm == "linear" or arm not in agg:
        continue
    line = f"{arm:<12}"
    for col, width in ((3, 14), (1, 12), (4, 13)):
        cur = mean_over_seeds(arm, col)
        k = sorted(set(cur) & set(base[col]))
        d = np.array([cur[c] - base[col][c] for c in k])
        if d.size == 0:
            continue
        b = rng.choice(d, size=(10000, d.size), replace=True).mean(axis=1)
        lo, hi = np.percentile(b, [2.5, 97.5])
        fmt = ".1f" if col == 3 else ".2f"
        line += f"{d.mean():>{width}{fmt}}"
        if col != 4:
            line += f"   [{lo:{fmt}}, {hi:{fmt}}]".rjust(22 if col == 3 else 20)
    print(line)

print()
print("=" * 104)
print("URTEIL")
print("=" * 104)
if "edm_rho1" in agg and "linear" in agg:
    dd = abs(agg["edm_rho1"]["ang_med"].mean() - agg["linear"]["ang_med"].mean())
    print(f"  KONTROLLE edm_rho1 vs linear: Differenz im Orientierungsmedian {dd:.2f} Grad")
    print("    (muss ~0 sein - beide erzeugen dasselbe Zeitgitter)"
          + ("  OK" if dd < 2 else "  <-- VERDAECHTIG, Auswertung pruefen"))
best = None
if "linear" in agg:
    for arm in ("edm_rho05", "edm_rho033"):
        if arm not in agg:
            continue
        d_ang = agg[arm]["ang_med"].mean() - agg["linear"]["ang_med"].mean()
        d_lt20 = agg[arm]["ang_lt20"].mean() - agg["linear"]["ang_lt20"].mean()
        d_cen = agg[arm]["cen_med"].mean() - agg["linear"]["cen_med"].mean()
        ok = (d_ang < -5) and (d_lt20 > 1) and (d_cen < 0.1)
        print(f"  {arm:<12} Orient. {d_ang:+.1f} Grad, <20 Grad {d_lt20:+.1f} Pp, "
              f"Schwerpunkt {d_cen:+.2f} A  -> {'ERFOLG' if ok else 'kein Effekt'}")
        if ok and (best is None or d_ang < best[1]):
            best = (arm, d_ang)
print()
if best:
    print(f"  => Das Zeitgitter hilft. Bester Arm: {best[0]}. Naechster Schritt: als neuen")
    print("     Sampling-Default uebernehmen und den 10-Seed-Vergleich damit wiederholen.")
else:
    print("  => Das Zeitgitter allein kompensiert die fruehe Rotationsblindheit NICHT.")
    print("     Damit ist der dedizierte Rotationskopf die erste Wahl (Vorschlag B1).")
if len(have) < len(SEEDS):
    print(f"\n  ACHTUNG: nur Seeds {have} ausgewertet. Bei einer Seed-Streuung von")
    print("  SD 1.55 A pro Komplex ist das vorlaeufig. Vor dem Urteil alle 3 Seeds abwarten.")
