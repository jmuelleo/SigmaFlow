"""Paper-Setup (graph.sample_conformer=true), 80 Seeds: Validitaet und Auswahl.

EINGABE (in diesem Verzeichnis)
    rd_<arm>cs2_seed0..79.csv           PoseBusters "redock" je Seed
    gncs/GNINA-SCORE-<modell>_<jobid>/  Vinardo-Affinitaet je Pose

AUSGABE (Thesis Visualisierungen/data/)
    validity_papersetup80.csv           Anteil je Ziehung, Wilson-Intervall
    selection_validity_papersetup80.csv Random / Top-1 / Oracle je k und Ziel

WARUM DIESE RECHNUNG SCHON OHNE RMSD LAEUFT
    Die SDF-Dateien der Seeds 40-79 liegen noch auf ARC; der RMSD entsteht erst
    lokal. Validitaet und affinitaetsbasierte Auswahl brauchen ihn nicht. Die
    gemeinsame Kenngroesse "< 2 A UND valide" folgt in build_thesis_datasets.py,
    sobald die Posen da sind.

ZWEI METHODISCHE PUNKTE
    Gepaart wird ueber die 209 Komplexe, nicht ueber die 16720 Posen. Zwei Seeds
    desselben Komplexes sind nicht unabhaengig -- ein Bootstrap ueber Posen
    macht die Intervalle systematisch zu eng.

    Bei Vinardo ist NEGATIV gut. top1() verlangt die Richtung deshalb ohne
    Vorgabewert: mit Vorgabewert waere der Fehler stumm und saehe nur nach
    einem schlechten Scorer aus.

AUFRUF   python papersetup80.py
"""
from __future__ import annotations

import csv
import glob
import os
import re

import numpy as np

# (Anzeigename, Praefix der redock-Tabellen, gnina-Modellname, gnina-Job-ID)
ARMS = [("Minimal", "minimalcs2", "sigmaflow_minimal", "8636670"),
        ("Separate", "exp110cs2", "exp110", "8636671"),
        ("SigmaDock", "sigmadockcs2", "sigmadock", "8636672")]
KS = (1, 2, 3, 5, 10, 20, 40, 80)
OUT = os.path.join("..", "..", "Thesis Visualisierungen", "data")
REP = 400
BOOT = 4000
RNG = np.random.default_rng(0)


def tf(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def spalten(rows):
    """-> (alle Checks, ligandenintrinsische, mit Protein)."""
    cols = list(rows[0])
    rmsd_col = next((c for c in cols if c.startswith("rmsd")), "__none__")
    geladen = {"mol_pred_loaded", "mol_true_loaded", "mol_cond_loaded"}
    checks = [c for c in cols
              if c not in ({"file", "molecule", "position", rmsd_col, ""} | geladen)]
    prot = [c for c in checks
            if any(t in c for t in ("protein", "cofactor", "water", "volume", "distance"))]
    return checks, [c for c in checks if c not in prot], prot


def komplex_id(pfad: str) -> str:
    return pfad.replace(chr(92), "/").split("/")[-1].split("__")[0]


def einlesen():
    """-> V (Validitaet je Pose), SC (Affinitaet je Pose), meta (Checkzahlen)."""
    V, SC, meta = {}, {}, {}
    for arm, key, gkey, jid in ARMS:
        V[arm] = {}
        for f in glob.glob(f"rd_{key}_seed*.csv"):
            s = int(re.search(r"seed(\d+)\.csv$", f).group(1))
            rows = list(csv.DictReader(open(f, encoding="utf-8", errors="replace")))
            checks, intra, prot = spalten(rows)
            meta = {"alle": len(checks), "intra": len(intra), "prot": len(prot)}
            for r in rows:
                V[arm][(komplex_id(r["file"]), s)] = (
                    all(tf(r[c]) for c in intra), all(tf(r[c]) for c in checks))
        # Die sampled-Scores liegen unter gncs/, die bound-Scores im
        # Arbeitsverzeichnis. Beide haben 16720 Zeilen -- ohne die Job-ID im
        # Pfad traefe der Glob das falsche Verzeichnis, und zwar lautlos.
        g = glob.glob(f"gncs/GNINA-SCORE-{gkey}_{jid}/gnina_scores_{gkey}.csv")
        if len(g) != 1:
            raise SystemExit(f"ABBRUCH: gnina-Datei nicht eindeutig fuer {gkey}: {g}")
        SC[arm] = {(r["complex"], int(r["seed"])): float(r["affinity"])
                   for r in csv.DictReader(open(g[0], encoding="utf-8"))}
    return V, SC, meta


def wilson(k: int, n: int):
    """95%-Intervall fuer einen Anteil. Wald bricht bei kleinen Anteilen zusammen."""
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def paar(A: np.ndarray, B: np.ndarray):
    """Bootstrap ueber Komplexe (Zeilen). -> (Differenz in pp, p, Aufloesungsgrenze)."""
    d = 100 * (B.mean() - A.mean())
    nc = A.shape[0]
    boot = np.empty(BOOT)
    for i in range(BOOT):
        idx = RNG.integers(0, nc, nc)
        boot[i] = 100 * (B[idx].mean() - A[idx].mean())
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    grenze = p < 1.0 / BOOT
    return d, max(p, 1.0 / BOOT), grenze


def top1(score: np.ndarray, hit: np.ndarray, k: int, hoeher_besser: bool) -> float:
    """Treffer, wenn aus k zufaelligen Posen die bestbewertete genommen wird.

    hoeher_besser hat bewusst keinen Vorgabewert, siehe Modulkopf.
    Gleichstaende werden zufaellig aufgeloest, sonst entschiede die
    Seed-Reihenfolge und die Kurven werden unmonoton.
    """
    nc, ns = score.shape
    s = score if hoeher_besser else -score
    acc = 0.0
    for _ in range(REP):
        idx = RNG.permuted(np.tile(np.arange(ns), (nc, 1)), axis=1)[:, :k]
        ss = np.take_along_axis(s, idx, axis=1) + RNG.random((nc, k)) * 1e-9
        best = np.take_along_axis(idx, ss.argmax(axis=1, keepdims=True), axis=1)
        acc += np.take_along_axis(hit, best, axis=1).mean()
    return 100 * acc / REP


def oracle(hit: np.ndarray, k: int) -> float:
    """Mindestens ein Treffer unter k, erwartungstreu ueber zufaellige Teilmengen."""
    nc, ns = hit.shape
    if k >= ns:
        return 100 * hit.any(axis=1).mean()
    acc = 0.0
    for _ in range(REP):
        idx = RNG.permuted(np.tile(np.arange(ns), (nc, 1)), axis=1)[:, :k]
        acc += np.take_along_axis(hit, idx, axis=1).any(axis=1).mean()
    return 100 * acc / REP


def main() -> int:
    V, SC, meta = einlesen()
    keys = sorted(set.intersection(*[set(V[a]) & set(SC[a]) for a, _, _, _ in ARMS]))
    cids = sorted({c for c, _ in keys})
    seeds = sorted({s for _, s in keys})
    if len(seeds) != 80:
        print(f"[WARN] {len(seeds)} Seeds statt 80")
    print(f"PoseBusters redock: {meta['alle']} Checks, davon {meta['intra']} "
          f"ligandenintrinsisch und {meta['prot']} mit Protein")
    print(f"{len(cids)} Komplexe x {len(seeds)} Seeds x {len(ARMS)} Arme "
          f"= {len(ARMS) * len(keys)} Posen, graph.sample_conformer=true\n")

    S = {a: np.array([[SC[a][(c, s)] for s in seeds] for c in cids])
         for a, _, _, _ in ARMS}
    Z = {"valid_ohne_protein":
         {a: np.array([[V[a][(c, s)][0] for s in seeds] for c in cids])
          for a, _, _, _ in ARMS},
         "valid_mit_protein":
         {a: np.array([[V[a][(c, s)][1] for s in seeds] for c in cids])
          for a, _, _, _ in ARMS}}

    # --- 1) Anteil je Ziehung -------------------------------------------------
    print("=== Anteil je Ziehung ===")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "validity_papersetup80.csv"), "w",
              encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "metric", "pct", "ci_lo", "ci_hi", "n_poses",
                    "never_hit_pct"])
        for name, M in Z.items():
            print(f"  {name}")
            for a, _, _, _ in ARMS:
                k, n = int(M[a].sum()), M[a].size
                lo, hi = wilson(k, n)
                nie = 100 * (~M[a].any(axis=1)).mean()
                print(f"    {a:<10} {100*k/n:6.2f} %  [{lo:5.2f}; {hi:5.2f}]"
                      f"   kein Treffer in 80: {nie:5.2f} %")
                w.writerow([a, name, f"{100*k/n:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                            n, f"{nie:.4f}"])

    # --- 2) Gepaarte Vergleiche ----------------------------------------------
    print("\n=== Gepaart, Bootstrap ueber die 209 Komplexe ===")
    for name, M in Z.items():
        print(f"  {name}")
        for x, y in (("Minimal", "Separate"), ("Minimal", "SigmaDock"),
                     ("Separate", "SigmaDock")):
            d, p, gr = paar(M[x], M[y])
            txt = f"< {1.0/BOOT:.5f}" if gr else f"= {p:.4f}"
            print(f"    {y:<10} - {x:<10} {d:+7.2f} pp   p {txt}")

    # --- 3) Auswahlkurven -----------------------------------------------------
    with open(os.path.join(OUT, "selection_validity_papersetup80.csv"), "w",
              encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "metric", "k", "random_pct", "top1_affinity_pct",
                    "oracle_pct"])
        for name, H in Z.items():
            print(f"\n=== Auswahl, Zielgroesse {name} ===")
            print(f"{'k':>4}" + "".join(f"{a:>34}" for a, _, _, _ in ARMS))
            print(f"{'':>4}" + "".join(f"{'Random':>11}{'Top-1':>11}{'Oracle':>12}"
                                       for _ in ARMS))
            for k in KS:
                zeile = f"{k:>4}"
                for a, _, _, _ in ARMS:
                    r, t, o = 100 * H[a].mean(), top1(S[a], H[a], k, False), oracle(H[a], k)
                    zeile += f"{r:>10.2f} {t:>10.2f} {o:>11.2f}"
                    w.writerow([a, name, k, f"{r:.4f}", f"{t:.4f}", f"{o:.4f}"])
                print(zeile)
            print("  Trefferquote (Top1-Random)/(Oracle-Random) bei k = 80")
            for a, _, _, _ in ARMS:
                r = 100 * H[a].mean()
                t, o = top1(S[a], H[a], 80, False), oracle(H[a], 80)
                q = 100 * (t - r) / (o - r) if o > r else float("nan")
                print(f"    {a:<10} {q:6.1f} %")
    print(f"\ngeschrieben nach {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
