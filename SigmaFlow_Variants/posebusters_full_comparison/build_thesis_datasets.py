"""Erzeugt die Datensaetze fuer die Thesis-Abbildungen aus den Rohtabellen.

EINGABE (alle in diesem Verzeichnis)
    pose_<arm>.csv                    complex,seed,rmsd   aus evaluate_run.py
    rd_<arm>_seed*.csv                PoseBusters "redock" je Seed
    GNINA-SCORE-*/gnina_scores_*.csv  Vinardo-Affinitaet je Pose

AUSGABE (Thesis Visualisierungen/data/)
    per_draw_80seeds.csv          Anteil je Ziehung, mit Wilson-Intervall
    selection_curves_80seeds.csv  Zufall / Top-1 nach Affinitaet / Oracle je k
    ranker_comparison_80seeds.csv alle fuenf Ranker, Zielgroesse RMSD < 2 A
    heuristic_grid_80seeds.csv    SigmaDocks Heuristik ueber ein Parametergitter

WARUM DIESES SKRIPT EXISTIERT
    Die Zahlen in RESULTS.md und die Zahlen hinter den Abbildungen muessen aus
    EINER Rechnung stammen. Zwei getrennte Skripte driften auseinander, und der
    Unterschied faellt erst auf, wenn eine Abbildung einer Tabelle widerspricht.
    Alle Zufallsschritte laufen mit festem Seed, default_rng(0).

ZWEI METHODISCHE PUNKTE, DIE IN DEN DATEN STECKEN
    Zirkularitaet: wer nach PoseBusters-Checks rankt und dann PB-Validitaet
    misst, misst sich selbst. Der Rankervergleich wird deshalb ausschliesslich
    gegen den RMSD ausgewertet -- der geht in keinen Ranker ein.

    Gleichstaende: die PB-Mittelwerte haben nur so viele Stufen wie es Checks
    gibt. Ohne zufaellige Aufloesung entschiede die Seed-Reihenfolge, und die
    Kurven werden unmonoton.

AUFRUF   python build_thesis_datasets.py
"""
from __future__ import annotations

import csv
import glob
import os
import re
import sys

import numpy as np

# Zwei Auswertungskonfigurationen, siehe RESULTS.md.
#   bound    graph.sample_conformer=false, Fragmente aus der gebundenen Pose.
#            80 Seeds, mit gnina-Scores.
#   sampled  graph.sample_conformer=true -- die Vorgabe des Papers. 40 Seeds,
#            ohne gnina (dafuer wurde nie gescort).
MODUS = sys.argv[1] if len(sys.argv) > 1 else "bound"
if MODUS == "bound":
    ARMS = [("Minimal", "minimal80", "sigmaflow_minimal"),
            ("Separate", "exp110_80", "exp110"),
            ("SigmaDock", "sigmadock80", "sigmadock")]
    RD = {"Minimal": "minimal80", "Separate": "exp110_80", "SigmaDock": "sigmadock80"}
    KS = (1, 2, 3, 5, 10, 20, 40, 80)
    GN = "GNINA-SCORE-*{gkey}_*"
    TAG = "80seeds"
    MIT_GNINA = True
elif MODUS == "sampled":
    # gnina-Scores liegen unter gncs/ (Jobs 8636670-72) und decken 0..79 ab;
    # der RMSD existiert bisher nur fuer 0..39, die Schnittmenge regelt das.
    ARMS = [("Minimal", "minimalcs", "sigmaflow_minimal"),
            ("Separate", "exp110cs", "exp110"),
            ("SigmaDock", "sigmadockcs", "sigmadock")]
    RD = {"Minimal": "minimalcs2", "Separate": "exp110cs2", "SigmaDock": "sigmadockcs2"}
    GN = "gncs/GNINA-SCORE-*{gkey}_86366*"
    KS = (1, 2, 3, 5, 10, 20, 40)
    TAG = "papersetup40"
    MIT_GNINA = True
elif MODUS == "sampled5":
    # Die vierte Zelle des Versuchsplans: Paper-Konformer UND fuenf Schritte.
    # Nie mit gnina gescort, deshalb kein Ranking.
    ARMS = [("Minimal", "minimal5cs", None),
            ("Separate", "exp1105cs", None),
            ("SigmaDock", "sigmadock5cs", None)]
    RD = {"Minimal": "minimaln5cs", "Separate": "exp110n5cs",
          "SigmaDock": "sigmadockn5cs"}
    GN = None
    KS = (1, 2, 3, 5, 10, 20, 40)
    TAG = "sampled5_40seeds"
    MIT_GNINA = False
elif MODUS == "nfe5":
    # bound, fuenf Integrationsschritte. Diese Posen wurden nie mit gnina
    # gescort, daher kein Ranking.
    ARMS = [("Minimal", "minimal5", None),
            ("Separate", "exp1105", None),
            ("SigmaDock", "sigmadock5", None)]
    RD = {"Minimal": "minimal5", "Separate": "exp1105", "SigmaDock": "sigmadock5"}
    GN = None
    KS = (1, 2, 3, 5, 10, 20, 40)
    TAG = "nfe5_40seeds"
    MIT_GNINA = False
else:
    raise SystemExit(f"MODUS unbekannt: {MODUS} (bound|sampled|sampled5|nfe5)")

OUT = os.path.join("..", "..", "Thesis Visualisierungen", "data")
REP = 400
RNG = np.random.default_rng(0)


def tf(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def spalten(rows: list[dict]) -> tuple[list[str], list[str], list[str]]:
    """-> (alle Checks, ligandenintrinsische, mit Protein)."""
    cols = list(rows[0])
    rmsd_col = next(c for c in cols if c.startswith("rmsd"))
    geladen = {"mol_pred_loaded", "mol_true_loaded", "mol_cond_loaded"}
    checks = [c for c in cols
              if c not in ({"file", "molecule", "position", rmsd_col, ""} | geladen)]
    prot = [c for c in checks
            if any(t in c for t in ("protein", "cofactor", "water", "volume", "distance"))]
    return checks, [c for c in checks if c not in prot], prot


def komplex_id(pfad: str) -> str:
    return pfad.replace(chr(92), "/").split("/")[-1].split("__")[0]


def einlesen():
    """-> V (Validitaet), RM (RMSD), SC (Affinitaet), PBa (mittlere Checkraten)."""
    V, RM, SC, PBa, meta = {}, {}, {}, {}, {}
    for arm, key, gkey in ARMS:
        V[arm], roh = {}, {}
        for f in glob.glob(f"rd_{RD[arm]}_seed*.csv"):
            s = int(re.search(r"seed(\d+)\.csv$", f).group(1))
            rows = list(csv.DictReader(open(f, encoding="utf-8", errors="replace")))
            checks, intra, prot = spalten(rows)
            meta = {"n_checks": len(checks), "n_intra": len(intra), "n_prot": len(prot)}
            for r in rows:
                cid = komplex_id(r["file"])
                V[arm][(cid, s)] = (all(tf(r[c]) for c in intra),
                                    all(tf(r[c]) for c in checks))
                roh[(cid, s)] = {"alle": np.mean([tf(r[c]) for c in checks]),
                                 "intra": np.mean([tf(r[c]) for c in intra]),
                                 "prot": np.mean([tf(r[c]) for c in prot])}
        RM[arm] = {(r["complex"], int(r["seed"])): float(r["rmsd"])
                   for r in csv.DictReader(open(f"pose_{key}.csv", encoding="utf-8"))}
        if MIT_GNINA:
            g = glob.glob(GN.format(gkey=gkey) + f"/gnina_scores_{gkey}.csv")
            if len(g) != 1:
                raise SystemExit(f"ABBRUCH: gnina-Datei nicht eindeutig fuer {gkey}: {g}")
            SC[arm] = {(r["complex"], int(r["seed"])): float(r["affinity"])
                       for r in csv.DictReader(open(g[0], encoding="utf-8"))}
        else:
            SC[arm] = None
        PBa[arm] = roh
    return V, RM, SC, PBa, meta


def wilson(k: int, n: int) -> tuple[float, float]:
    """95%-Intervall fuer einen Anteil. Wald bricht bei kleinen Anteilen zusammen."""
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def main() -> int:
    V, RM, SC, PBroh, meta = einlesen()
    if MIT_GNINA:
        keys = sorted(set.intersection(*[set(V[a]) & set(RM[a]) & set(SC[a])
                                         for a, _, _ in ARMS]))
    else:
        keys = sorted(set.intersection(*[set(V[a]) & set(RM[a]) for a, _, _ in ARMS]))
    cids = sorted({c for c, _ in keys})
    seeds = sorted({s for _, s in keys})
    NC, NS = len(cids), len(seeds)
    if len(keys) != NC * NS:
        raise SystemExit(f"ABBRUCH: {len(keys)} Paare, erwartet {NC * NS} "
                         f"-- Datensatz unvollstaendig")
    print(f"{NC} Komplexe x {NS} Seeds x {len(ARMS)} Arme = {len(ARMS) * len(keys)} Posen")
    print(f"Checks: {meta['n_intra']} intrinsisch + {meta['n_prot']} mit Protein "
          f"= {meta['n_checks']}")

    A = ({a: np.array([[SC[a][(c, s)] for s in seeds] for c in cids])
          for a, _, _ in ARMS} if MIT_GNINA else None)
    R = {a: np.array([[RM[a][(c, s)] for s in seeds] for c in cids]) for a, _, _ in ARMS}
    Vo = {a: np.array([[V[a][(c, s)][0] for s in seeds] for c in cids]) for a, _, _ in ARMS}
    Vm = {a: np.array([[V[a][(c, s)][1] for s in seeds] for c in cids]) for a, _, _ in ARMS}
    PB = {(a, g): np.array([[PBroh[a][(c, s)][g] for s in seeds] for c in cids])
          for a, _, _ in ARMS for g in ("alle", "intra", "prot")}

    METRIKEN = [
        ("rmsd_lt_1.0A", lambda a: R[a] < 1.0),
        ("rmsd_lt_2.0A", lambda a: R[a] < 2.0),
        ("rmsd_lt_2.5A", lambda a: R[a] < 2.5),
        ("rmsd_lt_3.0A", lambda a: R[a] < 3.0),
        ("pb_valid_no_protein", lambda a: Vo[a]),
        ("pb_valid_with_protein", lambda a: Vm[a]),
        ("lt2A_and_valid_no_protein", lambda a: (R[a] < 2.0) & Vo[a]),
        ("lt2A_and_valid_with_protein", lambda a: (R[a] < 2.0) & Vm[a]),
    ]
    os.makedirs(OUT, exist_ok=True)

    # --- 1) Anteil je Ziehung -----------------------------------------------
    pfad = os.path.join(OUT, f"per_draw_{TAG}.csv")
    with open(pfad, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "metric", "pct", "ci_lo", "ci_hi", "n_poses"])
        for arm, _, _ in ARMS:
            for name, f in METRIKEN:
                H = f(arm)
                k, n = int(H.sum()), H.size
                lo, hi = wilson(k, n)
                w.writerow([arm, name, f"{100 * k / n:.4f}",
                            f"{lo:.4f}", f"{hi:.4f}", n])
    print("geschrieben:", pfad)

    # --- 2) Auswahlkurven ----------------------------------------------------
    pfad = os.path.join(OUT, f"selection_curves_{TAG}.csv")
    with open(pfad, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        kopf = ["arm", "metric", "k", "random_pct", "oracle_pct"]
        if MIT_GNINA:
            kopf.insert(4, "top1_affinity_pct")
        w.writerow(kopf)
        for arm, _, _ in ARMS:
            for name, f in METRIKEN:
                H = f(arm)
                for k in KS:
                    ran, top, ora = [], [], []
                    for _ in range(REP if k < NS else 1):
                        idx = np.arange(NS) if k == NS else RNG.permutation(NS)[:k]
                        h = H[:, idx]
                        ran.append(h.mean())
                        ora.append(h.any(axis=1).mean())
                        if MIT_GNINA:
                            pick = A[arm][:, idx].argmin(axis=1)   # negativste Affinitaet
                            top.append(h[np.arange(NC), pick].mean())
                    zeile = [arm, name, k, f"{100 * np.mean(ran):.4f}",
                             f"{100 * np.mean(ora):.4f}"]
                    if MIT_GNINA:
                        zeile.insert(4, f"{100 * np.mean(top):.4f}")
                    w.writerow(zeile)
    print("geschrieben:", pfad)

    if not MIT_GNINA:
        # Ohne gnina-Scores gibt es weder Rankervergleich noch Heuristik.
        print("Modus sampled: Rankervergleich und Heuristik uebersprungen "
              "(keine gnina-Scores fuer diese Posen)")
        return 0

    def top1(score: np.ndarray, hit: np.ndarray, k: int, hoeher_besser: bool = True) -> float:
        s0 = score if hoeher_besser else -score
        out = []
        for _ in range(REP):
            idx = np.arange(NS) if k == NS else RNG.permutation(NS)[:k]
            s = s0[:, idx] + RNG.random((NC, len(idx))) * 1e-9   # Gleichstand zufaellig
            out.append(hit[:, idx][np.arange(NC), s.argmax(axis=1)].mean())
        return float(np.mean(out))

    # --- 3) Rankervergleich, Zielgroesse RMSD < 2 A --------------------------
    pfad = os.path.join(OUT, f"ranker_comparison_{TAG}.csv")
    with open(pfad, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "ranker", "k", "hit_rmsd_lt_2A_pct"])
        for arm, _, _ in ARMS:
            HIT = R[arm] < 2.0
            # Nullmatrix, nicht RNG.random(...): top1() addiert selbst Rauschen
            # zur Gleichstandsaufloesung, und das wird je Wiederholung neu
            # gezogen. Eine feste Zufallsmatrix waehlt dagegen bei k = NS in
            # allen Wiederholungen dieselbe Pose -- es wird nichts gemittelt,
            # und die Grundlinie schwankt um mehrere Punkte.
            ranker = [("random", np.zeros_like(A[arm]), True),
                      ("affinity_vinardo", A[arm], False),
                      ("pb_all", PB[(arm, "alle")], True),
                      ("pb_intrinsic", PB[(arm, "intra")], True),
                      ("pb_protein", PB[(arm, "prot")], True)]
            for nm, sc, hb in ranker:
                for k in KS:
                    w.writerow([arm, nm, k, f"{100 * top1(sc, HIT, k, hb):.4f}"])
            for k in KS:
                v = float(np.mean([HIT[:, (np.arange(NS) if k == NS
                                           else RNG.permutation(NS)[:k])].any(axis=1).mean()
                                   for _ in range(REP)]))
                w.writerow([arm, "oracle", k, f"{100 * v:.4f}"])
    print("geschrieben:", pfad)

    # --- 4) Heuristik-Gitter -------------------------------------------------
    # base * (score_bias + avg_pb ** pb_exponent), base = -Affinitaet.
    # Die beiden Parameter haben im SigmaDock-Repository KEINE Vorgabewerte,
    # deshalb ein Gitter statt eines Wertes. Ein einzelnes, nachtraeglich
    # ausgewaehltes bestes Feld waere wertlos.
    pfad = os.path.join(OUT, f"heuristic_grid_{TAG}.csv")
    with open(pfad, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "score_bias", "pb_exponent", "k", "hit_rmsd_lt_2A_pct"])
        for arm, _, _ in ARMS:
            HIT = R[arm] < 2.0
            base = -A[arm]
            for bias in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
                for expo in (0.5, 1.0, 2.0, 4.0):
                    sc = base * (bias + PB[(arm, "alle")] ** expo)
                    # KS[-1], nicht 80: bei 40 Seeds waere die Beschriftung
                    # sonst falsch, obwohl der Wert stimmt (permutation(40)[:80]
                    # liefert 40 Elemente).
                    w.writerow([arm, bias, expo, KS[-1],
                                f"{100 * top1(sc, HIT, KS[-1]):.4f}"])
    print("geschrieben:", pfad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
