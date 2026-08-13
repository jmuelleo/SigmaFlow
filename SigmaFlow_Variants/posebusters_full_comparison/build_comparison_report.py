"""Erzeugt den vollstaendigen SigmaFlow-vs-SigmaDock-Vergleichsbericht als .txt,
inklusive fertigem PyMOL-Code fuer repraesentative Komplexe.

Datengrundlage: das 10-Seed-Experiment (Jobs 8554147 / 8554149), je 209 Komplexe
x 10 Seeds = 2090 Posen pro Methode. Beide Methoden lasen dieselbe Eingabedatei,
beide sind auf dieselbe Rechenzeit gematcht (12h, max_epochs=6).

Vorzeichenkonvention durchgehend: positiv = SigmaFlow SCHLECHTER.

Usage: python build_comparison_report.py
"""

import glob
import os
from datetime import datetime
from math import comb

import numpy as np
import pandas as pd

from ligand_reference import best_copy, load_copies

OUT = "../../VERGLEICH_SigmaFlow_vs_SigmaDock.txt"
N_SEEDS = 10
RUNS = {"SigmaFlow": "seeds10_sigmaflow/last", "SigmaDock": "seeds10_sigmadock/last"}
PB_CSV = {"SigmaFlow": "posebusters_seed0_SigmaFlow.csv",
          "SigmaDock": "posebusters_seed0_SigmaDock.csv"}
L = []


def w(s=""):
    L.append(s)
    print(s)


def kabsch(P, Q):
    """-> (aligned_rmsd, rotation_angle_deg, centroid_distance)"""
    cP, cQ = P.mean(0), Q.mean(0)
    Pc, Qc = P - cP, Q - cQ
    U, _, Vt = np.linalg.svd(Qc.T @ Pc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    ali = float(np.sqrt(((Pc - Qc @ R.T) ** 2).sum(1).mean()))
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return ali, ang, float(np.linalg.norm(cP - cQ))


def mcnemar(x, y):
    n01 = int((x & ~y).sum())
    n10 = int((~x & y).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    return n01, n10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_ci(d, seed=0, n=20000):
    rng = np.random.default_rng(seed)
    b = rng.choice(d, size=(n, d.size), replace=True).mean(axis=1)
    return np.percentile(b, [2.5, 97.5])


# ---------------------------------------------------------------- load
cids, truth = [], {}
for tp in sorted(glob.glob("true_ligands/*_ligands.sdf")):
    c = os.path.basename(tp).replace("_ligands.sdf", "")
    m = load_copies(tp)
    if m:
        cids.append(c)
        truth[c] = m

cube = {}
for lbl, root in RUNS.items():
    R = np.full((len(cids), N_SEEDS), np.nan)   # raw RMSD
    A = np.full((len(cids), N_SEEDS), np.nan)   # aligned RMSD
    G = np.full((len(cids), N_SEEDS), np.nan)   # rotation angle
    C = np.full((len(cids), N_SEEDS), np.nan)   # centroid distance
    for s in range(N_SEEDS):
        for i, cid in enumerate(cids):
            h = glob.glob(os.path.join(root, f"seed_{s}", f"{cid}__*_seed0.sdf"))
            if not h:
                continue
            pc = load_copies(h[0])
            if not pc:
                continue
            P = pc[0].GetConformer().GetPositions()
            mt, raw, _ = best_copy(P, truth[cid])
            if mt is None:
                continue
            Q = mt.GetConformer().GetPositions()
            R[i, s] = raw
            A[i, s], G[i, s], C[i, s] = kabsch(P, Q)
    cube[lbl] = (R, A, G, C)

w("=" * 100)
w("SIGMAFLOW vs. SIGMADOCK — VOLLSTAENDIGER METRIKVERGLEICH")
w("=" * 100)
w(f"Erzeugt: {datetime.now().isoformat(timespec='seconds')}")
w("Datengrundlage : 10-Seed-Experiment, Jobs 8554147 (SigmaFlow) / 8554149 (SigmaDock)")
w(f"                 {len(cids)} PoseBusters-Komplexe x {N_SEEDS} Seeds = "
  f"{len(cids)*N_SEEDS} Posen je Methode")
w("Trainingsstand : beide 12h, max_epochs=6, compute-gematcht (SigmaFlow global_step 13.750)")
w("Referenz       : naechstgelegene kristallographische Kopie (ligand_reference.py)")
w("VORZEICHEN     : positiv = SigmaFlow SCHLECHTER")
w("")
w("WICHTIG: kein Confidence-Ranking. Das Paper erreicht 79.9% als Top-1 aus N Ziehungen,")
w("gerankt von einem trainierten Modell. Wir ranken nicht - alle Zahlen unten sind die")
w("rohe Verteilung einer Einzelziehung, fuer beide Methoden identisch behandelt.")

# ---------------------------------------------------------------- 1
w("")
w("=" * 100)
w("1. HAUPTMETRIKEN  (pro Seed berechnet, dann Mittel +- SD ueber die 10 Seeds)")
w("=" * 100)
w(f"{'Metrik':<40}{'SigmaFlow':>22}{'SigmaDock':>22}{'Delta':>14}")
w("-" * 100)


def per_seed(M, fn):
    return np.array([fn(M[:, s][np.isfinite(M[:, s])]) for s in range(N_SEEDS)])


rows = [
    ("RMSD Median (A)", 0, lambda v: np.median(v), "{:.2f}"),
    ("RMSD Mittel (A)", 0, lambda v: v.mean(), "{:.2f}"),
    ("Erfolg RMSD <= 1 A (%)", 0, lambda v: 100 * (v <= 1).mean(), "{:.1f}"),
    ("Erfolg RMSD <= 2 A (%)  <- Standard", 0, lambda v: 100 * (v <= 2).mean(), "{:.1f}"),
    ("Erfolg RMSD <= 5 A (%)", 0, lambda v: 100 * (v <= 5).mean(), "{:.1f}"),
    ("Ausreisser > 10 A (%)", 0, lambda v: 100 * (v > 10).mean(), "{:.1f}"),
    ("Ausreisser > 20 A (%)", 0, lambda v: 100 * (v > 20).mean(), "{:.1f}"),
    ("Schwerpunktabstand Median (A)", 3, lambda v: np.median(v), "{:.2f}"),
    ("Schwerpunkt <= 2 A (%)", 3, lambda v: 100 * (v <= 2).mean(), "{:.1f}"),
    ("RMSD n. Ausrichtung Median (A)", 1, lambda v: np.median(v), "{:.2f}"),
    ("Orientierungsfehler Median (Grad)", 2, lambda v: np.median(v), "{:.1f}"),
    ("Orientierung < 20 Grad (%)", 2, lambda v: 100 * (v < 20).mean(), "{:.1f}"),
    ("Orientierung > 160 Grad (%)", 2, lambda v: 100 * (v > 160).mean(), "{:.1f}"),
]
for name, idx, fn, fmt in rows:
    a = per_seed(cube["SigmaFlow"][idx], fn)
    b = per_seed(cube["SigmaDock"][idx], fn)
    sa = f"{fmt.format(a.mean())} +- {fmt.format(a.std(ddof=1))}"
    sb = f"{fmt.format(b.mean())} +- {fmt.format(b.std(ddof=1))}"
    w(f"{name:<40}{sa:>22}{sb:>22}{fmt.format(a.mean()-b.mean()):>14}")

w("")
w("Zufallsbaseline Orientierung: Haar-Winkelverteilung, Mittel 126.5 Grad, MEDIAN 132.3 Grad.")
w("(Beide Werte fuehren - der Median ist die richtige Referenz fuer Median-Vergleiche.)")

# ---------------------------------------------------------------- 2
w("")
w("=" * 100)
w("2. VERTEILUNG DES RMSD  (seed-gemittelt pro Komplex, n=209)")
w("=" * 100)
w(f"{'Lauf':<14}{'mean':>8}{'sd':>8}{'q10':>8}{'q25':>8}{'median':>9}{'q75':>8}{'q90':>8}{'max':>8}")
w("-" * 100)
for lbl in RUNS:
    v = np.nanmean(cube[lbl][0], axis=1)
    v = v[np.isfinite(v)]
    q = np.percentile(v, [10, 25, 50, 75, 90])
    w(f"{lbl:<14}{v.mean():>8.2f}{v.std(ddof=1):>8.2f}{q[0]:>8.2f}{q[1]:>8.2f}"
      f"{q[2]:>9.2f}{q[3]:>8.2f}{q[4]:>8.2f}{v.max():>8.2f}")

# ---------------------------------------------------------------- 3
w("")
w("=" * 100)
w("3. GEPAARTER VERGLEICH  (seed-gemittelt pro Komplex, Bootstrap-CI ueber 20.000 Ziehungen)")
w("=" * 100)
w(f"{'Groesse':<34}{'Median':>10}{'Mittel':>10}{'95%-CI':>22}{'SF besser':>12}{'Urteil':>16}")
w("-" * 100)
for name, idx in (("RMSD roh (A)", 0), ("RMSD ausgerichtet (A)", 1),
                  ("Orientierungsfehler (Grad)", 2), ("Schwerpunktabstand (A)", 3)):
    a = np.nanmean(cube["SigmaFlow"][idx], axis=1)
    b = np.nanmean(cube["SigmaDock"][idx], axis=1)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    lo, hi = boot_ci(d)
    sig = "signifikant" if not (lo < 0 < hi) else "n.s."
    w(f"{name:<34}{np.median(d):>+10.2f}{d.mean():>+10.2f}"
      f"   [{lo:+.2f}, {hi:+.2f}]{100*(d<0).mean():>11.0f}%{sig:>16}")

# ---------------------------------------------------------------- 4
w("")
w("=" * 100)
w("4. SEED-STREUUNG  (die wichtigste Warnung fuer jeden Einzelvergleich)")
w("=" * 100)
for lbl in RUNS:
    R = cube[lbl][0]
    med = np.nanmedian(R, axis=0)
    lt2 = 100 * np.nanmean(R <= 2, axis=0)
    sd = np.nanstd(R, axis=1, ddof=1)
    w(f"{lbl}")
    w(f"   Median-RMSD je Seed  : {med.min():.2f} bis {med.max():.2f} A  (SD {med.std(ddof=1):.3f})")
    w(f"   Erfolg <=2A je Seed  : {lt2.min():.1f}% bis {lt2.max():.1f}%  (SD {lt2.std(ddof=1):.2f})")
    w(f"   DERSELBE Komplex     : SD {np.nanmedian(sd):.2f} A, "
      f"Spannweite {np.nanmedian(np.nanmax(R,1)-np.nanmin(R,1)):.2f} A")
w("")
w("=> Die Streuung EINER Ziehung ist rund zehnmal so gross wie der Methodenunterschied.")
w("   Einzelziehungs-Vergleiche in diesem Feld lesen ueberwiegend Rauschen.")

# ---------------------------------------------------------------- 5
w("")
w("=" * 100)
w("5. BEST-OF-10  (Obergrenze, die ein perfektes Confidence-Ranking erreichen koennte)")
w("=" * 100)
w(f"{'Lauf':<14}{'<=2A ein Seed':>16}{'<=2A best-of-10':>18}{'Median ein Seed':>18}{'Median best-of-10':>20}")
w("-" * 100)
for lbl in RUNS:
    R = cube[lbl][0]
    best = np.nanmin(R, axis=1)
    w(f"{lbl:<14}{100*np.nanmean(R[:,0]<=2):>15.1f}%{100*np.nanmean(best<=2):>17.1f}%"
      f"{np.nanmedian(R[:,0]):>17.2f}A{np.nanmedian(best):>19.2f}A")

# ---------------------------------------------------------------- 6
w("")
w("=" * 100)
w("6. CHEMISCHE PLAUSIBILITAET + SYMMETRIEKORRIGIERTER RMSD  (PoseBusters, Seed 0)")
w("=" * 100)
try:
    pb = {k: pd.read_csv(v) for k, v in PB_CSV.items()}
    excl = ("position", "mol_cond", "molecule", "rmsd", "mol_pred_loaded", "mol_true_loaded")
    cols = [c for c in pb["SigmaFlow"].columns
            if c in pb["SigmaDock"].columns
            and set(pb["SigmaFlow"][c].dropna().unique()) <= {True, False}
            and not any(c.startswith(p) for p in excl) and "rmsd" not in c.lower()]
    rc = [c for c in pb["SigmaFlow"].columns if "rmsd" in c.lower()][0]
    w(f"{'Check':<32}{'SigmaFlow':>12}{'SigmaDock':>12}{'nur SF':>9}{'nur SD':>9}{'p':>10}{'Urteil':>22}")
    w("-" * 100)
    for c in cols + ["ALLE CHECKS", rc]:
        if c == "ALLE CHECKS":
            x = pb["SigmaFlow"][cols].all(axis=1).values
            y = pb["SigmaDock"][cols].all(axis=1).values
            nm = "ALLE CHECKS (= PB-valid)"
        else:
            x = pb["SigmaFlow"][c].fillna(False).values.astype(bool)
            y = pb["SigmaDock"][c].fillna(False).values.astype(bool)
            nm = "RMSD <= 2 A (symmetriekorr.)" if c == rc else c
        if x.all() and y.all():
            continue
        n01, n10, p = mcnemar(x, y)
        verd = "nicht unterscheidbar" if p >= 0.05 else ("SigmaFlow besser" if n01 > n10 else "SigmaDock besser")
        w(f"{nm:<32}{100*x.mean():>11.1f}%{100*y.mean():>11.1f}%{n01:>9}{n10:>9}{p:>10.4f}{verd:>22}")

    # THE headline metric of the PoseBusters paper
    w("")
    w("DIE STANDARD-KOPFZAHL DES FELDES:")
    for lbl in RUNS:
        valid = pb[lbl][cols].all(axis=1).values
        hit = pb[lbl][rc].fillna(False).values.astype(bool)
        w(f"   {lbl:<12} PB-valid UND RMSD<=2A : {100*(valid & hit).mean():>5.1f}%   "
          f"(PB-valid allein {100*valid.mean():.1f}%, RMSD<=2A allein {100*hit.mean():.1f}%)")
    vx = pb["SigmaFlow"][cols].all(axis=1).values & pb["SigmaFlow"][rc].fillna(False).values.astype(bool)
    vy = pb["SigmaDock"][cols].all(axis=1).values & pb["SigmaDock"][rc].fillna(False).values.astype(bool)
    n01, n10, p = mcnemar(vx, vy)
    w(f"   gepaart: nur SF {n01}, nur SD {n10}, p={p:.4f}  -> "
      f"{'nicht unterscheidbar' if p>=0.05 else ('SigmaFlow besser' if n01>n10 else 'SigmaDock besser')}")
    w("")
    w("VORBEHALT MULTIPLES TESTEN — bitte nicht ueberlesen:")
    w("  In dieser Tabelle werden 7 Hypothesen gegen dieselben Daten getestet. Bei")
    w("  Bonferroni-Korrektur liegt die Schwelle bei 0.05/7 = 0.0071.")
    w("    - RMSD <= 2 A (p=0.0041) UEBERLEBT die Korrektur -> gesicherter Befund.")
    w("    - Bindungslaengen (p=0.020) und Bindungswinkel (p=0.027) UEBERLEBEN NICHT.")
    w("      Als Tendenz zugunsten SigmaFlow berichten, NICHT als Nachweis.")
    w("  Zusaetzlich: alle Chemie-Checks stammen aus EINEM Seed. Die Seed-Streuung in")
    w("  Abschnitt 4 gilt sinngemaess auch hier.")
    w("")
    w("Hinweis: 8 protein-abhaengige mol_cond-Checks fehlen (Rezeptor-PDBs nicht lokal).")
    w("PB-valid bezieht sich hier auf die 15 liganden-intrinsischen Checks.")
except Exception as e:
    w(f"[PoseBusters-CSV nicht lesbar: {type(e).__name__}: {e}]")

# ---------------------------------------------------------------- 7 PyMOL
w("")
w("=" * 100)
w("7. PYMOL — REPRAESENTATIVE KOMPLEXE ZUM ANSEHEN")
w("=" * 100)
Rf = np.nanmean(cube["SigmaFlow"][0], axis=1)
Rd = np.nanmean(cube["SigmaDock"][0], axis=1)
Gf = np.nanmean(cube["SigmaFlow"][2], axis=1)
Af = np.nanmean(cube["SigmaFlow"][1], axis=1)
diff = Rf - Rd
ok = np.isfinite(diff)
idx_all = np.arange(len(cids))
picks = []


def add(name, i, why):
    if np.isfinite(Rf[i]):
        picks.append((name, cids[i], why))


add("BESTER SigmaFlow", int(idx_all[ok][np.argmin(Rf[ok])]), "kleinster RMSD ueberhaupt")
add("SigmaFlow GEWINNT", int(idx_all[ok][np.argmin(diff[ok])]), "groesster Vorsprung SF")
add("SigmaDock GEWINNT", int(idx_all[ok][np.argmax(diff[ok])]), "groesster Rueckstand SF")
med_i = int(idx_all[ok][np.argsort(Rf[ok])[len(Rf[ok]) // 2]])
add("TYPISCHER FALL", med_i, "Median-RMSD")
flip = ok & (Gf > 160) & (Af < 2.0)
if flip.any():
    add("KOPF-SCHWANZ-FLIP", int(idx_all[flip][np.argmin(Af[flip])]),
        "Orientierung >160 Grad, innere Geometrie aber gut")

w("")
for name, cid, why in picks:
    i = cids.index(cid)
    w(f"  {name:<20} {cid:<14} RMSD SF={Rf[i]:5.2f}  SD={Rd[i]:5.2f}  "
      f"Orient.SF={Gf[i]:5.1f} Grad  ausger.SF={Af[i]:4.2f}   ({why})")

# --- Fallordner schreiben: genau EIN Molekuel je Datei, keine Mehrdeutigkeit ---
from rdkit import Chem  # noqa: E402

CASE_DIR = "pymol_cases"
os.makedirs(CASE_DIR, exist_ok=True)
written = []
for name, cid, why in picks:
    i = cids.index(cid)
    sfp = glob.glob(f"seeds10_sigmaflow/last/seed_0/{cid}__*_seed0.sdf")
    sdp = glob.glob(f"seeds10_sigmadock/last/seed_0/{cid}__*_seed0.sdf")
    if not (sfp and sdp):
        continue
    mp = load_copies(sfp[0])[0]
    # die Referenzkopie, gegen die auch gerechnet wurde - NICHT einfach Kopie 0
    mt, _, cidx = best_copy(mp.GetConformer().GetPositions(), truth[cid])
    if mt is None:
        continue
    for tag, mol in (("TRUE", mt), ("SIGMAFLOW", mp), ("SIGMADOCK", load_copies(sdp[0])[0])):
        with Chem.SDWriter(os.path.join(CASE_DIR, f"{cid}_{tag}.sdf")) as wr:
            wr.write(mol)
    written.append((name, cid, why, i, cidx))

import json  # noqa: E402

cases_json = os.path.join(CASE_DIR, "cases.json")
registry = {}
if os.path.exists(cases_json):          # Faelle anderer Datensaetze erhalten
    try:
        registry = json.load(open(cases_json, encoding="utf-8"))
    except Exception:
        registry = {}
registry = {k: v for k, v in registry.items() if v.get("set") != "PoseBusters"}
for name, cid, why, i, cidx in written:
    registry[cid] = {"set": "PoseBusters", "rolle": name,
                     "rmsd_sf": round(float(Rf[i]), 2), "rmsd_sd": round(float(Rd[i]), 2),
                     "orient_sf": round(float(Gf[i]), 0), "aligned_sf": round(float(Af[i]), 2),
                     "kopie": int(cidx)}
with open(cases_json, "w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2)

w("")
w(f"Es wurden {len(written)} Faelle nach {CASE_DIR}/ geschrieben - je genau EIN Molekuel")
w("pro Datei (die Referenz ist die tatsaechlich verglichene Kristallkopie, nicht")
w("blind die erste). Damit gibt es in PyMOL keine Mehrdeutigkeit.")
w(f"Die Beschriftungen stehen in {CASE_DIR}/cases.json - das PyMOL-Skript liest sie von")
w("dort, neue Faelle erscheinen also automatisch, ohne dass dieser Bericht neu muss.")

w("")
w("=" * 100)
w("PYMOL — ALLES AB HIER EINFACH KOMPLETT KOPIEREN UND IN PYMOL EINFUEGEN")
w("=" * 100)
w("Nichts anpassen, nichts speichern. Der Block laedt Protein, wahren Liganden,")
w("SigmaFlow und SigmaDock. Danach mit  zeige('<KOMPLEX>')  zwischen den Faellen")
w("wechseln. Das Protein wird automatisch von der PDB geladen (Internet noetig).")
w("")
w("---8<--- AB HIER KOPIEREN ---8<---")
w("")
w("python")
w("import os, json")
w("from pymol import cmd, util")
w("")
w(f'BASE = r"{os.path.abspath(CASE_DIR)}"')
w("")
w("FAELLE = json.load(open(os.path.join(BASE, 'cases.json'), encoding='utf-8'))")
w("")
w("def zeige(cid, protein=True, oberflaeche=True):")
w('    """Zeigt Protein + wahren Liganden + SigmaFlow + SigmaDock fuer einen Komplex."""')
w("    cmd.delete('all')")
w("    cmd.load(os.path.join(BASE, cid + '_TRUE.sdf'),      'LIG_WAHR')")
w("    cmd.load(os.path.join(BASE, cid + '_SIGMAFLOW.sdf'), 'LIG_SIGMAFLOW')")
w("    cmd.load(os.path.join(BASE, cid + '_SIGMADOCK.sdf'), 'LIG_SIGMADOCK')")
w("    if protein:")
w("        try:")
w("            cmd.fetch(cid.split('_')[0], 'PROT', type='pdb', async_=0)")
w("            cmd.remove('PROT and not polymer')      # Wasser, Ionen, Kristallliganden weg")
w("            cmd.remove('PROT and hydro')")
w("        except Exception as e:")
w("            print('[WARN] Protein konnte nicht geladen werden:', e)")
w("    cmd.hide('everything')")
w("    cmd.show('sticks', 'LIG_*')")
w("    util.cbag('LIG_WAHR')          # Kohlenstoff gruen   = Kristallpose")
w("    util.cbac('LIG_SIGMAFLOW')     # Kohlenstoff cyan    = SigmaFlow")
w("    util.cbam('LIG_SIGMADOCK')     # Kohlenstoff magenta = SigmaDock")
w("    if protein and cmd.count_atoms('PROT'):")
w("        cmd.show('cartoon', 'PROT')")
w("        cmd.color('grey70', 'PROT')")
w("        cmd.set('cartoon_transparency', 0.5)")
w("        if oberflaeche:")
w("            cmd.select('TASCHE', 'byres (PROT within 6 of LIG_WAHR)')")
w("            cmd.show('surface', 'TASCHE')")
w("            cmd.set('transparency', 0.55)")
w("            cmd.color('wheat', 'TASCHE')")
w("            cmd.deselect()")
w("    cmd.bg_color('white')")
w("    cmd.set('ray_opaque_background', 0)")
w("    cmd.set('stick_radius', 0.14)")
w("    cmd.set('valence', 1)")
w("    cmd.orient('LIG_WAHR')")
w("    cmd.zoom('LIG_WAHR', 6)")
w("    m = FAELLE.get(cid, {})")
w("    print('=' * 78)")
w("    print(f\"{cid}  [{m.get('set','?')}]  {m.get('rolle','')}\")")
w("    print(f\"  RMSD SigmaFlow {m.get('rmsd_sf','?')} A | SigmaDock {m.get('rmsd_sd','?')} A\"")
w("          f\"  | Orientierungsfehler SigmaFlow {m.get('orient_sf','?')} Grad\")")
w("    print('  gruen = Kristallpose | cyan = SigmaFlow | magenta = SigmaDock')")
w("    if m.get('warnung'):")
w("        print('  [!] ' + m['warnung'])")
w("    print('=' * 78)")
w("")
w("def liste():")
w("    for s in sorted({v.get('set','?') for v in FAELLE.values()}):")
w("        print(f'--- {s} ---')")
w("        for k, v in FAELLE.items():")
w("            if v.get('set') != s:")
w("                continue")
w("            print(f\"  zeige(\\\"{k}\\\")   {v.get('rolle','')}: \"")
w("                  f\"SF {v.get('rmsd_sf','?')} A / SD {v.get('rmsd_sd','?')} A, \"")
w("                  f\"Orient. {v.get('orient_sf','?')} Grad\")")
w("")
w("cmd.extend('zeige', zeige)")
w("cmd.extend('liste', liste)")
w("python end")
w("")
w("liste")
if written:
    w(f'zeige("{written[0][1]}")')
w("")
w("---8<--- BIS HIER KOPIEREN ---8<---")
w("")
w("BEDIENUNG")
w("  liste                    zeigt alle verfuegbaren Faelle mit ihren Kennzahlen")
w('  zeige("7ZHP_IQY")        wechselt zu diesem Komplex')
w('  zeige("7ZHP_IQY", False) ohne Protein (schneller, kein Internet noetig)')
w("")
w("WORAUF DU ACHTEST")
w("  1. Liegt CYAN in derselben Tasche wie GRUEN? -> Translation stimmt fast immer")
w("     (Schwerpunkt <= 2 A bei 82% der Posen, besser als SigmaDock).")
w("  2. Zeigt CYAN bei gleicher Molekuelform in die GEGENRICHTUNG von GRUEN?")
w("     -> das ist der Kopf-Schwanz-Flip, der dokumentierte Hauptfehlermodus")
w("     (36.5% der Posen ueber 160 Grad Orientierungsfehler). Am deutlichsten")
w("     im Fall 'KOPF-SCHWANZ-FLIP' oben.")
w("  3. Sehen die Bindungslaengen von CYAN plausibel aus? -> ja, dort ist")
w("     SigmaFlow tendenziell sogar besser als SigmaDock (55.5% vs 46.4%).")
w("")
w("FALLS DAS PROTEIN NICHT LAEDT")
w("  cmd.fetch braucht Internet und schreibt in das aktuelle PyMOL-Arbeits-")
w("  verzeichnis. Alternativ das Protein von ARC holen:")
w("    /data/stat-cadd/shug8458/data/posebusters_paper/posebusters_benchmark_set/")
w("        <KOMPLEX>/<KOMPLEX>_protein.pdb")
w("  und im Skript cmd.fetch durch cmd.load(<pfad>, 'PROT') ersetzen.")
w("  Zur Not: zeige(\"...\", False) - die Ligandenvergleiche funktionieren auch ohne.")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print(f"\n\nGeschrieben nach: {os.path.abspath(OUT)}")
