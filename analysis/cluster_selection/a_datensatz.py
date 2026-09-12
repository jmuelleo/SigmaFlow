"""Teil A -- Clusterdatensatz: eine Zeile je Cluster, rund 90 Merkmale.

WAS HIER ENTSTEHT
    Fuer jede Zelle (Modell x Schrittzahl), jeden Komplex und jede
    Clusterschwelle werden die Posen geclustert und je Cluster ein
    Merkmalsvektor gebildet. Etiketten kommen aus dem RMSD zur Kristallpose
    und werden NIE als Merkmal verwendet.

DIE FUENF MERKMALSFAMILIEN
    S  Score-Struktur       Verteilung von Mixed Score, gnina und p_pb im
                            Cluster: Maximum, Median, Streuung, Schiefe, der
                            Abstand zwischen Maximum und Median ("winner
                            excess") und das Verhaeltnis dieses Abstands zur
                            eigenen Streuung ("extreme ratio").
    P  PoseBusters          Nicht nur "valide ja/nein", sondern jede der 24
                            Pruefungen einzeln als Bestehensquote, dazu die
                            mittlere Verletzungszahl und die Werte fuer Medoid
                            und bestbewertete Pose.
    G  Geometrie            Groesse, Kompaktheit, Medoidradius, Abstand zum
                            naechsten und zum groessten Konkurrenzcluster,
                            Silhouette, Zahl der Beinahe-Duplikate.
    K  Kontakte             Protein-Ligand-Wechselwirkungen je Pose, dann
                            aggregiert: Kontaktzahl, kontaktierte Reste,
                            polare und hydrophobe Kontakte, Ueberlappungen,
                            Vergrabungsgrad -- und die Kohaerenz der
                            kontaktierten Restemenge INNERHALB des Clusters.
    A  Ambiguitaet          Eigenschaften der Clusterlandschaft: Zahl der
                            Cluster, Entropie der Clustermassen, Vorsprung vor
                            dem naechsten Cluster, Zahl der Konkurrenten
                            innerhalb eines Score-Bandes.

WARUM JEDES MERKMAL ZUSAETZLICH RANGNORMIERT WIRD
    Absolute Werte sind zwischen Komplexen nicht vergleichbar -- eine
    Affinitaet von -9 ist bei einem grossen Liganden gewoehnlich und bei einem
    kleinen aussergewoehnlich. Die Entscheidung faellt aber IMMER innerhalb
    eines Komplexes. Deshalb bekommt jedes Merkmal eine Spalte `<name>__r`
    mit dem Perzentilrang unter den Clustern desselben Komplexes.

WARUM DIE KONTAKTE AUF EINEN AUSSCHNITT BESCHRAENKT WERDEN
    Ein Protein hat zehntausende Atome, der Ligand dreissig. Ein KD-Baum ueber
    das ganze Protein waere Verschwendung; gebaut wird er ueber die Atome im
    Umkreis von 12 Angstroem um die Vereinigung aller Posen des Komplexes.
    Das aendert kein Ergebnis, weil kein gemessener Kontakt weiter reicht.

Aufruf:
    python analysis/cluster_selection/a_datensatz.py --satz pb308
    python analysis/cluster_selection/a_datensatz.py --satz astex
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
_VAR = os.path.join(os.path.dirname(os.path.dirname(_HIER)), "SigmaFlow_Variants")
sys.path.insert(0, _VAR)
sys.path.insert(0, os.path.dirname(os.path.dirname(_HIER)))
import posencache  # noqa: E402
from zellen import SAETZE, lade_zelle, PRUEFUNGEN  # noqa: E402
from SigmaFlow_Evaluation.ranking.heuristic_score import PB_CHECKS  # noqa: E402

from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402
from scipy.stats import kurtosis, skew  # noqa: E402

ZELLEN = [("SigmaDock", 25), ("Minimal", 25), ("Separate", 25),
          ("Minimal", 5), ("Separate", 5)]
EPS = 1e-9

p = argparse.ArgumentParser()
p.add_argument("--satz", choices=sorted(SAETZE), default="pb308")
p.add_argument("--schwellen", default="2.0,1.0")
p.add_argument("--out", default=None)
a = p.parse_args()
SCHW = [float(x) for x in a.schwellen.split(",")]
AUS = a.out or os.path.join(_HIER, f"cluster_{a.satz}.csv")
SATZ = SAETZE[a.satz]


# ---------------------------------------------------------------- Protein
def protein_laden(code, zentrum, radius=12.0):
    """Schweratome des Proteins im Umkreis, mit Restkennung und Element.

    Eigener Parser statt einer Bibliothek: gebraucht werden nur Koordinaten,
    Element und Restkennung, und die stehen an festen Spalten des
    PDB-Formats. Wasserstoffe fliegen raus (viele Strukturen haben ohnehin
    keine), Wasser ebenfalls -- PoseBusters prueft Wasser getrennt, und als
    Kontaktpartner wuerde es die Zahlen dominieren.
    """
    pfad = os.path.join(SATZ["referenz"], code, f"{code}_protein.pdb")
    if not os.path.isfile(pfad):
        return None
    xyz, el, res = [], [], []
    with open(pfad, "r", errors="replace") as f:
        for z in f:
            if not z.startswith(("ATOM", "HETATM")):
                continue
            rn = z[17:20].strip()
            if rn in ("HOH", "WAT", "DOD"):
                continue
            e = (z[76:78].strip() or z[12:16].strip()[:1]).upper()
            if e == "H":
                continue
            try:
                xyz.append([float(z[30:38]), float(z[38:46]), float(z[46:54])])
            except ValueError:
                continue
            el.append(e)
            res.append(z[21] + z[22:27].strip())
    if not xyz:
        return None
    xyz = np.asarray(xyz)
    nah = np.linalg.norm(xyz - zentrum, axis=1) <= radius + 25.0
    if nah.sum() < 5:
        return None
    return (xyz[nah], np.asarray(el)[nah], np.asarray(res)[nah])


def kontakte(pose, prot, elem_lig):
    """Wechselwirkungsmerkmale EINER Pose gegen das Protein."""
    pxyz, pel, pres = prot
    baum = cKDTree(pxyz)
    aus = {}
    paare45 = baum.query_ball_point(pose, 4.5)
    paare40 = baum.query_ball_point(pose, 4.0)
    aus["kon40"] = float(sum(len(x) for x in paare40))
    aus["kon45"] = float(sum(len(x) for x in paare45))
    reste = set()
    for lst in paare45:
        reste.update(pres[i] for i in lst)
    aus["n_reste"] = float(len(reste))
    aus["vergraben"] = float(np.mean([len(x) > 0 for x in paare45]))
    # polare Kontakte: Ligand-N/O gegen Protein-N/O unter 3,5 A
    pol_l = np.isin(elem_lig, ["N", "O"])
    pol_p = np.isin(pel, ["N", "O"])
    npol = nhyd = 0
    for i, lst in enumerate(baum.query_ball_point(pose, 3.5)):
        if pol_l[i]:
            npol += int(pol_p[list(lst)].sum()) if lst else 0
    koh_l = elem_lig == "C"
    koh_p = pel == "C"
    for i, lst in enumerate(paare45):
        if koh_l[i]:
            nhyd += int(koh_p[list(lst)].sum()) if lst else 0
    aus["polar35"] = float(npol)
    aus["hydrophob45"] = float(nhyd)
    dmin, _ = baum.query(pose, k=1)
    aus["dmin"] = float(dmin.min())
    aus["n_eng22"] = float(sum(len(x) for x in baum.query_ball_point(pose, 2.2)))
    return aus, reste


# ------------------------------------------------------- Merkmalsbausteine
def verteilung(x, praefix, eps_band):
    """Struktur einer Score-Verteilung im Cluster."""
    n = len(x)
    s = np.sort(x)[::-1]
    med = float(np.median(x))
    q75, q25 = np.percentile(x, [75, 25])
    iqr = float(q75 - q25)
    d = {
        f"{praefix}_max": float(s[0]),
        f"{praefix}_med": med,
        f"{praefix}_mean": float(x.mean()),
        f"{praefix}_min": float(s[-1]),
        f"{praefix}_std": float(x.std(ddof=1)) if n > 1 else 0.0,
        f"{praefix}_iqr": iqr,
        f"{praefix}_q75": float(q75),
        f"{praefix}_q25": float(q25),
        f"{praefix}_top2": float(s[:2].mean()),
        f"{praefix}_top3": float(s[:3].mean()),
        f"{praefix}_gap12": float(s[0] - s[1]) if n > 1 else 0.0,
        # winner excess: wie weit steht der Beste ueber der eigenen Mitte?
        f"{praefix}_exzess": float(s[0] - med),
        f"{praefix}_extrem": float((s[0] - med) / (iqr + EPS)),
        f"{praefix}_trimm": float(np.mean(s[int(0.2 * n):max(int(0.8 * n), 1)]))
        if n >= 5 else float(x.mean()),
        f"{praefix}_schief": float(skew(x)) if n > 2 else 0.0,
        f"{praefix}_kurt": float(kurtosis(x)) if n > 3 else 0.0,
        f"{praefix}_nband": float(np.mean(x >= s[0] - eps_band)),
    }
    return d


def geometrie(dsub, dinter_min, dinter_gross, gr, n_ges, sil):
    n = len(dsub)
    iu = np.triu_indices(n, 1)
    paar = dsub[iu] if n > 1 else np.array([0.0])
    medoid = int(np.argmin(dsub.sum(axis=1)))
    rad = dsub[medoid]
    return {
        "g_gr": float(n),
        "g_relgr": float(n / n_ges),
        "g_loggr": float(np.log(n)),
        "g_intra_mean": float(paar.mean()),
        "g_intra_med": float(np.median(paar)),
        "g_intra_max": float(paar.max()),
        "g_medoid_rad": float(rad.mean()),
        "g_rad_p90": float(np.percentile(rad, 90)),
        "g_naechster": float(dinter_min),
        "g_zum_groessten": float(dinter_gross),
        "g_trennung": float(dinter_min / (paar.mean() + EPS)),
        "g_silhouette": float(sil),
        "g_duplikate": float(np.mean(paar < 0.5)),
        "g_medoid": medoid,
    }


# --------------------------------------------------------------- Hauptlauf
zeilen = []
for arm, nfe in ZELLEN:
    z = next((z for z in SATZ["zellen"] if z["arm"] == arm and z["nfe"] == nfe),
             None)
    if z is None:
        continue
    tab = lade_zelle(z, leise=True).set_index(["complex", "seed"])
    daten = posencache.hole(a.satz, arm, nfe, leise=True)
    print(f"{arm}, {nfe} Schritte: {len(daten)} Komplexe", flush=True)
    rmsd_csv = os.path.join(_VAR, f"rmsd_{a.satz}_nfe{nfe}.csv")
    rm = None
    if os.path.isfile(rmsd_csv):
        r = pd.read_csv(rmsd_csv)
        rm = r[r.arm == arm].set_index(["complex", "seed"])["rmsd"]

    for i_k, (code, v) in enumerate(daten.items(), 1):
        g = tab.loc[[(code, int(s)) for s in v["seed"]]]
        heur = g["heur"].to_numpy(float)
        gn = -g["affinity"].to_numpy(float)
        ppb = g["p_pb"].to_numpy(float)
        pruef = g[PRUEFUNGEN].to_numpy(bool)
        valid = np.asarray(v["valid"], bool)
        val5 = g[list(PB_CHECKS)].all(axis=1).to_numpy()
        acc = np.asarray(v["acc"], bool)
        beides = np.asarray(v["beides"], bool)
        d = v["d"].astype(float)
        xyz = v["xyz"].astype(float)
        n_ges = len(heur)
        if rm is not None:
            try:
                rmsd = rm.loc[[(code, int(s)) for s in v["seed"]]].to_numpy(float)
            except KeyError:
                rmsd = np.where(acc, 1.0, 3.0)
        else:
            rmsd = np.where(acc, 1.0, 3.0)

        # ---- Kontakte je Pose, einmal je Komplex ----------------------
        zentrum = xyz.reshape(-1, 3).mean(axis=0)
        prot = protein_laden(code, zentrum)
        kmerk, kreste = None, None
        if prot is not None:
            elem_lig = np.array(["C"] * xyz.shape[1])  # Fallback
            try:
                from rdkit import Chem, RDLogger
                RDLogger.DisableLog("rdApp.*")
                mk = Chem.MolFromMolFile(
                    os.path.join(SATZ["referenz"], code, f"{code}_ligand.sdf"),
                    sanitize=False, removeHs=True)
                if mk is not None and mk.GetNumAtoms() == xyz.shape[1]:
                    elem_lig = np.array([at.GetSymbol() for at in mk.GetAtoms()])
            except Exception:
                pass
            zeil, rest = [], []
            for pose in xyz:
                kk, rr = kontakte(pose, prot, elem_lig)
                zeil.append(kk)
                rest.append(rr)
            kmerk = pd.DataFrame(zeil)
            kreste = rest

        for S in SCHW:
            lab = (np.ones(n_ges, int) if n_ges <= 1 else
                   fcluster(linkage(squareform(d, checks=False),
                                    method="complete"),
                            t=S, criterion="distance"))
            cls = np.unique(lab)
            glob_best = float(heur.max())
            w_cluster = lab[int(np.argmax(heur))]
            # Clustermassen fuer die Ambiguitaetsmerkmale
            massen = np.array([(lab == c).sum() for c in cls], float) / n_ges
            ent = float(-(massen * np.log(massen)).sum())
            cl_max = {c: float(heur[lab == c].max()) for c in cls}
            sortiert = sorted(cl_max.values(), reverse=True)

            for c in cls:
                m = np.where(lab == c)[0]
                dsub = d[np.ix_(m, m)]
                andere = np.where(lab != c)[0]
                if len(andere):
                    dint = d[np.ix_(m, andere)]
                    dmin_inter = float(dint.min())
                    # Abstand zum groessten Konkurrenzcluster
                    grc = max((cc for cc in cls if cc != c),
                              key=lambda cc: (lab == cc).sum())
                    dgross = float(d[np.ix_(m, np.where(lab == grc)[0])].mean())
                    # Silhouette: eigener Zusammenhalt gegen naechsten Nachbarn
                    a_i = (dsub.sum(axis=1) / max(len(m) - 1, 1))
                    b_i = np.array([min(d[i, lab == cc].mean()
                                        for cc in cls if cc != c) for i in m])
                    sil = float(np.mean((b_i - a_i) /
                                        np.maximum(a_i, b_i) + EPS))
                else:
                    dmin_inter, dgross, sil = np.nan, np.nan, 0.0

                r = {"satz": a.satz, "arm": arm, "nfe": nfe, "S": S,
                     "complex": code, "cluster": int(c), "K": n_ges}
                # --- Etiketten (nur Analyse) --------------------------
                r["contains_correct"] = bool(beides[m].any())
                r["contains_acc"] = bool(acc[m].any())
                r["best_pose_correct"] = bool(
                    beides[m[int(np.argmax(heur[m]))]])
                r["correct_fraction"] = float(beides[m].mean())
                r["best_rmsd"] = float(rmsd[m].min())
                r["selected_by_ranker"] = bool(c == w_cluster)
                r["complex_hat_correct"] = bool(beides.any())
                r["ranker_richtig"] = bool(beides[int(np.argmax(heur))])

                # --- S: Score-Struktur --------------------------------
                r.update(verteilung(heur[m], "s_h", 0.5))
                r.update(verteilung(gn[m], "s_g", 0.5))
                r.update(verteilung(ppb[m], "s_p", 0.05))
                r["s_h_zu_global"] = float(heur[m].max() - glob_best)
                r["s_g_zu_global"] = float(gn[m].max() - gn.max())
                r["s_hg_diff"] = float(np.median(heur[m]) - np.median(gn[m]))
                r["s_korr_hv"] = (float(np.corrcoef(heur[m], valid[m].astype(float))[0, 1])
                                  if len(m) > 2 and valid[m].std() > 0 else 0.0)

                # --- P: PoseBusters -----------------------------------
                r["p_valid"] = float(valid[m].mean())
                r["p_valid5"] = float(val5[m].mean())
                verl = (~pruef[m]).sum(axis=1).astype(float)
                r["p_verl_mean"] = float(verl.mean())
                r["p_verl_min"] = float(verl.min())
                r["p_verl_med"] = float(np.median(verl))
                medoid_i = m[int(np.argmin(dsub.sum(axis=1)))]
                best_i = m[int(np.argmax(heur[m]))]
                r["p_verl_medoid"] = float((~pruef[medoid_i]).sum())
                r["p_verl_best"] = float((~pruef[best_i]).sum())
                r["p_valid_best"] = float(valid[best_i])
                r["p_valid_medoid"] = float(valid[medoid_i])
                for j, nm in enumerate(PRUEFUNGEN):
                    kurz = nm.replace("-", "_").replace(" ", "_")[:26]
                    r[f"pc_{kurz}"] = float(pruef[m, j].mean())

                # --- G: Geometrie -------------------------------------
                gg = geometrie(dsub, dmin_inter, dgross, c, n_ges, sil)
                gg.pop("g_medoid")
                r.update(gg)

                # --- K: Kontakte --------------------------------------
                if kmerk is not None:
                    sub = kmerk.iloc[m]
                    for sp in kmerk.columns:
                        r[f"k_{sp}_med"] = float(sub[sp].median())
                        r[f"k_{sp}_max"] = float(sub[sp].max())
                        r[f"k_{sp}_std"] = float(sub[sp].std(ddof=1)) \
                            if len(m) > 1 else 0.0
                        r[f"k_{sp}_medoid"] = float(kmerk.iloc[medoid_i][sp])
                        r[f"k_{sp}_best"] = float(kmerk.iloc[best_i][sp])
                    # Kohaerenz der kontaktierten Reste im Cluster
                    if len(m) > 1:
                        js = []
                        for ii in range(len(m)):
                            for jj in range(ii + 1, len(m)):
                                A, B = kreste[m[ii]], kreste[m[jj]]
                                u = len(A | B)
                                js.append(len(A & B) / u if u else 1.0)
                        r["k_kohaerenz"] = float(np.mean(js))
                    else:
                        r["k_kohaerenz"] = 1.0

                # --- A: Ambiguitaet -----------------------------------
                r["a_n_cluster"] = float(len(cls))
                r["a_masse_ent"] = ent
                r["a_masse_ent_norm"] = ent / np.log(len(cls)) if len(cls) > 1 else 0.0
                r["a_vorsprung"] = float(cl_max[c] - (sortiert[1] if c ==
                                          max(cl_max, key=cl_max.get) and
                                          len(sortiert) > 1 else sortiert[0]))
                r["a_n_im_band05"] = float(sum(1 for x in sortiert
                                               if x >= glob_best - 0.5))
                r["a_n_im_band10"] = float(sum(1 for x in sortiert
                                               if x >= glob_best - 1.0))
                zeilen.append(r)
        if i_k % 50 == 0:
            print(f"  {i_k}/{len(daten)}", flush=True)

t = pd.DataFrame(zeilen)

# ---- komplexinterne Rangnormierung -------------------------------------
# Die Entscheidung faellt immer innerhalb eines Komplexes; absolute Werte
# sind zwischen Komplexen nicht vergleichbar.
ETIKETT = {"contains_correct", "contains_acc", "best_pose_correct",
           "correct_fraction", "best_rmsd", "selected_by_ranker",
           "complex_hat_correct", "ranker_richtig"}
META = {"satz", "arm", "nfe", "S", "complex", "cluster", "K"}
MERK = [c for c in t.columns if c not in ETIKETT | META]
gr = t.groupby(["arm", "nfe", "S", "complex"])
for c in MERK:
    t[f"{c}__r"] = gr[c].rank(pct=True)

t.to_csv(AUS, index=False)
print(f"\n{len(t)} Cluster, {len(MERK)} Merkmale (+ Rangversionen) nach {AUS}")
print(f"  Zellen: {t.groupby(['arm','nfe']).size().to_dict()}")
print(f"  Schwellen: {sorted(t.S.unique())}")
print(f"  Etikettenrate contains_correct: {100*t.contains_correct.mean():.1f} %")
