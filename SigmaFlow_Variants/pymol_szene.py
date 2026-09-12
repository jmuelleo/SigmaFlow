"""PyMOL-Szenen, in denen man SIEHT, warum das Ranking funktioniert oder nicht.

WAS DIE ERSTE FASSUNG NICHT KONNTE
    `posen_verteilung.py --pymol` faerbt alle Posen einer Zelle gleich. Man
    sieht dann, WIE BREIT gestreut wird, aber nicht, ob die Streuung aus
    mehreren Moden besteht und welche davon richtig ist. Genau das ist aber
    die offene Frage: der Ranking-Hebel ist bei Minimal 1,66 gegen SigmaDocks
    1,38, obwohl die AUC leicht fuer SigmaDock spricht -- der Vorteil sitzt
    also an der SPITZE der Liste, nicht in der Gesamtordnung.

WAS DIESE FASSUNG ZEIGT
    Je Zelle drei Objekte statt einem:

      korrekt   RMSD < 2 A zur Kristallpose        gruen, duenne Linien
      falsch    alles andere                       grau, duenne Linien
      gewaehlt  die Pose, die der Mixed Score      magenta, dicke Sticks
                als Beste auswaehlt

    Damit ist auf einen Blick zu sehen:
      - Gibt es einen ENGEN falschen Modus? (grauer Klumpen)
      - Liegt die gewaehlte Pose im gruenen oder im grauen Klumpen?
      - Ist die richtige Pose ein schmaler Kanal oder eine breite Wolke?

    Die gewaehlte Pose ist IMMER enthalten, auch wenn ausgeduennt wird --
    sonst zeigte die Szene eine Auswahl, die gar nicht getroffen wurde.

WOHER DIE ZUORDNUNG KOMMT
    `acc` und `heur` stammen aus `zellen.py`, also aus derselben
    Redock-plus-gnina-Verknuepfung wie alle Tabellen. Keine zweite Rechnung,
    keine zweite Quelle. Der Schluessel ist (complex, seed); der Seed steht im
    Dateinamen.

Aufruf:
    python SigmaFlow_Variants/pymol_szene.py
    python SigmaFlow_Variants/pymol_szene.py --codes 6WTN_RXT --max-posen 80
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

_HIER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HIER)
from zellen import ZELLEN, lade_zelle  # noqa: E402

# Wo die Posen-SDF liegen. Der Etikettenteil "emergency" ist der ENDPUNKT;
# die "at_XXXh" sind Snapshots. Ein Glob ueber "*__sampled" wuerde sie
# mitnehmen und einen Zwischenstand als Ergebnis zeigen.
POSEN_LAUF = {
    ("SigmaDock", 25): ("SD_BASE_72H_s0_8648493", "sched239ep_emergency__nfe25__sampled"),
    ("SigmaDock", 5):  ("SD_BASE_72H_s0_8648493", "sched239ep_emergency__nfe5__sampled"),
    ("Minimal", 25):   ("SF_MIN_72H_s0_8653824", "sched255ep_emergency__nfe25__sampled"),
    ("Minimal", 5):    ("SF_MIN_72H_s0_8653824", "sched255ep_emergency__nfe5__sampled"),
    ("Separate", 25):  ("SF_2H_72H_s0_8668713", "sched255ep_emergency__nfe25__sampled"),
    ("Separate", 5):   ("SF_2H_72H_s0_8668713", "sched255ep_emergency__nfe5__sampled"),
}


def posenordner(lauf: str, tag: str) -> str:
    schnipsel = tag.split("__")[0]
    return os.path.join(_HIER, lauf, "learning_curve_cpu", tag, "results",
                        "posebusters308", f"snap_{schnipsel}")


p = argparse.ArgumentParser()
p.add_argument("--codes", nargs="*", default=None)
p.add_argument("--max-posen", type=int, default=60)
p.add_argument("--true", default=os.path.join(_HIER, "learning_curve_min", "true308"))
a = p.parse_args()

# Welche Zellen liegen lokal ueberhaupt vor?
verfuegbar = {}
for (arm, nfe), (lauf, tag) in POSEN_LAUF.items():
    w = posenordner(lauf, tag)
    if os.path.isdir(w):
        verfuegbar[(arm, nfe)] = w
if not verfuegbar:
    sys.exit("ABBRUCH: keine Posenverzeichnisse gefunden. poses_pymol.tgz entpackt?")
print(f"{len(verfuegbar)} Zellen lokal vorhanden\n")

# Welche Komplexe haben Posen in ALLEN vorhandenen Zellen?
mengen = []
for w in verfuegbar.values():
    mengen.append({os.path.basename(f).split("__")[0]
                   for f in glob.glob(os.path.join(w, "seed_*", "*.sdf"))})
gemeinsam = set.intersection(*mengen)
codes = sorted(a.codes) if a.codes else sorted(gemeinsam)
fehlend = [c for c in codes if c not in gemeinsam]
if fehlend:
    sys.exit(f"ABBRUCH: nicht in allen vorhandenen Zellen: {fehlend}")
print(f"Komplexe: {' '.join(codes)}\n")

print("Tabellen (fuer acc und Mixed Score):")
tabellen = {}
for z in ZELLEN:
    if (z["arm"], z["nfe"]) in verfuegbar:
        m = lade_zelle(z)
        tabellen[(z["arm"], z["nfe"])] = m[m["complex"].isin(codes)].copy()

FARBE_OK, FARBE_FALSCH, FARBE_WAHL = "green", "grey60", "magenta"

for code in codes:
    prot = os.path.join(a.true, code, f"{code}_protein.pdb")
    kris = os.path.join(a.true, code, f"{code}_ligands.sdf")
    if not (os.path.isfile(prot) and os.path.isfile(kris)):
        print(f"  {code}: Referenz fehlt unter {a.true} -- uebersprungen")
        continue

    z = [f"# {code} -- erzeugt von pymol_szene.py",
         "#",
         "#   gruen    Pose trifft (RMSD < 2 A)",
         "#   grau     Pose trifft nicht",
         "#   magenta  die Pose, die der Mixed Score auswaehlt",
         "#   gelb     Kristallpose",
         "#",
         "# Alle Zellgruppen starten AUS. Eine einschalten, z. B.:",
         "#   enable sigmadock25*",
         "",
         "bg_color white",
         "set ray_opaque_background, 0",
         "set orthoscopic, 1",
         "",
         f'load {prot.replace(os.sep, "/")}, rezeptor',
         "hide everything, rezeptor",
         "show surface, rezeptor",
         "color grey90, rezeptor",
         "set transparency, 0.75, rezeptor",
         "",
         f'load {kris.replace(os.sep, "/")}, kristall',
         "show sticks, kristall",
         "color yellow, kristall",
         "set stick_radius, 0.22, kristall",
         ""]

    zusammenfassung = []
    for (arm, nfe), w in verfuegbar.items():
        t = tabellen[(arm, nfe)]
        g = t[t["complex"] == code]
        if g.empty:
            continue
        acc_von = dict(zip(g["seed"], g["acc"]))
        gewaehlt_seed = int(g.loc[g["heur"].idxmax(), "seed"])

        dateien = sorted(glob.glob(os.path.join(w, "seed_*", f"{code}__*.sdf")),
                         key=lambda f: int(re.search(r"seed_(\d+)", f).group(1)))
        # Ausduennen, aber die GEWAEHLTE Pose immer behalten -- sonst zeigt
        # die Szene eine Auswahl, die so nie getroffen wurde.
        schritt = max(1, len(dateien) // a.max_posen)
        nimm = dateien[::schritt][:a.max_posen]
        gew_datei = next((f for f in dateien
                          if int(re.search(r"seed_(\d+)", f).group(1)) == gewaehlt_seed), None)
        if gew_datei and gew_datei not in nimm:
            nimm.append(gew_datei)

        pre = f"{arm}{nfe}".lower()
        n_ok = n_bad = 0
        z.append(f"# --- {arm} NFE {nfe}: {len(nimm)} von {len(dateien)} Posen")
        for f in nimm:
            s = int(re.search(r"seed_(\d+)", f).group(1))
            if s == gewaehlt_seed:
                continue                      # kommt unten als eigenes Objekt
            gruppe = "ok" if acc_von.get(s, False) else "bad"
            n_ok += gruppe == "ok"
            n_bad += gruppe == "bad"
            z.append(f'load {f.replace(os.sep, "/")}, {pre}_{gruppe}_{s:03d}')
        for gruppe, farbe in (("ok", FARBE_OK), ("bad", FARBE_FALSCH)):
            z += [f"group {pre}_{gruppe}, {pre}_{gruppe}_*",
                  f"show lines, {pre}_{gruppe}",
                  f"color {farbe}, {pre}_{gruppe}",
                  f"set line_width, 1.2, {pre}_{gruppe}",
                  f"disable {pre}_{gruppe}"]
        if gew_datei:
            trifft = acc_von.get(gewaehlt_seed, False)
            z += ["",
                  f"# gewaehlte Pose (seed {gewaehlt_seed}), "
                  f"{'TRIFFT' if trifft else 'trifft NICHT'}",
                  f'load {gew_datei.replace(os.sep, "/")}, {pre}_gewaehlt',
                  f"show sticks, {pre}_gewaehlt",
                  f"color {FARBE_WAHL}, {pre}_gewaehlt",
                  f"set stick_radius, 0.16, {pre}_gewaehlt",
                  f"disable {pre}_gewaehlt"]
        z.append("")
        zusammenfassung.append((f"{arm} {nfe}", n_ok, n_bad, gewaehlt_seed,
                                bool(acc_von.get(gewaehlt_seed, False))))

    z += ["orient kristall", "zoom kristall, 6", "",
          "# Zum Durchschalten:"]
    for (arm, nfe) in verfuegbar:
        pre = f"{arm}{nfe}".lower()
        z.append(f"#   disable *_ok *_bad *_gewaehlt; enable {pre}_ok {pre}_bad {pre}_gewaehlt")

    ziel = os.path.join(_HIER, f"{code}_szene.pml")
    with open(ziel, "w", encoding="utf-8") as fh:
        fh.write("\n".join(z) + "\n")

    print(f"\n{code} -> {os.path.basename(ziel)}")
    print(f"  {'Zelle':<14}{'gruen':>7}{'grau':>7}   gewaehlte Pose")
    for name, ok, bad, seed, trifft in zusammenfassung:
        print(f"  {name:<14}{ok:7d}{bad:7d}   seed {seed:<4d} "
              f"{'TRIFFT' if trifft else 'trifft nicht'}")
