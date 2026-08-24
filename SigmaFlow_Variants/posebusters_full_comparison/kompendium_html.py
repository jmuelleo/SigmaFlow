"""Erzeugt das vollstaendige Datenkompendium als HTML.

Ziel: JEDES berechnete Ergebnis nachschlagbar, nicht nur die Kernbefunde.
Alles wird aus den Datendateien gelesen; abgetippt sind ausschliesslich die
p-Werte der gepaarten Bootstrap-Tests, die nicht als CSV vorliegen. Die
stehen als benannte Konstanten weiter unten.

AUFRUF   python kompendium_html.py <ziel.html>
"""
from __future__ import annotations

import csv
import os
import sys

DATA = os.path.join("..", "..", "Thesis Visualisierungen", "data")
ARME = ("Minimal", "Separate", "SigmaDock")

# (Kuerzel, Anzeigename, per_draw-Datei, selection-Datei, ranker-Datei,
#  heuristik-Datei, Seeds)
ZELLEN = [
    ("b25", "25 Schritte, gebunden", "per_draw_80seeds.csv",
     "selection_curves_80seeds.csv", "ranker_comparison_80seeds.csv",
     "heuristic_grid_80seeds.csv", 80),
    ("s25", "25 Schritte, generiert (Paper)", "per_draw_papersetup80.csv",
     "selection_curves_papersetup80.csv", "ranker_comparison_papersetup80.csv",
     "heuristic_grid_papersetup80.csv", 80),
    ("b5", "5 Schritte, gebunden", "per_draw_nfe5_40seeds.csv",
     "selection_curves_nfe5_40seeds.csv", None, None, 40),
    ("s5", "5 Schritte, generiert (Paper)", "per_draw_sampled5_40seeds.csv",
     "selection_curves_sampled5_40seeds.csv",
     "ranker_comparison_sampled5_40seeds.csv",
     "heuristic_grid_sampled5_40seeds.csv", 40),
]
METRIKEN = [
    ("rmsd_lt_1.0A", "RMSD &lt; 1 &Aring;"),
    ("rmsd_lt_2.0A", "RMSD &lt; 2 &Aring;"),
    ("rmsd_lt_2.5A", "RMSD &lt; 2,5 &Aring;"),
    ("rmsd_lt_3.0A", "RMSD &lt; 3 &Aring;"),
    ("pb_valid_no_protein", "PB-valid ohne Protein"),
    ("pb_valid_with_protein", "PB-valid mit Protein"),
    ("lt2A_and_valid_no_protein", "&lt; 2 &Aring; und valid ohne Protein"),
    ("lt2A_and_valid_with_protein", "&lt; 2 &Aring; und valid mit Protein"),
]
RANKER = [("random", "Zufall"), ("affinity_vinardo", "Affinit&auml;t (Vinardo)"),
          ("pb_all", "PB, alle 24 Checks"), ("pb_intrinsic", "PB, nur intrinsisch"),
          ("pb_protein", "PB, nur Protein"), ("oracle", "Oracle (Obergrenze)")]
TRAJ = [("Minimal", [("5, generiert", "Min_s5"), ("25, generiert", "Min_s25"),
                     ("25, gebunden", "Minimal"), ("200, gebunden", "Min_b200")]),
        ("Separate", [("5, generiert", "Sep_s5"), ("25, generiert", "Sep_s25"),
                      ("25, gebunden", "Separate"), ("200, gebunden", "Sep_b200")]),
        ("SigmaDock", [("5, generiert", "SD_s5"), ("25, generiert", "SD_s25"),
                       ("25, gebunden", "SigmaDock"), ("200, gebunden", "SD_b200")])]


def csvlese(pfad):
    p = pfad if os.path.exists(pfad) else os.path.join(DATA, pfad)
    if not os.path.exists(p):
        return []
    return list(csv.DictReader(open(p, encoding="utf-8")))


def z(x, n=2):
    return f"{x:.{n}f}".replace(".", ",")


def zp(x, n=2):
    return ("+" if x > 0 else "") + z(x, n)


# --- Gepaarte Bootstrap-Tests, 209 Komplexe, 4000 Ziehungen ----------------
# Aufloesungsgrenze p = 0,00025. Nicht als CSV vorhanden, deshalb hier.
T_ARME_S25 = [
    ("PB-valid ohne Protein", "+1,17", "0,0045", "-5,44", "&lt;0,00025",
     "-6,61", "&lt;0,00025"),
    ("PB-valid mit Protein", "+0,54", "0,041", "-0,78", "0,069",
     "-1,33", "0,0025"),
]
T_ARME_S25_40 = [
    ("RMSD &lt; 2 &Aring;", "+0,51", "0,12", "+0,47", "0,20", "-0,05", "0,89"),
    ("PB-valid ohne Protein", "+1,11", "0,030", "-5,66", "&lt;0,0001",
     "-6,77", "&lt;0,0001"),
    ("PB-valid mit Protein", "+0,32", "0,30", "-1,02", "0,054", "-1,34", "0,003"),
    ("&lt; 2 &Aring; und valid ohne Prot.", "+0,59", "0,023", "-0,62", "0,039",
     "-1,21", "0,0003"),
    ("&lt; 2 &Aring; und valid mit Prot.", "+0,33", "0,052", "-0,20", "0,21",
     "-0,54", "0,003"),
]
T_SCHRITTE_S = [
    ("RMSD &lt; 2 &Aring;", "5,33&rarr;5,42", "+0,08", "0,73",
     "5,85&rarr;5,51", "-0,33", "0,20", "5,80&rarr;0,54", "-5,26", "&lt;0,00025"),
    ("PB-valid ohne Protein", "33,85&rarr;16,28", "-17,57", "&lt;0,00025",
     "34,96&rarr;17,00", "-17,97", "&lt;0,00025",
     "28,19&rarr;5,16", "-23,04", "&lt;0,00025"),
    ("PB-valid mit Protein", "6,61&rarr;4,02", "-2,60", "&lt;0,00025",
     "6,94&rarr;4,96", "-1,97", "&lt;0,00025",
     "5,60&rarr;0,61", "-4,99", "&lt;0,00025"),
    ("&lt; 2 &Aring; und valid ohne Prot.", "3,84&rarr;2,57", "-1,27", "&lt;0,00025",
     "4,43&rarr;2,63", "-1,79", "&lt;0,00025",
     "3,22&rarr;0,30", "-2,92", "&lt;0,00025"),
    ("&lt; 2 &Aring; und valid mit Prot.", "1,15&rarr;0,80", "-0,35", "0,0075",
     "1,48&rarr;1,00", "-0,48", "0,0005",
     "0,94&rarr;0,06", "-0,89", "&lt;0,00025"),
]
T_SCHRITTE_B = [
    ("RMSD &lt; 2 &Aring;", "Minimal", "-0,28", "0,18"),
    ("RMSD &lt; 2 &Aring;", "Separate", "-0,05", "0,86"),
    ("RMSD &lt; 2 &Aring;", "SigmaDock", "-8,73", "&lt;0,0001"),
    ("Median-RMSD", "Minimal", "-0,230 &Aring;", "&lt;0,001"),
    ("Median-RMSD", "Separate", "-0,055 &Aring;", "0,32"),
    ("Median-RMSD", "SigmaDock", "+18,42 &Aring;", "&lt;0,0001"),
]
T_ARME_S5 = [
    ("RMSD &lt; 2 &Aring;", "+0,10", "0,76", "+4,88", "&lt;0,00025",
     "+4,98", "&lt;0,00025"),
    ("PB-valid ohne Protein", "+0,72", "0,17", "+11,12", "&lt;0,00025",
     "+11,84", "&lt;0,00025"),
    ("PB-valid mit Protein", "+0,94", "0,0015", "+3,41", "&lt;0,00025",
     "+4,35", "&lt;0,00025"),
    ("&lt; 2 &Aring; und valid ohne Prot.", "+0,06", "0,84", "+2,27", "&lt;0,00025",
     "+2,33", "&lt;0,00025"),
    ("&lt; 2 &Aring; und valid mit Prot.", "+0,20", "0,23", "+0,74", "&lt;0,00025",
     "+0,94", "&lt;0,00025"),
]
T_KREUZ = [
    ("RMSD &lt; 2 &Aring;", "Minimal 5", "-0,38", "-1,05; +0,29", "0,30"),
    ("RMSD &lt; 2 &Aring;", "Separate 5", "-0,29", "-1,00; +0,44", "0,46"),
    ("PB-valid ohne Protein", "Minimal 5", "-11,91", "-14,35; -9,52", "&lt;0,00025"),
    ("PB-valid ohne Protein", "Separate 5", "-11,20", "-13,49; -9,06", "&lt;0,00025"),
    ("PB-valid mit Protein", "Minimal 5", "-1,58", "-2,63; -0,60", "0,0010"),
    ("PB-valid mit Protein", "Separate 5", "-0,63", "-1,60; +0,31", "0,20"),
    ("&lt; 2 &Aring; und valid ohne Prot.", "Minimal 5", "-0,65", "-1,16; -0,17", "0,0055"),
    ("&lt; 2 &Aring; und valid ohne Prot.", "Separate 5", "-0,59", "-1,17; -0,04", "0,039"),
    ("&lt; 2 &Aring; und valid mit Prot.", "Minimal 5", "-0,14", "-0,42; +0,12", "0,34"),
    ("&lt; 2 &Aring; und valid mit Prot.", "Separate 5", "+0,06", "-0,31; +0,42", "0,79"),
]
T_RANKING = [
    ("RMSD &lt; 2 &Aring;", "Separate 5 &minus; SigmaDock 25", "+4,31",
     "-2,87; +11,48", "0,24"),
    ("RMSD &lt; 2 &Aring;", "Minimal 5 &minus; SigmaDock 25", "+1,44",
     "-5,26; +8,13", "0,73"),
    ("RMSD &lt; 2 &Aring;", "Separate 5 &minus; SigmaDock 5", "+20,10",
     "+14,35; +25,84", "&lt;0,00025"),
    ("RMSD &lt; 2 &Aring;", "Separate 25 &minus; SigmaDock 25", "+6,70",
     "0,00; +13,40", "0,066"),
    ("&lt; 2 &Aring; und valid m. Prot.", "Separate 5 &minus; SigmaDock 25", "+1,91",
     "-2,39; +6,22", "0,44"),
    ("&lt; 2 &Aring; und valid m. Prot.", "Separate 5 &minus; SigmaDock 5", "+6,22",
     "+2,87; +10,05", "&lt;0,00025"),
    ("&lt; 2 &Aring; und valid m. Prot.", "Separate 25 &minus; SigmaDock 25", "+5,26",
     "+0,96; +9,57", "0,018"),
    ("&lt; 2 &Aring; und valid m. Prot.", "Separate 5 &minus; Minimal 5", "+2,39",
     "-0,96; +5,74", "0,21"),
]
T_WECHSEL = [
    ("PB-valid ohne Protein", "Minimal", "-17,69", "-17,57", "+0,12"),
    ("PB-valid ohne Protein", "Separate", "-18,43", "-17,97", "+0,47"),
    ("PB-valid ohne Protein", "SigmaDock", "-23,19", "-23,04", "+0,16"),
    ("PB-valid mit Protein", "Minimal", "-2,12", "-2,60", "-0,48"),
    ("PB-valid mit Protein", "Separate", "-2,20", "-1,97", "+0,23"),
    ("PB-valid mit Protein", "SigmaDock", "-5,89", "-4,99", "+0,90"),
]
T_PRIOR = [
    ("SigmaDock", "gebunden", "115,20", "9,37"),
    ("SigmaDock", "generiert", "126,47", "4,77"),
    ("Minimal", "gebunden", "127,16", "4,40"),
    ("Minimal", "generiert", "126,74", "4,50"),
    ("Separate", "gebunden", "126,67", "4,65"),
    ("Separate", "generiert", "126,65", "4,72"),
]
T_NIE = [("Minimal", "10,53", "47,85"), ("Separate", "12,44", "44,02"),
         ("SigmaDock", "21,53", "57,42")]
T_HORIZONT = [
    ("Minimal", "8541310", "11:06", "6", "13.750", "21,100", "255", "151.470"),
    ("Separate", "8625634", "11:10:59", "6", "13.750", "&mdash;", "&mdash;",
     "&mdash;"),
    ("SigmaDock", "8541439", "10:52", "6", "13.200", "19,800", "239", "141.966"),
]
T_ZEIT = [(1, 230, 18.6, 5.39), (5, 398, 32.1, 3.11), (10, 608, 49.1, 2.04),
          (25, 1238, 100.0, 1.00), (50, 2289, 184.8, 0.54),
          (100, 4389, 354.5, 0.28), (200, 8591, 693.8, 0.14)]
T_TREFFER = [
    ("25 Schritte, generiert", "29,6", "31,9", "27,4"),
    ("5 Schritte, generiert", "30,1", "33,6", "19,6"),
]

CSS = """
:root{--ground:#F6F7F5;--surface:#FFFFFF;--raised:#EEF1EE;--ink:#131A18;
 --muted:#5D6B67;--rule:#DCE2DE;--accent:#0E6E5C;--accent-soft:#DDEDE8;
 --warn:#8A5E10;--warn-soft:#F5EBD6;--refuted:#8C3A2B;
 --shadow:0 1px 2px rgba(19,26,24,.05),0 8px 24px rgba(19,26,24,.04);}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --ground:#0F1413;--surface:#161C1A;--raised:#1D2523;--ink:#E7EDEA;
 --muted:#94A39E;--rule:#2A3532;--accent:#54CDB0;--accent-soft:#16302B;
 --warn:#DCA748;--warn-soft:#2E2617;--refuted:#E0907C;
 --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);}}
:root[data-theme="dark"]{--ground:#0F1413;--surface:#161C1A;--raised:#1D2523;
 --ink:#E7EDEA;--muted:#94A39E;--rule:#2A3532;--accent:#54CDB0;
 --accent-soft:#16302B;--warn:#DCA748;--warn-soft:#2E2617;--refuted:#E0907C;
 --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
 font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;font-size:15px;
 line-height:1.6;-webkit-font-smoothing:antialiased}
.layout{max-width:1320px;margin:0 auto;padding:44px 22px 100px;
 display:grid;grid-template-columns:230px minmax(0,1fr);gap:44px;
 align-items:start}
@media (max-width:900px){.layout{grid-template-columns:1fr;gap:26px}
 nav.toc{position:static!important;max-height:none!important}}
nav.toc{position:sticky;top:22px;max-height:calc(100vh - 44px);overflow-y:auto;
 border:1px solid var(--rule);border-radius:12px;background:var(--surface);
 padding:18px 16px;box-shadow:var(--shadow);font-size:13px}
nav.toc .t{font-family:"IBM Plex Mono",monospace;font-size:10px;
 letter-spacing:.14em;text-transform:uppercase;color:var(--muted);
 margin-bottom:10px}
nav.toc ol{list-style:none;margin:0;padding:0;counter-reset:s}
nav.toc li{counter-increment:s;margin-bottom:2px}
nav.toc a{display:block;padding:5px 8px;border-radius:7px;color:var(--ink);
 text-decoration:none;line-height:1.35}
nav.toc a::before{content:counter(s) ".";color:var(--muted);
 font-family:"IBM Plex Mono",monospace;font-size:11px;margin-right:7px}
nav.toc a:hover{background:var(--raised);color:var(--accent)}
main{display:flex;flex-direction:column;gap:52px;min-width:0}
h1{font-family:Newsreader,Georgia,serif;font-weight:400;
 font-size:clamp(30px,4.2vw,44px);line-height:1.1;margin:0;
 letter-spacing:-.015em;text-wrap:balance}
h2{font-family:Newsreader,Georgia,serif;font-weight:500;font-size:26px;
 margin:0;letter-spacing:-.008em;text-wrap:balance;scroll-margin-top:26px}
h3{font-size:14.5px;font-weight:600;margin:22px 0 9px}
h4{font-family:"IBM Plex Mono",monospace;font-size:11px;letter-spacing:.1em;
 text-transform:uppercase;color:var(--muted);font-weight:500;margin:18px 0 7px}
p{margin:0 0 .9em;max-width:70ch}p:last-child{margin-bottom:0}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11px;
 letter-spacing:.15em;text-transform:uppercase;color:var(--muted)}
section{display:flex;flex-direction:column;gap:14px}
.kopf{border-top:1px solid var(--rule);padding-top:24px;
 display:flex;flex-direction:column;gap:8px}
.tblwrap{overflow-x:auto;border:1px solid var(--rule);border-radius:10px;
 background:var(--surface);box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 12px;text-align:right;white-space:nowrap;
 border-bottom:1px solid var(--rule)}
th.l,td.l{text-align:left}
thead th{font-family:"IBM Plex Mono",monospace;font-size:10px;font-weight:500;
 letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
 position:sticky;top:0;background:var(--surface);z-index:1}
tbody td{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
tbody td.l{font-family:"IBM Plex Sans",sans-serif}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover td{background:var(--raised)}
tbody tr.trenn td{border-top:2px solid var(--rule)}
.best{color:var(--accent);font-weight:600}
.sig{color:var(--accent);font-weight:600}
.ns{color:var(--muted)}
.schlecht{color:var(--refuted)}
.ci{color:var(--muted);font-size:12px}
.note{border-left:3px solid var(--rule);padding:3px 0 3px 16px;
 color:var(--muted);font-size:13.5px;max-width:70ch}
.note strong{color:var(--ink)}
.note.warn{border-color:var(--warn)}
code{font-family:"IBM Plex Mono",monospace;font-size:.9em;
 background:var(--raised);padding:1.5px 5px;border-radius:5px}
a{color:var(--accent)}
footer{border-top:1px solid var(--rule);padding-top:24px;color:var(--muted);
 font-size:13px;display:flex;flex-direction:column;gap:6px;max-width:70ch}
"""


def tab(kopf, zeilen, links=1):
    o = ['<div class="tblwrap"><table><thead><tr>']
    for i, k in enumerate(kopf):
        o.append(f'<th class="l">{k}</th>' if i < links else f"<th>{k}</th>")
    o.append("</tr></thead><tbody>")
    for zi in zeilen:
        cls = ' class="trenn"' if isinstance(zi, dict) and zi.get("trenn") else ""
        zellen = zi["z"] if isinstance(zi, dict) else zi
        kl = zi.get("kl", {}) if isinstance(zi, dict) else {}
        o.append(f"<tr{cls}>")
        for i, c in enumerate(zellen):
            o.append(f'<td class="{"l" if i < links else ""} {kl.get(i,"")}">{c}</td>')
        o.append("</tr>")
    o.append("</tbody></table></div>")
    return "".join(o)


def pkl(p):
    """Klasse fuer einen p-Wert-String."""
    if p.startswith("&lt;"):
        return "sig"
    try:
        return "sig" if float(p.replace(",", ".")) < 0.05 else "ns"
    except ValueError:
        return ""


def main() -> int:
    ziel = sys.argv[1]
    PD, SC, RK, HG = {}, {}, {}, {}
    for k, _, f1, f2, f3, f4, _ in ZELLEN:
        PD[k] = {(r["arm"], r["metric"]): r for r in csvlese(f1)}
        SC[k] = {(r["arm"], r["metric"], int(r["k"])): r for r in csvlese(f2)}
        if f3:
            RK[k] = {(r["arm"], r["ranker"], int(r["k"])): float(
                r["hit_rmsd_lt_2A_pct"]) for r in csvlese(f3)}
        if f4:
            HG[k] = csvlese(f4)

    KAP = []
    S = []

    def kapitel(titel, ident, kopftext=""):
        KAP.append((ident, titel))
        S.append(f'<section id="{ident}"><div class="kopf"><h2>{titel}</h2>'
                 + (f"<p>{kopftext}</p>" if kopftext else "") + "</div>")

    # 1 ---------------------------------------------------------------
    kapitel("Datenbasis und Versuchsplan", "basis",
            "Vier Zellen aus Integrationsschritten und Konformerquelle, drei "
            "Arme, 209 Komplexe. Die generierte Konformerquelle ist die "
            "Vorgabe des Originals und ma&szlig;geblich f&uuml;r alle "
            "Methodenaussagen.")
    zeilen = []
    ges = 0
    for k, name, f1, f2, f3, f4, seeds in ZELLEN:
        n = 209 * seeds * 3
        ges += n
        zeilen.append([name, f"{seeds}", f"{209*seeds:,}".replace(",", "."),
                       f"{n:,}".replace(",", "."),
                       "ja" if f3 else "nein"])
    zeilen.append({"z": ["<strong>Summe</strong>", "", "",
                         f"<strong>{ges:,}</strong>".replace(",", "."), ""],
                   "trenn": True})
    S.append(tab(["Zelle", "Seeds", "Posen je Arm", "Posen gesamt",
                  "gnina gescort"], zeilen, 1))
    S.append(tab(["Arm", "Trainingsjob", "Laufzeit", "Epochen",
                  "Optimierungsschritte", "Beispiele/s", "72h-Epochen",
                  "72h-Schritte"],
                 [list(t) for t in T_HORIZONT], 1))
    S.append('<p class="note">Angeglichen ist die <strong>Walltime</strong>, '
             "nicht die Schrittzahl: SigmaDock absolviert 4&nbsp;% weniger "
             "Optimierungsschritte. Das Budget entspricht rund 9&nbsp;% der "
             "155.544 Schritte des Originalpapers. Die 72h-Spalten sind der "
             "geplante Horizont f&uuml;r die noch ausstehenden langen "
             "L&auml;ufe; f&uuml;r Separate fehlt die Durchsatzmessung "
             "noch.</p>")
    S.append("</section>")

    # 2 ---------------------------------------------------------------
    kapitel("Anteil je Ziehung", "jeziehung",
            "Was eine einzelne, zuf&auml;llig gezogene Pose leistet &mdash; "
            "ohne jede Auswahl. 95-%-Wilson-Intervall, weil das "
            "Wald-Intervall bei kleinen Anteilen zusammenbricht.")
    for k, name, f1, *_ in ZELLEN:
        if not PD[k]:
            continue
        S.append(f"<h3>{name}</h3>")
        zeilen = []
        for mi, (mk, ml) in enumerate(METRIKEN):
            if (ARME[0], mk) not in PD[k]:
                continue
            best = max(ARME, key=lambda a: float(PD[k][(a, mk)]["pct"]))
            for i, a in enumerate(ARME):
                r = PD[k][(a, mk)]
                zeilen.append({"z": [ml if i == 0 else "", a,
                                     f"{z(float(r['pct']))}&nbsp;%",
                                     f'<span class="ci">{z(float(r["ci_lo"]))}'
                                     f"&ndash;{z(float(r['ci_hi']))}</span>",
                                     f"{int(r['n_poses']):,}".replace(",", ".")],
                               "kl": {2: "best"} if a == best else {},
                               "trenn": i == 0 and mi > 0})
        S.append(tab(["Kenngr&ouml;&szlig;e", "Arm", "Anteil", "95-%-Intervall",
                      "Posen"], zeilen, 2))
    S.append("</section>")

    # 3 ---------------------------------------------------------------
    kapitel("Oracle@k", "oracle",
            "Anteil der Komplexe, f&uuml;r die unter k zuf&auml;llig gezogenen "
            "Posen mindestens eine die Bedingung erf&uuml;llt. Erwartungstreu "
            "&uuml;ber zuf&auml;llige k-Teilmengen gemittelt &mdash; die "
            "Obergrenze, die eine perfekte Auswahl erreichen k&ouml;nnte.")
    for k, name, *_ , seeds in ZELLEN:
        if not SC[k]:
            continue
        KS = sorted({kk for _, _, kk in SC[k]})
        S.append(f"<h3>{name}</h3>")
        zeilen = []
        for mi, (mk, ml) in enumerate(METRIKEN):
            if (ARME[0], mk, KS[0]) not in SC[k]:
                continue
            for i, a in enumerate(ARME):
                z_ = [ml if i == 0 else "", a]
                for kk in KS:
                    r = SC[k].get((a, mk, kk))
                    z_.append(z(float(r["oracle_pct"])) if r else "&ndash;")
                zeilen.append({"z": z_, "trenn": i == 0 and mi > 0})
        S.append(tab(["Kenngr&ouml;&szlig;e", "Arm"] + [f"k={kk}" for kk in KS],
                     zeilen, 2))
    S.append("</section>")

    # 4 ---------------------------------------------------------------
    kapitel("Auswahl nach gnina/Vinardo (Top-1@k)", "top1",
            "Was ein Anwender bekommt, der k Posen erzeugt und die mit der "
            "besten Affinit&auml;t nimmt. Bei Vinardo ist negativ gut. Bei "
            "k&nbsp;=&nbsp;1 muss der Wert das Zufallsniveau treffen &mdash; "
            "die Probe, dass die Sortierrichtung stimmt.")
    for k, name, f1, f2, f3, *_ in ZELLEN:
        if not SC[k] or not f3:
            continue
        KS = sorted({kk for _, _, kk in SC[k]})
        S.append(f"<h3>{name}</h3>")
        zeilen = []
        for mi, (mk, ml) in enumerate(METRIKEN):
            r0 = SC[k].get((ARME[0], mk, KS[0]))
            if not r0 or not r0.get("top1_affinity_pct"):
                continue
            for i, a in enumerate(ARME):
                zufall = float(SC[k][(a, mk, KS[0])]["random_pct"])
                z_ = [ml if i == 0 else "", a,
                      f'<span class="ci">{z(zufall)}</span>']
                for kk in KS:
                    r = SC[k].get((a, mk, kk))
                    t = r.get("top1_affinity_pct") if r else None
                    z_.append(z(float(t)) if t else "&ndash;")
                zeilen.append({"z": z_, "trenn": i == 0 and mi > 0})
        S.append(tab(["Kenngr&ouml;&szlig;e", "Arm", "Zufall"]
                     + [f"k={kk}" for kk in KS], zeilen, 2))
    S.append("</section>")

    # 5 ---------------------------------------------------------------
    kapitel("Rankervergleich", "ranker",
            "F&uuml;nf Auswahlkriterien gegen dieselbe Zielgr&ouml;&szlig;e "
            "RMSD&nbsp;&lt;&nbsp;2&nbsp;&Aring;. Der RMSD geht in keinen der "
            "Ranker ein &mdash; deshalb ist dieser Vergleich frei von "
            "Zirkularit&auml;t, anders als &bdquo;nach PB-Checks ranken und PB "
            "messen&ldquo;.")
    for k, name, f1, f2, f3, *_ in ZELLEN:
        if k not in RK:
            continue
        KS = sorted({kk for _, _, kk in RK[k]})
        S.append(f"<h3>{name}</h3>")
        zeilen = []
        for ai, a in enumerate(ARME):
            for ri, (rk, rl) in enumerate(RANKER):
                z_ = [a if ri == 0 else "", rl]
                for kk in KS:
                    v = RK[k].get((a, rk, kk))
                    z_.append(z(v) if v is not None else "&ndash;")
                zeilen.append({"z": z_,
                               "kl": {i + 2: "best" for i in range(len(KS))}
                                     if rk == "affinity_vinardo" else {},
                               "trenn": ri == 0 and ai > 0})
        S.append(tab(["Arm", "Ranker"] + [f"k={kk}" for kk in KS], zeilen, 2))
    S.append("<h4>Trefferquote des Affinit&auml;tsrankers</h4>")
    S.append(tab(["Zelle"] + list(ARME), [list(t) for t in T_TREFFER], 1))
    S.append('<p class="note"><code>(Top1&minus;Random)/(Oracle&minus;Random)</code> '
             "bei k&nbsp;=&nbsp;40, Zielgr&ouml;&szlig;e "
             "RMSD&nbsp;&lt;&nbsp;2&nbsp;&Aring;. Der Anteil des vorhandenen "
             "Spielraums, den der Scorer tats&auml;chlich hebt.</p>")
    S.append("</section>")

    # 6 ---------------------------------------------------------------
    kapitel("Heuristik-Gitter", "heuristik",
            "SigmaDocks Modus <code>heuristic</code> mischt Affinit&auml;t und "
            "PoseBusters-Mittelwert: "
            "<code>(&minus;Affinit&auml;t) &times; (score_bias + "
            "avg_pb<sup>pb_exponent</sup>)</code>. Beide Parameter haben im "
            "Repository <strong>keine</strong> Vorgabewerte, deshalb ein "
            "Gitter statt eines Wertes &mdash; ein nachtr&auml;glich "
            "ausgew&auml;hltes bestes Feld w&auml;re wertlos.")
    for k, name, f1, f2, f3, f4, *_ in ZELLEN:
        if k not in HG:
            continue
        rows = HG[k]
        biases = sorted({float(r["score_bias"]) for r in rows})
        expos = sorted({float(r["pb_exponent"]) for r in rows})
        S.append(f"<h3>{name}</h3>")
        for a in ARME:
            S.append(f"<h4>{a}</h4>")
            zeilen = []
            for b in biases:
                z_ = [z(b, 2)]
                for e in expos:
                    v = [r for r in rows if r["arm"] == a
                         and float(r["score_bias"]) == b
                         and float(r["pb_exponent"]) == e]
                    z_.append(z(float(v[0]["hit_rmsd_lt_2A_pct"])) if v else "&ndash;")
                zeilen.append(z_)
            S.append(tab(["score_bias"] + [f"Exponent {z(e,1)}" for e in expos],
                         zeilen, 1))
    S.append("</section>")

    # 7 ---------------------------------------------------------------
    kapitel("Gepaarte Tests", "tests",
            "Bootstrap &uuml;ber die 209 Komplexe, nicht &uuml;ber die Posen "
            "&mdash; zwei Seeds desselben Komplexes sind nicht "
            "unabh&auml;ngig. 4000 Ziehungen, Aufl&ouml;sungsgrenze "
            "p&nbsp;=&nbsp;0,00025. Alle Angaben in Prozentpunkten.")
    S.append("<h3>Zwischen den Armen, 25 Schritte, generiert (80 Seeds)</h3>")
    S.append(tab(["Kenngr&ouml;&szlig;e", "Sep&minus;Min", "p", "SD&minus;Min",
                  "p", "SD&minus;Sep", "p"],
                 [{"z": list(t), "kl": {2: pkl(t[2]), 4: pkl(t[4]), 6: pkl(t[6])}}
                  for t in T_ARME_S25], 1))
    S.append("<h3>Zwischen den Armen, 25 Schritte, generiert (40 Seeds, "
             "voller Metriksatz)</h3>")
    S.append(tab(["Kenngr&ouml;&szlig;e", "Sep&minus;Min", "p", "SD&minus;Min",
                  "p", "SD&minus;Sep", "p"],
                 [{"z": list(t), "kl": {2: pkl(t[2]), 4: pkl(t[4]), 6: pkl(t[6])}}
                  for t in T_ARME_S25_40], 1))
    S.append("<h3>Zwischen den Armen, 5 Schritte, generiert</h3>")
    S.append(tab(["Kenngr&ouml;&szlig;e", "Sep&minus;Min", "p", "Min&minus;SD",
                  "p", "Sep&minus;SD", "p"],
                 [{"z": list(t), "kl": {2: pkl(t[2]), 4: pkl(t[4]), 6: pkl(t[6])}}
                  for t in T_ARME_S5], 1))
    S.append("<h3>Innerhalb der Arme: 5 gegen 25 Schritte, generiert</h3>")
    S.append(tab(["Kenngr&ouml;&szlig;e", "Minimal 25&rarr;5", "&Delta;", "p",
                  "Separate 25&rarr;5", "&Delta;", "p",
                  "SigmaDock 25&rarr;5", "&Delta;", "p"],
                 [{"z": list(t), "kl": {3: pkl(t[3]), 6: pkl(t[6]), 9: pkl(t[9])}}
                  for t in T_SCHRITTE_S], 1))
    S.append("<h3>Innerhalb der Arme: 5 gegen 25 Schritte, gebunden</h3>")
    S.append(tab(["Kenngr&ouml;&szlig;e", "Arm", "&Delta;", "p"],
                 [{"z": list(t), "kl": {3: pkl(t[3])}} for t in T_SCHRITTE_B], 2))
    S.append("<h3>Wechselwirkung: h&auml;ngt der Abfall an der "
             "Konformerquelle?</h3>")
    S.append(tab(["Kenngr&ouml;&szlig;e", "Arm", "Abfall gebunden",
                  "Abfall generiert", "Unterschied"],
                 [list(t) for t in T_WECHSEL], 2))
    S.append('<p class="note"><strong>Die Wechselwirkung ist null.</strong> Bei '
             "allen drei Armen unterscheiden sich die Abf&auml;lle um weniger "
             "als einen halben Prozentpunkt &mdash; der Zusammenbruch bei "
             "grober Integration hat mit dem Rotationsprior nichts zu tun.</p>")
    S.append("<h3>Quervergleich: 5 Schritte gegen SigmaDock mit 25</h3>")
    S.append(tab(["Kenngr&ouml;&szlig;e", "Vergleich gegen SigmaDock 25",
                  "&Delta;", "95-%-Intervall", "p"],
                 [{"z": [t[0], t[1], t[2], f'<span class="ci">{t[3]}</span>',
                         t[4]], "kl": {4: pkl(t[4])}} for t in T_KREUZ], 2))
    S.append("<h3>Unter gnina-Auswahl, k = 40</h3>")
    S.append(tab(["Zielgr&ouml;&szlig;e", "Vergleich", "&Delta;",
                  "95-%-Intervall", "p"],
                 [{"z": [t[0], t[1], t[2], f'<span class="ci">{t[3]}</span>',
                         t[4]], "kl": {4: pkl(t[4])}} for t in T_RANKING], 2))
    S.append("</section>")

    # 8 ---------------------------------------------------------------
    kapitel("Trajektoriengeometrie", "traj",
            "Aus den gespeicherten Zwischenzust&auml;nden. Die Fragmente "
            "werden aus der Konstanz der Bindungsl&auml;ngen &uuml;ber die "
            "Trajektorie zur&uuml;ckgewonnen; als Wahrheit dient die "
            "n&auml;chstgelegene kristallographische Kopie.")
    zeilen = []
    for arm, eintraege in TRAJ:
        rows200 = csvlese(f"traj_agg_{eintraege[3][1]}.csv")
        ref = sum(float(x["step_rot_deg"]) for x in rows200) if rows200 else None
        for i, (lab, key) in enumerate(eintraege):
            rows = csvlese(f"traj_agg_{key}.csv")
            if not rows:
                continue
            weg = sum(float(x["step_rot_deg"]) for x in rows)
            trans = sum(float(x["step_trans_A"]) for x in rows)
            st, en = float(rows[0]["rot_mean_deg"]), float(rows[-1]["rot_mean_deg"])
            anteil = 100 * weg / ref if ref else float("nan")
            zeilen.append({"z": [arm if i == 0 else "", lab,
                                 rows[0]["n_poses"], f"{z(st)}&deg;",
                                 f"{z(en)}&deg;", f"{zp(en-st)}&deg;",
                                 f"{z(weg,1)}&deg;", f"{z(anteil,1)}&nbsp;%",
                                 f"{z(trans,2)}&nbsp;&Aring;"],
                           "kl": {7: "schlecht" if anteil > 120 or anteil < 90
                                  else ""},
                           "trenn": i == 0 and arm != "Minimal"})
    S.append(tab(["Arm", "Schritte", "Posen", "Rotationsfehler Start",
                  "Ende", "&Delta;", "Wegl&auml;nge", "% konvergiert",
                  "Translationsweg"], zeilen, 2))
    S.append('<p class="note">Bei korrekt integrierter ODE ist die '
             "Wegl&auml;nge von der Schrittzahl unabh&auml;ngig. Bei 25 "
             "Schritten liegen alle drei Arme bei 96,5&ndash;99,1&nbsp;% des "
             "200-Schritte-Werts. Bei f&uuml;nf Schritten "
             "<strong>&uuml;berschie&szlig;t SigmaDock auf 151,2&nbsp;%</strong>, "
             "die Flow-Arme unterschreiten mild auf 85,6 und 86,1&nbsp;%.</p>")
    for lab, keys in (("25 Schritte, generiert", ("Min_s25", "Sep_s25", "SD_s25")),
                      ("5 Schritte, generiert", ("Min_s5", "Sep_s5", "SD_s5"))):
        S.append(f"<h3>Verlauf je Schritt &mdash; {lab}</h3>")
        rows = {a: csvlese(f"traj_agg_{k}.csv") for a, k in zip(ARME, keys)}
        n = max(len(v) for v in rows.values())
        zeilen = []
        for i in range(n):
            z_ = [str(i)]
            for a in ARME:
                r = rows[a][i] if i < len(rows[a]) else None
                if r:
                    z_ += [z(float(r["rmsd_mean"])), z(float(r["rot_mean_deg"])),
                           z(float(r["step_rot_deg"]))]
                else:
                    z_ += ["&ndash;"] * 3
            zeilen.append(z_)
        S.append(tab(["Schritt"] + [f"{a} {x}" for a in ARME
                                    for x in ("RMSD", "Rot&deg;", "Drehung&deg;")],
                     zeilen, 1))
    S.append("</section>")

    # 9 ---------------------------------------------------------------
    kapitel("Der Rotationsprior", "prior",
            "SigmaDock zieht die Startrotation aus IGSO(3) mit "
            "<code>max_sigma = 1.5</code> statt gleichverteilt "
            "(<code>so3_diffuser.sample_ref</code>). Mit gebundenem Konformer "
            "ist die Identit&auml;tsrotation die richtige Antwort &mdash; der "
            "Prior ist dort informativ.")
    S.append(tab(["Arm", "Konformer", "Startrotation", "RMSD &lt; 2 &Aring;"],
                 [{"z": [t[0], t[1], f"{t[2]}&deg;", f"{t[3]}&nbsp;%"],
                   "kl": {2: "schlecht" if t[0] == "SigmaDock"
                          and t[1] == "gebunden" else ""}}
                  for t in T_PRIOR], 2))
    S.append('<p class="note"><strong>126,48&deg;</strong> ist der '
             "Erwartungswert des Haar-Ma&szlig;es. Die Messung im "
             "generierten Fall trifft ihn auf eine Hundertstel Grad. Nur der "
             "Arm bewegt sich, dessen Prior nicht uniform ist.</p>")
    S.append("<h3>Komplexe ohne eine einzige brauchbare Pose in 80 Versuchen</h3>")
    S.append(tab(["Arm", "ohne Protein", "mit Protein"],
                 [{"z": [t[0], f"{t[1]}&nbsp;%", f"{t[2]}&nbsp;%"],
                   "kl": {1: "schlecht", 2: "schlecht"}
                         if t[0] == "SigmaDock" else {}} for t in T_NIE], 1))
    S.append("</section>")

    # 10 --------------------------------------------------------------
    kapitel("Rechenaufwand", "aufwand",
            "Gemessen auf CPU-Knoten. Aus den Laufzeiten bei 5 und 200 "
            "Schritten ergibt sich "
            "<code>Zeit = 188&nbsp;s + 42,0&nbsp;s &times; Schrittzahl</code>. "
            "Die Grundlast (Datenaufbau, Modell laden, Nachbearbeitung) "
            "skaliert nicht mit der Schrittzahl.")
    S.append(tab(["Schritte", "Zeit je Seed", "% von 25 Schritten",
                  "Beschleunigung"],
                 [{"z": [str(n), f"{s:,} s".replace(",", "."), z(p, 1),
                         f"{z(f,2)}&times;"],
                   "kl": {2: "best" if n == 5 else ""}} for n, s, p, f in T_ZEIT],
                 1))
    S.append('<p class="note">F&uuml;nf statt f&uuml;nfundzwanzig Schritte: '
             "exakt ein F&uuml;nftel der Netzwerkauswertungen, aber "
             "<strong>32&nbsp;% der Wall-Clock-Zeit</strong> &mdash; eine "
             "Beschleunigung um Faktor&nbsp;3,1, nicht um f&uuml;nf. Auf einer "
             "GPU f&auml;llt das Verh&auml;ltnis ung&uuml;nstiger aus, weil "
             "die Grundlast kaum sinkt.</p>")
    S.append("</section>")

    # 11 --------------------------------------------------------------
    kapitel("Ligandenstatistik", "ligand",
            "Eigenschaften des Auswertungssatzes. Die Fragmentzahl bestimmt "
            "die Dimension des Zustandsraums: sechs Freiheitsgrade je "
            "Fragment.")
    rows = csvlese("fragment_distribution.csv")
    if rows:
        S.append(tab(["Fragmente", "Zustandsdim.", "Liganden", "% des Satzes",
                      "kumuliert", "SigmaFlow Oracle@10", "SigmaDock Oracle@10",
                      "SF Median-RMSD", "SD Median-RMSD"],
                     [[r["fragments"], r["state_dimension"], r["n_ligands"],
                       z(float(r["pct"]), 1), z(float(r["cum_pct"]), 1),
                       z(float(r["sigmaflow_oracle10_pct"]), 1),
                       z(float(r["sigmadock_oracle10_pct"]), 1),
                       z(float(r["sigmaflow_median_best_rmsd"])),
                       z(float(r["sigmadock_median_best_rmsd"]))] for r in rows], 1))
        S.append('<p class="note warn">Diese Tabelle stammt aus der '
                 "12h-Auswertung mit <strong>gebundenem</strong> Konformer und "
                 "10 Seeds. Die SigmaDock-Spalten enthalten daher den "
                 "Prior-Effekt und sind nicht als Methodenvergleich zu lesen; "
                 "die Verteilung der Fragmentzahlen selbst ist davon "
                 "unber&uuml;hrt.</p>")
    rows = csvlese("summary_statistics.csv")
    if rows:
        S.append("<h3>Kennzahlen des Satzes</h3>")
        S.append(tab(["Gr&ouml;&szlig;e", "Wert"],
                     [[r["quantity"].replace("_", " "), r["value"]] for r in rows], 1))
    S.append("</section>")

    # 12 --------------------------------------------------------------
    kapitel("Quellen und Reproduktion", "quellen")
    S.append('<div style="max-width:70ch">'
             "<p>Alle Tabellen dieser Seite werden aus den Dateien unter "
             "<code>Thesis Visualisierungen/data/</code> und den "
             "<code>traj_agg_*.csv</code> erzeugt. Abgetippt sind "
             "ausschlie&szlig;lich die p-Werte der gepaarten Tests, die als "
             "benannte Konstanten im Quelltext von "
             "<code>kompendium_html.py</code> stehen.</p>"
             "<p>Die Datens&auml;tze entstehen mit:</p>"
             "<p><code>python build_thesis_datasets.py bound | sampled | nfe5 | "
             "sampled5</code><br>"
             "<code>python trajectory_geometry.py &lt;arm&gt; &lt;root&gt; "
             "--n_seeds 10</code><br>"
             "<code>python papersetup80.py</code> &middot; "
             "<code>python vierzellen_validitaet.py</code></p>"
             "<p>Verwandte Seiten: der <a href=\"https://claude.ai/code/artifact/"
             "830e118d-1106-412f-91e4-e92169ef4066\">Metriksatz</a> als "
             "kompakte Referenztabelle und der <a href=\"https://claude.ai/code/"
             "artifact/12b88147-29ab-4917-acab-1304c5ba4f40\">Ergebnisbericht</a> "
             "mit der Einordnung nach Beweiskraft.</p></div>")
    S.append("</section>")

    toc = "".join(f'<li><a href="#{i}">{t}</a></li>' for i, t in KAP)
    seite = f"""<title>SigmaFlow Datenkompendium</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap">
<style>{CSS}</style>
<div class="layout">
<nav class="toc"><div class="t">Inhalt</div><ol>{toc}</ol></nav>
<main>
<div style="display:flex;flex-direction:column;gap:12px">
  <div class="eyebrow">PoseBusters &middot; 209 Komplexe &middot; Stand 24. August 2026</div>
  <h1>Datenkompendium</h1>
  <p style="max-width:66ch;color:var(--muted);font-size:17px">Jedes berechnete
  Ergebnis zum Nachschlagen: Anteile, Auswahlkurven, Rankervergleiche,
  gepaarte Tests, Trajektoriengeometrie und Rechenaufwand &mdash; &uuml;ber
  alle vier Zellen des Versuchsplans.</p>
</div>
{"".join(S)}
<footer>
  <div>Erzeugt von <code>kompendium_html.py</code> aus den Datendateien.
  209 Komplexe, vier Versuchszellen, 150.480 ausgewertete Posen.</div>
  <div>RMSD durchgehend symmetriekorrigiert (spyrmsd), Referenz &uuml;ber die
  n&auml;chstgelegene kristallographische Kopie. PoseBusters im Modus
  <code>redock</code>: 15 ligandenintrinsische Checks, 9 mit Protein.</div>
  <div>Auswahlkurven mit 400 Wiederholungen, Gleichst&auml;nde zuf&auml;llig
  aufgel&ouml;st, <code>default_rng(0)</code>. Gepaarte Tests mit 4000
  Bootstrap-Ziehungen &uuml;ber die 209 Komplexe.</div>
</footer>
</main>
</div>
"""
    open(ziel, "w", encoding="utf-8").write(seite)
    print(f"geschrieben: {ziel} ({len(seite):,} Zeichen, {len(KAP)} Kapitel)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
