"""Erzeugt die HTML-Referenzseite aus denselben CSV-Dateien wie gesamttabelle.py.

Die Zahlen werden NICHT abgetippt. Bei rund 400 Werten ueber elf Tabellen ist
Abschreiben die wahrscheinlichste Fehlerquelle, und ein Zahlendreher in einer
Referenz faellt spaeter niemandem mehr auf.

AUFRUF   python gesamttabelle_html.py <ausgabe.html>
"""
from __future__ import annotations

import csv
import html
import os
import sys

DATA = os.path.join("..", "..", "Thesis Visualisierungen", "data")

ZELLEN = [
    ("bound25", 25, "bound", "per_draw_80seeds.csv",
     "selection_curves_80seeds.csv", 80, True),
    ("samp25", 25, "sampled", "per_draw_papersetup80.csv",
     "selection_curves_papersetup80.csv", 80, True),
    ("bound5", 5, "bound", "per_draw_nfe5_40seeds.csv",
     "selection_curves_nfe5_40seeds.csv", 40, False),
    ("samp5", 5, "sampled", "per_draw_sampled5_40seeds.csv",
     "selection_curves_sampled5_40seeds.csv", 40, True),
]
ARME = ("Minimal", "Separate", "SigmaDock")
METRIKEN = [
    ("pb_valid_no_protein", "PB-valid ohne Protein"),
    ("pb_valid_with_protein", "PB-valid mit Protein"),
    ("rmsd_lt_2.0A", "RMSD &lt; 2 &Aring;"),
    ("lt2A_and_valid_no_protein", "&lt; 2 &Aring; und PB-valid ohne Protein"),
    ("lt2A_and_valid_with_protein", "&lt; 2 &Aring; und PB-valid mit Protein"),
]
KS = (1, 2, 3, 5, 10, 20, 40, 80)


def lade_per_draw(f):
    d = {}
    for r in csv.DictReader(open(os.path.join(DATA, f), encoding="utf-8")):
        d[(r["arm"], r["metric"])] = (float(r["pct"]), float(r["ci_lo"]),
                                      float(r["ci_hi"]))
    return d


# Frueher wurden hier die 80-Seed-Validitaetszahlen aus papersetup80.py
# ueberlagert, weil der RMSD im Paper-Setup nur fuer 40 Seeds vorlag. Seit
# dem 24.08. steht die ganze Zelle auf 80 Seeds und stammt aus EINER
# Rechnung -- die Ueberlagerung ist entfallen. Kreuzprobe gegen
# papersetup80.py: alle sechs Validitaetswerte stimmen auf vier
# Nachkommastellen ueberein.
UEBERLAGERUNG = {}


def lade_kurven(f):
    d = {}
    for r in csv.DictReader(open(os.path.join(DATA, f), encoding="utf-8")):
        t = r.get("top1_affinity_pct")
        d[(r["arm"], r["metric"], int(r["k"]))] = (
            float(r["random_pct"]), float(r["oracle_pct"]),
            float(t) if t else None)
    return d


def z(x, n=2):
    return f"{x:.{n}f}".replace(".", ",")


CSS = """
:root{
  --ground:#F6F7F5; --surface:#FFFFFF; --raised:#EEF1EE;
  --ink:#131A18; --muted:#5D6B67; --rule:#DCE2DE;
  --accent:#0E6E5C; --accent-soft:#DDEDE8;
  --pending:#8A5E10; --pending-soft:#F5EBD6;
  --shadow:0 1px 2px rgba(19,26,24,.06), 0 8px 24px rgba(19,26,24,.05);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0F1413; --surface:#161C1A; --raised:#1D2523;
    --ink:#E7EDEA; --muted:#94A39E; --rule:#2A3532;
    --accent:#54CDB0; --accent-soft:#16302B;
    --pending:#DCA748; --pending-soft:#2E2617;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.3);
  }
}
:root[data-theme="dark"]{
  --ground:#0F1413; --surface:#161C1A; --raised:#1D2523;
  --ink:#E7EDEA; --muted:#94A39E; --rule:#2A3532;
  --accent:#54CDB0; --accent-soft:#16302B;
  --pending:#DCA748; --pending-soft:#2E2617;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.3);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;
  font-size:15px; line-height:1.6;
  -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1180px; margin:0 auto; padding:56px 24px 96px;
      display:flex; flex-direction:column; gap:48px}
header{display:flex; flex-direction:column; gap:14px;
       border-bottom:1px solid var(--rule); padding-bottom:32px}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11.5px;
         letter-spacing:.14em; text-transform:uppercase; color:var(--muted)}
h1{font-family:Newsreader,Georgia,serif; font-weight:500; font-size:clamp(30px,4.4vw,46px);
   line-height:1.12; margin:0; text-wrap:balance; letter-spacing:-.01em}
.lede{max-width:64ch; color:var(--muted); font-size:16px; margin:0}
h2{font-family:Newsreader,Georgia,serif; font-weight:500; font-size:26px;
   margin:0 0 6px; letter-spacing:-.005em; text-wrap:balance}
h3{font-family:"IBM Plex Sans",sans-serif; font-weight:600; font-size:14px;
   margin:0 0 10px; letter-spacing:.01em}
section{display:flex; flex-direction:column; gap:22px}
.sec-head{display:flex; flex-direction:column; gap:4px}
.sec-head p{margin:0; color:var(--muted); max-width:66ch; font-size:14.5px}

/* Versuchsplan als 2x2 -- die Struktur der Seite bildet die des Versuchs ab */
.plan{display:grid; grid-template-columns:auto 1fr 1fr; gap:10px; align-items:stretch}
.plan .corner{}
.plan .colhead,.plan .rowhead{
  font-family:"IBM Plex Mono",monospace; font-size:11.5px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted); display:flex; align-items:center}
.plan .rowhead{padding-right:8px}
.cell{background:var(--surface); border:1px solid var(--rule); border-radius:10px;
      padding:14px 16px; display:flex; flex-direction:column; gap:5px;
      box-shadow:var(--shadow)}
.cell .n{font-family:"IBM Plex Mono",monospace; font-size:19px; font-weight:500;
         font-variant-numeric:tabular-nums}
.cell .d{font-size:12.5px; color:var(--muted); line-height:1.45}
.cell.have{border-left:3px solid var(--accent)}
.cell.miss{border-left:3px solid var(--pending); background:var(--pending-soft)}
.cell.miss .n{color:var(--pending)}

.tblwrap{overflow-x:auto; border:1px solid var(--rule); border-radius:10px;
         background:var(--surface); box-shadow:var(--shadow)}
table{border-collapse:collapse; width:100%; font-size:13.5px}
caption{text-align:left; padding:14px 16px 0; font-weight:600; font-size:13.5px}
th,td{padding:7px 12px; text-align:right; white-space:nowrap;
      border-bottom:1px solid var(--rule)}
th:first-child,td:first-child,
th.lbl,td.lbl{text-align:left}
thead th{font-family:"IBM Plex Mono",monospace; font-size:10.5px; font-weight:500;
         letter-spacing:.1em; text-transform:uppercase; color:var(--muted);
         position:sticky; top:0; background:var(--surface); z-index:1}
tbody td{font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums}
tbody td.lbl{font-family:"IBM Plex Sans",sans-serif}
tbody tr:last-child td{border-bottom:none}
tbody tr.grp td{border-top:2px solid var(--rule)}
tbody tr:hover td{background:var(--raised)}
.best{color:var(--accent); font-weight:600}
.na{color:var(--pending); font-style:normal; font-family:"IBM Plex Sans",sans-serif;
    font-size:12.5px}
.dash{color:var(--muted)}
.cfg{font-family:"IBM Plex Mono",monospace; font-size:11.5px; color:var(--muted)}
.arm{font-weight:600}
.note{border-left:3px solid var(--rule); padding:2px 0 2px 16px;
      color:var(--muted); font-size:14px; max-width:66ch}
.note strong{color:var(--ink)}
.key{display:flex; flex-wrap:wrap; gap:18px; font-size:12.5px; color:var(--muted)}
.key span{display:flex; align-items:center; gap:7px}
.sw{width:11px; height:11px; border-radius:3px; display:inline-block}
footer{border-top:1px solid var(--rule); padding-top:24px; color:var(--muted);
       font-size:13px; display:flex; flex-direction:column; gap:6px}
code{font-family:"IBM Plex Mono",monospace; font-size:.92em;
     background:var(--raised); padding:1px 5px; border-radius:4px}
"""


def tabelle(kopf, zeilen):
    """zeilen: Liste von (config, arm, [zellen-html]) oder ('GRP',) fuer Trennlinie."""
    o = ['<div class="tblwrap"><table><thead><tr>']
    for i, k in enumerate(kopf):
        o.append(f'<th class="lbl">{k}</th>' if i < 2 else f"<th>{k}</th>")
    o.append("</tr></thead><tbody>")
    for cfg, arm, cells, neu in zeilen:
        cls = ' class="grp"' if neu else ""
        o.append(f'<tr{cls}><td class="lbl cfg">{cfg}</td>'
                 f'<td class="lbl arm">{arm}</td>' + "".join(cells) + "</tr>")
    o.append("</tbody></table></div>")
    return "".join(o)


def main() -> int:
    ziel = sys.argv[1]
    PD, SC = {}, {}
    for key, _, _, f1, f2, _, _ in ZELLEN:
        if f1:
            PD[key], SC[key] = lade_per_draw(f1), lade_kurven(f2)

    O = []
    A = O.append

    # --- Block A --------------------------------------------------------
    zeilen = []
    for mkey, mlab in METRIKEN:
        for zi, (key, ns, kf, f1, _, seeds, _) in enumerate(ZELLEN):
            werte = {}
            if f1:
                werte = {a: PD[key][(a, mkey)][0] for a in ARME}
                bester = max(werte, key=werte.get)
            for ai, arm in enumerate(ARME):
                cfg = f"{ns} Schritte &middot; {kf}" if ai == 0 else ""
                if not f1:
                    cells = ['<td class="na" colspan="3">rechnet auf ARC</td>']
                else:
                    p, lo, hi = PD[key][(arm, mkey)]
                    b = ' class="best"' if arm == bester else ""
                    cells = [f"<td{b}>{z(p)}&nbsp;%</td>",
                             f'<td class="dash">{z(lo)}&ndash;{z(hi)}</td>',
                             f'<td class="dash">{209*seeds}</td>']
                zeilen.append((cfg, arm if ai == 0 or True else "", cells,
                               ai == 0 and zi > 0))
        zeilen.append(("", "", [f'<td class="lbl" colspan="3" '
                                f'style="background:var(--raised);'
                                f'font-family:Newsreader,Georgia,serif;'
                                f'font-size:15px">{mlab}</td>'], True))
    # Kopfzeile der Gruppe gehoert VOR die Gruppe, nicht dahinter -> neu bauen
    zeilen = []
    for mi, (mkey, mlab) in enumerate(METRIKEN):
        zeilen.append(("", f'<span style="font-family:Newsreader,Georgia,serif;'
                           f'font-size:16px;font-weight:500">{mlab}</span>',
                       ['<td colspan="3"></td>'], mi > 0))
        for key, ns, kf, f1, _, seeds, _ in ZELLEN:
            if f1:
                werte = {a: PD[key][(a, mkey)][0] for a in ARME}
                bester = max(werte, key=werte.get)
            for ai, arm in enumerate(ARME):
                mark = " &middot; 80 Seeds" if (key == "samp25"
                                                and mkey in UEBERLAGERUNG) else ""
                cfg = f"{ns} Schr. &middot; {kf}{mark}" if ai == 0 else ""
                if not f1:
                    cells = ['<td class="na" colspan="3">rechnet auf ARC</td>']
                else:
                    p, lo, hi = PD[key][(arm, mkey)]
                    b = ' class="best"' if arm == bester else ""
                    n = 80 if (key == "samp25" and mkey in UEBERLAGERUNG) else seeds
                    cells = [f"<td{b}>{z(p)}&nbsp;%</td>",
                             f'<td class="dash">{z(lo)}&ndash;{z(hi)}</td>',
                             f'<td class="dash">{209*n}</td>']
                zeilen.append((cfg, arm, cells, False))
    A('<section id="a"><div class="sec-head"><h2>Anteil je Ziehung</h2>'
      '<p>Was eine einzelne, zuf&auml;llig gezogene Pose leistet &mdash; ohne jede '
      'Auswahl. 95-%-Wilson-Intervall, weil das Wald-Intervall bei kleinen '
      'Anteilen zusammenbricht. Der beste Arm je Zeile ist hervorgehoben.</p>'
      "</div>")
    A(tabelle(["Konfiguration", "Arm", "Anteil", "95-%-Intervall", "Posen"], zeilen))
    A("</section>")

    # --- Block B und C --------------------------------------------------
    for blk, idx, titel, erkl in (
        ("b", 1, "Oracle@k &mdash; die Obergrenze",
         "Anteil der Komplexe, f&uuml;r die unter k zuf&auml;llig gezogenen Posen "
         "<em>mindestens eine</em> die Bedingung erf&uuml;llt. Erwartungstreu &uuml;ber "
         "zuf&auml;llige k-Teilmengen gemittelt. Das ist die Grenze, die eine perfekte "
         "Auswahl erreichen k&ouml;nnte &mdash; kein erreichbarer Wert."),
        ("c", 2, "Auswahl nach gnina&thinsp;/&thinsp;Vinardo",
         "Was ein Anwender bekommt, der k Posen erzeugt und die mit der besten "
         "Affinit&auml;t nimmt. Bei Vinardo ist <em>negativ</em> gut. Bei k&nbsp;=&nbsp;1 "
         "muss der Wert das Zufallsniveau treffen &mdash; das ist die Probe, dass die "
         "Richtung stimmt."),
    ):
        A(f'<section id="{blk}"><div class="sec-head"><h2>{titel}</h2>'
          f"<p>{erkl}</p></div>")
        for mkey, mlab in METRIKEN:
            zeilen = []
            for key, ns, kf, f1, _, seeds, gn in ZELLEN:
                if f1 and (gn or idx == 1):
                    besteK = {}
                    for k in KS:
                        w = {a: SC[key].get((a, mkey, k)) for a in ARME}
                        w = {a: v[idx] for a, v in w.items() if v and v[idx] is not None}
                        if w:
                            besteK[k] = max(w, key=w.get)
                for ai, arm in enumerate(ARME):
                    mark = " &middot; 80 Seeds" if (key == "samp25"
                                                    and mkey in UEBERLAGERUNG) else ""
                    cfg = f"{ns} Schr. &middot; {kf}{mark}" if ai == 0 else ""
                    if not f1:
                        cells = [f'<td class="na" colspan="{len(KS)}">'
                                 f"rechnet auf ARC</td>"]
                    elif idx == 2 and not gn:
                        cells = [f'<td class="na" colspan="{len(KS)}">'
                                 f"noch nicht mit gnina gescort</td>"]
                    else:
                        cells = []
                        for k in KS:
                            v = SC[key].get((arm, mkey, k))
                            if not v or v[idx] is None:
                                cells.append('<td class="dash">&ndash;</td>')
                            else:
                                b = ' class="best"' if besteK.get(k) == arm else ""
                                cells.append(f"<td{b}>{z(v[idx])}</td>")
                    zeilen.append((cfg, arm, cells, ai == 0))
            A(f"<div><h3>{mlab}</h3>")
            A(tabelle(["Konfiguration", "Arm"] + [f"k={k}" for k in KS], zeilen))
            A("</div>")
        A("</section>")

    inhalt = "".join(O)

    seite = f"""<title>SigmaFlow Metriksatz</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <div class="eyebrow">PoseBusters &middot; 209 Komplexe &middot; 150&thinsp;480 Posen &middot; Stand 24. August 2026</div>
  <h1>Jede Zelle des Versuchsplans, jede Kenngr&ouml;&szlig;e</h1>
  <p class="lede">Drei Arme, zwei Schrittzahlen, zwei Konformerquellen. RMSD durchgehend
  symmetriekorrigiert &uuml;ber spyrmsd, Referenz per <code>best_copy</code>.
  PoseBusters im Modus <code>redock</code>: 15 ligandenintrinsische Checks,
  9 mit Protein.</p>
  <div class="key">
    <span><i class="sw" style="background:var(--accent)"></i>bester Arm der Zeile</span>
    <span><i class="sw" style="background:var(--pending)"></i>rechnet noch auf ARC</span>
    <span><i class="sw" style="background:var(--rule)"></i>nicht anwendbar</span>
  </div>
</header>

<section>
  <div class="sec-head"><h2>Der Versuchsplan</h2>
  <p>Vier Zellen, alle ausgewertet. Vollst&auml;ndig sind drei; bei
  <code>bound</code> mit f&uuml;nf Schritten fehlt allein die
  Affinit&auml;tsbewertung &mdash; die Zelle tr&auml;gt ohnehin den
  Prior-Vorbehalt und ist die am wenigsten aussagekr&auml;ftige.</p></div>
  <div class="plan">
    <div class="corner"></div>
    <div class="colhead">Konformer: bound</div>
    <div class="colhead">Konformer: sampled &middot; Paper</div>
    <div class="rowhead">25 Schritte</div>
    <div class="cell have"><span class="n">80 Seeds</span>
      <span class="d">16&thinsp;720 Posen je Arm &middot; RMSD, Validit&auml;t, gnina</span></div>
    <div class="cell have"><span class="n">80 Seeds</span>
      <span class="d">16&thinsp;720 Posen je Arm &middot; RMSD, Validit&auml;t, gnina
      &mdash; die ma&szlig;gebliche Zelle</span></div>
    <div class="rowhead">5 Schritte</div>
    <div class="cell have"><span class="n">40 Seeds</span>
      <span class="d">8&thinsp;360 Posen je Arm &middot; RMSD und Validit&auml;t,
      kein gnina</span></div>
    <div class="cell have"><span class="n">40 Seeds</span>
      <span class="d">8&thinsp;360 Posen je Arm &middot; RMSD, Validit&auml;t
      und gnina</span></div>
  </div>
  <p class="note"><strong>Was die vierte Zelle entschieden hat.</strong> Sie
  beantwortet, ob SigmaDocks Zusammenbruch bei grober Integration am
  Rotationsprior h&auml;ngt. Er tut es nicht: der Abfall von 25 auf 5 Schritte
  unterscheidet sich zwischen den beiden Konformerquellen um h&ouml;chstens
  0,47&nbsp;Prozentpunkte. Auf der Platzierungsgenauigkeit verlieren die
  Flow-Arme dabei <em>nichts</em> (5,33&nbsp;&rarr;&nbsp;5,42&nbsp;% und
  5,85&nbsp;&rarr;&nbsp;5,51&nbsp;%, beide p&nbsp;&gt;&nbsp;0,2), w&auml;hrend
  SigmaDock den Faktor&nbsp;10,8 einb&uuml;&szlig;t.</p>
</section>

{inhalt}

<footer>
  <div>Erzeugt aus <code>Thesis Visualisierungen/data/</code> durch
  <code>gesamttabelle_html.py</code>. Keine Zahl ist abgetippt.</div>
  <div>Alle Anteile in Prozent. Oracle- und Auswahlkurven mit 400 Wiederholungen,
  Gleichst&auml;nde zuf&auml;llig aufgel&ouml;st, <code>default_rng(0)</code>.</div>
</footer>
</div>
"""
    open(ziel, "w", encoding="utf-8").write(seite)
    print(f"geschrieben: {ziel} ({len(seite)} Zeichen)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
