#!/usr/bin/env python3
"""
Thesis-Abbildungen zur Fragmentierung der Liganden.

DREI ABBILDUNGEN

  A  Verteilung der Fragmentzahl ueber den Datensatz.
     Die Grundgroesse: die Zustandsdimension der Generierung ist D = 6F.

  B  Fragmentzahl gegen Ligandgroesse (Atome, Torsionen).
     Zeigt, wie stark F ueberhaupt durch die Molekuelgroesse bestimmt ist --
     und wo der Merge-Schritt zusammenfasst.

  C  Fragmentzahl gegen Dockingerfolg, beide Arme nebeneinander.
     Die eigentliche Frage: wird es mit mehr Fragmenten schwerer, und trifft
     das beide Verfahren gleich?

WAS BEWUSST NICHT PASSIERT
    Es wird nicht interpoliert und nichts geglaettet. Liganden ohne Messwert
    fehlen in der Abbildung und werden im Titel als Anzahl genannt. Bins mit
    wenigen Liganden bekommen ihre Fallzahl an die Saeule geschrieben -- bei
    n=1 ist ein Balken keine Rate, sondern eine Anekdote.

Aufruf:
    # nur Verteilung
    python -m visualization.plot_fragments --counts fragment_counts_posebusters.csv \
        --out-dir figures/

    # mit Erfolgsverknuepfung (per_complex_csv aus evaluate_run --per_complex_csv)
    python -m visualization.plot_fragments --counts fragment_counts_posebusters.csv \
        --perf sigmaflow=per_complex_SF.csv --perf sigmadock=per_complex_SD.csv \
        --out-dir figures/
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Zurueckhaltende, druckbare Palette. Ein Farbton je Arm, sonst Graustufen.
C_BAR = "#7BA7C7"
C_BAR_EDGE = "#3E6C8E"
C_SF = "#C1666B"
C_SD = "#4F8A8B"
C_GRID = "#DDDDDD"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


# Beschriftungen in beiden Sprachen. Die Thesis ist englisch, die Arbeitslogs
# sind deutsch -- beide Fassungen entstehen aus denselben Daten, damit keine
# Zahl beim Uebersetzen von Hand wandert.
L = {
 "de": {
  "ligands":"Liganden", "frags":"Fragmente je Ligand", "median":"Median",
  "mean":"Mittel", "atleast":"Anteil $\\geq$ k  [%]",
  "statedim":"Zustandsdimension  $D = 6F$",
  "title_a":"Fragmente je Ligand", "nomeas":"ohne Messwert",
  "heavy":"Schweratome", "tors":"Torsionsbindungen",
  "title_b":"Fragmentzahl gegen Ligandgroesse", "fragshort":"Fragmente",
  "jitter":"y-Jitter $\\pm$0.18 zur Sichtbarkeit",
  "succ":"Oracle@10  RMSD < 2 $\\AA$  [%]",
  "title_c1":"Erfolg gegen Fragmentzahl",
  "bestrmsd":"Median bester RMSD  [$\\AA$]",
  "title_c2":"Bester RMSD gegen Fragmentzahl",
  "openc":"Offene Kreise: weniger als {n} Liganden im Bin — Einzelfaelle, keine Rate.",
  "rigid":"{n} Liganden sind ein einziger starrer Koerper ($D = 6$)",
  "cumlab":"kumuliert  [%]",
 },
 "en": {
  "ligands":"Ligands", "frags":"Fragments per ligand", "median":"Median",
  "mean":"Mean", "atleast":"Fraction $\\geq$ k  [%]",
  "statedim":"State dimension  $D = 6F$",
  "title_a":"Fragments per ligand", "nomeas":"not measured",
  "heavy":"Heavy atoms", "tors":"Rotatable bonds",
  "title_b":"Fragment count vs. ligand size", "fragshort":"Fragments",
  "jitter":"y-jitter $\\pm$0.18 for visibility",
  "succ":"Oracle@10  RMSD < 2 $\\AA$  [%]",
  "title_c1":"Success vs. fragment count",
  "bestrmsd":"Median best RMSD  [$\\AA$]",
  "title_c2":"Best RMSD vs. fragment count",
  "openc":"Open circles: fewer than {n} ligands in bin — individual cases, not a rate.",
  "rigid":"{n} ligands are a single rigid body ($D = 6$)",
  "cumlab":"cumulative  [%]",
 },
}
T = L["de"]          # wird in main() gesetzt


def read_counts(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r.get("fragments"):
                continue          # Zeitlimit oder Parse-Fehler: bleibt leer
            rows.append({
                "complex": r["complex"],
                "fragments": int(r["fragments"]),
                "atoms": int(r["atoms"]) if r.get("atoms") else None,
                "torsions": int(r["torsions"]) if r.get("torsions") else None,
            })
    return rows


def read_perf(path: Path) -> dict[str, dict]:
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["complex"]] = {
                "rmsd_seed0": float(r["rmsd_seed0"]),
                "rmsd_best": float(r["rmsd_best"]),
                "success_2A_seed0": int(r["success_2A_seed0"]),
                "success_2A_oracle": int(r["success_2A_oracle"]),
            }
    return out


def read_joined(path: Path):
    """Liest die bereits verknuepfte fragments_vs_performance.csv.

    Damit ist der Abbildungsordner aus EINER Datei reproduzierbar; die beiden
    Ausgangstabellen muessen nicht mitgeliefert werden.
    """
    rows, perf = [], {"sigmaflow": {}, "sigmadock": {}}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r.get("fragments"):
                continue
            cid = r["complex"]
            rows.append({"complex": cid, "fragments": int(r["fragments"]),
                         "atoms": int(r["atoms"]) if r.get("atoms") else None,
                         "torsions": int(r["torsions"]) if r.get("torsions") else None})
            for arm in ("sigmaflow", "sigmadock"):
                if r.get(f"{arm}_rmsd_best"):
                    perf[arm][cid] = {
                        "rmsd_seed0": float(r[f"{arm}_rmsd_seed0"]),
                        "rmsd_best": float(r[f"{arm}_rmsd_best"]),
                        "success_2A_seed0": 0,
                        "success_2A_oracle": int(r[f"{arm}_success_2A_oracle"]),
                    }
    return rows, {k: v for k, v in perf.items() if v}


def fig_a_distribution(rows: list[dict], out: Path, n_missing: int,
                       set_label: str = "", cum_style: str = "panel") -> None:
    frags = [r["fragments"] for r in rows]
    lo, hi = min(frags), max(frags)
    bins = np.arange(lo - 0.5, hi + 1.5, 1)

    # Zwei Bauweisen fuer die kumulative Kurve. "panel" trennt sie sauber ab und
    # ist bei zwoelf Saeulen lesbarer; "twin" legt sie auf eine zweite y-Achse
    # ueber die Saeulen, wie in der Abbildungsbeschreibung vorgesehen.
    if cum_style == "twin":
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        ax2 = None
    else:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.2, 4.6), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1],
                                                   "hspace": 0.12})

    counts, _, patches = ax.hist(frags, bins=bins, color=C_BAR,
                                 edgecolor=C_BAR_EDGE, linewidth=0.8)
    for c, p in zip(counts, patches):
        if c:
            ax.text(p.get_x() + p.get_width() / 2, c + 0.6, f"{int(c)}",
                    ha="center", va="bottom", fontsize=7.5, color="#444444")

    # Median und Mittel als Linien MIT Legende statt als Text im Feld -- inline
    # gesetzte Beschriftungen ueberlagern bei diesem Wertebereich die Saeulen.
    med = float(np.median(frags))
    mean = float(np.mean(frags))
    ax.axvline(med, color="#333333", lw=1.2, ls="-", zorder=3,
               label=f'{T["median"]} {med:.0f}')
    ax.axvline(mean, color="#333333", lw=1.2, ls=":", zorder=3,
               label=f'{T["mean"]} {mean:.2f}')
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_ylim(0, max(counts) * 1.18)

    ax.set_ylabel(T["ligands"])
    title = f'{T["title_a"]}'
    if set_label:
        title += f' — {set_label}'
    title += f'  (n = {len(rows)})'
    if n_missing:
        title += f'   [{n_missing} {T["nomeas"]}]'
    ax.set_title(title, loc="left", fontsize=10)

    # Einzelfragment-Liganden: der interessanteste Rand der Verteilung, weil
    # dort D = 6 ist und die Zerlegung gar nichts zu tun hat. Ohne Hinweis
    # verschwindet die Saeule neben den grossen.
    if lo == 1:
        n1 = int(counts[0])
        ax.annotate(T["rigid"].format(n=n1),
                    xy=(1, n1), xytext=(1.9, max(counts) * 0.72),
                    fontsize=7.5, color="#444444", ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8,
                                    shrinkA=0, shrinkB=3))

    # Die Zustandsdimension gehoert nach OBEN. Unten kollidiert sie mit der
    # Achsenbeschriftung des kumulativen Panels.
    top = ax.secondary_xaxis("top", functions=(lambda x: 6 * x, lambda x: x / 6))
    top.set_xticks([6 * k for k in range(lo, hi + 1)])
    top.set_xlabel(T["statedim"], fontsize=8, labelpad=4)
    top.tick_params(labelsize=7)

    order = np.sort(frags)
    cum = np.arange(1, len(order) + 1) / len(order)
    if cum_style == "twin":
        # Zweite y-Achse rechts: kumulativer Anteil <= k, an den Saeulenmitten.
        axr = ax.twinx()
        ks = np.arange(lo, hi + 1)
        pct = [100 * np.mean(np.asarray(frags) <= k) for k in ks]
        axr.plot(ks, pct, color="#333333", lw=1.3, marker="o", ms=3, zorder=4)
        axr.set_ylabel(T["cumlab"], fontsize=8)
        axr.set_ylim(0, 105)
        axr.set_yticks([0, 25, 50, 75, 100])
        axr.grid(False)
        axr.spines["right"].set_visible(True)
        ax.set_xlabel(T["frags"])
        ax.set_xticks(range(lo, hi + 1))
    else:
        # Kumulativ als schmales Panel darunter -- "wie selten ist >= k".
        ax2.step(order, 100 * (1 - cum), where="post", color=C_BAR_EDGE, lw=1.3)
        ax2.set_ylabel(T["atleast"], fontsize=8)
        ax2.set_xlabel(T["frags"])
        ax2.set_xticks(range(lo, hi + 1))
        ax2.set_ylim(0, 100)
        ax2.set_yticks([0, 50, 100])

    fig.savefig(out / "A_fragment_distribution.png")
    fig.savefig(out / "A_fragment_distribution.pdf")
    plt.close(fig)
    print(f"  A  {out / 'A_fragment_distribution.png'}")


def fig_b_size(rows: list[dict], out: Path) -> None:
    have = [r for r in rows if r["atoms"] and r["torsions"] is not None]
    if not have:
        print("  B  uebersprungen (keine Atom-/Torsionsspalten)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, key, label in ((axes[0], "atoms", T["heavy"]),
                           (axes[1], "torsions", T["tors"])):
        x = np.array([r[key] for r in have], dtype=float)
        y = np.array([r["fragments"] for r in have], dtype=float)
        # Jitter nur auf der diskreten Achse, damit ueberlagerte Punkte sichtbar
        # bleiben. Der Betrag steht in der Legende, damit niemand ihn fuer
        # Messrauschen haelt.
        rng = np.random.default_rng(0)
        ax.scatter(x, y + rng.uniform(-0.18, 0.18, size=y.size), s=11,
                   color=C_BAR_EDGE, alpha=0.55, linewidths=0)
        ax.set_xlabel(label)
        ax.set_ylabel(T["fragshort"] if key == "atoms" else "")
        r = float(np.corrcoef(x, y)[0, 1])
        ax.set_title(f"Pearson r = {r:.2f}", loc="left", fontsize=9)
    axes[1].text(0.98, 0.03, T["jitter"],
                 transform=axes[1].transAxes, ha="right", fontsize=6.5,
                 color="#777777")
    fig.suptitle(T["title_b"], x=0.02, y=1.06, ha="left", fontsize=10)
    fig.savefig(out / "B_fragments_vs_size.png")
    fig.savefig(out / "B_fragments_vs_size.pdf")
    plt.close(fig)
    print(f"  B  {out / 'B_fragments_vs_size.png'}")


def fig_c_performance(rows: list[dict], perf: dict[str, dict[str, dict]],
                      out: Path, min_n: int = 3) -> None:
    if not perf:
        print("  C  uebersprungen (kein --perf uebergeben)")
        return
    frag_of = {r["complex"]: r["fragments"] for r in rows}
    colors = {"sigmaflow": C_SF, "sigmadock": C_SD}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    all_f = sorted({f for f in frag_of.values()})

    for arm, table in perf.items():
        by_f = defaultdict(list)
        for cid, m in table.items():
            if cid in frag_of:
                by_f[frag_of[cid]].append(m)
        col = colors.get(arm, "#666666")

        xs, rate, ns, med = [], [], [], []
        for f in all_f:
            ms = by_f.get(f, [])
            if not ms:
                continue
            xs.append(f)
            ns.append(len(ms))
            rate.append(100 * np.mean([m["success_2A_oracle"] for m in ms]))
            med.append(float(np.median([m["rmsd_best"] for m in ms])))

        solid = [i for i, n in enumerate(ns) if n >= min_n]
        ax1.plot([xs[i] for i in solid], [rate[i] for i in solid], "-o",
                 color=col, ms=4, lw=1.5, label=arm)
        thin = [i for i, n in enumerate(ns) if n < min_n]
        ax1.plot([xs[i] for i in thin], [rate[i] for i in thin], "o",
                 color=col, ms=4, mfc="white", lw=0)
        for i in thin:
            ax1.annotate(f"n={ns[i]}", (xs[i], rate[i]), fontsize=6,
                         xytext=(0, 5), textcoords="offset points",
                         ha="center", color=col)

        ax2.plot([xs[i] for i in solid], [med[i] for i in solid], "-o",
                 color=col, ms=4, lw=1.5, label=arm)
        ax2.plot([xs[i] for i in thin], [med[i] for i in thin], "o",
                 color=col, ms=4, mfc="white", lw=0)

    ax1.set_xlabel(T["frags"])
    ax1.set_ylabel(T["succ"])
    ax1.set_ylim(bottom=0)
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title(T["title_c1"], loc="left", fontsize=9)

    ax2.axhline(2.0, color="#999999", lw=0.9, ls="--")
    ax2.text(ax2.get_xlim()[1], 2.0, " 2 $\\AA$", fontsize=7,
             va="center", color="#777777")
    ax2.set_xlabel(T["frags"])
    ax2.set_ylabel(T["bestrmsd"])
    ax2.set_title(T["title_c2"], loc="left", fontsize=9)

    fig.text(0.02, -0.04, T["openc"].format(n=min_n), fontsize=6.5, color="#777777")
    fig.savefig(out / "C_fragments_vs_performance.png")
    fig.savefig(out / "C_fragments_vs_performance.pdf")
    plt.close(fig)
    print(f"  C  {out / 'C_fragments_vs_performance.png'}")


def write_join(rows: list[dict], perf: dict[str, dict[str, dict]], out: Path) -> None:
    """Eine Zeile je Ligand: Fragmentzahl neben dem Erfolg beider Arme.

    Das ist die Tabelle, mit der sich gezielt einzelne Liganden heraussuchen
    lassen -- die wenigfragmentierten ebenso wie die schwierigen.
    """
    arms = sorted(perf)
    dest = out / "fragments_vs_performance.csv"
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        cols = ["complex", "fragments", "state_dimension", "atoms", "torsions"]
        for a in arms:
            cols += [f"{a}_rmsd_seed0", f"{a}_rmsd_best", f"{a}_success_2A_oracle"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (-x["fragments"], x["complex"])):
            row = {"complex": r["complex"], "fragments": r["fragments"],
                   "state_dimension": 6 * r["fragments"],
                   "atoms": r["atoms"], "torsions": r["torsions"]}
            for a in arms:
                m = perf[a].get(r["complex"])
                if m:
                    row[f"{a}_rmsd_seed0"] = f"{m['rmsd_seed0']:.3f}"
                    row[f"{a}_rmsd_best"] = f"{m['rmsd_best']:.3f}"
                    row[f"{a}_success_2A_oracle"] = m["success_2A_oracle"]
            w.writerow(row)
    print(f"  CSV {dest}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", type=Path, default=None,
                    help="fragment_counts_*.csv aus arc/count_fragments.py")
    ap.add_argument("--joined", type=Path, default=None,
                    help="bereits verknuepfte fragments_vs_performance.csv -- "
                         "ersetzt --counts und --perf in einem")
    ap.add_argument("--lang", choices=["de", "en"], default="de")
    ap.add_argument("--perf", action="append", default=[], metavar="ARM=CSV",
                    help="per_complex_csv aus evaluate_run, mehrfach angebbar")
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    ap.add_argument("--set-label", default="",
                    help="Name des Datensatzes fuer den Titel, z.B. 'PoseBusters v2'")
    ap.add_argument("--cum-style", choices=["panel", "twin"], default="panel",
                    help="kumulative Kurve als eigenes Panel oder auf zweiter y-Achse")
    ap.add_argument("--min-n", type=int, default=3,
                    help="ab wieviel Liganden je Bin eine Rate als Linie gilt")
    args = ap.parse_args()

    global T
    T = L[args.lang]

    if not args.counts and not args.joined:
        print("FEHLER: --counts oder --joined angeben.")
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    perf: dict[str, dict[str, dict]] = {}
    if args.joined:
        total = sum(1 for _ in open(args.joined, encoding="utf-8")) - 1
        rows, perf = read_joined(args.joined)
    else:
        total = sum(1 for _ in open(args.counts, encoding="utf-8")) - 1
        rows = read_counts(args.counts)
    n_missing = total - len(rows)
    print(f"\n{len(rows)} Liganden mit Fragmentzahl"
          + (f", {n_missing} ohne Messwert (bleiben aussen vor)" if n_missing else ""))

    for spec in args.perf:
        if "=" not in spec:
            print(f"FEHLER: --perf braucht ARM=CSV, bekam '{spec}'")
            return 2
        arm, path = spec.split("=", 1)
        perf[arm] = read_perf(Path(path))
        print(f"  {arm}: {len(perf[arm])} Komplexe mit Metriken")

    print()
    fig_a_distribution(rows, args.out_dir, n_missing,
                       set_label=args.set_label, cum_style=args.cum_style)
    fig_b_size(rows, args.out_dir)
    fig_c_performance(rows, perf, args.out_dir, args.min_n)
    if perf:
        write_join(rows, perf, args.out_dir)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
