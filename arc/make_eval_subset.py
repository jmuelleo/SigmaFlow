"""Erzeugt den festen Auswertungs-Subset fuer die Lernkurve. EINMALIG.

WARUM EIN FESTER SUBSET
  metric(t) ist nur dann eine Kurve und kein Rauschen, wenn ueber alle
  Snapshots und beide Methoden DERSELBE Komplexsatz laeuft. Ein pro Lauf neu
  gezogener Subset erzeugt Sprunge, die wie Lernfortschritt aussehen.
  Der Subset-Vorbehalt ist in diesem Projekt schon einmal teuer geworden:
  RESULTS.md dokumentiert, dass derselbe Lauf auf dem 94er-Subset 0.36 A und
  auf dem 57er-Subset 0.40 A Uebergangsbindungsfehler zeigt -- allein durch
  die andere Komplexmenge.

WARUM STRATIFIZIERT UND NICHT EINFACH DIE ERSTEN 50
  Die PoseBusters-IDs sind alphabetisch nach PDB-Code sortiert, und der Code
  korreliert mit dem Hinterlegungsjahr. Die ersten 50 waeren damit eine
  Jahrgangsstichprobe, keine Zufallsstichprobe. Gezogen wird deshalb
  gleichmaessig ueber die sortierte Liste (systematische Stichprobe): das
  streut ueber den gesamten Bereich und ist bei festem Seed exakt
  reproduzierbar.

WARUM 50
  Der Stichprobenfehler der Erfolgsrate bei p ~ 0.05 und n = 50 ist
  sqrt(0.05*0.95/50) = 3.1 Prozentpunkte. Der gemessene Methodenunterschied
  liegt bei rund 5 Punkten (SF 4.4 % vs SD 9.8 % auf dem 12h-Stand). Ein
  50er-Subset kann diesen Unterschied also gerade eben aufloesen, einen
  Zuwachs von 1-2 Punkten zwischen zwei Snapshots dagegen NICHT.
  Die Kurve zeigt Tendenzen; Aussagen ueber einzelne Snapshotdifferenzen
  gehoeren in die volle Auswertung.

LEAKAGE
  Der Subset stammt aus PoseBusters, dem Testsatz. Er wird ausschliesslich
  BEOBACHTET. Alle Trainingsgroessen stehen vor dem ersten Snapshot in
  final_config.sh fest; es wird nichts daran angepasst. Mehrfaches Ablesen
  ist erlaubt, Tuning nicht.

    python arc/make_eval_subset.py --data_dir /data/.../data --n 50
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

OUT_DEFAULT = Path(__file__).resolve().parent / "eval_subset.txt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Wurzel mit posebusters/ darunter.")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--force", action="store_true",
                    help="Vorhandene Datei ueberschreiben. Nur mit gutem Grund.")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        # Der Subset darf sich NIE aendern, sobald eine Kurve darauf beruht.
        # Deshalb ist Ueberschreiben eine bewusste Handlung, kein Versehen.
        print(f"{args.out} existiert bereits — nicht angefasst.")
        print(f"  {sum(1 for _ in args.out.open()) } IDs, "
              f"sha256={hashlib.sha256(args.out.read_bytes()).hexdigest()[:16]}")
        print("  Zum Ersetzen: --force (macht alle bisherigen Kurven unvergleichbar).")
        return 0

    pb = Path(args.data_dir) / "posebusters"
    if not pb.is_dir():
        raise SystemExit(f"Nicht gefunden: {pb}")

    # Ein Komplex ist ein Unterverzeichnis mit einer *_protein.pdb darin.
    ids = sorted(d.name for d in pb.iterdir()
                 if d.is_dir() and any(d.glob("*_protein.pdb")))
    if not ids:
        raise SystemExit(f"Keine Komplexe unter {pb} gefunden.")
    print(f"PoseBusters: {len(ids)} Komplexe gefunden")

    n = min(args.n, len(ids))
    # Systematische Stichprobe ueber die sortierte Liste: Index
    # round(i * (N-1)/(n-1)). Deterministisch, ohne Zufallsgenerator, und
    # gleichmaessig ueber den gesamten Bereich.
    if n == 1:
        picked = [ids[0]]
    else:
        picked = [ids[round(i * (len(ids) - 1) / (n - 1))] for i in range(n)]
    picked = sorted(dict.fromkeys(picked))     # Duplikate raus, Reihenfolge fix

    args.out.write_text("\n".join(picked) + "\n", encoding="utf-8")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"geschrieben: {args.out}  ({len(picked)} IDs)")
    print(f"sha256: {digest}")
    print()
    print("Diese Pruefsumme gehoert in den Ergebnisbericht. Aendert sie sich,")
    print("sind alle vorher erzeugten Lernkurven untereinander unvergleichbar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
