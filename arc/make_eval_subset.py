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

# Kandidaten in der Reihenfolge, in der gesucht wird. Der erste ist der
# kanonische Satz -- derselbe, auf den ARC_TRUE_DIR in arc/_common.sh zeigt.
PB_KANDIDATEN = (
    "posebusters_paper/posebusters_benchmark_set",
    "posebusters_full/posebusters_benchmark_set",
    "posebusters",
    "",
)


def komplex_ids(d: Path) -> list:
    """Unterverzeichnisse, die wie ein Komplex aussehen: mit PDB UND SDF."""
    if not d.is_dir():
        return []
    out = []
    try:
        kinder = sorted(k for k in d.iterdir() if k.is_dir())
    except OSError:
        return []
    for k in kinder:
        if any(k.glob("*.pdb")) and any(k.glob("*.sdf")):
            out.append(k.name)
    return out


def find_pb_dir(data_dir: Path, explizit: Path | None):
    """Sucht das Verzeichnis mit den Komplexordnern, statt es zu konstruieren.

    WARUM DAS NOETIG IST
        Vorher stand hier `data_dir / "posebusters"`. Dieses Verzeichnis
        existiert zwar, enthaelt aber nur leere raw/- und processed/-Ordner --
        keinen einzigen Komplex. Der kanonische Satz liegt verschachtelt unter
        posebusters_paper/posebusters_benchmark_set/. Genau dieselbe
        Verwechslung ist in eval_snapshots.slurm, compare_rankers.slurm und
        evaluate_snapshot.py schon einmal aufgetreten und wurde dort mit
        ARC_TRUE_DIR behoben (arc/_common.sh:17-30); dieses Skript war die
        letzte Stelle, die den Pfad noch selbst zusammensetzt.

    Es wird jeder Kandidat geprueft und berichtet, was gefunden wurde. Ein
    leeres Verzeichnis gilt als NICHT gefunden -- sonst entstuende ein leerer
    Subset, und die Kurven waeren still leer geblieben.
    """
    if explizit is not None:
        ids = komplex_ids(explizit)
        if not ids:
            raise SystemExit(f"--dataset_dir enthaelt keine Komplexe: {explizit}")
        return explizit, ids

    versucht = []
    for rel in PB_KANDIDATEN:
        kand = data_dir / rel if rel else data_dir
        ids = komplex_ids(kand)
        versucht.append((kand, len(ids)))
        if ids:
            return kand, ids

    zeilen = "\n".join(f"    {p}  ->  {n} Komplexe" for p, n in versucht)
    raise SystemExit(
        "Kein Verzeichnis mit Komplexordnern gefunden. Geprueft wurde:\n"
        f"{zeilen}\n"
        "  -> mit --dataset_dir direkt darauf zeigen."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Datenwurzel. Der PoseBusters-Satz wird darunter GESUCHT, "
                         "nicht angenommen -- siehe find_pb_dir().")
    ap.add_argument("--dataset_dir", type=Path, default=None,
                    help="Direkt das Verzeichnis mit den Komplexordnern, "
                         "falls die Suche danebenliegt.")
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

    pb, ids = find_pb_dir(Path(args.data_dir), args.dataset_dir)
    print(f"PoseBusters: {len(ids)} Komplexe unter {pb}")

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
