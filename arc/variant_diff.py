"""Semantischer Diff zwischen SigmaFlow_Minimal und einer Variante.

WARUM DAS EIN EIGENES WERKZEUG BRAUCHT
  Die Varianten in diesem Projekt sind VOLLKOPIEN des Quellbaums, keine
  Patches. Das ist bequem, hat aber einen bekannten Preis: Kopien divergieren
  unbemerkt. Nach drei Monaten ist nicht mehr rekonstruierbar, ob ein
  Unterschied eine bewusste Methodenaenderung war oder ein vergessener
  Debug-Eingriff.

  Der wissenschaftliche Anspruch der finalen Laeufe ist aber, dass jede
  Variante eine EIN-FAKTOR-Ablation von Minimal ist. Diese Behauptung muss
  belegbar sein, nicht geglaubt. Genau das leistet dieser Report: er listet
  jede Abweichung auf, sodass sie entweder als beabsichtigt abgehakt oder als
  Confounder erkannt wird.

WAS ER IGNORIERT
  Nur Dinge, die den Lauf nachweislich nicht beeinflussen: __pycache__,
  .pyc, Ausgabeverzeichnisse, Checkpoints, Logs. Kommentare und Docstrings
  werden NICHT ignoriert -- eine geaenderte Dokumentationszeile ist harmlos,
  aber sie zu verstecken hiesse, dem Werkzeug beizubringen, Unterschiede zu
  verschweigen.

    python arc/variant_diff.py SigmaFlow_Minimal \\
        SigmaFlow_FM_Specific/EXP-102_heuristic_conditional_source
    python arc/variant_diff.py --all          # alle bekannten Varianten
"""

from __future__ import annotations

import argparse
import ast
import difflib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Nur Verzeichnisse/Endungen, die den Lauf nicht beeinflussen.
SKIP_DIRS = {"__pycache__", ".git", "experiments", "wandb", "outputs",
             "sampling_output", "results", ".ipynb_checkpoints", "notebooks"}
SKIP_SUFFIX = {".pyc", ".pyo", ".ckpt", ".pt", ".log", ".out", ".err",
               ".sdf", ".pdb", ".csv", ".json", ".png", ".pdf"}

# Diese Unterbaeume entscheiden ueber das Verhalten des Modells. Aenderungen
# ausserhalb sind meist Infrastruktur; innerhalb sind sie immer erklaerungs-
# beduerftig.
CRITICAL = ("src/sigmadock/diff/", "src/sigmadock/net/", "src/sigmadock/trainer.py",
            "src/sigmadock/data.py", "src/sigmadock/oracle.py", "src/sigmadock/chem/",
            # config.py fehlte hier zunaechst. Es haelt die Trainings-Defaults
            # (Loss-Gewichte, Optimizer, und in EXP-102 die Quellparameter) --
            # eine Aenderung dort ist so verhaltensrelevant wie eine im Modell,
            # sieht aber nach blosser Konfiguration aus. Genau solche Dateien
            # sind die, die ein Diff-Werkzeug uebersehen darf am wenigsten.
            "src/sigmadock/config.py", "src/sigmadock/sampling_setup.py",
            "scripts/train.py", "scripts/sample.py")


def rel_files(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not root.is_dir():
        return out
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        if p.suffix.lower() in SKIP_SUFFIX:
            continue
        out[p.relative_to(root).as_posix()] = p
    return out


def top_level_defs(path: Path) -> set[str]:
    """Namen aller Funktionen/Klassen/Methoden - fuer die Kurzfassung.

    ast.parse liest die Datei als Syntaxbaum statt sie auszufuehren; damit
    laesst sich fragen, WELCHE Definitionen es gibt, ohne Imports aufzuloesen.
    Bei Syntaxfehlern (etwa halbfertigem Code) wird leer zurueckgegeben statt
    zu werfen - ein Diff-Werkzeug soll auch kaputte Baeume noch vergleichen.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def changed_defs(a: Path, b: Path) -> tuple[set[str], set[str]]:
    da, db = top_level_defs(a), top_level_defs(b)
    return db - da, da - db          # (neu, entfernt)


def compare(base: Path, variant: Path, context: int, max_diff_lines: int) -> int:
    fa, fb = rel_files(base), rel_files(variant)
    only_base = sorted(set(fa) - set(fb))
    only_var = sorted(set(fb) - set(fa))
    common = sorted(set(fa) & set(fb))

    modified = []
    for r in common:
        ta = fa[r].read_bytes()
        tb = fb[r].read_bytes()
        if ta != tb:
            modified.append(r)

    def is_critical(r: str) -> bool:
        return any(r.startswith(c) for c in CRITICAL)

    crit_mod = [r for r in modified if is_critical(r)]
    crit_new = [r for r in only_var if is_critical(r)]
    crit_gone = [r for r in only_base if is_critical(r)]

    print("=" * 78)
    print(f"BASIS    {base}")
    print(f"VARIANTE {variant}")
    print("=" * 78)
    print(f"Dateien: {len(fa)} in Basis, {len(fb)} in Variante")
    print(f"  geaendert                 {len(modified):4d}   davon verhaltensrelevant {len(crit_mod)}")
    print(f"  nur in Variante           {len(only_var):4d}   davon verhaltensrelevant {len(crit_new)}")
    print(f"  nur in Basis (entfernt)   {len(only_base):4d}   davon verhaltensrelevant {len(crit_gone)}")

    if crit_gone:
        print("\n!! IN DER VARIANTE ENTFERNT (verhaltensrelevant) !!")
        for r in crit_gone:
            print(f"   - {r}")

    print("\n--- VERHALTENSRELEVANTE AENDERUNGEN " + "-" * 40)
    if not (crit_mod or crit_new):
        print("  keine. Die Variante unterscheidet sich nur in Infrastruktur.")
    for r in crit_new:
        print(f"\n  [NEU] {r}")
        defs = sorted(top_level_defs(fb[r]))
        if defs:
            print(f"        definiert: {', '.join(defs[:12])}"
                  + (" ..." if len(defs) > 12 else ""))
    for r in crit_mod:
        added, removed = changed_defs(fa[r], fb[r])
        print(f"\n  [GEAENDERT] {r}")
        if added:
            print(f"        neue Definitionen:      {', '.join(sorted(added))}")
        if removed:
            print(f"        entfernte Definitionen: {', '.join(sorted(removed))}")
        la = fa[r].read_text(encoding="utf-8", errors="replace").splitlines()
        lb = fb[r].read_text(encoding="utf-8", errors="replace").splitlines()
        diff = list(difflib.unified_diff(la, lb, lineterm="", n=context,
                                         fromfile="basis", tofile="variante"))
        n_add = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
        n_del = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))
        print(f"        {n_add} Zeilen hinzu, {n_del} entfernt")
        for line in diff[:max_diff_lines]:
            print(f"        {line}")
        if len(diff) > max_diff_lines:
            print(f"        ... {len(diff) - max_diff_lines} weitere Diffzeilen "
                  f"(--max-diff-lines erhoehen)")

    other_mod = [r for r in modified if not is_critical(r)]
    if other_mod:
        print("\n--- uebrige Aenderungen (Infrastruktur) " + "-" * 37)
        for r in other_mod:
            print(f"   ~ {r}")
    other_new = [r for r in only_var if not is_critical(r)]
    if other_new:
        print("\n--- nur in der Variante (Infrastruktur) " + "-" * 36)
        for r in other_new:
            print(f"   + {r}")

    print("\n" + "=" * 78)
    n_crit = len(crit_mod) + len(crit_new) + len(crit_gone)
    print(f"URTEIL: {n_crit} verhaltensrelevante Abweichung(en).")
    print("Jede davon muss im Variant-README als beabsichtigt begruendet sein,")
    print("sonst ist sie ein Confounder fuer den kontrollierten Vergleich.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default="SigmaFlow_Minimal")
    ap.add_argument("variant", nargs="?", default=None)
    ap.add_argument("--all", action="store_true",
                    help="alle Variantenbaeume unter SigmaFlow_FM_Specific vergleichen")
    ap.add_argument("--context", type=int, default=2)
    ap.add_argument("--max-diff-lines", type=int, default=80)
    args = ap.parse_args()

    base = (REPO / args.base).resolve()
    if not base.is_dir():
        print(f"Basisverzeichnis fehlt: {base}")
        return 2

    if args.all:
        roots = sorted(p for p in (REPO / "SigmaFlow_FM_Specific").glob("EXP-*")
                       if (p / "src").is_dir())
        if not roots:
            print("Keine Variante mit src/ unter SigmaFlow_FM_Specific gefunden.")
            print("Das ist der erwartete Zustand, solange kein Gate eine Variante "
                  "freigegeben hat.")
            return 0
        rc = 0
        for r in roots:
            rc |= compare(base, r, args.context, args.max_diff_lines)
            print()
        return rc

    if args.variant is None:
        print("Variante angeben oder --all benutzen.")
        return 2
    var = (REPO / args.variant).resolve()
    if not var.is_dir():
        print(f"Variantenverzeichnis fehlt: {var}")
        return 2
    return compare(base, var, args.context, args.max_diff_lines)


if __name__ == "__main__":
    raise SystemExit(main())
