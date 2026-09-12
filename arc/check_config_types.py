"""Die Konfigurationsdatei gegen die Dataclass pruefen, Namen UND Typen.

WARUM ES DIESE PRUEFUNG GIBT
    conf_flow_paper.yaml enthielt `max_lr_start: 1e-4`, woertlich aus
    SigmaDocks slurm.yaml uebernommen. In YAML 1.1 ist das ein STRING: der
    Float-Resolver von PyYAML verlangt einen Dezimalpunkt in der Mantisse,
    `1.0e-4` waere eine Zahl. AdamW bekam die Lernrate als Zeichenkette und
    starb mit
        TypeError: '<=' not supported between instances of 'float' and 'str'
    nach knapp sechs Minuten. Die Jobs 8688976 und 8688977 gingen so verloren.

    Die frueher benutzte Pruefung verglich nur die Schluesselnamen. Ein
    Schluessel kann aber richtig heissen und trotzdem den falschen Typ
    tragen, und train.py:49 uebernimmt ihn dann kommentarlos.

Aufruf:
    python arc/check_config_types.py arc/conf_flow_paper.yaml <config.py> ...
"""
import ast
import pathlib
import re
import sys

import yaml

VORGABE_CONFIGS = [
    "SigmaFlow_Minimal/src/sigmadock/config.py",
    "SigmaDock/src_sigmadock/config.py",
]


def felder(pfad, start="RunConfig"):
    """{Name: Annotation} aus RunConfig UND seinen Basisklassen.

    RunConfig erbt von TrainingConfig und StructuralConfig. `fields()` einer
    Dataclass enthaelt die geerbten Felder, und genau dagegen prueft
    train.py:46. Wer nur die eigenen Felder sammelt, haelt drei Viertel der
    gueltigen Schluessel faelschlich fuer unbekannt.
    """
    baum = ast.parse(pathlib.Path(pfad).read_text(encoding="utf-8"))
    klassen = {k.name: k for k in ast.walk(baum)
               if isinstance(k, ast.ClassDef)}
    if start not in klassen:
        raise SystemExit(f"{start} nicht gefunden in {pfad}")

    out, offen, gesehen = {}, [start], set()
    while offen:
        name = offen.pop()
        if name in gesehen or name not in klassen:
            continue
        gesehen.add(name)
        k = klassen[name]
        for s in k.body:
            if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
                out.setdefault(s.target.id, ast.unparse(s.annotation))
        offen += [b.id for b in k.bases if isinstance(b, ast.Name)]
    return out


def passt(wert, annotation):
    """Grobe, aber ausreichende Vertraeglichkeitspruefung."""
    a = annotation.replace(" ", "")
    if wert is None:
        return True
    if isinstance(wert, bool):
        return "bool" in a
    if isinstance(wert, int):
        return "int" in a or "float" in a
    if isinstance(wert, float):
        return "float" in a
    if isinstance(wert, str):
        return "str" in a or "Literal" in a or "Path" in a
    if isinstance(wert, (list, tuple)):
        return "list" in a or "tuple" in a or "Sequence" in a
    return True


def main():
    args = sys.argv[1:]
    yml = args[0] if args else "arc/conf_flow_paper.yaml"
    configs = args[1:] or VORGABE_CONFIGS

    cfg = yaml.safe_load(pathlib.Path(yml).read_text(encoding="utf-8"))
    print(f"{yml}: {len(cfg)} Schluessel\n")

    fehler = 0
    for c in configs:
        if not pathlib.Path(c).exists():
            print(f"uebersprungen, nicht vorhanden: {c}")
            continue
        f = felder(c)
        print(f"gegen {c}")
        for k, v in cfg.items():
            if k not in f:
                print(f"  STUMM VERWORFEN  {k}")
                fehler += 1
            elif not passt(v, f[k]):
                print(f"  TYP FALSCH       {k}: {v!r} ist "
                      f"{type(v).__name__}, erwartet {f[k]}")
                fehler += 1
        print("  in Ordnung" if fehler == 0 else "")

    # der konkrete Fallstrick noch einmal ausdruecklich
    roh = pathlib.Path(yml).read_text(encoding="utf-8")
    for zeile in roh.splitlines():
        if re.search(r":\s*[0-9]+e[-+]?[0-9]+\s*(#.*)?$", zeile):
            print(f"  ACHTUNG, YAML liest das als String: {zeile.strip()}")
            fehler += 1

    print(f"\n{'ALLES GUT' if fehler == 0 else str(fehler) + ' PROBLEM(E)'}")
    return 1 if fehler else 0


if __name__ == "__main__":
    sys.exit(main())
