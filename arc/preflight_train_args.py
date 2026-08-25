"""Prueft die Flags, die train_final_72h.slurm an scripts/train.py uebergibt.

WARUM
    Job 8634117 starb nach 46 Sekunden an
        train.py: error: unrecognized arguments: --accum_grad_batches 1
    Davor stand anderthalb Tage Wartezeit in der Warteschlange. Dieser Test
    findet dieselbe Fehlerklasse in Sekunden, ohne GPU und ohne Queue.

WIE
    Aus dem SLURM-Skript werden alle `--flag` des train.py-Aufrufs gelesen.
    Aus config.py wird der echte argparse-Parser instanziiert und nach seinen
    Optionen gefragt. Verglichen wird die Mengendifferenz.

    Der Parser wird NICHT ausgefuehrt, nur konstruiert -- der Test hat keine
    Nebenwirkungen und braucht weder Daten noch GPU.

AUFRUF
    python arc/preflight_train_args.py <MODEL> <CODE_DIR>
        z.B. python arc/preflight_train_args.py sigmaflow_minimal SigmaFlow_Minimal
    Rueckgabewert 0 = alle Flags bekannt, 1 = mindestens eines unbekannt.
"""
import argparse
import io
import os
import re
import sys

# Ueberschreibbar, damit sich der Pruefer selbst gegen eine aeltere Fassung
# testen laesst -- ohne diesen Gegentest weiss man nicht, ob er den Fehler,
# fuer den er gebaut wurde, ueberhaupt findet.
SKRIPT = os.environ.get(
    "PREFLIGHT_SLURM",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "train_final_72h.slurm"))


def flags_aus_skript(pfad: str) -> list[str]:
    """Die --flags des train.py-Aufrufs, in Reihenfolge des Auftretens."""
    text = io.open(pfad, encoding="utf-8").read()
    # Der Aufruf beginnt bei `"$PYTHON" scripts/train.py \` und endet an der
    # ersten Zeile, die nicht mit Backslash fortgesetzt wird.
    m = re.search(r'"\$PYTHON"\s+scripts/train\.py\s*\\\n', text)
    if m is None:
        raise SystemExit("ABBRUCH: train.py-Aufruf im SLURM-Skript nicht gefunden")
    rest = text[m.end():]
    zeilen = []
    for z in rest.splitlines():
        zeilen.append(z)
        if not z.rstrip().endswith("\\"):
            break
    block = "\n".join(zeilen)
    # Kommentarzeilen zaehlen nicht mit.
    block = "\n".join(z for z in block.splitlines() if not z.lstrip().startswith("#"))
    gefunden = set(re.findall(r"(?<![\w-])--[a-zA-Z][\w-]*", block))

    return sorted(gefunden)


def method_flags(pfad: str, modell: str) -> list[str]:
    """Die METHOD_FLAGS genau des angegebenen case-Zweigs.

    Die Flags stehen als Bash-Array im case-Block und werden erst zur Laufzeit
    expandiert -- `"${METHOD_FLAGS[@]}"` im Aufruf enthaelt kein einziges
    `--`. Ohne diesen Zusatz pruefte der Test genau die Flags NICHT, die sich
    zwischen den Armen unterscheiden. Wichtig ist die Zuordnung: jeder Arm
    bekommt NUR seinen eigenen Zweig, sonst meldet der Test Fehlalarme.
    """
    text = io.open(pfad, encoding="utf-8").read()
    m = re.search(r'case\s+"\$MODEL"\s+in(.*?)\nesac', text, re.S)
    if m is None:
        raise SystemExit("ABBRUCH: case-Block fuer MODEL nicht gefunden")
    block = m.group(1)

    zweige = re.split(r"\n\s{2}([a-z_]+)\)", block)
    # zweige = ['', name1, koerper1, name2, koerper2, ...]
    for i in range(1, len(zweige) - 1, 2):
        if zweige[i] != modell:
            continue
        koerper = "\n".join(z for z in zweige[i + 1].splitlines()
                            if not z.lstrip().startswith("#"))
        z = re.search(r"METHOD_FLAGS=\(([^)]*)\)", koerper)
        if z is None or not z.group(1).strip():
            return []
        return sorted(set(re.findall(r"(?<![\w-])--[a-zA-Z][\w-]*", z.group(1))))
    raise SystemExit(f"ABBRUCH: kein case-Zweig fuer MODEL={modell}")


def flags_aus_parser(code_dir: str) -> set[str]:
    """Alle Optionsnamen, die der echte Parser kennt."""
    src = os.path.join(code_dir, "src")
    if not os.path.isdir(src):
        raise SystemExit(f"ABBRUCH: {src} existiert nicht")
    sys.path.insert(0, src)
    try:
        from sigmadock import config as cfg
    except Exception as e:
        raise SystemExit(f"ABBRUCH: sigmadock.config nicht importierbar: {e}")

    # parse_args() ruft am Ende parser.parse_args() auf, was hier stoeren
    # wuerde. Deshalb wird der Parser abgefangen, bevor er parst.
    gesammelt: set[str] = set()
    echt = argparse.ArgumentParser.parse_args

    def abfangen(self, *a, **k):
        for handlung in self._actions:
            gesammelt.update(handlung.option_strings)
        raise _Fertig()

    class _Fertig(Exception):
        pass

    argparse.ArgumentParser.parse_args = abfangen
    try:
        cfg.parse_args()
    except _Fertig:
        pass
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = echt
    if not gesammelt:
        raise SystemExit("ABBRUCH: Parser lieferte keine Optionen")
    return gesammelt


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    modell, code_dir = sys.argv[1], sys.argv[2]

    uebergeben = flags_aus_skript(SKRIPT)
    eigene = method_flags(SKRIPT, modell)
    uebergeben = sorted(set(uebergeben) | set(eigene))
    bekannt = flags_aus_parser(code_dir)

    print(f"MODEL             : {modell}")
    print(f"Codebaum          : {code_dir}")
    print(f"Parser kennt      : {len(bekannt)} Optionen")
    print(f"Skript uebergibt  : {len(uebergeben)} Flags"
          f"{f' (davon {len(eigene)} methodenspezifisch)' if eigene else ''}")
    print()

    unbekannt = [f for f in uebergeben if f not in bekannt]
    for f in uebergeben:
        print(f"  {'FEHLT ' if f in unbekannt else 'ok    '} {f}")

    print()
    if unbekannt:
        print(f"FEHLGESCHLAGEN: {len(unbekannt)} unbekannte(s) Flag(s): "
              f"{', '.join(unbekannt)}")
        print("train.py wuerde mit rc=2 abbrechen, bevor ein Schritt laeuft.")
        return 1
    print("BESTANDEN: jedes uebergebene Flag ist dem Parser bekannt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
