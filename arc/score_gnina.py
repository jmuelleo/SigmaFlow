#!/usr/bin/env python3
"""
Vinardo-Affinitaeten fuer alle gesampelten Posen, wie SigmaDock sie zum
Ranken benutzt.

WAS DAS NACHBAUT
    SigmaDock rankt seine N_seeds Vorschlaege mit einer festen Heuristik, nicht
    mit einem gelernten Modell (Paper, Abschnitt 2.5). Der Code dafuer steht in
    src_sigmadock_chem/postprocessor.py und ruft gnina als externes Programm:

        gnina -r <protein> -l <pose> --autobox_ligand <pose> --score_only \\
              --scoring vinardo --cnn_scoring none

    --cnn_scoring none schaltet den neuronalen Teil ab; uebrig bleibt Vinardo,
    eine klassische empirische Scoringfunktion. Deshalb reicht CPU.

WARUM GEBUENDELT
    Ein Einzelaufruf dauert 3,7 s, davon fast alles Startzeit (Laden der
    CUDA-Bibliotheken). gnina bewertet aber alle Molekuele einer
    Multi-Model-SDF in EINEM Aufruf. Statt 209*N Aufrufen also 209, einer je
    Komplex mit allen Seeds darin. Das druckt die Startkosten um den Faktor N.

WICHTIG
    gnina laeuft NICHT auf Login-Knoten -- dort scheitert das Mappen von
    libcublasLt.so.12 am Speicherlimit. Auf Rechenknoten (auch reinen
    CPU-Knoten) laeuft es, wenn CUDA/12.6.0 mitgeladen ist.

Aufruf:
    python arc/score_gnina.py --sampling_root <root> --pdb_root <benchmark_set>
                              --out gnina_scores.csv [--max_seeds 40]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

AFF = re.compile(r"^Affinity:\s*([-\d.]+)")
INTRA = re.compile(r"^Intramolecular energy:\s*([-\d.]+)")


def collect(root: Path) -> dict[str, dict[int, Path]]:
    """complex -> {seed: sdf-Pfad}. Layout: <root>/**/seed_<k>/<CID>__*.sdf"""
    out: dict[str, dict[int, Path]] = defaultdict(dict)
    for sd in sorted(root.rglob("seed_*")):
        if not sd.is_dir():
            continue
        m = re.search(r"seed_(\d+)$", sd.name)
        if not m:
            continue
        seed = int(m.group(1))
        for f in sorted(sd.glob("*.sdf")):
            cid = f.name.split("__")[0]
            out[cid][seed] = f
    return out


def parse_multi(text: str) -> tuple[list[float], list[float]]:
    """Alle Affinity- und Intramolecular-Werte in Reihenfolge der Molekuele."""
    aff, intra = [], []
    for line in text.splitlines():
        m = AFF.match(line.strip())
        if m:
            aff.append(float(m.group(1)))
            continue
        m = INTRA.match(line.strip())
        if m:
            intra.append(float(m.group(1)))
    return aff, intra


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampling_root", required=True, type=Path)
    ap.add_argument("--pdb_root", required=True, type=Path,
                    help="posebusters_benchmark_set/ mit <CID>/<CID>_protein.pdb")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max_seeds", type=int, default=None)
    ap.add_argument("--gnina", default="gnina")
    a = ap.parse_args()

    if shutil.which(a.gnina) is None:
        print(f"FEHLER: '{a.gnina}' nicht im PATH. module load CUDA/12.6.0 gnina/1.3.2",
              file=sys.stderr)
        return 2

    data = collect(a.sampling_root)
    if not data:
        print(f"FEHLER: keine seed_*/*.sdf unter {a.sampling_root}", file=sys.stderr)
        return 2
    seeds_seen = sorted({s for d in data.values() for s in d})
    print(f"Komplexe: {len(data)}   Seeds: {len(seeds_seen)}  ({seeds_seen[0]}..{seeds_seen[-1]})")

    rows, n_ok, n_skip, n_mismatch = [], 0, 0, 0
    tmp = Path(tempfile.mkdtemp(prefix="gnina_"))

    for i, (cid, per_seed) in enumerate(sorted(data.items()), 1):
        prot = a.pdb_root / cid / f"{cid}_protein.pdb"
        if not prot.is_file():
            n_skip += 1
            continue
        seeds = sorted(per_seed)
        if a.max_seeds is not None:
            seeds = [s for s in seeds if s < a.max_seeds]
        if not seeds:
            n_skip += 1
            continue

        # Alle Posen dieses Komplexes in EINE Multi-Model-SDF, Reihenfolge = seeds
        merged = tmp / f"{cid}.sdf"
        with merged.open("w", encoding="utf-8") as fh:
            for s in seeds:
                txt = per_seed[s].read_text(encoding="utf-8", errors="replace")
                if not txt.rstrip().endswith("$$$$"):
                    txt = txt.rstrip() + "\n$$$$\n"
                fh.write(txt if txt.endswith("\n") else txt + "\n")

        cmd = [a.gnina, "-r", str(prot), "-l", str(merged),
               "--autobox_ligand", str(merged), "--score_only",
               "--scoring", "vinardo", "--cnn_scoring", "none", "--no_gpu"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            n_skip += 1
            continue
        aff, intra = parse_multi(r.stdout)

        # Nur verwerten, wenn die Zahl der Werte zur Zahl der Posen passt.
        # Sonst waere die Zuordnung Seed -> Score geraten, und ein falsch
        # zugeordneter Score ist schlimmer als ein fehlender.
        if len(aff) != len(seeds):
            n_mismatch += 1
            print(f"  [warn] {cid}: {len(aff)} Werte fuer {len(seeds)} Posen, uebersprungen")
            continue
        for k, s in enumerate(seeds):
            rows.append((cid, s, aff[k], intra[k] if k < len(intra) else ""))
        n_ok += 1
        if i % 25 == 0:
            print(f"  ...{i}/{len(data)}")

    shutil.rmtree(tmp, ignore_errors=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8", newline="") as fh:
        fh.write("complex,seed,affinity,intramolecular_energy\n")
        for c, s, af, ie in rows:
            fh.write(f"{c},{s},{af},{ie}\n")

    print()
    print(f"geschrieben: {a.out}  ({len(rows)} Posen aus {n_ok} Komplexen)")
    if n_skip:
        print(f"  uebersprungen (Protein fehlt / Timeout): {n_skip}")
    if n_mismatch:
        print(f"  uebersprungen (Wertzahl passt nicht):    {n_mismatch}")
    if rows:
        vals = [r[2] for r in rows]
        neg = sum(1 for v in vals if v < 0)
        vals_s = sorted(vals)
        print(f"  Affinitaet: median {vals_s[len(vals_s)//2]:.2f}, "
              f"min {vals_s[0]:.2f}, max {vals_s[-1]:.2f} kcal/mol")
        print(f"  davon negativ (guenstig): {neg}/{len(vals)} = {100*neg/len(vals):.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
