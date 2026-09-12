"""Haben die KAPUTTEN Komplexe einen ueberproportionalen Hebel auf den Verlust?

DIE FRAGE
    Rund 1,3 % der PDBBind-Komplexe sind geometrisch fehlerhaft (Atome an
    physikalisch unmoeglichen Positionen). Ein Mittelwertargument sagt, dass
    1,3 % nichts verschieben. Bei einem QUADRATISCHEN Verlust stimmt das aber
    nur, wenn die Residuen normal sind -- ein einzelner extremer Punkt hat
    Hebel proportional zu seinem Residuum, nicht zu seiner Haeufigkeit.

    Zwei moegliche Mechanismen, die dieses Skript beide erfasst:
      ZIEL   Das Vektorfeld wird je FRAGMENT gebildet. Liegt ein Ligandatom auf
             einem Proteinatom, sind die Fragmentpositionen trotzdem normal --
             ueber diesen Weg ist kein grosses Residuum zu erwarten.
      EINGABE Der Taschengraph enthaelt dann Kanten mit absurd kurzen
             Abstaenden. Deren Merkmale koennen das Netz stoeren, und das
             schlaegt sehr wohl auf den Verlust durch.

    Gemessen wird der echte Trainingsverlust je Komplex mit einem vorhandenen
    Checkpoint: `losses["loss_trans"]` und `losses["loss_R"]` aus
    `SigmaLightningModule.forward`, also genau die Groessen, die
    `trainer.py:_shared_step` gewichtet zusammenzieht.

WARUM MEHRERE ZIEHUNGEN JE KOMPLEX
    Der Verlust haengt von der gezogenen Zeit t und der Quelle ab. Eine
    einzelne Ziehung waere reines Rauschen. Je Komplex werden N Ziehungen mit
    FESTEN Seeds gemittelt, und dieselben Seeds gelten fuer alle Komplexe --
    so unterscheiden sich die Werte nur durch den Komplex.

WARUM EINE KONTROLLGRUPPE STATT DES GANZEN DATENSATZES
    19.037 Komplexe x N Ziehungen waeren auf CPU Tage. Gebraucht wird nur der
    Vergleich: alle kaputten plus eine Zufallsstichprobe unauffaelliger.

Aufruf:
    python arc/verlust_je_komplex.py --ckpt <pfad.ckpt> --liste <kaputt.txt> \
        --kontrolle 750 --ziehungen 4 --chunks 10 --chunk 0 --out <csv>
"""
import argparse
import os
import sys
import warnings
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

from sigmadock.config import (  # noqa: E402
    RunConfig,
    StructuralConfig,
    get_experiment_config,
    update_config_from_args,
)
from sigmadock.data import SigmaDataset  # noqa: E402
from sigmadock.datafronts import MetaFront  # noqa: E402
from sigmadock.utils import load_from_scratch  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--liste", required=True, help="Textdatei, ein PDB-Code je Zeile")
p.add_argument("--exp", nargs="+", default=["pdbbind-general"])
p.add_argument("--data_dir", default=os.environ.get("ARC_DATA"))
p.add_argument("--config", default=None)
p.add_argument("--kontrolle", type=int, default=750)
p.add_argument("--ziehungen", type=int, default=4)
p.add_argument("--chunks", type=int, default=1)
p.add_argument("--chunk", type=int, default=0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", required=True)
a = p.parse_args()

# --- Konfiguration wie in scripts/train.py --------------------------------
cfg = RunConfig()
if a.config:
    with open(a.config) as f:
        roh = yaml.safe_load(f) or {}
    gueltig = {f.name for f in fields(RunConfig)}
    cfg = replace(cfg, **{k: v for k, v in roh.items()
                          if k in gueltig and v is not None
                          and k not in ("data_dir", "exp_dir", "betas")})
sc = update_config_from_args(StructuralConfig(), cfg)
kw = dict(sc.__dict__)
kw["get_mol_info"] = True     # fuer den Komplexnamen
kw["force_retry"] = False     # ein Fehler darf NICHT still ersetzt werden
kw["verbose"] = False
kw["sample_conformer"] = False

front = MetaFront([get_experiment_config(n, root_dir=Path(a.data_dir)) for n in a.exp])
if len(front) == 0:
    sys.exit(f"ABBRUCH: Datafront leer fuer {a.exp} unter {a.data_dir}")
ds = SigmaDataset(datafront=front, seed=a.seed, **kw)

# --- Welche Komplexe? -----------------------------------------------------
# Die Liste darf zwei Formen haben:
#   "code"        -> Gruppe "kaputt"
#   "code,gruppe" -> beliebig viele benannte Gruppen
# Zwei Defektarten wirken ueber VERSCHIEDENE Wege und gehoeren getrennt:
#   extern  Atome liegen auf Proteinatomen -> verfaelscht den EINGABEGRAPHEN
#   intern  der Ligand selbst ist kaputt   -> verfaelscht das ZIEL
gruppe_von = {}
for zeile in open(a.liste):
    zeile = zeile.strip()
    if not zeile or zeile.startswith("#"):
        continue
    code, _, g = zeile.partition(",")
    gruppe_von[code.strip().lower()] = (g.strip() or "kaputt")
kaputt = set(gruppe_von)

namen = [os.path.basename(os.path.dirname(str(front[i][1]))).lower()
         for i in range(len(ds))]
idx_kaputt = [i for i, n in enumerate(namen) if n in kaputt]
rest = [i for i, n in enumerate(namen) if n not in kaputt]
rng = np.random.default_rng(a.seed)
idx_kontrolle = sorted(rng.choice(rest, size=min(a.kontrolle, len(rest)),
                                  replace=False).tolist())
for i in idx_kontrolle:
    gruppe_von[namen[i]] = "kontrolle"

from collections import Counter  # noqa: E402
print("Gruppen:", dict(Counter(gruppe_von[namen[i]]
                               for i in idx_kaputt + idx_kontrolle)))

alle = sorted(set(idx_kaputt) | set(idx_kontrolle))
meine = alle[a.chunk::a.chunks]
print(f"Datensatz    : {len(ds)} Komplexe")
print(f"Liste        : {len(kaputt)} Codes, davon {len(idx_kaputt)} gefunden")
print(f"Kontrolle    : {len(idx_kontrolle)}")
print(f"Block {a.chunk} von {a.chunks}: {len(meine)} Komplexe "
      f"x {a.ziehungen} Ziehungen", flush=True)
if not idx_kaputt:
    sys.exit("ABBRUCH: kein einziger Code der Liste im Datensatz gefunden. "
             "Gross-/Kleinschreibung oder falsches Experiment?")

# --- Modell ---------------------------------------------------------------
lm = load_from_scratch(Path(a.ckpt), load_ema=False, strict=True)
lm.eval()
geraet = "cuda" if torch.cuda.is_available() else "cpu"
lm.to(geraet)
print(f"Geraet: {geraet}", flush=True)

from torch_geometric.data import Batch  # noqa: E402

zeilen = []
for n, i in enumerate(meine):
    for z in range(a.ziehungen):
        # DIESELBEN Seeds fuer jeden Komplex: der Unterschied darf nur vom
        # Komplex kommen, nicht von der Ziehung.
        torch.manual_seed(a.seed + z)
        np.random.seed((a.seed + z) % (2**31 - 1))
        try:
            d = ds[i]
        except Exception as e:      # noqa: BLE001
            print(f"  [{i}] Ausnahme beim Laden: {e}", flush=True)
            break
        if d is None:
            break
        name = os.path.basename(os.path.dirname(str(d.mol_info["ligand_path"])))
        b = Batch.from_data_list([d]).to(geraet)
        try:
            with torch.no_grad():
                _, losses = lm(b)
        except Exception as e:      # noqa: BLE001
            print(f"  [{i}] Ausnahme im Vorwaertslauf: {e}", flush=True)
            break
        zeilen.append({
            "komplex": name.lower(),
            "idx": i,
            "ziehung": z,
            "gruppe": gruppe_von.get(name.lower(), "kontrolle"),
            "loss_trans": float(losses["loss_trans"].mean()),
            "loss_R": float(losses["loss_R"].mean()),
        })
    if (n + 1) % 50 == 0:
        print(f"  {n + 1}/{len(meine)} ...", flush=True)

if not zeilen:
    sys.exit("ABBRUCH: keine einzige Zeile entstanden.")

t = pd.DataFrame(zeilen)
t["total"] = t["loss_trans"] + 0.5 * t["loss_R"]   # Gewichte aus trainer.py
t.to_csv(a.out, index=False)
print(f"\n{len(t)} Zeilen nach {a.out}")

m = t.groupby(["komplex", "gruppe"], as_index=False)[
    ["loss_trans", "loss_R", "total"]].mean()
for sp in ("loss_trans", "loss_R", "total"):
    print(f"\n{sp}:")
    ref = m.loc[m["gruppe"] == "kontrolle", sp]
    print(f"  {'Gruppe':<14} {'n':>5} {'Median':>10} {'Mittel':>10} "
          f"{'90 %':>10} {'Verh. Median':>13}")
    for g, d in m.groupby("gruppe"):
        v = d[sp]
        vh = (v.median() / max(ref.median(), 1e-12)) if len(ref) else float("nan")
        print(f"  {g:<14} {len(v):5d} {v.median():10.5f} {v.mean():10.5f} "
              f"{np.percentile(v, 90):10.5f} {vh:13.2f}")
