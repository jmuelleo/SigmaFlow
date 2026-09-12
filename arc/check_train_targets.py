"""Sind die TATSAECHLICHEN TRAININGSZIELE PB-valide?

WOZU
    `arc/crystal_pb.slurm` hat die KRISTALLPOSEN geprueft: PDBBind 83,89 %.
    Das ist ein Stellvertreter. Das echte Trainingsziel ist in vier von fuenf
    Faellen NICHT die Kristallpose, sondern eine neu erzeugte,
    torsionsoptimierte Konformation, die darauf ausgerichtet wurde
    (`data.py:434`). Mit `pb_check: false` wird deren Gueltigkeit NIE geprueft.
    Sie koennte schlechter sein als die Kristallpose, weil die
    Torsionsoptimierung ligandinterne Verletzungen einfuehren kann, die in der
    hinterlegten Struktur nicht stehen.

    Dieses Skript misst sie direkt: es laesst den ECHTEN Datenpfad laufen und
    bustet `mol_info["aligned"]` -- genau das Molekuel, das in
    `fragment_and_annotate` geht und damit die Zielkoordinaten liefert.

WARUM KEIN NACHBAU
    Der Datensatz wird ueber `SigmaDataset` mit `get_mol_info=True` gebaut,
    die Konfiguration ueber `StructuralConfig` + `update_config_from_args`
    genau wie in `scripts/train.py:141-142`. Eine zweite Fassung der
    Ausrichtungslogik gaebe es damit nicht.

WARUM JE INDEX NEU GESEEDET WIRD
    `data.py:434` und `:450` ziehen aus dem globalen torch-RNG (Ausrichtung
    mit p = 0,8; Annahme einer ungueltigen mit p = 0,5). Ohne festen Seed je
    Index waeren zwei Laeufe (pb_check an/aus) nicht vergleichbar. Mit
    `torch.manual_seed(SEED + idx)` sehen beide Arme DIESELBEN
    Kandidatenkonformationen, und der Unterschied ist genau die Wirkung von
    pb_check.

WARUM force_retry=False
    Bei einem Fehler liefert `__getitem__` sonst still einen ANDEREN,
    zufaelligen Komplex nach (`_try_again`). Dann stuende in der Tabelle eine
    Pose unter dem falschen Namen. Fehler werden hier gezaehlt, nicht ersetzt.

Aufruf (siehe arc/train_targets_pb.slurm):
    python arc/check_train_targets.py --exp pdbbind-general --pb_check false \
        --chunks 20 --chunk 0 --out /pfad/ziel_0.csv
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
from posebusters import PoseBusters
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

p = argparse.ArgumentParser()
p.add_argument("--exp", nargs="+", default=["pdbbind-general"])
p.add_argument("--data_dir", default=os.environ.get("ARC_DATA"))
p.add_argument("--config", default=None, help="YAML wie conf/training/slurm.yaml")
p.add_argument("--pb_check", choices=["true", "false"], required=True)
p.add_argument("--chunks", type=int, default=1)
p.add_argument("--chunk", type=int, default=0)
p.add_argument("--stride", type=int, default=1, help="nur jeden n-ten Komplex")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", required=True)
a = p.parse_args()

# --- Konfiguration genau wie scripts/train.py -----------------------------
cfg = RunConfig()
if a.config:
    with open(a.config) as f:
        roh = yaml.safe_load(f) or {}
    gueltig = {f.name for f in fields(RunConfig)}
    cfg = replace(cfg, **{k: v for k, v in roh.items()
                          if k in gueltig and v is not None and k not in ("data_dir", "exp_dir", "betas")})
sc = update_config_from_args(StructuralConfig(), cfg)

# pb_check ist der einzige bewusste Eingriff.
kw = dict(sc.__dict__)
kw["pb_check"] = (a.pb_check == "true")
kw["get_mol_info"] = True
kw["force_retry"] = False
kw["verbose"] = False
kw["sample_conformer"] = False   # Trainingszweig, nicht Inferenz
kw["random_rotation"] = False    # Rotation aendert die Chemie nicht, aber die
                                 # Vergleichbarkeit der Koordinaten

# root_dir MUSS ein Path sein: config.py:871 ruft `root_dir.exists()`.
# In train.py kommt es als Path aus RunConfig; hier kaeme es als str durch.
front = MetaFront([get_experiment_config(n, root_dir=Path(a.data_dir)) for n in a.exp])
if len(front) == 0:
    sys.exit(f"ABBRUCH: Datafront leer fuer {a.exp} unter {a.data_dir}. "
             f"Pfad aufloesen, nicht raten.")
ds = SigmaDataset(datafront=front, seed=a.seed, **kw)

alle = list(range(0, len(ds), a.stride))
meine = alle[a.chunk::a.chunks]
print(f"Datenquelle : {a.exp}  ({len(ds)} Komplexe, Schritt {a.stride})")
print(f"pb_check    : {kw['pb_check']}")
print(f"Block {a.chunk} von {a.chunks}: {len(meine)} Komplexe", flush=True)

pb = PoseBusters(config="redock")
zeilen, fehler, ohne_ausrichtung = [], 0, 0

for n, i in enumerate(meine):
    # Paarung zwischen den beiden Armen: gleicher Seed, gleiche Kandidaten.
    torch.manual_seed(a.seed + i)
    np.random.seed((a.seed + i) % (2**31 - 1))
    try:
        d = ds[i]
    except Exception as e:      # noqa: BLE001
        print(f"  [{i}] Ausnahme: {e}", flush=True)
        fehler += 1
        continue
    if d is None or d.mol_info is None:
        fehler += 1
        continue

    mi = d.mol_info
    rmsd = float(d.alignment_rmsd)
    if rmsd == 0.0:
        ohne_ausrichtung += 1
    try:
        r = pb.bust(mol_pred=mi["aligned"], mol_true=mi["original"], mol_cond=mi["pocket"])
    except Exception as e:      # noqa: BLE001
        print(f"  [{i}] bust fehlgeschlagen: {e}", flush=True)
        fehler += 1
        continue

    z = r.iloc[0].to_dict()
    z["komplex"] = os.path.basename(os.path.dirname(str(mi["ligand_path"])))
    z["idx"] = i
    z["alignment_rmsd"] = rmsd
    z["alignment_energy_delta"] = float(d.alignment_energy_delta)
    z["ausgerichtet"] = rmsd > 0.0
    zeilen.append(z)

    if (n + 1) % 200 == 0:
        print(f"  {n + 1}/{len(meine)} ...", flush=True)

if not zeilen:
    sys.exit(f"ABBRUCH: keine einzige Zeile entstanden ({fehler} Fehler).")

t = pd.DataFrame(zeilen)
t.to_csv(a.out, index=False)
print(f"\n{len(t)} Zeilen geschrieben nach {a.out}")
print(f"  Fehler/uebersprungen : {fehler}")
print(f"  ohne Ausrichtung (rohe Kristallpose) : {ohne_ausrichtung} "
      f"({100 * ohne_ausrichtung / max(len(meine), 1):.1f} %)")
