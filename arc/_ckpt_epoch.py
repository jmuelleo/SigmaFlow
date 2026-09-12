"""Liest Epoche und global_step aus einem Lightning-Checkpoint.

WARUM ALS EIGENE DATEI
    Die Alternative waere ein Heredoc innerhalb von arc/neuer_punkt.sh. Ein
    Heredoc in einem Skript, das selbst per Heredoc oder Copy-Paste in eine
    Shell wandert, geht regelmaessig kaputt -- am 2026-08-28 mehrfach passiert.
    Eine Datei laesst sich ausserdem einzeln testen.

WARUM DIE META-DATEI NICHT REICHT
    `snapshot_meta()` in arc/train_final_72h.slurm liest die Epoche aus dem
    LAUFENDEN stdout-Log zum Kopierzeitpunkt. Der kopierte `last.ckpt` stammt
    aber vom letzten gespeicherten Zustand und hinkt hinterher -- gemessen am
    2026-08-28 bei allen geprueften Snapshots, um 3 bis 15 Epochen. Wer die
    Kurve gegen die meta-Epoche auftraegt, verschiebt Punkte um bis zu einen
    halben Tag Training.

WARUM weights_only=False
    Der Checkpoint enthaelt Hyperparameter als Objekte des jeweiligen Arms.
    Deshalb muss der passende src-Baum auf dem PYTHONPATH liegen; der Aufrufer
    setzt ihn. Bei SigmaDock steckt das Paket in myenv, nicht in sigmaflow_env.

Aufruf:
    python arc/_ckpt_epoch.py <pfad/zum/checkpoint.ckpt>
"""
import sys

import torch


def main() -> int:
    if len(sys.argv) != 2:
        print("Aufruf: python arc/_ckpt_epoch.py <checkpoint.ckpt>", file=sys.stderr)
        return 2
    pfad = sys.argv[1]
    try:
        d = torch.load(pfad, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 -- die Ursache soll sichtbar bleiben
        print(f"     CHECKPOINT nicht lesbar: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"     CHECKPOINT epoch={d.get('epoch')}  global_step={d.get('global_step')}")

    # Hyperparameter aus dem ARTEFAKT, nicht aus einer YAML oder meta.txt.
    # trainer.py ruft save_hyperparameters(), Lightning legt sie unter
    # "hyper_parameters" ab. batch_size ist dort der BEREITS GETEILTE Wert je
    # GPU (config.py:251 teilt in __post_init__ durch world_size); die globale
    # Batch ist deshalb batch_size * world_size.
    hp = d.get("hyper_parameters") or {}
    if not isinstance(hp, dict):
        hp = getattr(hp, "__dict__", {}) or {}
    interessant = ("batch_size", "world_size", "accum_grad_batches", "max_epochs",
                   "max_steps", "max_lr_start", "min_lr_start", "init_lr_start",
                   "ema_halflife", "pb_check", "random_rotation",
                   "include_protein_ligand_interactions", "t_emb_scale",
                   "train_exps", "val_exps", "seed")
    gefunden = {k: hp[k] for k in interessant if k in hp}
    if gefunden:
        for k in interessant:
            if k in gefunden:
                print(f"       {k:36s} = {gefunden[k]}")
        b, w = gefunden.get("batch_size"), gefunden.get("world_size")
        if isinstance(b, int) and isinstance(w, int):
            print(f"       {'-> globale Batch (batch_size*world_size)':36s} = {b * w}")
    else:
        print(f"       keine bekannten Hyperparameter im Checkpoint "
              f"(vorhandene Schluessel: {sorted(hp)[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
