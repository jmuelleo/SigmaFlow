#!/bin/bash
#
# Diese Datei wird normalerweise ge-source-t, nicht ausgefuehrt.
# Ein 'set -euo pipefail' auf Top-Level wuerde die Shell-Optionen des
# AUFRUFENDEN Skripts veraendern -- ein unsichtbarer Seiteneffekt, der
# spaeter niemand mehr zuzuordnen ist (submit_final.sh laeuft bewusst
# ohne -e, weil es interaktiv nachfragt). Strict mode gilt deshalb nur,
# wenn die Datei direkt aufgerufen wird: 'bash arc/final_config.sh'.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    set -euo pipefail
fi

# =============================================================================
# FINALE TRAININGSKONFIGURATION — die EINZIGE Datei, die nach dem
# Throughput-Sweep (arc/throughput_sweep.slurm) angefasst wird.
#
# Alle langen Laeufe lesen von hier. Nach Gate 1 werden hier drei Zahlen
# gesetzt und danach wird KEIN Jobskript mehr editiert.
#
# WICHTIG: Diese Werte gelten fuer ALLE Varianten gleichzeitig
# (SD_BASE, SF_MIN, SF_SRC, SF_CONF_INT). Fairness haengt daran, dass sie
# genau einmal definiert sind.
# =============================================================================

# --- Nach dem Sweep zu setzen -------------------------------------------------

# Globale, effektive Batch-Size. Ziel ist 32 (Originalwert des Papers,
# Appendix E.3: "batch size of 32" auf 4 GPUs, config.py teilt durch world_size
# ⇒ 8 pro A100). Auf einer GPU heisst effektiv 32 also BATCH_SIZE=32.
export FINAL_BATCH_SIZE="${FINAL_BATCH_SIZE:-32}"

# Falls Batch 32 nicht in den Speicher passt: echte Batch kleiner, dafuer
# Akkumulation. FINAL_BATCH_SIZE * FINAL_ACCUM muss 32 ergeben.
export FINAL_ACCUM="${FINAL_ACCUM:-1}"

# "32" | "bf16-mixed" | "16-mixed". Nur uebernehmen, wenn der Sweep zeigt:
# kein NaN, keine relevante Loss-Abweichung, echter Durchsatzgewinn.
export FINAL_PRECISION="${FINAL_PRECISION:-32}"

# "highest" schaltet TF32 AB (train.py:72). "high" schaltet es an.
export FINAL_CUDA_PRECISION="${FINAL_CUDA_PRECISION:-high}"

# Aus dem gemessenen Durchsatz abzuleiten, NICHT zu raten.
#   Original (Paper E.3): max 256 Epochen.
#   Ziel: 128 = 50 % des Originalbudgets, falls der Durchsatz es hergibt.
#   Rechnung: max_epochs = floor(72h * samples_per_s * 3600 / 19443)
# Der Scheduler wird hierauf kalibriert (max_steps -> LR-Verlauf), deshalb
# darf der Wert NICHT grosszuegig hoch gesetzt werden: ein nie abgeschlossener
# Cosine-Anneal ist genau der Fehler, der die Laeufe vom 2026-08-07 entwertet
# hat (STATUS.md, "KONFIGURATIONSFEHLER GEFUNDEN").
export FINAL_MAX_EPOCHS="${FINAL_MAX_EPOCHS:-128}"

# --- Fest, nicht nach dem Sweep zu aendern ------------------------------------

# 1x pro Epoche, wie im Original (config.py:149 default = None = pro Epoche).
# Die 24h-Jobs benutzten 50 Schritte, also ~49 Validierungen je Epoche auf dem
# vollen PoseBusters-Satz ohne limit_val_batches. Der Sweep misst, was das
# kostet; solange es keinen Grund dagegen gibt, gilt der Originalwert.
# Als Schrittzahl ausgedrueckt, damit Snapshots und Checkpoints planbar sind:
#   steps_per_epoch = 19443 / (BATCH * ACCUM)
export FINAL_VAL_EVERY_N_EPOCHS="${FINAL_VAL_EVERY_N_EPOCHS:-1}"

export FINAL_NUM_WORKERS="${FINAL_NUM_WORKERS:-6}"
export FINAL_SEED="${FINAL_SEED:-0}"

# Loss-Gewichte: SigmaDocks Produktionswerte, auf beiden Seiten identisch.
export FINAL_TRANS_WEIGHT="${FINAL_TRANS_WEIGHT:-2.0}"
export FINAL_ROT_WEIGHT="${FINAL_ROT_WEIGHT:-0.5}"

# Walltime und Partition. `medium` ist laut 00_preflight.sh <= 48h, fuer 72h
# wird `long` gebraucht. Beides ist PENDING ARC VERIFICATION.
export FINAL_WALLTIME="${FINAL_WALLTIME:-72:00:00}"
export FINAL_PARTITION="${FINAL_PARTITION:-long}"
export FINAL_GPU="${FINAL_GPU:-gpu:l40s:1}"
export FINAL_CPUS="${FINAL_CPUS:-8}"
export FINAL_MEM="${FINAL_MEM:-32G}"

# Archiv-Snapshots — die Lernkurve faellt gratis aus dem langen Lauf ab.
# Stunden nach Trainingsbeginn (ABSOLUT, nicht kumulativ), an denen last.ckpt
# weggesichert wird. Alle 6 h ueber 72 h = 12 Snapshots je Lauf.
#
# KOSTEN, damit die Zahl nicht unbesehen erhoeht wird: ein Checkpoint des
# EquiformerV2-Backbones liegt in der Groessenordnung mehrerer hundert MB.
# 12 Snapshots x 2 Baselines sind also einige GB pro Seed — unkritisch, aber
# nicht null. Speichern ist billig, EVALUIEREN ist teuer (siehe
# arc/eval_snapshots.slurm): deshalb wird haeufiger gespeichert als evaluiert.
export FINAL_SNAPSHOT_HOURS="${FINAL_SNAPSHOT_HOURS:-6 12 18 24 30 36 42 48 54 60 66 72}"

# Welche Snapshots die VOLLE Auswertung bekommen (10 Seeds, 209 Komplexe).
# Der Rest bekommt nur die billige Auswertung auf dem festen Subset.
export FINAL_FULL_EVAL_HOURS="${FINAL_FULL_EVAL_HOURS:-6 24 48 72}"

# --- Screening-Modus ----------------------------------------------------------
# Ein 6h-Screening ist NICHT ein eigenstaendiges 6h-Training, sondern die
# ERSTEN 6 STUNDEN DESSELBEN Trainingsprogramms, das ein langer Lauf haette.
# max_epochs (und damit der LR-Schedule) bleibt deshalb unveraendert auf
# FINAL_MAX_EPOCHS; nur die SLURM-Walltime ist kuerzer.
#
# Wuerde man stattdessen max_epochs auf das 6h-Budget kalibrieren, liefe ein
# vollstaendiger Cosine-Anneal in 6 h durch — ein ANDERES Trainingsprogramm,
# dessen Ergebnis mit dem 6h-Snapshot des langen Laufs nicht vergleichbar ist.
# Genau das war der Fehler der Laeufe vom 2026-08-07.
export SCREEN_WALLTIME="${SCREEN_WALLTIME:-06:30:00}"   # 6 h + Puffer fuer Setup
export SCREEN_PARTITION="${SCREEN_PARTITION:-short}"    # <= 12 h
export SCREEN_SNAPSHOT_HOURS="${SCREEN_SNAPSHOT_HOURS:-1 2 3 4 5 6}"

# --- Abgeleitete Groessen (nur informativ, nicht editieren) -------------------
export FINAL_EFFECTIVE_BATCH=$(( FINAL_BATCH_SIZE * FINAL_ACCUM ))

print_final_config() {
    cat <<EOF
[final_config] effektive Batch   = ${FINAL_EFFECTIVE_BATCH}  (${FINAL_BATCH_SIZE} x ${FINAL_ACCUM})
[final_config] precision         = ${FINAL_PRECISION}  (cuda_precision=${FINAL_CUDA_PRECISION})
[final_config] max_epochs        = ${FINAL_MAX_EPOCHS}   -> LR-Schedule wird hierauf kalibriert
[final_config] val               = alle ${FINAL_VAL_EVERY_N_EPOCHS} Epoche(n)
[final_config] seed              = ${FINAL_SEED}
[final_config] walltime/part.    = ${FINAL_WALLTIME} auf ${FINAL_PARTITION}, ${FINAL_GPU}
[final_config] Snapshots (h)     = ${FINAL_SNAPSHOT_HOURS}
EOF
    if [ "${FINAL_EFFECTIVE_BATCH}" -ne 32 ]; then
        echo "[final_config] WARNUNG: effektive Batch ${FINAL_EFFECTIVE_BATCH} != 32 (Originalwert)."
    fi
}
