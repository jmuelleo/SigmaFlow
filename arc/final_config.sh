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

# --- Trainingshorizont: EINZIGE QUELLE DER WAHRHEIT ---------------------------
#
# HIER STEHT BEWUSST KEINE ZAHL.
#
# Frueher stand hier ein Default (zuletzt 128), und im Runbook stand
# unabhaengig davon 36. Beide waren aus einem ANDEREN Durchsatzregime
# abgeleitet (Batch 8, --debug, ohne TF32) und beide waren fuer die finale
# Konfiguration falsch. Ein plausibel aussehender Default ist hier
# gefaehrlicher als gar keiner: er wird uebernommen, ohne dass jemand die
# Herleitung nachrechnet.
#
# Der Horizont kommt deshalb ausschliesslich aus arc/final_horizon.env,
# das von arc/calculate_final_epochs.py aus einer GEMESSENEN Stufe-2-
# Durchsatzzahl erzeugt wird. Fehlt die Datei, bleibt FINAL_MAX_EPOCHS leer
# und submit_final.sh verweigert den Start.
#
# Asymmetrie, die die ganze Sorgfalt begruendet (numerisch bestaetigt in
# arc/test_scheduler_horizon.py):
#   Horizont zu GROSS  -> Lauf endet bei ~5.5x der Ziel-Lernrate. UNGUELTIG.
#   Horizont zu KLEIN  -> Lernrate clampt am Minimum. Gueltig, nur weniger
#                         effizient. Deshalb wird konsequent abgerundet.
HORIZON_FILE="${HORIZON_FILE:-$(dirname "${BASH_SOURCE[0]}")/final_horizon.env}"
if [ -f "$HORIZON_FILE" ]; then
    # shellcheck source=/dev/null
    source "$HORIZON_FILE"
    export FINAL_HORIZON_SOURCE="$HORIZON_FILE"
else
    export FINAL_HORIZON_SOURCE="FEHLT"
fi

# Armspezifische Horizonte. Ob beide Arme dieselbe Epochenzahl bekommen,
# entscheidet arc/calculate_final_epochs.py anhand des gemessenen
# Durchsatzunterschieds (Regel: <= 5 % -> gemeinsame Zahl aus dem
# langsameren Arm; > 5 % -> getrennte Zahlen bei gleichem Zeitbudget).
# resolve_horizon_for_model() unten setzt FINAL_MAX_EPOCHS und FINAL_MAX_STEPS
# auf die Werte des jeweiligen Arms.
export FINAL_MAX_EPOCHS="${FINAL_MAX_EPOCHS:-}"
export FINAL_MAX_STEPS="${FINAL_MAX_STEPS:-}"

# Echte Laenge der Trainings-Datafront. NICHT die Paper-Zahl 19443 -- die gilt
# fuer PDBBind v2020 vor unserer Filterung. Wird vom Throughput-Sweep Stufe 2
# gemessen und in final_horizon.env geschrieben.
export FINAL_N_TRAIN="${FINAL_N_TRAIN:-}"

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

# --- EXP-102: Quellverteilung (nur MODEL=sigmaflow_source) --------------------
# "haar"       Kontrollbedingung, bit-identisch zu EXP-100
# "pocket_pca" Zentrum aus Hauptachsen Konformer -> Tasche  (die Hypothese)
# "constant"   festes Zentrum: misst KONZENTRATION ohne Konditionierung
export SOURCE_MODE="${SOURCE_MODE:-pocket_pca}"

# Medianwinkel der Quelle zum Zentrum. Haar hat 132.3 Grad.
# WIRD AUS arc/exp101_distance_audit.py GESETZT, NICHT GERATEN:
# sinnvoll ist der dort gemessene Median von H1. Ist H1 z.B. 110 Grad, waere
# eine Quelle mit Median 110 genau so breit wie die Heuristik verlaesslich ist.
# Enger zu waehlen heisst, der Heuristik mehr zu glauben als die Messung hergibt.
export SOURCE_MEDIAN_DEG="${SOURCE_MEDIAN_DEG:-132.3}"

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

# -----------------------------------------------------------------------------
# resolve_horizon_for_model <modelname>
#
# Setzt FINAL_MAX_EPOCHS und FINAL_MAX_STEPS auf die Werte des gewaehlten Arms
# und prueft, dass der gespeicherte Horizont zur AKTUELLEN Konfiguration passt.
#
# Der Staleness-Guard ist der eigentliche Zweck. Ein Horizont, der unter
# Batch 32 gemessen wurde, ist unter Batch 16 + accum 2 numerisch ein anderer;
# ohne diese Pruefung wuerde eine spaetere Aenderung an FINAL_BATCH_SIZE eine
# alte Zahl still weiterverwenden. Genau diese Klasse von Fehler soll hier
# unmoeglich werden.
#
# Rueckgabe: 0 wenn alles stimmt, 1 sonst (mit Meldung auf stderr).
# -----------------------------------------------------------------------------
resolve_horizon_for_model() {
    local model="$1"
    local key ep st

    case "$model" in
        sigmadock)          key="SIGMADOCK" ;;
        sigmaflow_minimal)  key="SIGMAFLOW_MINIMAL" ;;
        sigmaflow_source)   key="SIGMAFLOW_SOURCE" ;;
        sigmaflow_conf)     key="SIGMAFLOW_CONF" ;;
        *) echo "[horizon] unbekanntes Modell: $model" >&2; return 1 ;;
    esac

    if [ "$FINAL_HORIZON_SOURCE" = "FEHLT" ]; then
        {
            echo "[horizon] FEHLER: arc/final_horizon.env fehlt."
            echo "[horizon] Der Trainingshorizont MUSS aus einer gemessenen"
            echo "[horizon] Stufe-2-Durchsatzzahl kommen. Vorgehen:"
            echo "[horizon]   1) MODEL=<arm> STAGE=2 BATCH=<gewaehlt> sbatch arc/throughput_sweep.slurm"
            echo "[horizon]   2) python arc/calculate_final_epochs.py ... --write-env arc/final_horizon.env"
        } >&2
        return 1
    fi

    # Indirekte Variablenexpansion: aus dem Namen einen Wert lesen.
    eval "ep=\${FINAL_MAX_EPOCHS_${key}:-}"
    eval "st=\${FINAL_MAX_STEPS_${key}:-}"

    if [ -z "$ep" ] || [ -z "$st" ]; then
        echo "[horizon] FEHLER: FINAL_MAX_EPOCHS_${key} / FINAL_MAX_STEPS_${key} fehlen in ${FINAL_HORIZON_SOURCE}" >&2
        return 1
    fi

    # Staleness-Guard: passt der gespeicherte Horizont zur aktuellen Batch?
    if [ -n "${FINAL_HORIZON_EFFECTIVE_BATCH:-}" ] &&
       [ "${FINAL_HORIZON_EFFECTIVE_BATCH}" -ne "${FINAL_EFFECTIVE_BATCH}" ]; then
        {
            echo "[horizon] FEHLER: gespeicherter Horizont gilt fuer effektive Batch"
            echo "[horizon]         ${FINAL_HORIZON_EFFECTIVE_BATCH}, aktuell konfiguriert ist ${FINAL_EFFECTIVE_BATCH}."
            echo "[horizon]         Durchsatz neu messen und final_horizon.env neu erzeugen."
        } >&2
        return 1
    fi

    # Konsistenzprobe: der gespeicherte Horizont muss zur Formel passen.
    local expect=$(( (FINAL_N_TRAIN / FINAL_EFFECTIVE_BATCH) * ep ))
    if [ "$st" -ne "$expect" ]; then
        {
            echo "[horizon] FEHLER: FINAL_MAX_STEPS_${key}=${st} passt nicht zu"
            echo "[horizon]         floor(${FINAL_N_TRAIN}/${FINAL_EFFECTIVE_BATCH}) * ${ep} = ${expect}."
            echo "[horizon]         Datei von Hand editiert? Neu erzeugen lassen."
        } >&2
        return 1
    fi

    export FINAL_MAX_EPOCHS="$ep"
    export FINAL_MAX_STEPS="$st"
    return 0
}

print_final_config() {
    cat <<EOF
[final_config] effektive Batch   = ${FINAL_EFFECTIVE_BATCH}  (${FINAL_BATCH_SIZE} x ${FINAL_ACCUM})
[final_config] precision         = ${FINAL_PRECISION}  (cuda_precision=${FINAL_CUDA_PRECISION})
[final_config] max_epochs        = ${FINAL_MAX_EPOCHS:-<NICHT GESETZT>}
[final_config] max_steps         = ${FINAL_MAX_STEPS:-<NICHT GESETZT>}   -> LR-Schedule wird hierauf kalibriert
[final_config] n_train (gemessen)= ${FINAL_N_TRAIN:-<NICHT GESETZT>}
[final_config] Horizontquelle    = ${FINAL_HORIZON_SOURCE}
[final_config] val               = alle ${FINAL_VAL_EVERY_N_EPOCHS} Epoche(n)
[final_config] seed              = ${FINAL_SEED}
[final_config] walltime/part.    = ${FINAL_WALLTIME} auf ${FINAL_PARTITION}, ${FINAL_GPU}
[final_config] Snapshots (h)     = ${FINAL_SNAPSHOT_HOURS}
EOF
    if [ "${FINAL_EFFECTIVE_BATCH}" -ne 32 ]; then
        echo "[final_config] WARNUNG: effektive Batch ${FINAL_EFFECTIVE_BATCH} != 32 (Originalwert)."
    fi
}
