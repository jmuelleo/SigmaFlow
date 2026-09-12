#!/bin/bash -l
#
# EIN NEUER LERNKURVENPUNKT, VON DER PRUEFUNG BIS ZUM FERTIGEN PAKET.
#
# WARUM ES DIESES SKRIPT GIBT
#   Am 2026-08-28 wurde diese Kette sieben Mal von Hand gefahren. Dabei sind
#   zwei Fehler passiert, die beide NICHT als Fehler aufgefallen waeren:
#
#   1. TRUE_DIR/BENCH auf ARC_TRUE_DIR gelassen. Das ist der 209er-Satz;
#      `posebusters308`-Zellen ergeben damit still 153 statt 307 Komplexe.
#      evaluate_run bricht bei fehlenden Referenzstrukturen NICHT ab.
#   2. Beinahe 20 Aufgaben Rechenzeit fuer einen Snapshot, der md5-identisch
#      mit einem bereits ausgewerteten war. Weil `monitor=val_loss` mit
#      `save_top_k=3` nur bei Verbesserung speichert, sind Dubletten der
#      Normalfall: bei SigmaDock waren vier von elf Snapshots doppelt.
#
#   Beides ist hier strukturell ausgeschlossen. Der Referenzsatz steht fest
#   und wird gegengeprueft; `pruefen` bricht bei einer Dublette ab, bevor
#   irgendetwas eingereicht wird.
#
# WARUM DREI UNTERBEFEHLE STATT EINEM DURCHLAUF
#   Zwischen Sampling und Auswertung liegt eine SLURM-Warteschlange. Ein
#   Skript, das darauf wartet, blockiert entweder die interaktive Zuteilung
#   oder pollt sinnlos. Die Trennung macht ausserdem sichtbar, wo man steht.
#
# AUFRUF
#   bash arc/neuer_punkt.sh pruefen   <RUN_DIR> <SNAP_TAG>
#   bash arc/neuer_punkt.sh sampeln   <RUN_DIR> <SNAP_TAG> <MODEL>
#   bash arc/neuer_punkt.sh abschluss <RUN_DIR> <SNAP_TAG> <MODEL>
#
#   RUN_DIR  Name unter arc_runs/, z.B. SF_MIN_72H_s0_8653824
#   SNAP_TAG ohne "snap_" und ohne ".ckpt", z.B. sched255ep_at_042h
#   MODEL    sigmadock | sigmaflow_minimal | sigmaflow_twohead
#
# `abschluss` darf mehrfach laufen: solange die Redock-Arrays noch rechnen,
# endet es mit einem Hinweis statt mit einem Paket. evaluate_run ist
# idempotent, es entsteht kein Schaden.
#
# Kein `set -e`: `module load` und `conda activate` liefern hier regelmaessig
# einen Rueckgabewert != 0. Jeder Schritt wird stattdessen einzeln geprueft.

set -uo pipefail

REPO_ROOT="/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow"
# shellcheck source=/dev/null
source "${REPO_ROOT}/arc/_common.sh"

# DER REFERENZSATZ. Bewusst nicht ueberschreibbar -- das ist der halbe Zweck.
TRUE308="${ARC_DATA}/posebusters_v2_308"

N_SEEDS=10
N_NFE=2            # Schrittzahlen 25 und 5, Reihenfolge wie in eval_snapshots_cpu.slurm
SOLL_POSEN=3080    # 10 Seeds x 308 Dateien
SOLL_KOMPLEXE=300  # Untergrenze; erwartet sind 307, 7XPO_UPG liefert nie eine Pose
SOLL_RD_ZEILEN=3090

CMD="${1:?pruefen | sampeln | abschluss}"
RUN_NAME="${2:?RUN_DIR-Name unter arc_runs/}"
TAG="${3:?Snapshot-Tag ohne snap_ und ohne .ckpt}"
RUN="${ARC_RUNS}/${RUN_NAME}"
SNAP="${RUN}/snapshots/snap_${TAG}.ckpt"

[ -d "$TRUE308" ] || { echo "FEHLER: Referenzsatz fehlt: ${TRUE308}" >&2; exit 2; }
N_REF="$(ls -1 "$TRUE308" | wc -l)"
[ "$N_REF" -eq 308 ] || { echo "FEHLER: ${TRUE308} hat ${N_REF} Eintraege statt 308." >&2; exit 2; }

# ------------------------------------------------------------- Hilfsfunktionen

# Index des Snapshots in der sortierten Liste. EXAKT dieselbe Bildung wie in
# eval_snapshots_cpu.slurm -- weicht sie ab, zeigt der Array-Bereich auf ein
# anderes Modell, und das faellt hinterher niemandem mehr auf.
snap_index() {
    local i=0 c
    while IFS= read -r c; do
        if [ "$(basename "$c" .ckpt)" = "snap_${TAG}" ]; then
            echo "$i"
            return 0
        fi
        i=$(( i + 1 ))
    done < <(find "${RUN}/snapshots" -maxdepth 1 -name 'snap_*.ckpt' | sort)
    return 1
}

zelle()   { echo "${RUN}/learning_curve_cpu/${TAG}__nfe${1}__sampled"; }
rdzelle() { echo "${RUN}/posebusters_redock_curve/${TAG}__nfe${1}__sampled"; }

# ------------------------------------------------------------------- pruefen
if [ "$CMD" = "pruefen" ]; then
    [ -f "$SNAP" ] || { echo "FEHLER: ${SNAP} existiert nicht." >&2; exit 1; }
    MD5="$(md5sum "$SNAP" | cut -d" " -f1)"
    echo "[pruef] ${TAG}"
    echo "[pruef] md5 = ${MD5}"

    # Gegen alle anderen Snapshots desselben Laufs vergleichen. Ein Treffer
    # heisst: dasselbe Modell liegt bereits unter einem anderen Stundennamen.
    DUP=""
    while IFS= read -r c; do
        [ "$c" = "$SNAP" ] && continue
        if [ "$(md5sum "$c" | cut -d" " -f1)" = "$MD5" ]; then
            DUP="$(basename "$c")"
            break
        fi
    done < <(find "${RUN}/snapshots" -maxdepth 1 -name 'snap_*.ckpt' | sort)

    if [ -n "$DUP" ]; then
        echo "[pruef] ABBRUCH: md5-identisch mit ${DUP}."
        echo "        Kein neues Modell. Nicht sampeln, in arme.py nicht eintragen."
        exit 3
    fi

    IDX="$(snap_index)" || { echo "FEHLER: ${TAG} nicht in der Snapshot-Liste." >&2; exit 1; }
    VON=$(( IDX * N_NFE * N_SEEDS ))
    BIS=$(( VON + N_NFE * N_SEEDS - 1 ))
    echo "[pruef] neu. Snapshot-Index ${IDX}  ->  --array=${VON}-${BIS}"
    echo "[pruef] weiter:  bash arc/neuer_punkt.sh sampeln ${RUN_NAME} ${TAG} <MODEL>"
    exit 0
fi

MODEL="${4:?MODEL: sigmadock | sigmaflow_minimal | sigmaflow_twohead}"

# Umgebung und Quellbaum je Arm -- dieselbe Zuordnung wie in
# eval_snapshots_cpu.slurm. Der SigmaDock-Checkpoint braucht beim Entpicklen
# das Paket `sigmadock` aus myenv; sigmaflow_env kennt es nicht.
case "$MODEL" in
  sigmadock)         PY_ENV="$ARC_SD_ENV"; PY_PATH="" ;;
  sigmaflow_minimal) PY_ENV="$ARC_SF_ENV"; PY_PATH="${ARC_REPO}/SigmaFlow_Minimal/src" ;;
  sigmaflow_twohead) PY_ENV="$ARC_SF_ENV"
                     PY_PATH="${ARC_REPO}/SigmaFlow_FM_Specific/EXP-110_two_head_vector_field/src" ;;
  *) echo "MODEL unbekannt: ${MODEL}" >&2; exit 1 ;;
esac

# ------------------------------------------------------------------- sampeln
if [ "$CMD" = "sampeln" ]; then
    IDX="$(snap_index)" || { echo "FEHLER: ${TAG} nicht in der Snapshot-Liste." >&2; exit 1; }
    VON=$(( IDX * N_NFE * N_SEEDS ))
    BIS=$(( VON + N_NFE * N_SEEDS - 1 ))
    echo "[samp] Index ${IDX}, Aufgaben ${VON}-${BIS}"
    cd "$ARC_REPO" || exit 1
    RUN_DIR="$RUN" MODEL="$MODEL" N_SEEDS="$N_SEEDS" \
        EXPERIMENT=posebusters308 CONFORMER=sampled \
        sbatch --array="${VON}-${BIS}" arc/eval_snapshots_cpu.slurm
    echo "[samp] wenn durch:  bash arc/neuer_punkt.sh abschluss ${RUN_NAME} ${TAG} ${MODEL}"
    exit 0
fi

# ----------------------------------------------------------------- abschluss
if [ "$CMD" = "abschluss" ]; then
    for N in 25 5; do
        C="$(find "$(zelle "$N")" -name "*.sdf" 2>/dev/null | wc -l)"
        printf "[ab] nfe%-3s %5s Posen (Soll %s)\n" "$N" "$C" "$SOLL_POSEN"
        [ "$C" -eq "$SOLL_POSEN" ] || { echo "[ab] ABBRUCH: Sampling unvollstaendig." >&2; exit 4; }
    done

    # Redock nur einreichen, wenn es noch keine Ergebnisse gibt -- sonst
    # startet ein zweiter Aufruf von `abschluss` die Arrays erneut.
    RD_DA=1
    for N in 25 5; do
        [ "$(ls "$(rdzelle "$N")"/rd_*.csv 2>/dev/null | wc -l)" -eq 10 ] || RD_DA=0
    done
    if [ "$RD_DA" -eq 0 ]; then
        echo "[ab] Redock einreichen"
        cd "$ARC_REPO" || exit 1
        for N in 25 5; do
            MODEL="$MODEL" BENCH="$TRUE308" \
            SAMPLING_ROOT="$(zelle "$N")" OUT_DIR="$(rdzelle "$N")" \
            sbatch --array=0-9 arc/posebusters_redock.slurm
        done
    else
        echo "[ab] Redock liegt bereits vor, nicht erneut eingereicht"
    fi

    echo "[ab] Epoche -- meta.txt zum Vergleich, massgeblich ist der Checkpoint"
    grep -E "walltime_h|^epoch" "${RUN}/snapshots/snap_${TAG}.meta.txt" | sed "s/^/     meta: /"

    module load Mamba
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$PY_ENV"

    PYTHONPATH="${PY_PATH}:${ARC_REPO}:${PYTHONPATH:-}" python "${ARC_REPO}/arc/_ckpt_epoch.py" "$SNAP"

    echo "[ab] RMSD-Auswertung"
    for N in 25 5; do
        D="$(zelle "$N")"
        PYTHONPATH="${ARC_REPO}:${PYTHONPATH:-}" python -m SigmaFlow_Evaluation.evaluate_run \
            --sampling_root "$D" --true_dir "$TRUE308" --label "${TAG}__nfe${N}" \
            --out_json "$D/evaluation.json" --per_complex_csv "$D/per_complex.csv" \
            --per_pose_csv "$D/per_pose.csv" > "$D/evaluate.log" 2>&1
        RC=$?
        if [ "$RC" -ne 0 ]; then
            echo "[ab] nfe${N}: FEHLER rc=${RC}" >&2
            tail -5 "$D/evaluate.log" >&2
            exit 5
        fi
        NP=$(( $(wc -l < "$D/per_pose.csv") - 1 ))
        NC=$(( $(wc -l < "$D/per_complex.csv") - 1 ))
        printf "[ab] nfe%-3s %5s Posen, %s Komplexe (erwartet %s / 307)\n" \
               "$N" "$NP" "$NC" "$SOLL_POSEN"
        if [ "$NC" -lt "$SOLL_KOMPLEXE" ]; then
            echo "[ab] ABBRUCH: nur ${NC} Komplexe. Falscher Referenzsatz?" >&2
            exit 6
        fi
    done
    conda deactivate

    echo "[ab] Redock-Ergebnisse"
    OK=1
    for N in 25 5; do
        D="$(rdzelle "$N")"
        NF="$(ls "$D"/rd_*.csv 2>/dev/null | wc -l)"
        NZ="$(cat "$D"/rd_*.csv 2>/dev/null | wc -l)"
        printf "[ab] nfe%-3s %2s Dateien %6s Zeilen (Soll 10 / %s)\n" \
               "$N" "$NF" "$NZ" "$SOLL_RD_ZEILEN"
        if [ "$NF" -ne 10 ] || [ "$NZ" -ne "$SOLL_RD_ZEILEN" ]; then OK=0; fi
    done
    if [ "$OK" -ne 1 ]; then
        echo "[ab] Redock laeuft noch. Diesen Unterbefehl spaeter erneut aufrufen."
        exit 0
    fi

    PAKET="/data/stat-cadd/shug8458/punkt_${TAG}.tgz"
    LISTE="$(mktemp)"
    cd "$ARC_RUNS" || exit 1
    for N in 25 5; do
        C="${TAG}__nfe${N}__sampled"
        ls "${RUN_NAME}/learning_curve_cpu/$C"/per_pose.csv \
           "${RUN_NAME}/learning_curve_cpu/$C"/per_complex.csv \
           "${RUN_NAME}/learning_curve_cpu/$C"/evaluation.json \
           "${RUN_NAME}/posebusters_redock_curve/$C"/rd_*.csv
    done > "$LISTE"
    tar czf "$PAKET" -T "$LISTE"
    echo "[ab] Paket: ${PAKET}  ($(wc -l < "$LISTE") Dateien, Soll 26)"
    rm -f "$LISTE"
    echo "[ab] lokal holen:"
    echo "     scp shug8458@htc-login.arc.ox.ac.uk:${PAKET} <ziel>"
    exit 0
fi

echo "Unbekannter Unterbefehl: ${CMD}" >&2
exit 1
