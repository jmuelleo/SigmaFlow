# Gemeinsame Hilfsfunktionen fuer alle ARC-Jobs.
# Wird von den .slurm-Skripten per `source` eingebunden, ist selbst kein Job.
#
# Zweck: jeder Lauf soll Monate spaeter noch rekonstruierbar sein. Dafuer
# schreibt `write_run_metadata` ein festes Set von Dateien in das Laufverzeichnis,
# BEVOR das Training startet - sonst fehlen sie genau dann, wenn der Job stirbt.

# Feste Orte auf ARC. An genau einer Stelle definiert.
export ARC_USER_ROOT="/data/stat-cadd/shug8458"
export ARC_REPO="${ARC_USER_ROOT}/SigmaFlow_Development_JulianMueller/SigmaFlow"
export ARC_SIGMADOCK="${ARC_USER_ROOT}/SigmaDock_Reproduction_JulianMueller/sigmadock"
export ARC_DATA="${ARC_USER_ROOT}/data"
export ARC_RUNS="${ARC_USER_ROOT}/arc_runs"
export ARC_SF_ENV="${ARC_USER_ROOT}/sigmaflow_env"
export ARC_SD_ENV="${ARC_USER_ROOT}/myenv"

# wandb-Haerten. Job 8525523 verlor 36 Minuten an
#   ServicePollForTokenError: Failed to read port info after 30.0 seconds
# weil wandbs Hilfsprozess seine Portdatei nicht rechtzeitig auf dem
# knoteneigenen /tmp veroeffentlichen konnte. Drei Gegenmassnahmen: Timeout
# hoch, TMPDIR weg vom geteilten /tmp, und ein Preflight VOR dem teuren
# Datafront-Aufbau (steht in den einzelnen Skripten).
setup_wandb_hardening() {
    export WANDB__SERVICE_WAIT=300
    export TMPDIR="${ARC_USER_ROOT}/tmp/${SLURM_JOB_ID:-manual}"
    mkdir -p "$TMPDIR"
    export WANDB_DIR="$TMPDIR"
    # Nebenwirkung, die man kennen sollte: TMPDIR liegt jetzt auf NFS, deshalb
    # erzeugen die DataLoader-Worker beim Beenden harmlose
    #   OSError: [Errno 16] Device or resource busy: '.nfsXXXX'
    # Tracebacks NACH "`Trainer.fit` stopped". Das ist Rauschen, kein Fehler.
}

# $1 = RUN_ID, $2 = Codeverzeichnis, $3 = Python-Interpreter
write_run_metadata() {
    local run_id="$1" code_dir="$2" py="$3"
    export RUN_DIR="${ARC_RUNS}/${run_id}_${SLURM_JOB_ID:-manual}"
    mkdir -p "$RUN_DIR"

    {
        echo "run_id          = ${run_id}"
        echo "slurm_job_id    = ${SLURM_JOB_ID:-manual}"
        echo "slurm_job_name  = ${SLURM_JOB_NAME:-n/a}"
        echo "submitted_from  = ${SLURM_SUBMIT_DIR:-n/a}"
        echo "node            = $(hostname)"
        echo "partition       = ${SLURM_JOB_PARTITION:-n/a}"
        echo "gres            = ${SLURM_JOB_GRES:-n/a}"
        echo "start_utc       = $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "code_dir        = ${code_dir}"
        echo "python          = ${py}"
        echo "python_version  = $($py --version 2>&1)"
        echo "data_dir        = ${ARC_DATA}"
    } > "${RUN_DIR}/run_metadata.txt"

    echo "${SLURM_JOB_ID:-manual}" > "${RUN_DIR}/slurm_job_id.txt"

    # Git-Zustand des Codeverzeichnisses. `|| true`, weil nicht jedes
    # Codeverzeichnis auf ARC ein Git-Repo sein muss (die Variantenordner
    # wurden historisch per scp hochgeladen).
    {
        echo "# git -C ${code_dir}"
        git -C "$code_dir" rev-parse HEAD 2>/dev/null || echo "KEIN GIT-REPO"
        echo "# describe"
        git -C "$code_dir" describe --tags --always --dirty 2>/dev/null || true
        echo "# status --short (leer = sauber)"
        git -C "$code_dir" status --short 2>/dev/null || true
    } > "${RUN_DIR}/git_commit.txt"

    # Umgebung. pip freeze ist der teuerste Teil (~10 s) und trotzdem das,
    # was man spaeter am haeufigsten braucht.
    {
        echo "CONDA_PREFIX=${CONDA_PREFIX:-n/a}"
        echo "PYTHONPATH=${PYTHONPATH:-n/a}"
        echo "---- nvidia-smi ----"
        nvidia-smi 2>&1 || echo "nvidia-smi nicht verfuegbar"
    } > "${RUN_DIR}/environment.txt"
    "$py" -m pip freeze > "${RUN_DIR}/pip_freeze.txt" 2>/dev/null || \
        echo "pip freeze fehlgeschlagen" > "${RUN_DIR}/pip_freeze.txt"

    # Der komplette Aufrufkontext, inklusive des Skripts selbst.
    cp "${BASH_SOURCE[1]:-$0}" "${RUN_DIR}/submitted_script.sh" 2>/dev/null || true

    echo "[meta] Laufverzeichnis: ${RUN_DIR}"
}

# Nach dem Training: welchen experiments/-Ordner hat Lightning angelegt?
# train.py schreibt nach experiments/sigmadock/<timestamp>/ relativ zum cwd und
# nennt den Zeitstempel nur auf stdout. Wir suchen deshalb den neuesten Ordner,
# der NACH dem Startmarker entstanden ist - das ist eindeutiger als stdout zu parsen.
record_experiment_dir() {
    local code_dir="$1" marker="$2"
    local found
    found="$(find "${code_dir}/experiments" -mindepth 2 -maxdepth 2 -type d -newer "$marker" 2>/dev/null | sort | tail -1)"
    if [ -n "$found" ]; then
        echo "$found" > "${RUN_DIR}/experiment_dir.txt"
        ln -sfn "$found" "${RUN_DIR}/experiment" 2>/dev/null || true
        echo "[meta] Experimentordner: $found"
        ls -la "${found}/checkpoints" 2>/dev/null | tee "${RUN_DIR}/checkpoints_listing.txt" || true
    else
        echo "KEIN experiments/-Ordner gefunden" > "${RUN_DIR}/experiment_dir.txt"
        echo "[meta] WARNUNG: kein neuer Experimentordner gefunden."
    fi
    echo "end_utc         = $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RUN_DIR}/run_metadata.txt"
}
