# ARC Return Runbook

**Stand 2026-08-18.** Nur der operative Ablauf. Begründungen stehen in
`CURRENT_RESEARCH_STATE.md`.

Der kritische Pfad ist:

```
Throughput messen  ->  Horizont berechnen  ->  Preflight  ->  72h-Paar submitten
```

Alles andere läuft **parallel dazu**, nicht davor.

---

# FIRST COMMANDS WHEN ARC RETURNS

```bash
cd /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow
git pull origin main
git rev-parse --short HEAD

bash arc/00_preflight.sh

sacct -j 8583394,8583395 --format=JobID%14,JobName%14,State,ExitCode,Elapsed,Start,End

python audits/test_rotation_gradient_reachability.py
python arc/test_scheduler_horizon.py
bash   arc/test_train_rc.sh
python arc/test_aggregate_learning_curve.py
python arc/compare_final_configs.py --fail-on-confounder

cd SigmaFlow_Minimal
python ../arc/probe_datafront_size.py \
    --data-dir /data/stat-cadd/shug8458/data \
    --train pdbbind-general --val posebusters --test posebusters \
    --json-out ../arc/datafront_sizes.json
cd ..

sbatch arc/exp101_distance_audit.slurm

column -s, -t < /data/stat-cadd/shug8458/arc_runs/THROUGHPUT-*/throughput_*.csv
```

---

# Phase A — ARC ist zurück

```bash
cd /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow
git pull origin main
git rev-parse --short HEAD

# A1  Laufzeitumgebung und Zielpartition
bash arc/00_preflight.sh

# A2  Alle lokalen Gates auf ARC bestaetigen
python audits/test_rotation_gradient_reachability.py
python arc/test_scheduler_horizon.py
bash   arc/test_train_rc.sh
python arc/test_aggregate_learning_curve.py
python arc/compare_final_configs.py --fail-on-confounder
python visualization/tests/test_visualization.py

# A3  Stufe-1-Throughput pruefen
sacct -j 8583394,8583395 --format=JobID%14,JobName%14,State,ExitCode,Elapsed,Start,End
column -s, -t < /data/stat-cadd/shug8458/arc_runs/THROUGHPUT-*/throughput_*.csv

# A4  Echte Trainingsdatengroesse messen
cd SigmaFlow_Minimal
python ../arc/probe_datafront_size.py \
    --data-dir /data/stat-cadd/shug8458/data \
    --train pdbbind-general --val posebusters --test posebusters \
    --json-out ../arc/datafront_sizes.json
cd ..
```

Aus A3 wählen: höchste Batch mit `status=OK` und VRAM-Abstand, ob TF32
(`--cuda_precision high`) trägt, ob bf16 nötig ist. Passt Batch 32 nicht in den
Speicher, dann kleinere physische Batch **mit Akkumulation** auf effektiv 32 —
der Scheduler-Horizont ist dafür abgesichert.

---

# Phase B — finale Trainingskonfiguration bestimmen

```bash
# B1  Stufe 2: absoluter Durchsatz je Arm (Zweipunktmessung)
MODEL=sigmaflow_minimal STAGE=2 BATCH=<B> ACCUM=<A> \
    EXTRA="--cuda_precision high" sbatch arc/throughput_sweep.slurm
MODEL=sigmadock         STAGE=2 BATCH=<B> ACCUM=<A> \
    EXTRA="--cuda_precision high" sbatch arc/throughput_sweep.slurm

# B2  Ergebnisse: samples_per_s und n_train
cat /data/stat-cadd/shug8458/arc_runs/THROUGHPUT-*/stage2_*.json

# B3  Horizont berechnen und einfrieren
python arc/calculate_final_epochs.py \
    --arm sigmaflow_minimal --samples-per-s <SF aus B2> \
    --arm sigmadock         --samples-per-s <SD aus B2> \
    --n-train <aus B2> --physical-batch <B> --accum <A> \
    --budget-hours 70 \
    --throughput-source "stage2 jobs <IDs>" \
    --write-env arc/final_horizon.env

cat arc/final_horizon.env
```

`--budget-hours 70` ist die reine Optimierungszeit innerhalb der 72h-Zuteilung;
die übrigen 2 h decken Datafront-Aufbau, Snapshots und Abbau. Beide Arme
benutzen dieselbe Definition. Die Epochenpolitik — gemeinsame Zahl bei ≤ 5 %
Durchsatzunterschied, sonst getrennte Horizonte bei gleichem Zeitbudget — wird
vom Skript entschieden, ausgedruckt und in `final_horizon.env` festgehalten.

---

# Phase C — finales Basispaar starten

```bash
# C1  Preflight, beide Arme muessen GREEN sein
bash arc/final_72h_preflight.sh sigmaflow_minimal
bash arc/final_72h_preflight.sh sigmadock

# C2  Trockenlauf
DRY_RUN=1 bash arc/submit_final.sh sigmaflow_minimal
DRY_RUN=1 bash arc/submit_final.sh sigmadock

# C3  Start
bash arc/submit_final.sh sigmaflow_minimal
bash arc/submit_final.sh sigmadock

# C4  Ueberwachen
squeue -u $USER --format="%.12i %.9P %.20j %.8T %.10M %.11l %R"
tail -f /data/stat-cadd/shug8458/arc_runs/slurm_logs/<jobid>.out
ls -la /data/stat-cadd/shug8458/arc_runs/*/snapshots/
```

**Fortsetzen nach Walltime-Abbruch** — `FINAL_MAX_EPOCHS` dabei **nicht**
ändern. Ein Resume-Guard liest `max_steps` aus dem Checkpoint und bricht bei
Abweichung ab:

```bash
RESUME=/pfad/zu/last.ckpt bash arc/submit_final.sh sigmaflow_minimal
```

---

# Phase D — billige Parallelarbeit, während das Training läuft

```bash
# D1  IS-1-Gate (CPU-only, umgeht die GPU-Queue)
sbatch arc/exp101_distance_audit.slurm

# D2  Fester Auswertungssubset -- einmalig, danach NIE aendern
python arc/make_eval_subset.py --data_dir /data/stat-cadd/shug8458/data --n 50

# D3  Echte Fragmentzahlen je Komplex -> counts.csv
#     ad-hoc: is_lig = where(frag_idx_map != -1)
#             frags  = unique(frag_idx_map[is_lig]).numel()
#             state_dimension = 6 * frags

# D4  Faelle fuer die qualitative Abbildung waehlen
python -m visualization.select_cases --counts-csv counts.csv \
    --results-dir SigmaFlow_Variants/posebusters_full_comparison \
    --out visualisations/selection.csv

# D5  PyMOL-Kette gegen ECHTES Sampling validieren (12h-Checkpoint).
#     Vorlage fuer das Sampling-Snippet ausgeben lassen:
python -m visualization.extract_from_sampling
```

Beim Einsetzen des Snippets ist dreierlei entscheidend: `all_pos` ist
**skaliert und taschenzentriert** — `build_trajectory_state` rechnet mit
`dimensional_scale` und `pocket_com` nach Ångström zurück; `model` und
`time_kind` müssen zusammenpassen (`sigmaflow`/`ode_t` gegen
`sigmadock`/`diffusion_t`, das sind **nicht** dieselbe physikalische Größe);
und ohne `crystal_positions=batch.ref_pos[...]` bleiben alle Fehlerspalten leer.

```bash
# D6  .npz nach lokal holen (wenige MB), dann ARC-unabhaengig weiterarbeiten
scp shug8458@arc:.../trajectories/*.npz ./trajectories/

python -m visualization.build_case \
    --complex 6VTA_AKN \
    --sigmaflow trajectories/6VTA_AKN_sigmaflow_12h.npz \
    --sigmadock trajectories/6VTA_AKN_sigmadock_12h.npz \
    --receptor  receptors/6VTA_AKN.pdb \
    --out       visualisations/6VTA_AKN

pymol visualisations/6VTA_AKN/view_trajectory_sigmaflow.pml
pymol visualisations/6VTA_AKN/view_static.pml
```

In PyMOL: `scrub`, `ghost`, `final`, `frag 3`, oder `scene scrub|ghost|final`.

---

# Phase E — Snapshots auswerten, sobald sie entstehen

```bash
# E1  Billige Kurve ueber ALLE Snapshots eines Laufs
RUN_DIR=/data/stat-cadd/shug8458/arc_runs/SF_MIN_72H_s0_<jobid> \
    MODEL=sigmaflow_minimal sbatch arc/eval_snapshots.slurm
RUN_DIR=/data/stat-cadd/shug8458/arc_runs/SD_BASE_72H_s0_<jobid> \
    MODEL=sigmadock sbatch arc/eval_snapshots.slurm

# E2  Einzelner Snapshot mit voller Provenienz (SHA256, EMA, Anneal-Stand)
python arc/evaluate_snapshot.py --arm sigmaflow_minimal \
    --checkpoint <run>/snapshots/snap_sched<E>ep_at_024h.ckpt \
    --data-dir /data/stat-cadd/shug8458/data \
    --subset arc/eval_subset.txt --run

# E3  Lernkurve beider Arme
python arc/aggregate_learning_curve.py \
    --curve-csv sigmaflow_minimal <run_sf>/learning_curve/curve_sigmaflow_minimal.csv \
    --curve-csv sigmadock         <run_sd>/learning_curve/curve_sigmadock.csv \
    --out-csv learning_curve.csv --out-md learning_curve.md

# E4  Die alten, ausannealten Laeufe GETRENNT halten
python arc/aggregate_learning_curve.py --annealed-endpoint \
    --report-json sigmaflow_minimal <alt-12h>/snapshot_report.json \
    --out-csv annealed_endpoints.csv

# E5  Denselben Komplex ueber die Trainingszeit ansehen
for H in 012 024 036 072; do
  python arc/evaluate_snapshot.py --arm sigmaflow_minimal \
      --checkpoint <run>/snapshots/snap_sched<E>ep_at_${H}h.ckpt \
      --data-dir /data/stat-cadd/shug8458/data --run --out-dir traj_${H}h
done
# danach je Zeitpunkt eine .npz erzeugen (D5) und lokal bauen (D6)
```

---

# Phase F — Erweiterung, nur bei bestandenem IS-1

```bash
# Medianwinkel aus IS-1 uebernehmen, NICHT den Haar-Default 132.3 stehen lassen
MODEL=sigmaflow_source SOURCE_MODE=pocket_pca SOURCE_MEDIAN_DEG=<aus IS-1> \
    MODE=screen bash arc/submit_final.sh sigmaflow_source
```

Zeigt IS-1 keine belastbare Anreicherung gegenüber Haar, wird IS-2 **nicht**
gebaut und keine GPU-Zeit dafür verwendet.

---

# Optional, nicht auf dem kritischen Pfad

```bash
# Die zwei fehlenden SigmaDock-Punkte der NFE-Kurve (50 und 100 Schritte).
# Vorher den EXAKT gleichen Checkpoint wie in Task 0-4 aus dem alten Log holen:
cd /data/stat-cadd/shug8458/SigmaDock_Reproduction_JulianMueller/sigmadock
grep -i 'ckpt\|loaded from' slurm_logs/8562618_0.out | head
CKPT_DIR=<Pfad daraus> sbatch --array=5-6 slurm/nfe_sweep_sigmadock.sh
```

Die Wall-Clock-Achse dieses Sweeps ist durch kalten Dateisystem-Cache
kontaminiert und nicht verwertbar. NFE selbst ist unberührt.

---

# Was nie passieren darf

- `FINAL_MAX_EPOCHS=36` oder `=128` als Default ins Repository zurückschreiben.
- `arc/final_horizon.env` von Hand schreiben. Sie wird erzeugt; ein editierter
  Wert wird durch die Nachrechnung von `max_steps` erkannt und abgelehnt.
- Snapshots aus dem 72h-Lauf mit den alten, eigenständig ausannealten
  6h/12h-Läufen in eine Kurve legen.
- `posebusters_ligandonly_SigmaDock_12h.csv` benutzen — nur `_lrfix.csv` gilt.
- Den Auswertungssubset nach dem ersten Snapshot ändern.
- Am Modellcode in `SigmaFlow_Minimal/` oder `SigmaDock/` etwas für
  Visualisierung, Logging, Durchsatz oder Snapshots ändern.
