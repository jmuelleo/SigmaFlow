# ARC Runbook & Current State

> **Single Source of Truth** für den Projektstand und alles, was auf ARC als
> Nächstes passiert. Stand 2026-08-13.
>
> Bei Widersprüchen zu `STATUS.md` gilt dieses Dokument für den *aktuellen*
> Stand, `STATUS.md` für die *Historie*.

---

## 0. Wenn ARC wieder läuft — die ersten fünf Befehle

```bash
# 1. Lokal (auf dem Laptop): die 7 Commits sind noch nicht auf GitHub.
#    Ohne diesen Schritt hat ARC weder SigmaFlow_Minimal noch EXP-100.
cd /c/Users/julia/Documents/SigmaFlow && git push origin main

# 2. Auf ARC: Code holen
ssh <arc>            # dein üblicher Login
cd /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow
git pull origin main && git rev-parse --short HEAD

# 3. Preflight — prüft Pfade, Envs, Partitionen, GPU-Klasse, Frame-Fix. Kein Job.
bash arc/00_preflight.sh

# 4. Inventur — existiert doch schon eine gültige 24h-Baseline?
bash arc/01_inventory_existing_runs.sh

# 5. Erster echter Job: EXP-100 auf ARC absichern (~40 min, short)
sbatch arc/train_exp100_sanity.slurm
```

Danach weiter bei [Abschnitt 9](#9-copy-paste-commands-when-arc-is-back).

---

## 0a. Reihenfolge der finalen Kampagne

Die Schritte sind so sortiert, dass **jede Messung vor der Entscheidung
kommt, die sie tragen soll**. Kein Schritt darf vorgezogen werden, weil sein
Ergebnis sonst geraten statt gemessen wäre.

| # | Job | Dauer | Wozu | Blockiert |
|---|---|---|---|---|
| 1 | `arc/00_preflight.sh` | — | Pfade, Envs, Partitionen, GPU-Klasse | alles |
| 2 | `arc/train_exp100_sanity.slurm` | ~40 min | EXP-100 läuft auf ARC | 5 |
| 3 | `sbatch arc/exp101_source_audit.slurm` | ≤2 h, **keine GPU** | Gate für den Source-Strang | EXP-102 |
| 4 | `sbatch arc/throughput_sweep.slurm` | ~6 h | Batch/Precision/Durchsatz messen | 5 |
| 5 | `bash arc/submit_final.sh <model>` | 72 h | die eigentlichen Läufe | Auswertung |

Schritt 3 braucht **weder GPU noch Checkpoint** und kann parallel zu 2 und 4
laufen — er ist der billigste Job der ganzen Kampagne und entscheidet trotzdem
über einen kompletten Experimentstrang.

```bash
# 3. Source-Audit (Gate für EXP-102). Standard misst gegen EXP-100.
sbatch arc/exp101_source_audit.slurm                        # beide Hälften
DATASET=pdbbind-core sbatch arc/exp101_source_audit.slurm     # 10-min-Vorlauf
TREE=minimal         sbatch arc/exp101_source_audit.slurm     # nur Translation

# 4. Durchsatz messen — DANACH arc/final_config.sh ausfüllen.
sbatch arc/throughput_sweep.slurm

# 5. Finale Läufe. Erst nach Schritt 4, sonst ist max_epochs geraten.
DRY_RUN=1 bash arc/submit_final.sh sigmadock          # anzeigen, nicht submitten
bash arc/submit_final.sh sigmadock
bash arc/submit_final.sh sigmaflow_minimal
```

### Warum Schritt 4 nicht übersprungen werden darf

`final_config.sh` liefert bewusst nur **Platzhalter**. `FINAL_MAX_EPOCHS=128`
ist eine Zielgröße, keine gemessene: bei den bisher beobachteten
2.94 Samples/s wären 128 Epochen ≈ 3.3× zu langsam für 72 h. Ob die
Beschleunigung reicht, entscheiden die drei ungemessenen Hebel des Sweeps —
`--debug` (detect_anomaly + NaN-Callback), `cuda_precision` (TF32) und
`val_check_interval` (bisher ~49 Validierungen pro Epoche).

Wird der Sweep übersprungen, läuft das Training in einen Scheduler, dessen
`max_steps` aus einer falschen Epochenzahl stammt — die LR-Kurve passt dann
nicht zur tatsächlichen Laufzeit, und das ist **nach** 72 h nicht mehr
reparierbar.

---

## 1. Projektstand

### 1.1 SigmaDock (Referenz, unverändert)

| | |
|---|---|
| Ort lokal | `SigmaDock/` |
| Ort ARC | `/data/stat-cadd/shug8458/SigmaDock_Reproduction_JulianMueller/sigmadock` |
| Conda-Env | **`myenv`**, aktiviert mit `source activate` — **nicht** `sigmaflow_env` |
| Status | eingefroren, read-only laut `CLAUDE.md` |
| Trainings-Config | `--train_exps pdbbind-general --val_exps posebusters --batch_size 8 --trans_score_weight 2.0 --rot_score_weight 0.5 --rot_score_method space --rot_score_scaling rms` |
| Sampling-Config | `experiment=posebusters graph.sample_conformer=false ode.num_steps=25` |

**Bekannte offene Punkte.** Die Env-Trennung ist keine Kosmetik: `sigmaflow_env`
hat SigmaFlow_Development in seine site-packages installiert, dort löst
`import sigmadock` auf das SigmaFlow-Paket auf und stirbt mit
`ModuleNotFoundError: No module named 'sigmadock.diff.denoiser'`. Genau das hat
Job 8512799 nach 5:44 gekillt.

### 1.2 SigmaFlow Minimal

| | |
|---|---|
| Ort | `SigmaFlow_Minimal/` |
| Tag / Commit | `sigflow-minimal-baseline-v1` / `267fb69` |
| Status | **eingefroren**, wird nicht mehr verändert |
| Mathematik | R³ linear, SO(3) geodätisch, Body-Frame durchgängig, Frame-Fix verifiziert |
| Lokale Tests | Oracle-Vektorfeld exakt; Schrittzahl 5→200 wirkungslos; Gauge- und SE(3)-Invarianz |

**Offene ARC-Validierungen:** `cosine_by_t`-Diagnose und globale
SE(3)-Äquivarianz (eingereicht, wegen Wartung nie gelaufen).

### 1.3 EXP-100 State Reparameterization

| | |
|---|---|
| Ort | `SigmaFlow_FM_Specific/EXP-100_state_reparam/` |
| Commit | `0fdd9f5` |
| Status | **`LOCALLY VALIDATED — READY FOR ARC`** |

**Geändert** (3 Dateien, alle Kategorie A):
`data.py` (rein additiv: `ref_conf_pos`), `sigma_flow_generator.py`
(`get_fragment_com_and_rot_reparam`, `R_1`→`R_ref` an 6 Stellen),
`sampling.py` (6 Aufrufstellen), plus die neue `diff/state_reparam.py`.

**Unverändert:** alle 18 `net/`-Dateien, Loss, Optimizer, LR, Integrator, NFE,
Source-Verteilungen, `SigmaFlow_Minimal`, `SigmaDock`.

**Bestandene lokale Tests:** vier Skripte, 0 Fehlschläge — Mathematik
(7 Kategorien), Indexabbildung (chemisch + geometrisch), Pipeline (T1–T6),
echter Trainingsschritt mit EquiformerV2 inkl. beider Sampler.

**Fehlende ARC-Validierung:** GPU/AMP, echte Datensätze statt der 10
Dummy-Komplexe, Durchsatzkosten der zusätzlichen ETKDGv3+MMFF-Generierung, und
der eigentliche Trainingsvergleich.

> **Präzisierung gegenüber einer früheren Formulierung.** Es hieß, EXP-100 sei
> an der Inferenz ein exakter No-Op. Das gilt nur für `sample_conformer=true`.
> **Alle** Sampling-Skripte setzen `graph.sample_conformer=false`. In diesem
> Modus ändert EXP-100 auch die Inferenz: `C_F` wird der leakage-freie
> Konformer statt der Fragmentgeometrie aus der gebundenen Pose. Das bleibt
> eine kohärente Einzelintervention, aber sie wirkt auf beiden Seiten.

### 1.4 FM Research Roadmap

Details: [`FM_SOURCE_ROADMAP.md`](SigmaFlow_FM_Specific/FM_SOURCE_ROADMAP.md),
Priorisierung: [`SIGMAFLOW_RESEARCH_ROADMAP.md`](SIGMAFLOW_RESEARCH_ROADMAP.md).

| ID | Inhalt | Status |
|---|---|---|
| EXP-101 | Source Distance Audit (reine Messung) | Skript fertig, **ARC-ready** |
| EXP-102 | heuristische konditionierte Quelle | `NOT STARTED` |
| EXP-103 | gelernte konditionierte Quelle | `NOT STARTED` |
| EXP-104 | Multi-Sampling | Audit-Skript fertig |
| EXP-105 | Confidence Ranking | `NOT STARTED` |
| EXP-106 | Early Pruning | `NOT STARTED` |

Keine davon wird in dieser Runde gestartet.

---

## 2. Inventur bestehender Läufe

### 2.1 Die 24h-Läufe vom 2026-08-07

| Job | Modell | Ende | Status |
|---|---|---|---|
| 8465054 | SigmaFlow | `TIMEOUT` nach `1-00:00:17`, Exit `0:0` | **NOT VALID** |
| 8465055 | SigmaDock | `TIMEOUT` nach `1-00:00:17`, Exit `0:0` | **NOT VALID as matched baseline** |

**SigmaFlow 8465054 — zwei unabhängige Ausschlussgründe:**

1. **Vor dem Frame-Fix.** Die Ursache (Welt- vs. Körperrahmen in der
   Rotationsvorhersage) wurde erst am **2026-08-09** gefunden, der Lauf war am
   **2026-08-07**. Der Fix war real und wirksam (Lokalitätslücke +1.1° → −12.8°).
2. **Unkalibrierter LR-Zeitplan.** Das Skript benutzte `--max_epochs 1000`.

**SigmaDock 8465055 — der Code ist in Ordnung, das Budget nicht:**
Derselbe `--max_epochs 1000`. Ein Vergleich gegen eine neue, LR-kalibrierte
SigmaFlow-Seite wäre unfair — nicht wegen des Modells, sondern wegen des
Lernratenverlaufs. Fairness geht vor Ersparnis von 24 GPU-Stunden.

### 2.2 Warum `max_epochs` das Budget ist, nicht die Walltime

`scripts/train.py:218`:

```python
max_steps = args.max_epochs * len(train_datafront) // (args.batch_size * args.world_size)
```

`max_steps` speist den LR-Scheduler. Die kalibrierte Reihe:

| Budget | `--max_epochs` | ≈ Schritte |
|---|---|---|
| 6 h | 3 | ~7 100 |
| 12 h | 6 | ~14 250 |
| **24 h** | **12** | **~28 500** |

Mit `max_epochs=1000` wäre der Zeitplan auf ~2,4 Mio Schritte ausgelegt; gelaufen
sind ~28 000. Die Lernrate blieb praktisch konstant. Das ist genau der Fehler,
den die spätere „lrfix"-Reihe behoben hat.

**Antwort auf die Frage Walltime vs. Budget:** Es ist **Variante C** — eine feste
Zahl Epochen, die zum Zeitbudget kalibriert ist, mit der Walltime als
Sicherheitsnetz. Beide Seiten bekommen dieselbe Walltime, dieselbe GPU-Klasse
und dasselbe `max_epochs`.

### 2.3 Verdikt

| | Gültiger 24h-Lauf vorhanden? |
|---|---|
| **SigmaDock** | **NO** — 8465055 hat `max_epochs=1000`, nicht vergleichbar → neuer Lauf **RUN-B24-SD** |
| **SigmaFlow Minimal** | **NO** — 8465054 ist vor-Frame-Fix *und* `max_epochs=1000` → neuer Lauf **RUN-B24-SF** |
| **EXP-100** | **NO** — existiert noch nie auf ARC → **RUN-B24-E100**, erst nach dem Sanity-Lauf |

`arc/01_inventory_existing_runs.sh` prüft dieses Verdikt auf ARC nach.
**Falls es widerspricht** — etwa weil ein Log doch `--max_epochs 12` zeigt —
nicht neu starten, sondern dieses Dokument korrigieren.

---

## 3. Fairness der 24h-Läufe

| Aspekt | RUN-B24-SF | RUN-B24-SD | identisch? |
|---|---|---|---|
| Datensatz | pdbbind-general | pdbbind-general | ✅ |
| Val/Test | posebusters | posebusters | ✅ |
| Seed | 0 | 0 | ✅ |
| Batch size | 8 | 8 | ✅ |
| `max_epochs` (= LR-Zeitplan) | 12 | 12 | ✅ |
| `val_check_interval` | 50 | 50 | ✅ |
| Loss-Gewichte | 2.0 / 0.5 | 2.0 / 0.5 | ✅ |
| Early stopping | aus | aus | ✅ |
| Präzision, Scheduler, Weight decay, EMA | Default `config.py` | Default `config.py` | ✅ |
| GPU-Klasse | `l40s`, 1 GPU | `l40s`, 1 GPU | ✅ |
| Walltime / Partition | 24 h / `medium` | 24 h / `medium` | ✅ |
| Checkpointing | `save_top_k=3`, `save_last` | dito | ✅ |
| Conda-Env | `sigmaflow_env` | `myenv` | ⚠️ **methodenspezifisch, zwingend** |
| `--rot_score_method space` | – | ✅ | ⚠️ **methodenspezifisch** |
| `--rot_score_scaling rms` | – | ✅ | ⚠️ **methodenspezifisch** |

Die drei markierten Abweichungen sind unvermeidbar: die Env-Trennung aus
technischen Gründen (§1.1), die beiden `rot_score_*`-Flags, weil sie SigmaDocks
Score-Parametrisierung auf SO(3) steuern und in einem Flow-Matching-Modell kein
Gegenstück haben. Sie sind Teil der bekannten guten SigmaDock-Konfiguration.

**Was nicht garantiert ist:** gleiche *Anzahl Optimierungsschritte in 24 h*.
Beide laufen bis zur Walltime; wenn eine Methode pro Schritt langsamer ist,
macht sie weniger Schritte. Das ist gewollt — verglichen wird **gleiche reale
Rechenzeit auf gleicher Hardware**, nicht gleiche Schrittzahl.
`arc/summarize_arc_results.py` gibt beide Zahlen aus, damit der Unterschied
sichtbar bleibt.

---

## 4. Checkpointing und Resume

`ModelCheckpoint(save_top_k=3, monitor=loss_val/total, save_last=True)` bei
`val_check_interval=50`. `last.ckpt` wird also alle 50 Schritte überschrieben —
ein Knotenausfall kostet höchstens wenige Minuten.

**Resume wird unterstützt** (`--resume_from_checkpoint <pfad>`), mit einer
Falle, die schon einmal zugeschlagen hat:

> `--resume_from_checkpoint` schreibt in **denselben** Experimentordner. Mit
> `save_top_k=3` verdrängen die neuen Checkpoints die alten. Nach einem Resume
> sind die ursprünglichen Checkpoints **weg**. Wer den Vorher-Zustand behalten
> will, muss den Ordner vorher kopieren.

```bash
# Frischer Lauf
sbatch arc/train_sigmaflow_minimal_24h.slurm

# Fortsetzen nach Knotenausfall — CKPT aus arc_runs/<RUN>/experiment_dir.txt
CKPT=/data/.../experiments/sigmadock/<ts>/checkpoints/last.ckpt
cp -r "$(dirname "$CKPT")" "$(dirname "$CKPT")_backup_$(date +%s)"   # gegen die Falle oben
sbatch --export=ALL,RESUME="$CKPT" arc/train_sigmaflow_minimal_24h.slurm
```

⚠️ Die Skripte nehmen `RESUME` derzeit **nicht** entgegen — sie starten immer
frisch. Für ein Resume muss `--resume_from_checkpoint "$RESUME"` von Hand in den
`train.py`-Aufruf ergänzt werden. Bewusst so gelassen: ein automatisches Resume
hätte im Fehlerfall stillschweigend einen falschen Checkpoint fortgesetzt.

---

## 5. Sampling und Evaluation

### 5.1 Zwei Vergleichsarten, nicht mischen

| | Was | Wie |
|---|---|---|
| **Controlled** | gleiche NFE auf beiden Seiten | `NUM_STEPS=25` überall |
| **Method-default** | jede Methode mit ihrem Vorgabewert | SigmaDock: Diffusions-Sampler; SigmaFlow: ODE |

Beide sind wissenschaftlich legitim, beantworten aber verschiedene Fragen.
Getrennt berichten.

### 5.2 Raw vs. Ranked

`arc/sample_pb_seeds.slurm` setzt `postprocessing.scoring=null` — das ist die
**RAW/Generator**-Auswertung ohne methodeneigenes Ranking. Für die
**Ranked**-Variante dasselbe Skript mit gesetztem `scoring` in ein *anderes*
`OUT_ROOT` fahren. Nur so lässt sich später sagen, was vom Generator kommt und
was vom Ranking.

### 5.3 Metriken

Vorhanden und belastbar: RMSD (raw, aligned, symmetriekorrigiert), <2 Å, <5 Å,
Median, Centroid-Fehler, Rotationsfehler je Fragment, PoseBusters (28 Checks),
Oracle@K.

> **`TFD OPTIONAL / REQUIRES ROBUST SANITIZATION`** — die vorhandene
> TFD-Implementierung hat weiterhin eine nennenswerte Fehlerrate durch
> Sanitisierungsprobleme. Nicht als Hauptmetrik führen.

---

## 6. Erwartete Ausgabepfade

```
/data/stat-cadd/shug8458/arc_runs/
├── slurm_logs/<jobid>.out|.err
├── RUN-B24-SF_<jobid>/       run_metadata.txt, git_commit.txt, config,
│                             slurm_job_id.txt, environment.txt, pip_freeze.txt,
│                             train_stdout.log, experiment_dir.txt, experiment -> …
├── RUN-B24-SD_<jobid>/
├── RUN-B24-E100_<jobid>/
├── EXP100-SANITY_<jobid>/    zusätzlich test_*.log, real_data_check.log
├── EXP101-DISTANCE_<jobid>/
├── AUDIT-<model>_<jobid>/    oracle_k_*.log, symmetry_*.log
└── sampling/<model>_nfe<N>/results/posebusters/<model>/seed_<k>/*.sdf
```

Die Checkpoints selbst bleiben, wo Lightning sie hinschreibt
(`<code_dir>/experiments/sigmadock/<timestamp>/checkpoints/`); jedes
Laufverzeichnis enthält den aufgelösten Pfad in `experiment_dir.txt` und einen
Symlink `experiment`.

---

## 7. „Done"-Kriterien

**RUN-B24-SF / RUN-B24-SD / RUN-B24-E100 gelten als fertig, wenn:**

- [ ] `sacct` zeigt `TIMEOUT` mit ExitCode `0:0` (bei 24h-Walltime erwartet) oder `COMPLETED`
- [ ] `experiment_dir.txt` zeigt auf einen existierenden Ordner
- [ ] `checkpoints/last.ckpt` existiert und ist < 30 min vor dem Kill geschrieben
- [ ] keine NaN im Trainingslog (`summarize_arc_results.py` zählt sie)
- [ ] Schrittzahl plausibel (~28 500 bei `max_epochs=12`, oder dokumentiert warum weniger)
- [ ] alle fünf Metadatendateien vorhanden
- [ ] Fragment-Mass-Skips und `Failed to parse pocket`-Skips notiert (Vergleichbarkeit beider Seiten)
- [ ] Sampling startet mit diesem Checkpoint fehlerfrei

**EXP100-SANITY zusätzlich:**

- [ ] alle vier Testskripte auf ARC grün
- [ ] `ref_conf_pos` auf echten Daten endlich, korrekte Form
- [ ] `R_1`-Median > 20° auf echten Daten (sonst wirkt EXP-100 nicht)
- [ ] Durchsatz notiert — kostet die Konformergenerierung Schritte/s?

---

## 8. Empfohlene ARC-Reihenfolge

Gegenüber dem ursprünglichen Vorschlag in zwei Punkten geändert:

**Die Diagnose kommt vor die 24h-Läufe.** `SIGMAFLOW_RESEARCH_ROADMAP.md` §2.1
zeigt, dass SigmaFlows Rotationskanal auf Zufallsniveau arbeitet. Die
`cosine_by_t`-Diagnose kostet Minuten und entscheidet, welcher Forschungsast
überhaupt sinnvoll ist. Sie vor 48 GPU-Stunden zu fahren ist fast gratis.

**EXP-100 24h ist optional und wird erst nach dem Sanity-Lauf entschieden.**

| Phase | Schritt | Kosten |
|---|---|---|
| **1 Baseline validity** | Preflight + Inventur | Minuten |
| | `cosine_by_t`-Diagnose (Rotationskanal) | Minuten |
| | EXP-100 Sanity | ~40 min |
| **2 Baselines** | RUN-B24-SD | 24 h |
| | RUN-B24-SF | 24 h (parallel) |
| | RUN-B24-E100 *(optional)* | 24 h |
| **3 Evaluate** | Sampling, 10 Seeds je Modell | ~3 h je Array |
| | PoseBusters + RMSD + Rotation | Minuten |
| | Oracle@K-Audit | Minuten |
| **4 Research decisions** | EXP-101 Distance Audit | ~1 h |
| | Symmetrie-Audit *(lokal bereits beantwortet)* | Minuten |
| | Entscheidung: Rotationskopf vs. konditionierte Quelle | – |

---

## 9. COPY-PASTE COMMANDS WHEN ARC IS BACK

Pfade sind echt, keine Platzhalter außer den Checkpoint-Pfaden, die erst nach
dem Training feststehen.

### Phase 0 — Code auf ARC bringen

```bash
# --- 1. LOKAL auf dem Laptop ---
cd /c/Users/julia/Documents/SigmaFlow
git status
git push origin main

# --- 2. AUF ARC ---
cd /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow
git pull origin main
git rev-parse --short HEAD          # muss den lokalen HEAD zeigen

# --- 3. Preflight (kein Job, ändert nichts) ---
bash arc/00_preflight.sh
# Endet mit READY oder einer Blockerliste. Bei Blockern nicht weitermachen.

# --- 4. Inventur bestehender Läufe ---
bash arc/01_inventory_existing_runs.sh
```

### Phase 1 — Diagnose und Absicherung

```bash
# 5. Rotationskanal-Diagnose (Minuten, entscheidet die Forschungsrichtung)
cd /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow
sbatch SigmaFlow_Minimal/slurm/diag_rotation_completion.sh
squeue -j $!   # oder: squeue -u $USER

# 6. EXP-100 auf ARC absichern (~40 min)
sbatch arc/train_exp100_sanity.slurm
```

Prüfen:

```bash
squeue -u $USER -o "%.12i %.10P %.24j %.8T %.10M %.12l %R"
sacct -j <JOBID> --format=JobID,JobName%24,State,ExitCode,Elapsed
tail -40 /data/stat-cadd/shug8458/arc_runs/slurm_logs/<JOBID>.out
```

### Phase 2 — die beiden 24h-Baselines

```bash
# 7. Beide gleichzeitig einreichen — gleiche Warteschlange, gleiche Bedingungen
sbatch arc/train_sigmadock_24h.slurm
sbatch arc/train_sigmaflow_minimal_24h.slurm

# 8. Optional, NUR wenn Schritt 6 grün war
sbatch arc/train_exp100_24h.slurm

# 9. Überwachen
squeue -u $USER -o "%.12i %.10P %.24j %.8T %.10M %.12l %R"
```

### Phase 3 — Sampling (nach dem Training)

```bash
# 10. Checkpointpfade auflösen
python arc/summarize_arc_results.py
# Gibt am Ende "CKPT=..." je Lauf aus. Diese Pfade unten einsetzen.

# 11. Sampling, 10 Seeds je Modell, gleiche NFE (controlled comparison)
MODEL=sigmadock         NUM_STEPS=25 CKPT=<CKPT_SD>   sbatch --array=0-9 arc/sample_pb_seeds.slurm
MODEL=sigmaflow_minimal NUM_STEPS=25 CKPT=<CKPT_SF>   sbatch --array=0-9 arc/sample_pb_seeds.slurm
MODEL=exp100            NUM_STEPS=25 CKPT=<CKPT_E100> sbatch --array=0-9 arc/sample_pb_seeds.slurm
```

### Phase 4 — Auswertung

```bash
# 12. Oracle@K und Symmetrie je Modell
TRUE_DIR=/data/stat-cadd/shug8458/data/posebusters
MODEL=sigmaflow_minimal NFE=25 TRUE_DIR=$TRUE_DIR sbatch arc/post_sampling_audits.slurm
MODEL=sigmadock         NFE=25 TRUE_DIR=$TRUE_DIR sbatch arc/post_sampling_audits.slurm

# 13. EXP-101 Source Distance Audit
sbatch arc/exp101_distance_audit.slurm

# 14. Gesamtübersicht
python arc/summarize_arc_results.py
```

### Phase 5 — Ergebnisse auf den Laptop holen

```bash
# Auf dem Laptop, nicht auf ARC
scp -r <arc>:/data/stat-cadd/shug8458/arc_runs/RUN-B24-*   ./arc_results/
scp -r <arc>:/data/stat-cadd/shug8458/arc_runs/AUDIT-*     ./arc_results/
scp -r <arc>:/data/stat-cadd/shug8458/arc_runs/sampling    ./arc_results/
```

---

## 10. Job-Status lesen

| Status | Bedeutung | Zu tun |
|---|---|---|
| `PD` + `Priority` / `Resources` | normal, wartet auf freie Knoten | warten |
| `PD` + **`ReqNodeNotAvail`** | angeforderte Knoten in Wartung oder reserviert | oft löst es sich selbst; hält es an, `--time` senken (24 h passt nicht in `short`) oder Partition wechseln |
| `PD` + `QOSMaxJobsPerUserLimit` | zu viele eigene Jobs | warten oder `scancel` auf unwichtige |
| `R` | läuft | `tail -f` auf das Log |
| `TIMEOUT` mit ExitCode `0:0` | **bei 24h-Läufen erwartet**, kein Fehler | Checkpoint prüfen |
| `FAILED` mit `0:0` | meist ein Sanity-Gate hat gegriffen | `.err` lesen |
| `CANCELLED` ohne eigenes `scancel` | Wartung oder Knotenausfall | Resume, §4 |

Nützlich:

```bash
squeue -u $USER -o "%.12i %.10P %.24j %.8T %.10M %.12l %R"
sacct -S today -u $USER --format=JobID,JobName%24,Partition,State,ExitCode,Elapsed
scontrol show job <JOBID> | head -30
sinfo -o "%20P %10a %12l %8D %s"
```

**Während der Wartung:** laufende Jobs nicht anfassen. Neue Jobs bleiben
`PD/ReqNodeNotAvail`, bis die Knoten zurück sind.

---

## 11. Offene Blocker, die nur ARC klären kann

| Blocker | Betrifft |
|---|---|
| Läuft EXP-100 auf GPU und echten Daten? | EXP-100 |
| Kostet die Konformergenerierung Durchsatz? | EXP-100 |
| Shrinkage oder Rauschen im Rotationskanal? | die gesamte Forschungspriorisierung |
| Globale SE(3)-Äquivarianz des trainierten Modells | Minimal |
| Existiert `l40s` noch als GPU-Klasse? | alle Skripte |
| Senkt eine Heuristik die Rotationsdistanz? | EXP-101 → EXP-102 |

---

## 12. Nach den Baselines — Forschungsreihenfolge

Aus `SIGMAFLOW_RESEARCH_ROADMAP.md`, hier nur die Reihenfolge:

1. **Rotationskanal-Diagnose** — entscheidet zwischen den beiden Ästen
2. **EXP-101 Distance Audit** — Abbruchkriterium für den Quellen-Strang
3. **Genau ein Ast:** direkter Rotationskopf *(bei Rauschen)* **oder**
   heuristische konditionierte Quelle *(bei Shrinkage)*
4. **Multi-Sampling A vs. C** — billig, nutzt das vorhandene Modell
5. **Confidence Ranking** — nur wenn 4 die Lücke bestätigt *und* der Generator repariert ist
6. **Probability Path** — mittlere Priorität, warnender Präzedenzfall (Zeitgewichtung war wirkungslos)
7. ~~Symmetry-aware Rotation~~ — **gemessen widerlegt**, Kategorie D
