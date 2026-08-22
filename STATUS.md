# SigmaFlow — Development Status

Diese Datei ist das "Lesezeichen" für den Projektfortschritt. Am Anfang jeder
neuen Session: diese Datei zuerst lesen, dann nahtlos weitermachen. Am Ende
jeder Session (oder vor einer Pause): diese Datei aktualisieren.

**Paralleles Vorhaben, separat getrackt:** seit 2026-07-28 läuft zusätzlich
ein vollständiger, Zeile-für-Zeile-Code-Walkthrough des gesamten Repos (alle
66 `.py`-Dateien, ~20.225 Zeilen) zu reinen Lernzwecken für den User — siehe
`CODE_WALKTHROUGH.md` für den Fortschritts-Tracker. Das ist unabhängig vom
inhaltlichen Entwicklungsfortschritt unten (kein neuer Code, keine
Änderungen) und sollte bei Session-Start ebenfalls geprüft werden, falls der
User dort weitermachen will.

---

# ⏭️ HIER WEITERMACHEN (Stand 2026-08-22)

## Wo es steht

**EXP-110 ist gelaufen, gesampelt und ausgewertet.** Training `8625634`
(`COMPLETED`, 11:10:59, 6 Epochen, 99,11 % des Anneals), Sampling `8629345`
(10 Seeds auf CPU, 2090 Posen), Auswertung liegt vor.

**Ergebnis in einem Satz:** EXP-110 schlägt Minimal bei Top-1 signifikant
(7,2 % gegen 2,4 %, p = 0,021) und erreicht SigmaDocks Ziehungskonsistenz
exakt (Oracle@10/Oracle@1 = 4,80 gegen 4,71), hebt die Obergrenze aber
**nicht** an (Oracle@10 34,4 % gegen 30,1 %, p = 0,27 nicht signifikant,
SigmaDock 47,4 %). Der zweite Kopf beseitigt Ziehungsvarianz, statt mehr
Komplexe loesbar zu machen. Details in `RESULTS.md`.

**Das 72h-Paar ist weiterhin blockiert**, aber nicht mehr an der Kapazität,
sondern an einem Skriptfehler, der jetzt behoben ist. Stufe 2 der
Durchsatzmessung muss neu abgeschickt werden.

| Strang | Zustand |
|---|---|
| **A — 72h-Paar** | wartet auf Stufe 2 der Durchsatzmessung, Fix liegt vor |
| **B — EXP-110 Zwei-Kopf** | **12h-Lauf fertig**, Sampling läuft |

## Nächster Schritt

```bash
cd /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow && git pull
MODEL=sigmaflow_minimal STAGE=2 BATCH=32 EXTRA="--cuda_precision high" sbatch arc/throughput_sweep.slurm
MODEL=sigmadock         STAGE=2 BATCH=32 EXTRA="--cuda_precision high" sbatch arc/throughput_sweep.slurm
```

Bei OOM auf `BATCH=16 ACCUM=2` ausweichen, das erhält die effektive
Batchgröße 32. Danach `calculate_final_epochs.py`, Preflight, 72h-Paar.

**Termin:** Abgabe 2026-09-14. Zwei mal 72 h seriell sind sechs Tage plus
Auswertung, Start also spätestens **2026-08-29**. ARC-Support hat am 21.08.
die Durchsatzjobs priorisiert und sie liefen sofort an; der Kontakt ist also
offen und nutzbar.

## ARC-Fahrplan: 40 Seeds vollständig machen (Stand 2026-08-22 abends)

Reihenfolge einhalten. Schritt 1 blockiert alles, was SigmaDock betrifft.

### Stand der Posen

| Arm | Seeds auf ARC | Ort |
|---|---:|---|
| sigmaflow_minimal | **40** (8360 SDF) | `arc_runs/sampling/sigmaflow_minimal__controlled__raw__nfe25__cpu` |
| exp110 | **40** (8360 SDF) | `arc_runs/sampling/exp110__controlled__raw__nfe25__cpu` |
| sigmadock | **0** | Verzeichnis existiert, ist leer |

**SigmaDock fehlt vollständig.** Job 8629868 (Seeds 0–9) und Array 8629911
(Seeds 10–39) sind beide am selben Fehler gescheitert: die Skripte übergaben
unbedingt `ode.num_steps`, SigmaDock hat aber keinen `ode`-Block, der
Schlüssel heißt dort `diffusion.num_steps`. Behoben in Commit `789efdc`.

Die SigmaDock-Posen, auf denen alle bisherigen Zahlen beruhen, liegen nur
lokal (`eval_sd_root/`, 2090 SDF) und stammen aus einem Lauf mit dem Tag
`last`, dessen Verzeichnis auf ARC nicht mehr auffindbar ist. Herkunft daher
**nicht belegbar** — der Neulauf ersetzt sie.

### Schritt 1: SigmaDock neu sampeln

Probe `8630732` läuft bereits mit dem korrigierten Skript. Erst prüfen:

```bash
D=/data/stat-cadd/shug8458/arc_runs/sampling/sigmadock__controlled__raw__nfe25__cpu
sacct -j 8630732 --format=JobID%16,State%12,Elapsed,ExitCode -X
ls "$D"/results/posebusters/*/seed_0/*.sdf | wc -l    # muss 209 sein
```

Nur wenn beides stimmt (COMPLETED und 209):

```bash
cd /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow
CKPT_SD=/data/stat-cadd/shug8458/SigmaDock_Reproduction_JulianMueller/sigmadock/experiments/sigmadock/0-08-11_17-57-05/checkpoints/last.ckpt
MODEL=sigmadock CKPT="$CKPT_SD" sbatch --array=1-39 arc/sample_pb_seeds_cpu.slurm
```

`CKPT_SD` muss in **derselben Shell** gesetzt werden wie das `sbatch`.

### Schritt 2: Validität mit Protein, alle Arme, 40 Seeds

Script `arc/posebusters_redock.slurm` (Commit `5981fdd`). Erst eine Probe:

```bash
MODEL=exp110 sbatch --array=0 arc/posebusters_redock.slurm
```

Der Task druckt die drei Raten ins Log. Wenn plausibel:

```bash
MODEL=exp110            sbatch --array=1-39 arc/posebusters_redock.slurm
MODEL=sigmaflow_minimal sbatch --array=0-39 arc/posebusters_redock.slurm
MODEL=sigmadock         sbatch --array=0-39 arc/posebusters_redock.slurm   # nach Schritt 1
```

Ergebnis: `arc_runs/posebusters_redock/<model>/rd_<model>_seed<N>.csv`

### Schritt 3: Prüfen, dann herunterladen

Immer beides — SLURM COMPLETED ist kein Nachweis:

```bash
for M in sigmaflow_minimal exp110 sigmadock; do
  D=/data/stat-cadd/shug8458/arc_runs/sampling/${M}__controlled__raw__nfe25__cpu
  R=/data/stat-cadd/shug8458/arc_runs/posebusters_redock/${M}
  echo "$M: $(find $D -type d -name 'seed_*' 2>/dev/null | wc -l) Seeds, \
$(find $D -name '*.sdf' 2>/dev/null | wc -l) SDF, \
$(ls $R/*.csv 2>/dev/null | wc -l) redock-Tabellen"
done
```

Soll je Arm: **40 Seeds, 8360 SDF, 40 Tabellen**.

Packen, mit Zählung und Prüfsumme (ein früherer Transfer war abgeschnitten
und fiel nur an der Dateizahl auf):

```bash
cd /data/stat-cadd/shug8458/arc_runs
for M in sigmaflow_minimal exp110 sigmadock; do
  tar czf ~/${M}_40seeds.tar.gz -C sampling/${M}__controlled__raw__nfe25__cpu results
  echo "$M SDF: $(tar tzf ~/${M}_40seeds.tar.gz | grep -c '\.sdf$')  \
$(du -h ~/${M}_40seeds.tar.gz | cut -f1)  $(md5sum ~/${M}_40seeds.tar.gz | cut -d' ' -f1)"
done
tar czf ~/redock_40seeds.tar.gz -C /data/stat-cadd/shug8458/arc_runs posebusters_redock
echo "redock CSV: $(tar tzf ~/redock_40seeds.tar.gz | grep -c '\.csv$')  \
$(md5sum ~/redock_40seeds.tar.gz | cut -d' ' -f1)"
```

Lokal (Zielverzeichnis `SigmaFlow_Variants/posebusters_full_comparison/`):

```
scp shug8458@arc-login.arc.ox.ac.uk:~/{sigmaflow_minimal,exp110,sigmadock}_40seeds.tar.gz \
    shug8458@arc-login.arc.ox.ac.uk:~/redock_40seeds.tar.gz \
    "C:/Users/julia/Documents/SigmaFlow/SigmaFlow_Variants/posebusters_full_comparison/"
```

MD5-Summen mitschicken, sie werden lokal gegengeprüft.

### Was danach lokal gerechnet wird

`evaluate_run.py --per_pose_csv` → `error_budget.py` → `threshold_and_overlap.py`.
Alle drei liegen im Repo und laufen auf 40 Seeds unverändert.

**Der entscheidende Test:** Separate gegen Minimal lag bei +0.09 Erfolgen je
Komplex, Intervall [−0.02, +0.21], p = 0.13; auf der stetigen RMSD-Skala
−0.15 Å bei p = 0.079. Bei vierfacher Datenmenge halbiert sich der
Standardfehler. Wird die Richtung signifikant, ist Separate doch besser; geht
der Punktschätzer gegen null, ist das Nullergebnis bestätigt. **Beides ist
verwertbar.**

### Nicht vergessen

- `arc/score_gnina.slurm` (Ranking) ist vorbereitet, aber nie gelaufen. Erst
  entscheiden, ob es die Rechenzeit wert ist.
- Stage-2-Durchsatzjobs `8628662`/`8628663` hängen weiter auf `PD (Priority)`.
  Sie blockieren `FINAL_MAX_EPOCHS` und damit das 72h-Paar.
  **Spätester sinnvoller Start: 2026-08-29.**

## EXP-110 endgültig ausgewertet: Nullergebnis (2026-08-22, abends)

Vollständige Auswertung über **10 Seeds** und **mit Proteinprüfung**
(PoseBusters `redock`). Maßgeblicher Stand, Details in `RESULTS.md`.

| Hypothese zu SigmaFlow-Separate | Ergebnis |
|---|---|
| bessere Genauigkeit als Minimal | **nein**, p = 0.16 |
| bessere Rotation als Minimal | **nein**, −0.06° bei ±1.5° |
| bessere Validität als Minimal | **nein**, p = 0.72 |
| bessere Validität als SigmaDock | **nein**, p = 0.78 |

**Der Zwei-Kopf-Ansatz ist bei 12 h Budget ein sauberes Nullergebnis.** Das
ist berichtbar und braucht keine Rettung. Was bleibt, ist der
Verfahrensvergleich: SigmaDock trifft rund doppelt so oft unter 2 Å
(11.3 % gegen 5.3 / 6.2 %, p = 8.7e-14), und alle drei Arme erzeugen weit
überwiegend Posen, die ins Protein ragen (`minimum_distance_to_protein`
besteht bei 16–20 %).

Kernzahlen, 209 Komplexe × 10 Seeds je Arm (Minimal / Separate / SigmaDock):

| Größe | Werte |
|---|---|
| RMSD < 2 Å | 5.3 / 6.2 / **11.3 %** |
| PB-valid ohne Protein | 34.4 / 35.8 / 29.3 % |
| **PB-valid mit Protein** | **7.0 / 6.7 / 6.9 %** (alle gleich) |
| beides, mit Protein | 1.4 / 1.5 / 2.1 % |
| P(valid \| genau), mit Protein | 27.3 / 24.8 / 18.2 % (p = 0.071) |

**Drei Befunde sind an diesem Tag gekippt**, alle weil die Auswertung
weniger prüfte als die Zielgröße: der Top-1-Vorsprung galt nur für Seed 0,
der Validitätsvorsprung nur ohne Protein, und die Rangfolge der gemeinsamen
Metrik stand nach 4 Seeds falsch herum. **Regel: keine Zahl in die Thesis,
die nicht über alle Seeds und mit vollständiger Prüfung gerechnet ist.**

Werkzeuge dafür liegen im Repo: `redock_report.py`, `run_redock.py`,
`rotation_translation_compare.py` (alle in
`SigmaFlow_Variants/posebusters_full_comparison/`). Die 209 Protein-PDBs
liegen lokal unter `pb_proteins/`, md5 des Archivs
38cf9027b29e46a6284b0f62b4bba339.

### Offen aus diesem Strang

- 40-Seed-Erweiterung und gnina-Ranking sind vorbereitet (`arc/score_gnina.slurm`,
  `SigmaFlow_Evaluation/rank_and_report.py`), aber **nicht abgeschickt**.
  Angesichts des Nullergebnisses ist der Nutzen fraglich; erst entscheiden,
  ob EXP-110 überhaupt weiterverfolgt wird.
- Symmetrievorbehalt bei den absoluten Rotationswinkeln ungeprüft
  (`symmetry_flip_analysis.py` läge bereit).

## EXP-110: das erste echte Ergebnis (2026-08-22)

> **ÜBERHOLT.** Beruht auf Seed 0 und auf Validität ohne Protein. Siehe den
> Abschnitt darüber.


Job `8625634`, Commit `870fcc0`.

| | |
|---|---|
| Laufzeit | 11:10:59 von 12:00:00 |
| Epochen | 6 von 6 |
| Schritte | 14.150 von 14.277 = **99,11 % des Anneals** |
| Checkpoint | 273 MB, `trans_block` ✓, `rot_block` ✓, kein `force_block`, EMA ✓, Optimierer ✓, Scheduler ✓ |
| Ordner | `experiments/sigmadock/0-08-21_21-24-48/` |

**Gemessene Schrittrate: 0,3515/s gegen 0,3443/s beim Minimal-Referenzlauf
8541310. Die Zwei-Kopf-Variante war also 2,1 % SCHNELLER, nicht langsamer.**

Das widerlegt beide Vorabschätzungen, und die Lehre gilt für den Rest des
Projekts:

| Schätzverfahren | Vorhersage für 6 Epochen |
|---|---|
| Parameterzahl (+12,82 %) | 12,51 h, hätte nicht gepasst |
| CPU-Benchmark (Faktor 1,36) | 15,09 h, weit daneben |
| **gemessen auf GPU** | **11,18 h** |

**Parameterzahl ist kein Laufzeitproxy, und CPU-Verhältnisse übertragen sich
nicht auf GPU.** Der zusätzliche Attention-Block verschwindet dort in der
Parallelität. Für künftige Varianten zählt nur die GPU-Messung. Die
Entscheidung, `--max_epochs` bei 6 zu lassen, war richtig.

Vor dem Lauf wurden in sechs Auditskripten **167 Checks** grün gefahren, und
drei echte Blocker gefunden, die den Job in den ersten Minuten getötet hätten:
fehlendes `conf/experiments/`, zwei Restschlüssel in `forward()`, und eine tote
Entpackung in `trainer.py`. Alle drei fielen erst beim ersten echten Lauf auf,
keiner bei Codeinspektion.

## Durchsatz Stufe 2: Bug gefunden, Fix liegt vor (2026-08-22)

`8625082` und `8625083` meldeten `sacct: COMPLETED`, waren aber **gescheitert**:
`FAIL_rc2` in allen drei Messläufen, `peak_vram = 3 MiB`, also ohne dass die
GPU je benutzt wurde. Wieder ein Fall, in dem SLURMs `COMPLETED` nichts belegt.

**Ursache:** Stufe 2 übergibt `--accum_grad_batches`, Stufe 1 nicht. Das Feld
existiert in `RunConfig`, ist aber in **keinem** der drei Bäume als CLI-Flag
registriert (79 Attribute im Namespace, keines davon). `argparse` bricht mit
rc=2 ab. Stufe 2 ist damit noch nie gelaufen.

**Fix** in `7454c01`, ohne Eingriff in die Trainings-Config: bei `ACCUM=1` wird
gar nichts übergeben, bei `ACCUM>1` über den `--config`-YAML-Weg, den
`train.py` unterstützt. Gegen die echte Auflösungslogik geprüft.

Was Stufe 1 (`8606626`/`8606627`) hergab, bleibt gültig: Batch 16 schlägt Batch
8 deutlich und passt mit 20 GB bequem, TF32 bringt 3–4 %, bf16 ist kaputt,
beide Arme laufen im Gleichschritt. Nicht verwertbar waren die Absolutwerte
(enthalten den Datafront-Aufbau) und die beiden `TIMEOUT`-Konfigurationen
(konfundiert, weil Stufe 1 kein Priming hat und es die ersten beiden Läufe
traf). Offen bleibt allein, ob Batch 32 auf `pdbbind-general` in den Speicher
passt.

## Evaluationssatz: geklärt und entschieden (2026-08-21)

Die Frage „warum 209 und nicht 308" ist beantwortet. Das Zenodo-Archiv auf ARC
enthält alle **428** Komplexe, entpackt waren **209** — eine **abgebrochene
Entpackung**, kein Filter. Belege in `SigmaFlow_Evaluation/reference/README.md`.

Die vollen 428 liegen zusätzlich unter `data/posebusters_full/` (214 MB), das
alte 209er-Verzeichnis ist **unberührt**, damit alle bisherigen Ergebnisse
reproduzierbar bleiben.

**Entscheidung:** vorerst bei den 209 bleiben. Auswertung auf den offiziellen
308 erst am Ende, falls Zeit bleibt; sie kostet rund 44 GPU-h je Modell an
neuem Sampling. Der billige Zwischenschritt bleibt verfügbar: auf die **151**
Komplexe filtern, die in v2 enthalten sind. Kostet keine GPU-Zeit und entfernt
alle 57 Strukturen, die die Benchmark-Autoren wegen Kristallkontakten
verworfen haben.

## Strang A — 72h-Paar

### 🟢 Launch-Verdikt: GREEN

| Gate | Stand |
|---|---|
| Rotationsmathematik (11 Angriffsvektoren, 16 Checks) | sauber |
| Frame-Kette / adjungierter Transport | exakt bis 6.7e-16 |
| J1 Checkpoint-Provenienz | **geschlossen** |
| J2 Code-Baum-Identität | **geschlossen** — nur CRLF/LF |
| Gradient-Erreichbarkeit Rotation | **geschlossen, GREEN** (15/15) |
| EMA-Politik | entschieden, deckt sich mit dem Paper |

`audits/test_rotation_gradient_reachability.py`: `‖∂L_rot/∂f‖ = 2.41e+01`
gegen `‖∂L_trans/∂f‖ = 7.85e+00` — der Rotationsgradient ist **größer**,
nicht kleiner. 100 % der Atome bekommen ihn.
`cos(∂L_rot/∂f, ∂L_trans/∂f) = +0.0000` bestätigt numerisch, dass
Translation und Rotation orthogonale Projektionen desselben Feldes sind.
**H-dead ist ausgeschlossen.** Die drei Null-Ablationen heißen „kein Signal
zum Umgewichten", **nicht** „kein Gradient".

### ⏭️ Ablauf beim Abschicken

```bash
cd /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow
git pull origin main
bash arc/00_preflight.sh                                    # READY erwarten
python audits/test_rotation_gradient_reachability.py        # Exit 0 erwarten

# Durchsatz messen, dann:
python arc/calculate_final_epochs.py ... --write-env        # schreibt final_horizon.env
bash arc/final_72h_preflight.sh sigmaflow_minimal
bash arc/final_72h_preflight.sh sigmadock

DRY_RUN=1 bash arc/submit_final.sh sigmaflow_minimal
DRY_RUN=1 bash arc/submit_final.sh sigmadock
bash arc/submit_final.sh sigmaflow_minimal
bash arc/submit_final.sh sigmadock
```

Grobe Erwartung aus dem 12h-Lauf: 13.750 Steps in 11,1 h bei Batch 8
→ 2,78 Beispiele/s → ~90k Steps in 72 h → **~36 Epochen**.
**NICHT höher setzen** — ein unvollendeter Cosine-Anneal hat schon die Läufe
vom 2026-08-07 entwertet.

⚠️ **Bei Resume `FINAL_MAX_EPOCHS` NICHT erhöhen** — `configure_optimizers`
baut den Scheduler aus dem aktuellen Wert neu und wendet den
wiederhergestellten Step-Zähler auf eine gestreckte Kurve an
(ungeplanter Warm Restart).

---

## Strang B — EXP-110 Zwei-Kopf, 12h

Getrennte Köpfe für Translation und Rotation auf geteiltem Rumpf, Mittel-
Pooling je Fragment, Rahmenwechsel `R_t^T hat(ω) R_t` bleibt. +12,82 %
Parameter (16.899.826 gesamt).

**Rechenzeitgematcht, nicht schrittgematcht:** 12:00:00 gegen den
12h-Minimal-Lauf 8541310. Die Walltime wird **nicht** erhöht, auch wenn das
größere Modell weniger Schritte schafft — das ist Teil des Vergleichs.
Ziel sind **nicht** sechs Epochen, das Budget sind 12 Stunden.

```bash
cd /data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow
git pull
cd SigmaFlow_FM_Specific/EXP-110_two_head_vector_field
mkdir -p slurm_logs
sbatch slurm/train_two_head_12h.slurm
```

Nach etwa einer Stunde die Schrittrate prüfen — bleibt sie deutlich unter
1260 Steps/h, wird der Anneal nicht fertig, und man weiß es früh:

```bash
grep -oE "Epoch [0-9]+" slurm_logs/<jobid>.out | tail -1
```

Abbruchsicherheit ist geprüft: Checkpoint alle 50 Trainingsbatches
(**nicht** an Epochengrenzen), maximaler Verlust ~2,4 Minuten, atomares
Schreiben, `RUN_STATUS.json` unterscheidet `COMPLETED` von
`WALLTIME_ODER_SIGTERM`. Alles Weitere in der Varianten-`STATUS.md`.

---

## Offene Punkte (keine Blocker)

1. SigmaDock-Checkpoint/EMA-Provenienz noch nicht per `torch.load` geprüft.
2. **Trajektorien-Export** (CORE 3) nicht implementiert. SigmaFlow sammelt
   bereits `all_pos` (`sampling.py:380/464/478`), `sample.py` gibt
   `trajectory: [T, N_lig, 3]` heraus. Nur ein Writer fehlt; `R_t`/`trans_t`
   je Schritt müssten append-only ergänzt werden. SigmaDock-Seite ungeprüft.
3. **Evaluationssatz: unsere 209 stammen aus PoseBusters v1, nicht v2.**
   Geklärt am 2026-08-20 mit der offiziellen 308er-Liste
   (`SigmaFlow_Evaluation/reference/`): Schnittmenge 151, bei uns 57 Komplexe,
   die v2 wegen **Kristallkontakten** entfernt hat, 157 v2-Komplexe fehlen uns.
   Wirkung gemessen: auf den 151 steigen beide Arme (SF 30.3→33.8 %,
   SD 47.6→53.0 % Oracle@10), der Methodenunterschied bleibt stabil. Offen ist
   die Entscheidung, ob primär auf den 151 ausgewertet wird. Kostet keine
   GPU-Zeit. Betrifft **nur die Auswertung, nicht das Training**
   (`--train_exps pdbbind-general`).
   **Ursache 2026-08-21 geklärt:** abgebrochene Entpackung. Das Archiv auf ARC
   enthält alle 428 Komplexe, entpackt sind 209. Die vollen Daten sind also
   ohne Download da; für eine Auswertung auf den offiziellen 308 fehlt nur
   neues Sampling (rund 44 GPU-h je Modell).
4. `texprobe/` im Repo-Root ist Müll aus einer Werkzeugprüfung, untracked.
5. SigmaDock-NFE-Tasks `--array=5-6` zurückgestellt.
6. Oracle-Verhältnis-Spalte in `aggregate_learning_curve.py` fehlt noch.

**Erledigt seit dem letzten Lesezeichen:** Fragmentzahl-Histogramm (Job
8607523, `Thesis Visualisierungen/`), IS-1 (negativ, CORE 4 geschlossen),
vier ARC-Blocker gefunden und behoben, EXP-110 gebaut und auditiert.

---

# 📁 HISTORISCH — Theory-Phase (Stand 2026-08-15)

## Aktuelle Arbeitsphase: **Theory Summary `Texte/theory.tex`**

Seit einigen Sessions läuft **kein** Implementierungsstrang mehr, sondern der
Ausbau des Theory Summary. Der ARC-Strang ruht (keine Jobs starten, alles
GPU-Pflichtige gilt als `PENDING ARC VALIDATION`).

### 🔑 Prioritätenordnung, vom User am 2026-08-15 ausdrücklich festgelegt

```
Theory first  →  Architecture understanding  →  Implementation mapping
```

Das Dokument ist **primär ein Lehrbuch**, nicht Code-Dokumentation. Maßstab
für jeden Abschnitt:

> Könnte ein mathematisch starker MSc-Student, der dieses Gebiet vorher nicht
> gelernt hat, danach die Hauptideen selbst erklären und die zentralen
> Gleichungen rekonstruieren?

Und: **Der Leser darf das Repository nicht brauchen.** Code illustriert
Theorie, ist aber nie ihre Voraussetzung. Wenn ein Abschnitt zu
implementation-heavy wird, trennen in *Main theory* und *Implementation note:
SigmaDock/SigmaFlow*.

**Kein Seitenlimit.** 220–300 Seiten sind ausdrücklich in Ordnung, solange es
kein Padding ist. Lieber 20 Seiten, die einen schweren Übergang verständlich
machen, als 2 Seiten elegante Mathematik für Experten.

**Papers aktiv nutzen** (`papers/`, nicht `paper/`), nicht aus dem Gedächtnis
schreiben. Vorhanden:
- Holderrieth & Erives — *An Introduction to Flow Matching and Diffusion
  Models* (84 S.) ← didaktische Leitreferenz
- Song et al. 2021 — *Score-Based Generative Modeling through SDEs*
- Chen & Lipman 2024 — *Flow Matching on General Geometries*
- Bortoli et al. 2022 — *Riemannian Score-Based Generative Modelling*
- Yim et al. 2023 — SE(3) diffusion / SE(3) flow matching (2 Paper)
- Prat et al. 2026 — SigmaDock

**Nicht vorhanden** (bei Bedarf beschaffen): DDPM, SMLD, originales
Flow-Matching-Paper (Lipman et al.), Equiformer/EquiformerV2.

### ✅ Phase 1 abgeschlossen (2026-08-15): Diffusion didaktisch ausgebaut

Commit `cd16b4c`. 181 → **196 Seiten**. Details im Session-Log unten.

### ✅ Phase 2 abgeschlossen (2026-08-16): Flow Matching didaktisch ausgebaut

196 → **230 Seiten**. Kapitel 11 (Euclidean FM) komplett neu aufgebaut:
7 → 39 Seiten, 20 Sections statt 4. Alle 13 User-Punkte umgesetzt.
Details im Session-Log unten. Neues Audit-Skript
`audits/flow_matching_theory_audit.py` (27 Checks, alle grün).

### ✅ Phase 3 abgeschlossen (2026-08-16): Geometry Revisit + Riemannian FM

230 → **263 Seiten**. Kapitel 3 (Geometrie) gezielt erweitert, Kapitel 12
(Riemannian FM) komplett neu aufgebaut: 12 → 45 Seiten, 4 → 16 Sections.
Details im Session-Log unten. Neues Audit-Skript
`audits/riemannian_fm_theory_audit.py` (35 Checks, alle grün).

### ✅ Phase 4 abgeschlossen (2026-08-16): globaler Konsolidierungs-Audit

263 → **277 Seiten**. Keine neue Theorie, sondern Kohärenz: Dependency-,
Notations-, Redundanz- und Dichte-Audit über alle 16 Kapitel, plus die
strukturellen Navigationselemente, die bei 277 Seiten gefehlt haben.
Details im Session-Log unten.

### ✅ Phase 5 abgeschlossen (2026-08-16): die drei letzten Blocker

277 → **286 Seiten**. `cleveref` repariert (`aliascnt`),
EquiformerV2-Worked-Example in Kapitel 9, Part VIII strukturell bereinigt.
Details im Session-Log unten.

---

## 🏁 THEORY SUMMARY = FINISHED THEORY DOCUMENT (2026-08-16)

Der Theoriestrang ist **abgeschlossen**. Das Dokument wird ab jetzt als
stabile theoretische Basis behandelt, nicht mehr als Baustelle.

**Vom User ausdrücklich als Nicht-Blocker akzeptiert, bewusst offen:**
1. Vollständiger Fokker–Planck-Beweis (bräuchte Itôs Lemma). Empfehlung aus
   dem Phase-4-Audit angenommen: **nicht** ergänzen.
2. Zweiseitige Kopplung / OT-CFM. `caution:two-sided-conditioning` reicht.

**Regel für kommende Sitzungen:** keine Theory-Erweiterungen vorschlagen.
Änderungen am Dokument nur auf ausdrückliche Anforderung.

### ⏭️ NÄCHSTER SCHRITT

Zurück zum **Implementierungs-/ARC-Strang** (ruht seit 2026-08-13). Siehe
den historischen Block unten; der dortige oberste offene Punkt ist der
`sdf_regex`-Bug beim Sampling.

### Offene Lücken, selbst benannt

1. **Fokker–Planck ist motiviert, aber nicht bewiesen.** Der ½-Faktor ist über
   das Faltungs-/Taylor-Argument erklärt, der Transportterm über
   Massenerhaltung. Ein sauberer Beweis bräuchte Itôs Lemma, das nach
   User-Vorgabe §47/§48 weggelassen wurde. Ca. 4 Seiten, falls gewünscht.
2. **Zweiseitige Kopplung (OT-CFM) nur benannt, nicht repariert.**
   `caution:two-sided-conditioning`: bedingt man auf das Paar `(x_0,x_1)`,
   ist der Conditional Path zu *jedem* `t` ein Dirac, hat also keine Dichte —
   `thm:continuity` greift nicht. Reparierbar per `σ_min`-Glättung oder
   schwacher Formulierung; im Dokument bewusst als Lücke markiert statt
   überklebt. Für SigmaFlow irrelevant (dort unabhängige Kopplung).
3. **cleveref nennt Definitionen/Propositionen „Theorem".** Dokumentweite
   Altlast: alle Theorem-Umgebungen teilen sich per `[theorem]` denselben
   Zähler, und cleveref unterscheidet nach Zähler, nicht nach Umgebung.
   Folge: `\Cref{def:...}` rendert als „Theorem 11.16", `\Cref{caution:...}`
   als „Theorem 12.13". Fix wäre `aliascnt` oder `thmtools`/`\declaretheorem`
   im Präambel-Block — betrifft ~1000 Querverweise in allen 13 Kapiteln.
   In Phase 2 zurückgestellt, in Phase 3 vom User ausdrücklich
   zurückgestellt (§74). Durch Kapitel 12 (viele Verweise auf Definitionen
   und Cautions) inzwischen deutlich sichtbarer. **Entscheidung steht aus.**
4. **Kosmetische Overfull-hboxes.** Kapitel 11: 25 pt im geerbten Beweis von
   `thm:continuity`. Kapitel 12: max. 18 pt (eine `\file{}`-Zeile). Beides
   unter dem bestehenden Dokumentstandard (anderswo 340 pt und 110 pt).

### Mehr Worked Examples (User-Wunsch §46, laufend)

- Geometrie: S¹, S², Rotation um z-Achse, Exp/Log einer 90°-Rotation
- Diffusion: 1D-Gauß ✅, diskreter Random Walk ✅, VP-SDE in 1D ✅,
  Score einer Gauß ✅
- Flow Matching ✅ (Phase 2): 1D-Paar `x_0=-1,x_1=3`, Gauß-Source → fester
  Datenpunkt (inkl. ODE-Lösung von Hand), 2D-Zweipunkt-Beispiel mit
  geschlossener tanh-Form für das marginale Feld
- Geometrie ✅ (Phase 3): S¹, S² (Exp/Log in geschlossener Form, mit
  Zahlen), Großkreise als Geodäten, Polarkoordinaten als Volumen- und
  Divergenz-Sanity-Check, 90°-Rotation um z
- Riemannian FM ✅ (Phase 3): S¹ (170°→−170°, Wrap-Falle), S² (Viertelbogen,
  vollständig), SO(3) (90°, jeder Matrixeintrag ausgerechnet),
  Produkt-Zustand (ein Fragment, r und R gemeinsam)

---

# 📁 HISTORISCH (Stand 2026-08-13) — Implementierungsstrang, ruht

## 🔴 BUG GEFUNDEN (2026-08-13): `sdf_regex` der Experimentdatei wird beim Sampling IMMER verworfen

`sampling_setup.py::build_sampling_datafront`:

```python
ec = get_experiment_config(str(cfg.experiments.name), data_dir)  # laedt z.B. posebusters.yaml
if "sdf_regex" in cfg.experiments:      # <- IMMER wahr, base.yaml definiert den Schluessel
    ec.sdf_regex = str(cfg.experiments.sdf_regex)
```

Die Bedingung sollte "nur wenn der Nutzer etwas per CLI gesetzt hat" bedeuten.
Weil `conf/sampling/base.yaml:15` aber `sdf_regex: ".*ligand.sdf$"` fest
definiert, greift sie immer. **Der Wert aus `conf/experiments/*.yaml` ist beim
Sampling toter Code.**

**Wirkung, empirisch bestaetigt** (aufgeloeste Config aus `predictions.yaml`
zeigt `.*ligand.sdf$`, obwohl `posebusters.yaml` `.*ligands.sdf$` sagt):
- Alle Sampling-Laeufe ohne Override lasen `<cid>_ligand.sdf` — die
  KANONISCHE Einzelkopie. Zufaellig die richtige Datei.
- Der einzige abweichende Lauf war 8553984 (alter SigmaDock-Vergleich), der
  `experiments.sdf_regex=".*ligands.sdf$"` explizit uebergab und dadurch die
  Mehrkopie-Datei las. **Das erzeugte die 42 Phantom-Ausreisser.**
- **Gute Nachricht fuer Ziel 1:** seeds10, Schritt-Sweep, rho-Sweep und die
  12h-Laeufe lasen alle dieselbe kanonische Datei. Datenprovenienz geklaert.

**⚠️ FIX NUR ALS PAAR ANWENDEN.** Nur den Guard zu reparieren macht es
SCHLIMMER: dann greift wieder `posebusters.yaml` mit `.*ligands.sdf$`, also
die Mehrkopie-Datei, und das Artefakt waere der Normalzustand.

```yaml
# conf/sampling/base.yaml
experiments:
  sdf_regex: null                 # null = Wert der Experimentdatei benutzen
# conf/experiments/posebusters.yaml
sdf_regex: ".*ligand.sdf$"        # SINGULAR, kanonische Einzelkopie
```
```python
# sampling_setup.py
if nonempty_cfg_str(cfg.experiments.get("sdf_regex")):
    ec.sdf_regex = str(cfg.experiments.sdf_regex)
```

Betroffen ist NUR `sdf_regex`; `pdb_regex` steht nicht in `base.yaml`, dessen
Guard greift korrekt. **Nebenbefund:** `dummy_crossdock.yaml` nutzt
`sdf_regex: "query_.*\.sdf$"` — ueber den Sampling-Pfad ebenfalls
ueberschrieben, das Cross-Docking-Experiment findet seine Query-SDFs also
vermutlich nie. Nicht dringend, aber notiert.


## 🚨🚨🚨 2026-08-13: die 42 SigmaDock-Ausreisser haben NIE EXISTIERT

Auswertungsartefakt, kein Modellfehler. Vollstaendig nachgewiesen, nicht
vermutet. Kette der Belege, jede einzeln gemessen:

1. Der alte Vergleichslauf (Job 8553984, `sampling_output_pb_full_12h`)
   uebergab `experiments.sdf_regex=".*ligands.sdf$"` und las damit die
   PLURAL-Datei. **84 der 209 `_ligands.sdf` enthalten mehrere Ligandenkopien**
   (bis zu 6). Der seeds10-Lauf liess das Flag weg, nahm per Default die
   Singular-Datei mit genau einer Kopie.
2. Alle Auswertungsskripte vergleichen gegen die ERSTE Kopie, weil `load_mol`
   das erste Molekuel der Datei zurueckgibt.
3. **Alle 42 Ausreisser stammen aus Mehrkopie-Dateien, keiner aus einer
   Einkopie-Datei.** Bei 42 von 42 liegt die beste Uebereinstimmung auf einer
   anderen Kopie als Nummer 0.
4. Gegen die RICHTIGE Kopie gemessen faellt der Fehler von 45-80 A auf
   **Median 3.44 A** (29 von 42 unter 5 A) — normaler Leistungsbereich.
5. Zerlegung des Fehlers: 99% ist reine Verschiebung, Schwerpunktabstand
   Median 36.5 A, RMSD nach Ausrichtung 1.71 A — also BESSER als die 1.91 A
   der unauffaelligen Komplexe. Gute Posen am falschen Ort.

**Konsequenz 1: Argument 1 der Verteidigungsliste ("Robustheit, 0 gegen 42")
ist ersatzlos gestrichen.** Beide Methoden haben null Ausreisser jenseits
20 A, ueber 2090 Posen je Methode.

**Konsequenz 2: latenter Fehler im Auswertungswerkzeug.** `load_mol` nimmt
willkuerlich Kopie 0 — bei 40% der Komplexe eine Wahl, nicht die Wahrheit.
Betrifft `full_metrics.py`, `fragment_locality.py`, `seed_variance.py`,
also auch alle SigmaFlow-Zahlen. SigmaFlow ist nur zufaellig unauffaellig
(liest die Singular-Datei, dort steht Kopie 0). **Fix:** RMSD als Minimum
ueber alle Kopien, wie in der Docking-Literatur ueblich — oder fuer beide
Seiten konsequent die Singular-Datei als Referenz.

**Nicht behaupten:** "der alte SigmaDock-Lauf war kaputt". Er war es nicht.
Er wurde falsch gemessen. Das Modell lieferte durchgehend vernuenftige Posen.

Zweiter Unterschied der beiden Laeufe, Wirkung noch nicht quantifiziert: der
alte Lauf setzte zusaetzlich `graph.fragmentation_strategy=canonical`. Eine
andere Fragmentierung ist ein anderes generatives Problem (jedes Fragment ein
eigener starrer Koerper mit eigener SE(3)-Transformation). Erklaert
vermutlich die ~1.02 A Grunddifferenz, die auch die 167 unauffaelligen
Komplexe zwischen den beiden Laeufen zeigen.

## ✅ Beide Job-Arrays vom 2026-08-12 sind fertig und geprueft

Alle `COMPLETED`, ExitCode 0:0, und die Ausgaben gegengezaehlt (nicht nur der
Slurm-State, siehe Falle unten):
- **8554147** `sigmaflow-pb-seeds10`, 10 Tasks → 10 x 209 = 2090 SDF ✅
- **8554149** `sigmadock-pb-seeds10`, 10 Tasks → 10 x 209 = 2090 SDF ✅
- **8554148** `sigmaflow-pb-stepsweep`, 21 Tasks → 7 x 627 = 4389 SDF ✅
- Kein `Traceback`/`Error`/`WARNING` in irgendeinem `.err`; jede `.out`
  enthaelt genau einmal `frame fix verified`.

Daten liegen lokal ausgepackt in `posebusters_full_comparison/`:
`seeds10_sigmaflow/`, `seeds10_sigmadock/`, `stepsweep/`.

## ✅ Seed-Varianz ausgewertet (`seed_variance.py`, 2026-08-13)

**Kernbefund: das Rauschen einer Einzelziehung ist ~10x so gross wie der
Methodenunterschied.** Derselbe Komplex bewegt sich zwischen zwei Seeds im
Median um 4.46 A (SigmaFlow) bzw. 4.59 A (SigmaDock); der gepaarte
Methodenunterschied betraegt 0.47 A. Jeder fruehere Einzelziehungs-Vergleich
dieses Projekts las damit Rauschen.

| | SigmaFlow | SigmaDock |
|---|---|---|
| Median-RMSD je Seed | 4.56–5.70 A | 4.15–4.87 A |
| Anteil <2 A je Seed | 1.4–6.7% | 5.7–13.4% |
| Mittel <2 A ueber 10 Seeds | 4.4% | 9.8% |
| Ausreisser >20 A | 0 | 0 |
| Best-of-10 <2 A | 29.2% | 45.9% |

SigmaFlows bester Seed (6.7%) liegt UEBER SigmaDocks schlechtestem (5.7%) —
mit Seed-Glueck haette man in beide Richtungen "gewinnen" koennen.

**Gepaart auf Seed-Mittelwerten:** SigmaFlow minus SigmaDock **+0.47 A**,
Bootstrap-CI **[+0.36, +0.59]**, schliesst Null aus. SigmaFlow bei 29% der
Komplexe besser. Das ist die erste methodisch saubere Aussage des Projekts:
SigmaDock ist gesichert besser, um einen kleinen Betrag.

Konsistenzcheck geht auf: die Seed-zu-Seed-Streuung der Aggregate entspricht
reiner Binomialstreuung (n=209, p=0.044 → erwartet 1.42, beobachtet 1.75
Prozentpunkte; p=0.098 → erwartet 2.05, beobachtet 2.17). Kein unerklaerter
Streuungsanteil.

Reproduzierbarkeit verifiziert: SigmaFlow alt ↔ neuer seed_0 **209/209 Posen
bitgleich**. Das Sampling-Geruest ist deterministisch; die SigmaDock-Differenz
war deshalb ein echter Konfigurationsunterschied, kein Rauschen.

## ✅ Kopien-Fix umgesetzt (2026-08-13)

Neues gemeinsames Modul **`posebusters_full_comparison/ligand_reference.py`**:
`load_copies()` liest ALLE Kopien, `best_copy()` waehlt die naechstgelegene.
Umgestellt: `seed_variance.py`, `full_metrics.py`, `fragment_locality.py`.
**Noch NICHT umgestellt** (gleicher Fehler, niedrigere Prioritaet):
`bond_length_check.py`, `placement_vs_bond_error.py`, `placement_dummy.py`,
`run_posebusters_ligandonly.py`.

Alle 84 Mehrkopie-Dateien enthalten verifiziert IDENTISCHE Topologie, also
echte Kristallkopien — Minimum-ueber-Kopien ist damit zulaessig und ist die
uebliche Docking-Konvention. Bias ehrlich benennen: kann einen Fehler nur
senken, nie heben; wird aber jedem Lauf gleich gewaehrt.

Wirkung, gemessen: beim seeds10-Lauf gewinnt in **0 von 84** Faellen eine
andere Kopie als 0, beim alten Plural-Lauf in **44 von 84**.

**Die Seed-Varianz-Zahlen oben aendern sich durch den Fix NICHT** (die
seeds10-Laeufe lesen die Singular-Datei und landen ohnehin bei Kopie 0) —
sie waren bereits korrekt.

**`full_metrics.py 12h` aendert sich dagegen deutlich, SigmaDock-Seite:**

| | vorher (Kopie 0) | jetzt (naechste Kopie) |
|---|---|---|
| Ausreisser >20 A | 42 | **0** (max 9.8 A) |
| Anteil <2 A | 4.8% | **6.7%** |
| Median-RMSD | 5.35 A | **4.23 A** |
| gematchte Teilmenge | 94 | **100** Komplexe |

SigmaFlow 12h unveraendert: 1.9% unter 2 A, Median 4.78 A, 0 Ausreisser.
Lokalitaetsluecke jetzt: SigmaFlow 12h −15.0°, SigmaDock 12h −18.6°
(vorher −20.4°).

## ✅ Stepsweep ausgewertet (`stepsweep_curve.py`, 2026-08-13)

**Die Kurve ist flach. Mehr Integrationsschritte helfen nicht.**

| Schritte | Median roh | <2 A | ausgerichtet |
|---|---|---|---|
| 5 | 4.61 ± 0.06 | 3.7% | 2.11 |
| 25 (geerbt) | 4.76 ± 0.19 | 4.1% | 2.01 |
| 100 | 5.17 ± 0.18 | 4.8% | 1.99 |
| 200 | 4.93 ± 0.24 | 3.7% | 1.99 |

Gepaart gegen den Default von 25: 10/15/50/100/200 alle **nicht
signifikant**, jedes Bootstrap-CI schliesst Null ein. Auch der ausgerichtete
RMSD bleibt bei ~2 A — nicht einmal die innere Geometrie profitiert.

5 Schritte sind formal signifikant BESSER (−0.199 A, CI [−0.391, −0.007],
besser bei 67%), aber **nicht als Effekt behaupten**: obere CI-Grenze
praktisch auf Null, und bei sechs Vergleichen gegen dieselbe Referenz
ueberlebt das keine Korrektur fuer multiples Testen. Korrekte Aussage:
**keine Schrittzahl zwischen 5 und 200 ist nachweisbar besser als eine
andere.**

Groessenordnung: ein Komplex streut ueber 3 Seeds mit SD 1.36 A, ueber alle
7 Schrittzahlen (seed-gemittelt) nur mit Spanne 1.66 A — bei reinem Rauschen
waere die erwartete Spanne schon ~2.4 A. Die Schrittzahl traegt nichts bei.

**Fuer Argument 2 (Sampling-Kosten) ist das die erste echte Substanz:**
SigmaFlow liefert bei 5 Schritten dasselbe wie bei 25, kann also **5x
billiger sampeln**. **Einschraenkung, nicht ueberdehnen:** SigmaDock wurde
NICHT mit 5 Schritten getestet. Belegt ist "SigmaFlow braucht die geerbten
25 nicht", nicht "SigmaFlow braucht weniger Schritte als SigmaDock". Dafuer
fehlt der Gegen-Sweep auf der SigmaDock-Seite.

## ✅ Berichte korrigiert (2026-08-13)

`RESULTS.md` und `ERGEBNISSE_FrameFix_vs_SigmaDock.txt` sind fertig
korrigiert. Vorgehen: widerlegte Aussagen NICHT geloescht, sondern als
widerrufen markiert und die alten Zahlen danebengestellt — damit spaeter
nachvollziehbar bleibt, was geprueft und verworfen wurde.

- `ERGEBNISSE...txt` hat einen neuen **Abschnitt 0** ganz vorn (Widerruf +
  Nachtrag), auf den alle korrigierten Stellen verweisen.
- `RESULTS.md` hat zwei neue Abschnitte: Seed-Varianz und Schritt-Sweep.

**Auch der paarweise Kernbefund kippte** (war nicht vorhergesehen): der
Satz "auf einem typischen Komplex sind beide gleichauf" (Median der
Differenz 0.03 Å, Mittel −6.73 Å zugunsten SigmaFlow) war vollstaendig
Artefakt. Korrekt: **Median +0.48 Å, Mittel +0.77 Å CI [+0.44, +1.10]
zugunsten SigmaDock**, SigmaFlow besser bei 37%. Unabhaengig bestaetigt
durch die 10-Seed-Rechnung (+0.39 / +0.47 Å, CI [+0.36, +0.59], 29%).

**`validity_significance.py` musste NICHT neu gerechnet werden.**
`run_posebusters_ligandonly.py` wurde auf die naechstgelegene Kopie
umgestellt und beide CSVs neu erzeugt (SigmaDock 44 Referenzen umgestellt,
SigmaFlow 0). Das `<2 Å`-Urteil kippte bei **0 von 44** — PoseBusters
rechnet symmetriekorrigiert und war gegen das Artefakt immun. Alle p-Werte
unveraendert, `p=0.0347` eingeschlossen.

## ⏭️ Naechste Schritte

1. ✅ **ERLEDIGT 2026-08-13.** `bond_length_check.py` und
   `placement_vs_bond_error.py` umgestellt und neu gerechnet.
   `placement_dummy.py` ist NICHT betroffen — es arbeitet auf dem
   Dummy-Datensatz, liest bewusst die Singular-Datei (mit begruendendem
   Kommentar im Code) und alle 10 Dateien dort enthalten genau ein Molekuel.
   Bewusst unveraendert gelassen.

   **Zwei weitere Berichtsaussagen fielen dabei:**
   - *Uebergangsbindungen, Trade-off-Lesart:* lautete "SigmaFlow hat weniger
     betroffene Bindungen, SigmaDock die kleinere Abweichung". Korrigiert hat
     SigmaDock **beides** — 12.8% gegen 13.1% betroffene Bindungen (vorher
     14.9%) UND 1.22 gegen 2.68 Å Mittel. Es gibt keinen Trade-off.
   - *Fragment-Platzierung:* SigmaDock-Schwerpunktfehler 12.92/6.39 →
     **6.26/5.29 Å**, Rotationsfehler 118.7° → **116.4°**, Gesamt-RMSD
     13.57/6.55 → **6.40/5.59 Å**. Kernaussage bleibt: beide nahe der
     Zufallsgrenze 126.5°, beide 0 Komplexe unter 2 Å bei 12h.

   **Gegengeprueft und STANDGEHALTEN:** die Uebergangsbindungs-Aussage des
   Frame-Fix-Berichts (SigmaFlow 0.36 Å vs SigmaDock 0.38 Å auf der
   korrigierten 118er-Teilmenge, vorher 0.36/0.37 auf 94) — innermolekulare
   Distanzen sind gegen die Kopienwahl unempfindlich.

   ⚠️ **NICHT nachgerechnet: alle 24h-Zeilen.** Die zugehoerigen
   Vorhersageordner liegen nicht lokal. Betroffen sind die Aussagen ueber
   SigmaDocks schrumpfenden Einzelausreisser (18.08 → 5.33 Å) und ueber
   `7T1D_E7K`/`6TW5_9M2` als "komplex-spezifisches strukturelles Problem" —
   `6TW5_9M2` ist einer der 42 Phantom-Ausreisser (3 Kopien, alt 68.9 Å,
   korrigiert 4.29 Å). In RESULTS.md entsprechend markiert.

2. **Billig, schliesst Argument 2 ab:** SigmaDock-Stepsweep als Gegenstueck
   (~10 min GPU). Erst damit laesst sich "SigmaFlow braucht weniger Schritte
   als SigmaDock" belegen statt nur "SigmaFlow braucht die geerbten 25 nicht".
3. Danach der alte Punkt: **Frame-Fix nach `SigmaFlow_Development/`
   zurueckportieren** (lebt weiterhin NUR in `SigmaFlow_Variants/d_frame_fix/`).
4. **Der eigentliche Engpass bleibt die absolute Orientierung.** Der
   Schritt-Sweep hat die Sampling-Aufloesung als Ursache AUSGESCHLOSSEN: das
   Modell integriert sein Vektorfeld sauber, das Vektorfeld zeigt nur nicht
   auf die richtige Stelle.

---

# Aeltere Notizen (Stand 2026-08-12)

**Die zwei Job-Arrays** (2026-08-12 hochgeladen und gestartet — erledigt,
siehe oben):
- `d_frame_fix/slurm/sample_pb_seeds10.sh` + das SigmaDock-Gegenstück
  `sample_pb_seeds10_sigmadock.sh` — je 10 unabhängige Sampling-Seeds bei 25
  Schritten. **Zweck:** die Seed-Varianz messen. Bisher steht JEDER Vergleich
  auf EINEM Zug pro Methode; wir wissen daher nicht, ob die gemessenen
  Unterschiede überhaupt außerhalb des Rauschens liegen.
- `d_frame_fix/slurm/sample_pb_stepsweep.sh` — 7 Schrittzahlen
  (5/10/15/25/50/100/200) × 3 Seeds. **Zweck:** SigmaFlow hat die 25 Schritte
  von SigmaDock GEERBT; das Paper hat sie für Diffusion optimiert, nicht für
  eine ODE.

**Auswertung dieser beiden steht noch aus** — sie braucht zwei neue Skripte:
Streuung pro Komplex über die 10 Seeds, und die Kurve RMSD-über-Schrittzahl
mit Fehlerbalken.

6h- und 12h-Stufe sind komplett gefahren, gesampelt und
ausgewertet. Zahlen und Vorbehalte: RESULTS.md, "12h-Stufe: sauberer
compute-gematchter Vergleich" und "Vollständiger Metrikvergleich".

## 🚨 Wichtigste Einordnung (2026-08-12): kein belegter Vorteil von SigmaFlow

Nach gepaarten Signifikanztests (`validity_significance.py`) ist der **einzige
statistisch gesicherte Unterschied** zwischen SigmaFlow und SigmaDock die
Erfolgsquote unter 2 Å — **8.1% gegen 2.9%, p=0.035, zugunsten SigmaDock**.
Chemische Plausibilität: nicht unterscheidbar (alle p ≥ 0.076) — **beides am
2026-08-13 gegen das Kopien-Artefakt geprüft und unverändert gültig**
(PoseBusters rechnet symmetriekorrigiert; bei 44 umgestellten Referenzen
kippte das <2-Å-Urteil bei 0).

~~Typischer Komplex: nicht unterscheidbar (Median der RMSD-Differenz
0.03 Å).~~ **KORRIGIERT 2026-08-13:** Das war ebenfalls das Artefakt.
Korrekt ist **+0.48 Å Median zugunsten SigmaDock**, Mittel +0.77 Å
CI [+0.44, +1.10]; über 10 Seeds gemittelt +0.39 Å Median, +0.47 Å Mittel
CI [+0.36, +0.59]. SigmaDock ist auch auf dem typischen Komplex besser —
aber der Unterschied ist mit ~0.5 Å auf einem Fehlerniveau von ~4.5 Å klein.

Verteidigbare Argumente für SigmaFlow, in dieser Reihenfolge:
1. ~~**Robustheit** — 0 gegen 42 Komplexe jenseits 20 Å.~~ **GESTRICHEN
   2026-08-13.** Die 42 Ausreißer waren ein Auswertungsartefakt
   (Mehrkopie-Referenzdateien, siehe ganz oben). Beide Methoden haben null
   Ausreißer jenseits 20 Å über je 2090 Posen. Dieses Argument existiert
   nicht mehr.
2. **Sampling-Kosten** — Schritt-Sweep gelaufen, Auswertung steht aus.
3. **Weniger Hyperparameter** (kein Rauschplan, keine Score-Skalierung).
4. **Erweiterbarkeit** (freie Quellverteilung, exakte Likelihood) —
   Möglichkeit, kein gemessener Vorteil.

NICHT behaupten: "chemisch bessere Moleküle" (n.s.), "genauer" (falsch),
"Flow Matching gewinnt bei Konvergenz" (keine Evidenz).

Das Projektziel war eine minimalinvasive Ersetzung, nicht SigmaDock zu
schlagen. "Generativen Prozess ausgetauscht, nichts kaputtgemacht" ist das
korrekte Ergebnis — ausführlich in
`ERGEBNISSE_FrameFix_vs_SigmaDock.txt`, Abschnitt 2e.

## Stand in einem Satz

Der Frame-Fix war real und wirksam (Lokalitätslücke +1.1° → −12.8°,
Übergangsbindungsfehler 1.20 → 0.36 Å, damit gleichauf mit SigmaDock). Mehr
Rechenzeit hilft danach nur noch wenig, und alle drei Loss-Varianten sind
tot. Der Engpass ist jetzt die **absolute** Orientierung/Verankerung, nicht
die innere Geometrie.

## Abgeschlossene Läufe

| Job | Lauf | Ergebnis |
|---|---|---|
| 8512798 | SigmaFlow 6h ohne Fix | Lücke +1.1° (keine Lokalität) |
| 8530243 | SigmaFlow 6h Frame-Fix | Lücke −12.8° |
| 8541310 | SigmaFlow 12h Frame-Fix | Lücke −15.0°, <2Å 1.9% |
| 8512922 | SigmaDock 6h | Lücke −22.0° |
| 8541439 | SigmaDock 12h | Lücke −20.4°, <2Å 4.8% |
| 8540758 | + Variante a (time weight) | −12.8° = ECHTES Nullergebnis |
| 8534746 | + Variante b (rot-data-space) | −12.8° = echtes Nullergebnis |
| 8534747 | + Variante c (anchor-dist) | No-Op, nie getestet |

Das 12h-Paar ist compute-gematcht (`max_epochs=6`, SigmaFlow global_step
13.750). **Noch offen:** SigmaDocks `global_step` gegenprüfen — sein höchster
Top-k-Checkpoint zeigt 13.200. Bei ~4% Abweichung ändert sich inhaltlich
nichts, es gehört aber dokumentiert statt als "gleich" verbucht:
`python -c "import torch; ck=torch.load('<sigmadock>/experiments/sigmadock/0-08-11_17-57-05/checkpoints/last.ckpt', map_location='cpu', weights_only=False); print(ck['epoch'], ck['global_step'])"`

## Nächste Schritte, in dieser Reihenfolge

1. **Frame-Fix nach `SigmaFlow_Development/` zurückportieren.** Er lebt
   weiterhin NUR in `SigmaFlow_Variants/d_frame_fix/`. Die Hauptcodebasis
   trägt den Rahmenfehler noch. Einzeiler in
   `sigma_flow_generator.py::_compute_vector_field`, davor Oracle- und
   Smoke-Test wie beim ersten Mal.
2. **Den echten Engpass angehen: absolute Orientierung.** Der absolute
   Fragment-Rotationsfehler liegt bei SigmaFlow über alle Budgets an der
   Zufallsgrenze (~123-128°), während die relative Orientierung
   funktioniert. Das Modell lernt innere Geometrie, aber keine Verankerung
   in der Bindetasche. **Vor jedem weiteren Compute-Einsatz diagnostizieren,
   nicht mehr Stunden draufwerfen** — die 6h→12h-Verdopplung hat <2Å nur
   von 1.0 auf 1.9% bewegt.
3. **KEINE weiteren Loss-Varianten**, bis Punkt 2 verstanden ist. Drei
   Versuche, drei Nullergebnisse (bzw. ein No-Op).
4. **Variante c reparieren** — nur falls ihre Idee nach Punkt 2 noch
   relevant erscheint. Sie braucht ein `pos_1_hat` MIT Gradientenpfad,
   siehe unten "Variante c ist ein NO-OP".
5. **3-Tage-Stufe: Freigabe NICHT mehr automatisch einholen.** Die
   Compute-Skalierung von 6h→12h spricht dagegen, dass Laufzeit allein
   reicht. Wirksamere Hebel wären Batch-Size 32 (Paper-Wert statt unserer 8)
   und/oder DDP über mehrere GPUs. Erst Punkt 2.

## Werkzeuge (neu, reproduzierbar)

- `posebusters_full_comparison/full_metrics.py` — RMSD, Ausreißer,
  Übergangsbindungen, Fragmentmetriken. **RUNS-Liste klein halten:** der
  Matched-Subset schrumpft mit jedem Lauf (3 Läufe → 94 Komplexe,
  4 Läufe → 57).
- `posebusters_full_comparison/fragment_locality.py` — nur die
  Lokalitätslücke, braucht KEIN Matching, steht auf n≈133.
- `ERGEBNISSE_FrameFix_vs_SigmaDock.txt` (Repo-Root) — Ergebnisbericht für
  den User inkl. fertigem PyMOL-Code.

## ⚠️ Zwei Fallen, die diese Sessions Zeit gekostet haben

1. **Subset-abhängige Zahlen nicht über Sessions vergleichen.** Derselbe
   Lauf (SigmaFlow 6h) hat 0.36 Å Bindungsfehler auf dem 94er-Subset und
   0.40 Å auf dem 57er; absoluter Rotationsfehler 128.1° vs. 126.1°. Ohne
   den Subset dazuzuschreiben liest sich das wie Fortschritt.
2. **Dateinamen unterscheiden sich zwischen den Codebasen:** SigmaDock
   schreibt `..._ligands_seed0.sdf` (Plural), SigmaFlow
   `..._ligand_seed0.sdf` (Singular). Ein festes Namensmuster lädt für eine
   der beiden Seiten stillschweigend nichts. Immer `glob` benutzen — das
   traf sowohl das PyMOL-Skript als auch einen meiner Prüfbefehle, der
   dadurch "0 verschieden" meldete, obwohl er gar nichts verglichen hatte.
3. `tar --exclude='experiments'` ist nicht an den Pfadanfang gebunden und
   verschluckt auch `conf/experiments/`. Muster binden:
   `--exclude='<ordner>/experiments'`, danach Dateiliste gegen einen
   funktionierenden Ordner diffen.
4. Die Lightning-Abschlusszeile lautet ``​`Trainer.fit` stopped:`` MIT
   Backticks — nach `"stopped:"` suchen.
5. Top-k-Checkpointnamen sind bei konstantem `val_loss=0.0000` keine
   verlässliche Quelle für die Endschrittzahl. `global_step` aus
   `last.ckpt` lesen.
6. **Wall-Clock aus einem Job-Array ist keine Kostenmessung.** Im Stepsweep
   (`--array=0-20%5`) reichte die Laufzeit nur von 1:26 bis 15:03, obwohl
   zwischen 5 und 200 ODE-Schritten Faktor 40 an Arbeit liegt: die frühen
   Indizes liefen zu fünft parallel und teilten sich Dateisystem und CPU.
   Nur die zuletzt gelaufenen Tasks (Seed 2) skalieren sauber
   (~1.3 min Fixkosten + ~2.4 s je Schritt). Für echte Sampling-Kosten ein
   Extra-Lauf mit `%1` oder die reine GPU-Zeit aus den Logs.
7. **Mehrere Ligandenkopien pro Referenzdatei** (84 von 209, bis zu 6).
   `Chem.SDMolSupplier` liefert sie der Reihe nach; das erste Molekül zu
   nehmen ist eine WILLKÜR, keine Referenz. Hat 42 Phantom-Ausreißer
   erzeugt, die als Robustheitsvorteil in den Bericht gewandert waren.
   RMSD immer als Minimum über alle Kopien bilden.

---

## 🔖 PAUSE-PUNKT #14 (2026-08-05) — AKTUELL, zuerst lesen

**Kontext:** Neue Session, losgelöst vom Trainingslauf-Thema aus #13 (das
bleibt unverändert offen, siehe dort). Auftrag: ein vollständiges,
unvoreingenommenes, kritisches Audit von SigmaFlow gegen SigmaDock, mit
Fokus auf (a) ob SigmaFlow die Triangulation-Constraints korrekt nutzt und
(b) ob SigmaFlow strukturell längere Bindungen an Fragmentübergängen baut
als SigmaDock. Anschließend, nach Vorliegen einer empirisch gestützten
Bond-Length-Hypothese: erster Kausalitätstest ("Test 1").

### ✅ Vollständiges Audit erstellt: `AUDIT_SigmaDock_vs_SigmaFlow.txt` (Repo-Root)

Arbeitsteilig erstellt (zwei unabhängige Analyse-Durchläufe + eigene
Verifikation, u.a. echter Abgleich der lokalen `SigmaDock/`-Referenz gegen
das tatsächliche GitHub-Repo `alvaroprat97/sigmadock` per `gh api` — 5
Kerndateien identisch, Referenz vertrauenswürdig). Kernergebnisse:

- **Triangulationsmechanismus**: code-identisch zwischen SigmaDock und
  SigmaFlow (`fragmentation.py`/`processing.py`, `diff`-Exit-Code 0),
  identisch konfiguriert (`ignore_triangulation: false`, auch in echten
  persistierten Trainingsläufen bestätigt). Es ist in BEIDEN Methoden ein
  weiches Graph-Kanten-Signal fürs Netzwerk, kein hartes geometrisches
  Constraint, keine Reprojektion — das war im Original schon so. Kein
  Implementierungsfehler bei SigmaFlow.
- **Bond-Length-Hypothese**: EMPIRISCH BESTÄTIGT, nicht nur Spekulation —
  drei unabhängige Messungen (RDKit an echten SDF-Sampling-Outputs, 10
  Komplexe) zeigen übereinstimmend: SigmaFlow baut an den
  (ehemaligen) Torsionsbindungen zwischen Fragmenten deutlich längere/
  stärker gebrochene Bindungen als SigmaDock (1.5-2x größere
  Maximallängen, 2.5-3x mehr Bindungen >2.0 Å). Das korrigiert die
  bisherige Einschätzung in RESULTS.md/#10 ("kein Unterschied"), die auf
  einem binären PoseBusters-Pass/Fail-Kriterium mit Deckeneffekt beruhte
  (0% bei allen Methoden verdeckt reale Größenunterschiede).
- Einziger echter Code-Unterschied mit potenzieller Kausalrolle: SigmaDock
  gewichtet den Trainings-Score zeitabhängig (`1/score_scaling(t)²`, aus
  dem Diffusions-Rauschplan), SigmaFlows Flow-Matching-Loss ist eine rohe,
  ungewichtete MSE über alle `t` gleich (bewusste, kommentierte
  Design-Entscheidung: "no time dependent scaling needed, unlike the
  diffusion score", `sigma_flow_generator.py:884`). Widerlegt: fehlender
  Reprojektionsschritt, seltenere Nutzung der Triangulation, falsche
  Implementierung — nichts davon trifft zu (Mechanismus ist identisch).
- Nebenbefund: SO(3)-Winkel-Clamp-Bugfix (`so3_utils.py::Omega`,
  `[-0.99,0.99]`→`[-1+1e-7,1-1e-7]`, bereits in #12 dokumentiert) wurde
  tiefer analysiert — ist eine echte Verbesserung (aktiviert einen
  bereits vorhandenen, aber durch den alten Clamp nie erreichbaren
  `mask_pi`-Sicherheitszweig in `rotation_vector_from_matrix`), keine
  plausible Erklärung für längere Bindungen.

### ✅ Test 1 durchgeführt: Oracle-Vektorfeld widerlegt ODE-Integration als Ursache

**Idee:** `sampler(..., use_true_vector_field=True)` ersetzt bei jedem
ODE-Schritt die Netzwerk-Vorhersage durch das exakte geschlossene
Vektorfeld (`u_t=(x_1-x_t)/(1-t)` bzw. SO(3)-Analogon) — das Netzwerk wird
dabei gar nicht aufgerufen. Das isoliert: liegt die zu lange
Übergangsbindung an der ODE-Integrations-/Rekonstruktions-MECHANIK selbst,
oder an dem, was das trainierte Netzwerk vorhersagt?

**Setup (neu, diese Session):** lokale Python-Umgebung auf diesem
Windows-Rechner von Grund auf aufgebaut (`pip install rdkit torch
torch_geometric matplotlib pillow hydra-core omegaconf pytorch-lightning
wandb biopython spyrmsd posebusters tqdm e3nn` — vorher war nur Basis-Python
installiert), `scripts/sample.py` lief danach sauber lokal auf CPU (kein
ARC/GPU nötig, da `use_true_vector_field=True` das Netzwerk überspringt).
Genutzter Checkpoint: `experiments/sigmadock/0-07-21_11-01-39/checkpoints/
last.ckpt` (früher, nur ~15-20 Trainingsschritte — bewusst irrelevant,
siehe unten).

**Ergebnis (RDKit-Bindungslängen-Messung, 10 Komplexe, identisch zur
Audit-Methodik):**

| Lauf | mean Übergangsbindung | max |
|---|---|---|
| Oracle-Vektorfeld (Netzwerk übersprungen) | **praktisch exakt** (≤0.0002 Å Abweichung, 0 von 278 Bindungen als Übergang auffällig) | — |
| Netzwerk-Vorhersage, selber (kaum trainierter) Checkpoint | 6.93 Å | 12.18 Å |
| (zum Vergleich, SigmaFlow prodhparams, aus Audit) | 2.67 Å | 8.64 Å |
| (zum Vergleich, SigmaDock prodhparams, aus Audit) | 1.81 Å | 5.12 Å |

**Schlussfolgerung:** Die ODE-Integration + starre Fragment-Rekonstruktion
ist NICHT der Fehler — mit dem exakten Vektorfeld rekonstruiert derselbe
Code (25 Euler-Schritte, `power`-Diskretisierung, identisches Batching/
Kanten-Handling wie beim echten Sampling) die wahre Struktur bis auf
Fließkomma-Rauschen exakt, auch an den Fragmentübergängen — unabhängig vom
verwendeten (hier: kaum trainierten) Checkpoint, da die Checkpoint-Gewichte
in diesem Modus gar nicht befragt werden. Das **widerlegt die
1/(1-t)-Singularitäts-Hypothese (E.4 aus dem Audit) endgültig** als
Erklärung. Der Fehler sitzt vollständig in der NETZWERK-VORHERSAGE:
schlechter trainiert → schlechter (6.93 Å), besser trainiert → näher am
Oracle (2.67 Å), aber selbst beim besten verfügbaren Checkpoint bleibt eine
Lücke zum Oracle (~0 Å) UND zu SigmaDock (1.81 Å). Das stützt die
Trainings-Loss-Hypothese (E.3, fehlende zeitabhängige Gewichtung) deutlich,
beweist aber noch nicht, dass GENAU das (statt allgemein "SigmaFlow lernt
Fragmentübergänge schlechter") ursächlich ist — dafür ist Test 3 nötig.

### ✅ Neue Ordnerstruktur angelegt: `SigmaFlow_MinimalChange` / `SigmaFlow_Adapted`

Auf User-Wunsch, um chirurgische Konversion (Projektziel laut `CLAUDE.md`
§1) und künftige, darüber hinausgehende Verbesserungen sauber zu trennen:

- **`SigmaFlow_MinimalChange/`** (neu, exakte Kopie von
  `SigmaFlow_Development/` zum Zeitpunkt 2026-08-05, inkl. `experiments/`-
  Checkpoints): eingefrorener Referenzstand der rein chirurgischen
  Flow-Matching-Konversion (das, was das obige Audit beschreibt). Test 1
  wurde hier durchgeführt. Bleibt unverändert, dient als Vergleichsbasis.
- **`SigmaFlow_Adapted/`** (neu, ebenfalls exakte Kopie zum selben
  Zeitpunkt): Sandbox für größere, über den chirurgischen Eingriff
  hinausgehende Änderungen. **Test 3 (zeitabhängige Loss-Gewichtung für
  Flow Matching einführen) wird hier implementiert**, nicht in
  `SigmaFlow_MinimalChange`.
- **`SigmaFlow_Development/`** (bestehend, unverändert): Rolle für künftige
  Sessions noch nicht neu festgelegt — bis auf Weiteres unangetastet
  gelassen, nicht gelöscht/umbenannt.
- Alle drei Ordner waren zum Zeitpunkt der Kopie (vor jeder Test-1-Änderung)
  identisch, je 1.9 GB (dominiert von `experiments/`-Checkpoints).

### ✅ Paper-Recherche zu Test 3 (Fork, `papers/`-Verzeichnis geprüft statt geraten)

Bevor eine zeitabhängige Loss-Gewichtung implementiert wurde: geprüft, ob
die Flow-Matching-Literatur (`papers/`) eine theoretische Grundlage dafür
hergibt (Pflicht laut `CLAUDE.md` §2). Ergebnis: Holderrieth & Erives und
Chen & Lipman (RFM) definieren den CFM-Loss durchgehend UNGEWICHTET
(`w(t)=1`). Das einzige Paper mit einem zeitabhängigen Faktor ist **Yim et
al. 2023 (FrameFlow, SE(3)-Flow-Matching für Protein-Backbones)** — deren
geclipptes `1/(1-t)²` folgt aber rein algebraisch aus IHRER
Netzwerk-Parametrisierung (Netzwerk sagt x̂1 direkt vorher, nicht das
Vektorfeld) und ist bei genauerem Nachrechnen **mathematisch äquivalent**
zu einem UNGEWICHTETEN v-Raum-Loss, wie SigmaFlow ihn bereits hat — bis auf
den Clip (reine Stabilitätsmaßnahme, kein Genauigkeits-Gewinn). Es gibt
also **keine saubere theoretische Stütze** in der geprüften Literatur für
"SigmaFlow fehlt eine zeitabhängige Gewichtung". Klarstellung/Korrektur
einer eigenen Fehlüberlegung währenddessen: das Draufmultiplizieren eines
`1/(1-t)²`-artigen Faktors AUF SigmaFlows bestehenden (bereits im v-Raum
formulierten) Loss ist trotzdem ein echter, anderer (nicht äquivalenter)
Loss — nur eben ohne FrameFlow-Herleitung dahinter. **Test 3 ist damit
explizit als unbewiesene EMPIRISCHE EXPLORATION eingeordnet**, nicht als
theoretisch fundierte Korrektur.

### ✅ Variante (a) implementiert + geprüft: zeitabhängige Loss-Gewichtung (`SigmaFlow_Variants/a_time_weighting/`)

User hat `compute_losses()` in `sigma_flow_generator.py` selbst geändert
(Lehr-Workflow, Claude hat Interface/Mathematik erklärt und danach
reviewt): neue Methode

```python
def _time_weight(self, t: torch.Tensor, cap_t: float = 0.9) -> torch.Tensor:
    t_clamped = torch.clamp(t, max=cap_t)
    return 1.0 / (1.0 - t_clamped) ** 2
```

wird in `compute_losses()` als `weight = self._time_weight(t_batch)`
berechnet und NACH dem Quadrieren mit `loss_trans`/`loss_R` multipliziert
(`weight * (pt-tt).pow(2).sum(-1)` bzw. analog für Rotation) — korrekt,
da eine Gewichtung VOR dem Quadrieren `weight` selbst mitquadrieren würde
(anderer Fehler, der in der Diskussion mit dem User geklärt wurde).

**Review bestanden, zweifach getestet:**
1. Isolierter Unit-Test von `_time_weight` (Grenzwerte `t=0, 0.01, 0.5,
   0.9, 0.99, 0.999999, 1.0`): alle Werte endlich (max. 100, durch
   `cap_t=0.9` beschränkt, auch exakt bei `t=1.0` kein Blow-up), Shape
   korrekt, Broadcast gegen `[B x F]`-Loss-Werte korrekt.
2. Echter End-to-End-Smoke-Test (`scripts/train.py`, 2 Epochen, CPU,
   `dummy_train`, `--debug --offline_run`, lokal in
   `SigmaFlow_Variants/a_time_weighting/`): lief fehlerfrei durch, endliche
   Loss-Werte (`loss_train/total=40.77`, `loss_val/loss_trans=71.16`, kein
   NaN/Crash).

**Cap-Wert `cap_t=0.9`** ist wie bei Yim et al. eine reine
Stabilitätsmaßnahme (verhindert Explosion bei `t→1`, da `sample_time()`
`t`-Werte beliebig nah an 1 zieht) — keine theoretisch hergeleitete Zahl,
bei Bedarf später als Hyperparameter exponierbar (aktuell hartkodierter
Default-Parameter der Methode).

### ✅ Variante (a) auf ARC eingereicht

`SigmaFlow_Variants/a_time_weighting/slurm/train_dummy_overfit_gpu_3h_time_weighting.sh`
geschrieben (analog `train_dummy_overfit_gpu_3h_prodhparams.sh`, gleiche
Produktions-Hyperparameter `trans_score_weight=2.0 rot_score_weight=0.5
rot_vector_field_scaling=rms`, plus explizites `PYTHONPATH`-Override, da
`sigmaflow_env` sonst standardmäßig den unveränderten Code aus
`SigmaFlow_Development` laden würde statt dieser Variante — Sicherheitscheck
via `sigmadock loaded from: ...`-Print im Log). `sample_dummy.sh` im
selben Ordner ebenfalls auf die richtigen Pfade korrigiert (für den
Sampling-Schritt danach). Hochgeladen (`scp -r`) und via `sbatch`
eingereicht (Job 8443875, 2026-08-05) — ARC war ausgelastet, Job wartet in
der Queue, Startzeitpunkt unklar. **Noch nicht verifiziert:** ob die
`sigmadock loaded from`-Zeile tatsächlich auf den richtigen Pfad zeigt
(User schaut rein, sobald der Job läuft).

### ❌ Variante (b) (x̂1-Reparametrisierung) verworfen, NICHT implementiert

Vor der Implementierung durchgerechnet (User-Frage "wieso sollte das
überhaupt einen Effekt haben, wenn's äquivalent ist" führte zur
entscheidenden Klärung): für SigmaFlows Parametrisierung gilt exakt
`‖x̂1_pred - x1_true‖² = (1-t)² · ‖v_pred - v_true‖²` — ein UNGEWICHTETER
x̂1-Loss ist also mathematisch äquivalent zu einem `(1-t)²`-GEWICHTETEN
v-Raum-Loss, der Fehler nahe `t=1` (die Problemzone) damit RUNTER- statt
hochgewichtet — analytisch die falsche Richtung, kein Implementierungs-
Detail, das ein 3h-ARC-Lauf erst zeigen müsste. FrameFlows tatsächlicher
Loss kompensiert das mit einem geclippten `1/(1-t)²`-Faktor, ist dadurch
aber (bis auf den Clip-Bereich `t>0.9`) selbst wieder fast äquivalent zu
Variante (a)/dem aktuellen Loss — eine Kombination aus beidem hätte also
im Wesentlichen (a) über einen komplizierteren Umweg reproduziert, keine
neue Hypothese getestet.

**Entscheidung (mit User):** verworfen. Code (unveränderte
MinimalChange-Basis, keine Loss-Änderung vorgenommen) nach
`SigmaFlow_Variants/not_implemented/b_x1_reparam/` verschoben, mit
`WARUM_NICHT_IMPLEMENTIERT.md` (Kurzfassung der obigen Herleitung) als
Archiv der Design-Diskussion.

### ✅✅ Test 2 durchgeführt: ODE-Schrittzahl bestätigt wirkungslos (zweite, unabhängige Widerlegung von E.4)

Sampling desselben `prodhparams`-Checkpoints (`experiments/sigmadock/
0-07-25_21-40-56/checkpoints/last.ckpt`) mit `ode.num_steps=10` bzw. `=100`
statt des Defaults `25` (Skripte `slurm/sample_dummy_10steps.sh`/
`_100steps.sh` in `SigmaFlow_MinimalChange/`, kein Retraining, reine
Sampling-Config-Änderung). RDKit-Bindungslängen-Messung (gleiche Methodik
wie Test 1/Audit Teil 5.2, 10 Komplexe):

| Lauf | n Übergangsbindungen | mean | median | max | >3.0 Å | >5.0 Å |
|---|---|---|---|---|---|---|
| 10 Schritte | 41 | 2.675 Å | 2.489 Å | 8.546 Å | 13 | 3 |
| 25 Schritte (Baseline) | 40 | 2.672 Å | 2.456 Å | 8.638 Å | 13 | 3 |
| 100 Schritte | 40 | 2.670 Å | 2.457 Å | 8.747 Å | 13 | 3 |

**Praktisch identisch über einen 10-fachen Schrittzahl-Bereich.** Bestätigt
unabhängig von Test 1 (dort: exaktes Vektorfeld → exakte Rekonstruktion;
hier: reales Netzwerk-Feld, Diskretisierung variiert → kein Effekt): Kandidat
E.4 (`1/(1-t)²`-Singularität/Diskretisierungsfehler) ist damit **zweifach,
über zwei verschiedene Experimente**, als Erklärung für die
Bindungslängen-Differenz widerlegt. Der Fehler sitzt vollständig in der
Netzwerk-Vorhersage selbst (Trainings-Frage, nicht Sampling-Frage).

### ⚠️ Echter Skript-Bug gefunden + gefixt: stiller Fehlschlag als `COMPLETED` getarnt

Erste Einreichung von Job 8443875 (Variante a, `train_dummy_overfit_gpu_3h_
time_weighting.sh`) meldete `sacct`-Status `COMPLETED, ExitCode=0:0`, lief
aber nur 5:04 Min statt der vorgesehenen 2:45h. Ursache: das Skript enthielt
`--rot_vector_field_scaling rms`, kopiert aus der älteren
`train_dummy_overfit_gpu_3h_prodhparams.sh`-Vorlage — dieser CLI-Flag wurde
aber bereits in PAUSE-PUNKT #12 als toter Code aus `config.py` entfernt.
`scripts/train.py` brach sofort mit `argparse`-Fehler ab; da das Skript kein
`set -e` hatte, lief die letzte (immer erfolgreiche) `echo`-Zeile trotzdem
durch, und SLURM sah nur deren Exit-Code. **Exakt das Muster, das schon in
PAUSE-PUNKT #13 als Risiko notiert, aber nie behoben wurde** — jetzt real
aufgetreten. Bemerkenswert: der Grep-Check aus PAUSE-PUNKT #13 ("keine
Treffer... auch nicht in den SLURM-Skripten") hat diesen Fund nicht
verhindert — entweder war die Prüfung unvollständig, oder die Datei wurde
danach nochmal mit dem alten Flag angelegt; nicht abschließend geklärt.

**Gefixt:** `--rot_vector_field_scaling rms` aus allen drei betroffenen
Skripten entfernt (`a_time_weighting/slurm/train_dummy_overfit_gpu_3h_
time_weighting.sh`, `SigmaFlow_MinimalChange/slurm/train_pdbbind_general_6h.sh`
— Letzteres war noch nicht submittet, Fehler rechtzeitig gefunden). `set
-euo pipefail` in allen drei neuen Skripten dieser Session ergänzt
(inkl. des SigmaDock-6h-Skripts, dort zur Sicherheit, der eigentliche Fehler
betraf nur die SigmaFlow-Skripte, da SigmaDocks `--rot_score_method`/
`--rot_score_scaling` echte, nicht entfernte Flags sind). Variante (a)
danach neu submittet (Job-ID noch nicht bestätigt/geprüft).
**Lehre, jetzt wirklich verstanden statt nur notiert:** `set -euo pipefail`
gehört in JEDES `slurm/*.sh` in diesem Repo, nicht nur in neu geschriebene —
die älteren Skripte (`train_dummy_overfit_gpu_3h_prodhparams.sh` etc.) haben
es weiterhin nicht.

### ✅ ARC-Infrastruktur für den großen Lauf geklärt (mehrere offene Punkte aus #13 beantwortet)

- **Partitions-Zeitlimits** (User hat ARC-Doku geprüft): `short` = 12h max,
  `medium` = 48h max, `long` = 30 Tage max, `devel` = 10 Min (nur Tests),
  `interactive` = 24h. Damit ist `short` für den geplanten 6h/12h/24h-Stufen-
  plan durchgehend ausreichend, `long` für den späteren 3-/7-Tage-Lauf.
- **Gestuftes grünes Licht für den großen Lauf** (User-Info, neu): 6h → 12h →
  max. 24h auf Dummy-Daten/PDB-Subsets vs. SigmaDock, **erst danach**
  Freigabe für 3-Tage-, dann 7-Tage-Lauf. Ändert die Priorität aus #13
  (7-Tage-Infrastruktur) — der 6h-Lauf ist jetzt der unmittelbar nächste
  Schritt, nicht mehr blockiert durch die Mehrtages-Fragen.
- **PDBbind-Datenpfade auf ARC prospektiv geprüft** (`ls`/`find`, nicht
  angenommen): `pdbbind/general-set/` ist befüllt (~19k Komplexe, Dateinamen
  passen exakt zur Config-Regex, z.B. `10gs_pocket.pdb`/`10gs_ligand.sdf`).
  `pdbbind/refined-set/` und `pdbbind/core-set/` sind dagegen LEER (Link-Count
  2, kein Unterordner — nie aus `pdbbind/raw/P-L.tar.gz` extrahiert).
  `posebusters_paper/posebusters_benchmark_set/` (209 Komplexe) und
  `.../astex_diverse_set/` (85 Komplexe) sind befüllt, Dateinamen passen zur
  `posebusters.yaml`/`astex.yaml`-Regex (`ligands.sdf`-Plural beachtet).
- **Angepasster Split für den 6h-Lauf:** da `refined-set`/`core-set` leer
  sind, `train_exps=pdbbind-general` + `val/test_exps=posebusters` statt
  SigmaDocks eigentlich vorgesehenem `[general,refined]`/`[core]`/
  `[posebusters,astex]`-Split (`SigmaDock/conf/training/slurm.yaml`).
- **Zwei 6h-SLURM-Skripte fertig** (`SigmaFlow_MinimalChange/slurm/
  train_pdbbind_general_6h.sh`, `train_sigmadock_pdbbind_general_6h.sh`):
  SigmaFlow mit den validierten `prodhparams`-Gewichten (siehe Sweep-Zahlen
  oben), SigmaDock mit seiner eigenen Produktionsconfig (`trans_score_
  weight=2.0, rot_score_weight=0.5, rot_score_method=space,
  rot_score_scaling=rms`, direkt aus `SigmaDock/conf/training/slurm.yaml`
  verifiziert). `--batch_size 8`, `--val_check_interval 200` sind
  ungetestete Schätzungen (General-Set-Komplexe sind größer als die 10
  Dummy-Komplexe), im Skript als solche markiert. SigmaDock-Skript läuft im
  separaten ARC-Klon aus PAUSE-PUNKT #9. **Noch nicht submittet.**

### ✅✅✅ Test 3 durchgeführt: Variante (a) (Zeitgewichtung) schließt die Bindungslängen-Lücke NICHT

Nach dem Flag-Fix (`--rot_vector_field_scaling rms` entfernt, `set -euo
pipefail` ergänzt) lief Job 8447496 sofort wieder in 6s fehl (identisches
Muster — separat nicht weiter untersucht, da der direkt danach gestartete
Retry, Job **8447571**, sauber durchlief). 8447571 lief bis zur Walltime
(2:45h, Epoche 632, `TIMEOUT`-Status — hier KEIN Fehlschlag, sondern
erwartetes Verhalten, da `Early stopping is disabled` und das Skript
bewusst bis zum Zeitlimit trainiert). `last.ckpt` wurde 3 Minuten vor dem
Kill sauber gespeichert (`experiments/sigmadock/0-08-06_09-42-56/
checkpoints/last.ckpt`).

**Nebenbefund, geklärt:** die drei `checkpoint-step=*-val_loss=0.0000.ckpt`-
Dateinamen sind ein reiner Kosmetik-Bug, kein Auswahlfehler — `scripts/
train.py:311-319` übergibt `monitor=args.monitor_metric` (korrekt
`"loss_val/total"`, `config.py:160`) an `ModelCheckpoint` für die
tatsächliche `save_top_k`-Auswahl, aber der `filename`-String ist
hartkodiert `"checkpoint-{step:02d}-{val_loss:.4f}"` — der Platzhalter
`val_loss` existiert nicht unter diesem Namen in den geloggten Metriken,
Lightning füllt daher `0.0000` ein. Vorbestehend (nicht durch diese
Session verursacht), betrifft vermutlich alle bisherigen Läufe mit diesem
Skript. Nicht dringend gefixt.

Gesampelt (Job 8448647, `sample_dummy.sh`, `COMPLETED`, `sigmadock loaded
from`-Check bestätigt korrekten Variante-a-Code), SDFs per `scp`
heruntergeladen, mit einem neu geschriebenen RDKit-Skript (`bond_length_
check_time_weighting.py`, Scratchpad, Methodik identisch zu Audit Teil
8.5/9.2: pro Bindung `|vorhergesagte Länge - wahre Länge| > 0.02 Å` =
"Übergangsbindung") gegen die bestehenden SigmaFlow-/SigmaDock-
`prodhparams`-Baselines verglichen — Baseline-Zahlen reproduzieren dabei
exakt die im Audit dokumentierten Werte (Skript methodisch verifiziert):

| Lauf | n Übergangsbindungen | mean | median | max |
|---|---|---|---|---|
| SigmaFlow, **time_weighting (Variante a)** | 41 | 2.816 Å | 2.290 Å | 7.066 Å |
| SigmaFlow, prodhparams (Baseline, ungewichtet) | 40 | 2.672 Å | 2.456 Å | 8.638 Å |
| SigmaDock, prodhparams (Baseline) | 35 | 1.811 Å | 1.485 Å | 5.120 Å |

**Ergebnis: kein Effekt.** Variante (a) liegt praktisch auf demselben
Niveau wie die ungewichtete SigmaFlow-Baseline (41 vs. 40 Übergangs-
bindungen, mean 2.82 Å vs. 2.67 Å — tendenziell sogar leicht schlechter),
der Abstand zu SigmaDock (1.81 Å) bleibt in beiden Fällen etwa gleich
groß. Deckt sich mit der fehlenden theoretischen Stütze aus der
Paper-Recherche weiter oben — die Zeitgewichtungs-Hypothese (E.3) ist
damit auch empirisch widerlegt, nicht nur theoretisch unbegründet.

**Stand der Kausalitätsfrage nach drei Tests:** ODE-Integrations-Mechanik
(Test 1: Oracle-Vektorfeld rekonstruiert exakt; Test 2: Schrittzahl
wirkungslos) UND globale Zeitgewichtung des Trainingsloss (Test 3, s.o.)
sind beide als Erklärung ausgeschlossen. Der Fehler sitzt nachweisbar in
der gelernten Netzwerk-Vorhersage selbst (Test 1 zeigt das indirekt: je
besser trainiert, desto näher am Oracle, aber auch der beste bisherige
Checkpoint bleibt hinter SigmaDock zurück) — aber WARUM SigmaFlows
Netzwerk an den Fragmentübergängen ein ungenaueres Vektorfeld lernt als
SigmaDocks Netzwerk einen Score, ist nach diesen drei Tests weiterhin
nicht kausal geklärt (siehe Antwort an den User unten für die aktuell
plausibelste, aber unbewiesene Erklärung).

### 💡 Idee (zurückgestellt, noch nicht implementiert): gezielter Anker-Atom-Distanz-Loss

Diskutiert nach Test 3, User-Entscheidung: erstmal zurückgestellt, zuerst die
6h-Läufe (s.u.) auswerten und schauen, ob das Problem dort in derselben
Deutlichkeit auftritt, bevor daran weitergearbeitet wird. Festgehalten,
damit die Ausarbeitung nicht verloren geht:

**Plausible (unbewiesene) Erklärung, warum SigmaFlow überhaupt schlechter
lernt als SigmaDock, trotz identischer Architektur/Daten/Triangulation:**
SigmaDock sagt einen Diffusions-Score vorher, dessen Größe bei kleinem
Rauschen (t nah am fertigen Zustand) intrinsisch explodiert (`~1/σ(t)²`) —
das zwingt das Netzwerk architekturbedingt zu hoher Präzision in der
Endphase, unabhängig von jeder expliziten Loss-Gewichtung. SigmaFlows
CondOT-Vektorfeld-Ziel (`x1-x0`) ist dagegen konstant in `t` und hat diese
eingebaute Verschärfung nicht. Das würde erklären, warum Test 3
(Gewichtung draufmultiplizieren) wirkungslos war: das zugrunde liegende
Regressionsproblem unterscheidet sich in seiner Struktur, nicht nur in
seiner Gewichtung — das lässt sich nicht durch Reskalieren nachbauen.
Alternative, ebenfalls nicht ausgeschlossene Erklärung: Netzwerkkapazität
(dasselbe Netz könnte für die FM-Zielgröße an genau dieser geometrischen
Größe grundsätzlich schwerer zu trainieren sein) — dafür gibt es noch
keinen eigenen Test.

**Die Idee selbst:** statt einer globalen Zeitgewichtung (Test 3, negativ)
einen zusätzlichen, RÄUMLICH gezielten Loss-Term einbauen, der direkt den
Abstand zwischen den beiden Anker-Atomen einer (ehemals) geschnittenen
Torsionsbindung bestraft — dem exakten Ort, an dem die RDKit-Messung das
Problem verortet. Die Atompaare sind bereits bekannt:
`fragmentation.py::fragment_ligand()`s `frag_map["torsional_bonds"]`
(Zeile ~245) enthält pro geschnittener Bindung das Paar `(anchor1,
anchor2)`.

**Wie man an eine "aktuelle Vorhersage" der Anker-Position kommt:** aus der
Netzwerk-Vorhersage `pred_u_t_trans`/`pred_u_t_R` bei Zeit `t` lässt sich
(dieselbe Reparametrisierung, die für Variante (b) durchgerechnet und
verworfen wurde — hier als Werkzeug wiederverwendet, nicht als Ersatz-Loss)
ein `trans_1_pred`/`R_1_pred`-Schätzwert pro Fragment ableiten
(`x̂1 = x_t + (1-t)·v_pred`-artig). Dieselbe starre Transformation, die
`get_transformations_from_rototranslations` (`sigma_flow_generator.py:298`)
auf echte Atome anwendet, liefert damit angewandt auf die beiden
Anker-Atome ihre vorhergesagten 3D-Positionen `p̂_A`, `p̂_B` — bei JEDEM
`t`, nicht nur bei `t=1`.

**Zwei Formulierungs-Varianten:**
- **Direkter Abstands-Loss** (empfohlen): `(‖p̂_A - p̂_B‖ - ‖A_true -
  B_true‖)²`. Bestraft nur die relative Anordnung der zwei Fragmente
  zueinander, trifft exakt das, was die RDKit-Messung als "Übergangs-
  bindung" mist, unempfindlich gegen globale Verschiebung. Erfordert die
  Kreuz-Fragment-Paarbildung explizit im Loss.
- **Positions-Loss auf Anker-Atomen**: `‖p̂_A - A_true‖² + ‖p̂_B -
  B_true‖²`. Einfacher (reine Erweiterung der bestehenden COM-Loss-Logik),
  aber nur ein indirekter Proxy für die Bindungslänge.

**Offene Frage, VOR der Implementierung zu klären (Schritt 1: erst
inspizieren):** überlebt die `torsional_bonds`-Atompaar-Liste aus der
Fragmentierung bis zum Zeitpunkt von `compute_losses()` im `batch`-Objekt,
oder wird sie nur beim Graph-Aufbau (`data.py`) verwendet und danach
verworfen? Falls letzteres, müsste sie zusätzlich durchgereicht werden —
das würde den Eingriff größer machen als der reine Loss-Term.

### ✅✅✅ 6h-Real-Daten-Vergleich ausgewertet: Bindungslängen-Lücke DREHT SICH UM

Beide 6h-Läufe (SigmaFlow Job 8447572, SigmaDock Job 8447498, beide
`train_exps=pdbbind-general`/`val_exps=posebusters`, identische
Produktions-Hyperparameter je Methode, s.o.) liefen sauber bis zur vollen
Walltime (`TIMEOUT` bei exakt 6:00:0x, kein Fehlschlag). Checkpoints:
`SigmaFlow_Development/experiments/sigmadock/0-08-06_09-42-56/checkpoints/
last.ckpt`, `SigmaDock_Reproduction_JulianMueller/sigmadock/experiments/
sigmadock/0-08-06_09-26-06/checkpoints/last.ckpt`.

**Fairness der Trainingsläufe verifiziert** (auf expliziten User-Wunsch, vor
der Auswertung): beide exakt 6:00h gelaufen, beide Checkpoints <1 Min vor
Kill gespeichert, kein NaN/Crash in beiden Logs, `sigmadock loaded from`
zeigt in beiden Fällen auf den richtigen Code-Pfad. Einzige Asymmetrie:
SigmaDock übersprang mehr `[WARN] Fragment mass below 2`-Batches (170) als
SigmaFlow (111) — ein Datenqualitätsproblem im gemeinsamen
`pdbbind-general`-Datensatz, das BEIDE Methoden trifft; falls überhaupt ein
Bias, dann zulasten SigmaDock (weniger effektive Trainingsschritte), nicht
zugunsten SigmaFlow.

**Gesampelt** auf 10 zufällig gewählten, NIE gesehenen PoseBusters-Komplexen
(`5S8I_2LY, 5SAK_ZRY, 5SD5_HWI, 5SIS_JSM, 6T88_MWQ, 6TW5_9M2, 6VTA_AKN,
6XAF_GDP, 6XG5_TOP, 6XUM_30L`) — ein echter Generalisierungstest, kein
Auswendiglernen wie bei den 10 Dummy-Komplexen. Wichtige Methodik-Lektion
unterwegs: der erste Versuch nutzte `graph.sample_conformer=true` (frisch
generierter Konformer statt Kristallpose als Fragmentierungs-Basis) — das
bricht die RDKit-Bindungslängen-Messmethodik (Intra-Fragment-Bindungen sind
dann nicht mehr garantiert längenerhaltend gegenüber der wahren Pose),
sichtbar daran, dass beide unabhängig trainierten Methoden EXAKT dieselbe
Anzahl "Übergangsbindungen" pro Komplex zeigten (Artefakt des gemeinsamen,
seed-deterministischen Fresh-Konformers, nicht der Modelle). Korrigiert
durch Neu-Sampling mit `graph.sample_conformer=false` (wie bei allen
bisherigen Tests), verifiziert per Code-Inspektion (`data.py:156-159`: die
"Sample conformer is set to True"-Logzeile erscheint nur im `True`-Fall,
ihr Fehlen im korrigierten Lauf bestätigt `False`).

**Ergebnis (RDKit-Bindungslängen-Messung, identische Methodik wie
Test 1/2/3, `TOL=0.02 Å`, 264 Bindungen über 10 Komplexe):**

| Lauf | n Übergangsbindungen | mean | median | max |
|---|---|---|---|---|
| **SigmaFlow**, 6h, echte Generalisierung | **37** | **2.93 Å** | 2.56 Å | **10.83 Å** |
| **SigmaDock**, 6h, echte Generalisierung | **46** | **13.09 Å** | 3.29 Å | **81.73 Å** |

**SigmaFlow schneidet hier BESSER ab als SigmaDock** — umgekehrt zum
bisherigen Befund auf den 10 auswendig gelernten Dummy-Komplexen (Audit,
Test 1-3: SigmaFlow ~2.7 Å vs. SigmaDock ~1.8 Å mean). SigmaDocks
13.09-Å-Mittelwert wird von einem einzelnen katastrophalen Ausreißer
(81.73 Å, konzentriert auf Komplex `6TW5_9M2`: 12 von 34 Bindungen dort
Übergänge, gegenüber nur 4 von 34 bei SigmaFlow) dominiert — kein
"leicht zu lange", sondern ein komplett zerschossenes Atom. Bei 8 von 10
Komplexen sind sich beide Methoden dagegen fast identisch (gleiche Anzahl
Übergangsbindungen) — plausibel bei nur 6h Training (deutlich weniger
Feinschliff als die stark überangepassten Dummy-Checkpoints), beide
Methoden sind bei den "leichteren" Komplexen ähnlich grob und
unterscheiden sich vor allem dort, wo es schwer wird.

**Einordnung, mit Vorsicht:** dieser Befund relativiert die bisherige
Erzählung ("SigmaFlow baut systematisch längere Übergangsbindungen")
deutlich, beweist aber nicht ihr Gegenteil — Stichprobe ist klein (10
Komplexe, 1 Ausreißer dominiert den Mittelwert), und beide Checkpoints sind
mit 6h nur kurz trainiert (nicht das Konvergenz-Regime der
Dummy-Overfit-Tests). Naheliegende, unbewiesene Hypothese: der bisherige
Befund könnte stärker mit ÜBERANPASSUNGSVERHALTEN an die 10 Dummy-Komplexe
zusammenhängen als mit einer grundsätzlichen Flow-Matching-Schwäche an
Fragmentübergängen — dafür bräuchte es aber einen direkten Vergleich bei
längerem Training auf echten Daten (12h/24h-Stufe), nicht nur diese
6h-Momentaufnahme.

### ✅ Fairness des 6h-Vergleichs verifiziert (auf User-Wunsch, vor dem Festhalten der Ergebnisse)

Vier Punkte geprüft, alle bestanden:
1. Beide Läufe exakt 6:00:0x gelaufen (`TIMEOUT`), beide Checkpoints <1 Min
   vor Kill gespeichert, kein NaN/Crash in beiden Logs.
2. `[WARN] Fragment mass below 2`-Skips: SigmaFlow 111/~7550 Steps (1.47%),
   SigmaDock 170/~7850 Steps (2.17%) — klein, für beide vergleichbar, falls
   überhaupt ein Bias dann zulasten SigmaDock.
3. **Herkunft dieser Warnung geklärt:** der zugrundeliegende `assert
   M.min() >= 2` (`sigma_flow_generator.py:823`) existiert wortgleich im
   Original (`SigmaDock/src_sigmadock/src_sigmadock_diff/denoiser -
   UMBAUEN.py:950-951`, per Grep verifiziert) — geerbte SigmaDock-
   Design-Eigenschaft (Starrkörper-Physik braucht Mindestmasse pro
   Fragment), kein SigmaFlow-Bug. Der `try/except`-Skip-Wrapper drumherum
   (`trainer.py:320-328`, macht aus dem harten Crash ein `[WARN]`) existiert
   im Original NICHT — wurde vor dieser Session ergänzt (uncommitted lokale
   Änderung), nötig weil die 10 kuratierten Dummy-Komplexe nie einen so
   entarteten Fall enthielten.
4. **Der exakt selbe Skip-Wrapper existiert auch in der separaten
   SigmaDock-ARC-Kopie** (`SigmaDock_Reproduction_JulianMueller/sigmadock/
   src/sigmadock/trainer.py`, per Grep byte-identisch bestätigt, sogar
   derselbe Kommentar-Verweis auf `STATUS.md PAUSE-PUNKT #14) — beide
   Trainingsläufe wurden also unter identischen Bedingungen gefahren.

### ⏳ 12h-Stufe gestartet: Fortsetzung der 6h-Checkpoints (nicht neu von Null)

Auf User-Entscheidung: **Fortsetzen statt neu starten** (spart Compute,
kumulativ 12h Training). Neue Skripte `train_pdbbind_general_resume_6more.sh`
(SigmaFlow + SigmaDock-Analog, beide mit `--resume_from_checkpoint`, das
laut `scripts/train.py:81-95` sowohl den Checkpoint als auch das zugehörige
`EXP_DIR`/W&B-Run automatisch weiterführt statt einen neuen Ordner
anzulegen). Jeweils nur 6 WEITERE Stunden angefordert (nicht 12h) — `short`
(12h-Limit) reicht damit komfortabel, keine Partitions-Grenzfall-Frage.

**Erster Versuch (Jobs 8451879/8451880) FAILED nach ~5 Min bei beiden** —
echter Fehlschlag, kein stiller Erfolg (`set -euo pipefail` griff). Ursache
(`.err`-Log, nicht `.out`): `_pickle.UnpicklingError: Weights only load
failed` beim Wiederherstellen des vollen Trainer-Zustands
(`trainer.fit(..., ckpt_path=...)`). **Root Cause:** PyTorch ≥2.6 hat den
Default von `torch.load(weights_only=...)` von `False` auf `True`
umgestellt (Sicherheits-Härtung) — verweigert seitdem das Laden eigener
Klassen aus dem Checkpoint (`sigmadock.oracle.HParams` bei SigmaFlow,
`pathlib.PosixPath` bei SigmaDock), außer man erlaubt sie explizit. Trat
nur beim FORTSETZEN auf (voller Trainer-Zustand inkl. Optimizer), nicht
beim ursprünglichen 6h-Start (kein Checkpoint zu laden) oder beim Sampling
(anderer Lade-Pfad). **Gefixt:** `Trainer.fit()` akzeptiert `weights_only`
direkt als Parameter (per `inspect.signature` verifiziert) —
`scripts/train.py`s Resume-Zweig ruft jetzt `trainer.fit(...,
weights_only=False)` (unbedenklich, da immer der eigene, vertrauenswürdige
Checkpoint). Gepatcht in BEIDEN Codebasen (SigmaFlow lokal editiert + hochgeladen,
SigmaDock direkt auf ARC per `sed`, da kein lokaler Klon existiert).

Submittet 2026-08-06: **Job 8452085** (SigmaFlow, resumed von
`experiments/sigmadock/0-08-06_09-42-56/checkpoints/last.ckpt`), **Job
8452088** (SigmaDock, resumed von `experiments/sigmadock/
0-08-06_09-26-06/checkpoints/last.ckpt`). Beide liefen sauber die vollen 6
weiteren Stunden (`TIMEOUT`, 21:1x→03:1x über Nacht) — kumulativ 12h
Training. **Fairness auch für diese Stufe verifiziert:** kein NaN/Crash in
beiden Logs (nur die Standard-"NaN Check Callback enabled"-Zeile),
Fragment-mass-Warnungen im zweiten 6h-Abschnitt SigmaFlow 147/~7950 Schritte
(≈1.85%) vs. SigmaDock 172/~7950 (≈2.16%) — ähnliche Größenordnung wie im
ersten Abschnitt, kein neuer Bias. Checkpoints frisch gespeichert kurz vor
Kill (SigmaFlow Step 15500, SigmaDock Step 15800 — beide ca. verdoppelt
gegenüber dem 6h-Stand, konsistent).

### ✅✅ 12h-Ergebnis ausgewertet: 6h-Führung von SigmaFlow schmilzt deutlich

Dieselben 10 PoseBusters-Komplexe, dieselbe Methodik (`sample_conformer=
false`, RDKit-Bindungslängen-Messung, `TOL=0.02 Å`):

| Lauf | n Übergangsbindungen | mean | median | max |
|---|---|---|---|---|
| SigmaFlow, 6h | 37 | 2.93 Å | 2.56 Å | 10.83 Å |
| **SigmaFlow, 12h** | **36** | 2.59 Å | 2.31 Å | 6.77 Å |
| SigmaDock, 6h | 46 | 13.09 Å | 3.29 Å | 81.73 Å |
| **SigmaDock, 12h** | 43 | **1.26 Å** | **1.31 Å** | **3.17 Å** |

**SigmaDocks katastrophaler 81.7-Å-Ausreißer (`6TW5_9M2`) ist mit mehr
Training praktisch verschwunden** (max jetzt 3.17 Å) — die Anzahl der
Übergangsbindungen bei diesem Komplex blieb bei 12 (unverändert), aber ihre
Schwere sank massiv. SigmaFlow veränderte sich dagegen kaum: die
Pro-Komplex-Anzahl der Übergangsbindungen ist zwischen 6h und 12h fast
identisch (nur `6XUM_30L` um 1 verbessert), nur die Beträge sanken moderat.

**Ergebnis jetzt gemischt, nicht mehr eindeutig:**
- SigmaFlow gewinnt bei der ANZAHL (36 vs. 43)
- SigmaDock gewinnt jetzt klar bei der SCHWERE (mean/median/max alle
  niedriger)

SigmaDock hat sich von 6h auf 12h deutlich stärker verbessert als
SigmaFlow. Der 6h-Befund ("SigmaFlow klar vor SigmaDock") ist damit **nicht
bestätigt** — er war vermutlich stark von SigmaDocks einzelnem, noch nicht
ausgeheiltem Trainings-Ausreißer getrieben, nicht von einem grundsätzlichen
Methodenunterschied.

### Nebenuntersuchung: warum sampelt SigmaFlow langsamer als SigmaDock?

User-Frage nach Beobachtung 5:38 Min (SigmaFlow) vs. 52s (SigmaDock) für
dieselben 10 Komplexe. Per Code-Vergleich geprüft (nicht spekuliert):
- **Schrittzahl identisch:** beide `num_steps: 25` (`conf/sampling/
  base.yaml` in beiden Bäumen).
- **Solver identisch:** beide `solver: euler`.
- **Pro-Schritt-Struktur identisch:** `SigmaFlow_Development/src/sigmadock/
  diff/sampling.py::sampler()` und die Original-Referenz (`SigmaDock/
  src_sigmadock/src_sigmadock_diff/sampling - UMBAUEN.py::sampler()`) rufen
  BEIDE pro Schritt genau EINEN Netzwerk-Forward-Pass auf
  (`_predict_vector_field_step`/`_reverse_step`) plus eine geschlossene
  "wahres Feld/wahrer Score"-Berechnung fürs Loss-Logging (kein Netzwerk-
  Aufruf, nur Formel) — strukturell identisches Muster in beiden Bäumen,
  keine Asymmetrie gefunden.
- Direkt gemessen (aus dem Log, `Predicting DataLoader 0`-Zeile): 27s
  (SigmaFlow) vs. 7s (SigmaDock) für den reinen Vorhersageschritt — ein
  echter ~4x-Unterschied, der aber algorithmisch NICHT erklärt werden
  konnte. Beide Jobs liefen laut `squeue` zufällig auf demselben Knoten
  (`htc-g065`), zeitgleich gestartet — GPU-Konkurrenz auf demselben Knoten
  ist ein möglicher, nicht verifizierter Störfaktor.
- **Offen, nicht abschließend geklärt:** die Restdifferenz zwischen
  Gesamtlaufzeit (338s vs. 52s) und der gemessenen Vorhersagezeit (27s vs.
  7s) liegt vor der `Predicting`-Zeile (Environment-Aktivierung,
  Imports, Modellaufbau) und ist im Log nicht zeitgestempelt — Ursache
  unbekannt, für die Bindungslängen-Ergebnisse irrelevant (betrifft nur
  Sampling-Geschwindigkeit, nicht Trainings-Fairness oder Ergebnisqualität).

### ✅✅✅ Vollständiges PoseBusters-Set (209 Komplexe) ausgewertet — 10-Komplexe-Befund bestätigt, jetzt statistisch robust

Auf User-Wunsch, um die Kleine-Stichprobe-Problematik zu beheben (der
6h→12h-Vergleich hatte gezeigt, wie stark 10 Komplexe von einem einzelnen
Ausreißer dominiert werden können): dasselbe Sampling+Mess-Verfahren auf
ALLEN 209 PoseBusters-Komplexen wiederholt, mit den 12h-Checkpoints. Neue
Skripte `sample_posebusters_full.sh` (SigmaFlow + SigmaDock-Analog, gleiche
Methodik wie `sample_posebusters_10.sh` nur ohne Whitelist) und
`bond_length_check_posebusters_full.py` (Scratchpad, Komplexliste wird aus
dem Ordner ermittelt statt hartkodiert, robust gegen Parse-Fehler/
Bindungszahl-Mismatches). Beide Sampling-Jobs liefen sauber und schnell
(SigmaFlow 6:34 Min, SigmaDock 2:02 Min, Jobs 8464306/8464307) — bei 209
statt 10 Komplexen skaliert die Zeit weit unterlinear, wie vermutet
(Fixkosten pro Job dominieren gegenüber Pro-Komplex-Kosten). Wahre
Referenz-Liganden aus `posebusters_paper/posebusters_benchmark_set/` per
`tar` gebündelt heruntergeladen (209 Dateien einzeln per `scp` wäre
unpraktikabel gewesen).

**Ergebnis (5576 Bindungen über alle 209 Komplexe, KEIN Komplex
übersprungen — 0 Parse-Fehler, 0 Bindungszahl-Mismatches):**

| Lauf | n Übergangsbindungen | % aller Bindungen | mean | median | max |
|---|---|---|---|---|---|
| **SigmaFlow**, 12h | 728 | 13.1% | 2.68 Å | 2.23 Å | 15.73 Å |
| **SigmaDock**, 12h | 830 | 14.9% | 1.25 Å | 1.22 Å | 18.08 Å |

**Bestätigt das gemischte 10-Komplexe-Ergebnis bei 12h, jetzt mit
statistisch robuster Basis (5576 statt 264 Bindungen):**
- SigmaFlow hat WENIGER betroffene Bindungen (13.1% vs. 14.9%)
- SigmaDock hat KLEINERE Abweichungen, wenn eine Bindung betroffen ist
  (mean/median etwa halb so groß wie bei SigmaFlow)
- Beide haben noch einzelne schwere Ausreißer (SigmaFlow max 15.73 Å,
  SigmaDock max 18.08 Å) — keine Methode ist frei davon; die schlimmsten
  Komplexe unterscheiden sich zwischen den Methoden (z.B. `7T1D_E7K`: 26
  von 39 Bindungen bei SigmaDock betroffen, deutlich schlimmer als bei
  SigmaFlow für diesen Komplex).

**Einordnung:** dies ist jetzt ein belastbares Ergebnis (volle Stichprobe,
kein Ausreißer-Artefakt), aber weiterhin **kein klarer Sieger** — echter
Trade-off zwischen Häufigkeit (SigmaFlow besser) und Schwere (SigmaDock
besser) der Fragmentübergangs-Fehler, bei nur 12h Training für beide. Die
ursprüngliche Audit-Erzählung ("SigmaFlow baut systematisch längere
Übergangsbindungen als SigmaDock") ist damit für echte
Generalisierungs-Checkpoints NICHT bestätigt — sie galt nur für die stark
überangepassten 10-Dummy-Komplexe-Checkpoints.

### ✅ 24h-Stufe gestartet: FRISCH von Null (kein Fortsetzen)

User-Entscheidung: 24h-Läufe diesmal **komplett neu von Null trainiert**,
nicht fortgesetzt — auch weil die 6h/12h-Checkpoints ohnehin nicht mehr
existieren (siehe oben: `--resume_from_checkpoint` schreibt in denselben
Ordner, `save_top_k=3` hat die alten Checkpoint-Dateien beim Fortsetzen
verdrängt). Ziel: prüfen, ob der bei 12h beobachtete Trade-off (SigmaFlow
weniger, aber größere Übergangsbindungs-Abweichungen; SigmaDock mehr, aber
kleinere) bei 24h Training bestehen bleibt.

Neue Skripte `train_pdbbind_general_24h.sh` (SigmaFlow + SigmaDock-Analog),
identische Hyperparameter wie die 6h/12h-Läufe, `--partition=medium` (24h
überschreitet `short`s 12h-Limit; `medium` erlaubt bis 48h). Eigene neue
Experiment-Ordner (kein `--resume_from_checkpoint`).

Submittet 2026-08-07: **Job 8465054** (SigmaFlow), **Job 8465055**
(SigmaDock), beide `medium`, `--time=24:00:00`, frisch von Null (eigene
neue Experiment-Ordner).

### ✅✅✅ 24h-Ergebnis ausgewertet (2026-08-08): 12h-Trade-off bestätigt, bleibt bestehen

Beide Jobs sauber `TIMEOUT` nach exakt 24h (`sacct`: `1-00:00:17`,
ExitCode `0:0`). **Fairness verifiziert** wie bei 6h/12h: kein NaN/Crash in
beiden Logs (nur die erwartete `CANCELLED ... DUE TO TIME LIMIT`-Zeile im
`.err`); Fragment-mass-Skips SigmaFlow 564, SigmaDock 694 (vergleichbare
Größenordnung); Checkpoints frisch vor Kill gespeichert (SigmaFlow 09:53,
SigmaDock 09:33, beide <30 Min vor `CANCELLED AT 09:59:23`). **Eine neue,
nicht sofort erklärte Asymmetrie:** SigmaFlow zeigt ~30
`Failed to parse pocket`-Skips (verstreut über verschiedene PDB-Komplexe,
bekannte harmlose Warnung, s. PAUSE-PUNKT #4/Runde 2), SigmaDock zeigt
**0** über den ganzen Lauf — bei vergleichbarer Step-Zahl (SigmaFlow
~31700, SigmaDock ~31000+) auf demselben Datensatz eher unwahrscheinlich
als reiner Zufall der Sampling-Reihenfolge, aber auch nicht crash-relevant
(<0.1% der Steps) — als offene Beobachtung notiert, nicht weiter verfolgt.

Checkpoints: `experiments/sigmadock/0-08-07_10-03-37/checkpoints/last.ckpt`
(SigmaFlow, im `SigmaFlow_Development`-Baum) bzw.
`experiments/sigmadock/0-08-07_09-59-31/checkpoints/last.ckpt` (SigmaDock,
im separaten `sigmadock`-ARC-Baum). Sampling auf dem vollen 209-Komplexe-
PoseBusters-Set mit `sample_posebusters_full.sh` (Jobs 8489346/8489347,
je ~1 Min), Download der Vorhersagen per `tar`+`scp` (lokales Windows-
`scp.exe` unterstützt keine Remote-Wildcards mehr — Workaround: erst
`tar czf` auf ARC, dann einzelne Archivdatei runterladen). RDKit-
Bindungslängen-Messung (`bond_length_check_posebusters_full.py`,
lokal ausgeführt) lokal unter
`SigmaFlow_Variants/posebusters_full_comparison_24h/`.

**Ergebnis (5576 Bindungen über alle 209 Komplexe, 0 Parse-Fehler, 0
Bindungszahl-Mismatches — direkt vergleichbar mit der 12h-Tabelle oben):**

| Lauf | n Übergangsbindungen | % aller Bindungen | mean | median | max |
|---|---|---|---|---|---|
| SigmaFlow, 12h | 728 | 13.1% | 2.68 Å | 2.23 Å | 15.73 Å |
| **SigmaFlow, 24h** | 733 | 13.1% | 2.41 Å | 1.93 Å | 14.22 Å |
| SigmaDock, 12h | 830 | 14.9% | 1.25 Å | 1.22 Å | 18.08 Å |
| **SigmaDock, 24h** | 813 | 14.6% | 1.40 Å | 1.40 Å | 5.33 Å |

**Einordnung — der 12h-Trade-off bleibt bei 24h qualitativ unverändert
bestehen:** SigmaFlow weiterhin weniger betroffene Bindungen (13.1% vs.
14.6%), SigmaDock weiterhin die kleinere typische Abweichung, wenn eine
Bindung betroffen ist (mean/median ≈1.4 Å vs. ≈2.1–2.4 Å). Kein
Seitenwechsel zwischen 12h und 24h — die ursprüngliche Frage des Users
("bleibt der Trade-off bei 24h bestehen?") ist damit mit Ja beantwortet.

Zwei bemerkenswerte Verschiebungen mit mehr Training:
- SigmaDocks schlimmster Ausreißer schrumpft deutlich (18.08 → 5.33 Å) —
  grobe Einzelfehler werden mit mehr Training seltener/kleiner.
- **`7T1D_E7K` (26/39 Bindungen) und `6TW5_9M2` (12/34 Bindungen) sind bei
  SigmaDock bei 24h exakt identisch zu den 12h-Full-Set-Zahlen** — diese
  zwei Komplexe sind jetzt über mindestens zwei unabhängige
  Trainingsstände (12h und 24h, jeweils frisch bzw. fortgesetzt trainiert)
  identisch schlecht, was eher für ein strukturelles/architekturelles
  Problem bei genau diesen Komplexen spricht als für Untertraining.

### ✅✅✅ Erweiterung (2026-08-08): volle PoseBusters-Chemie-Checks (209 Komplexe, 12h+24h) — SigmaDock zieht mit mehr Training klar vorne weg

Die Bindungslängen-Metrik oben misst nur, WIE SCHLIMM eine bereits als
"Übergangsbindung" markierte Bindung abweicht — nicht, welcher ANTEIL der
Komplexe insgesamt als chemisch plausibel durchgeht. Auf User-Nachfrage
("ist SigmaFlow insgesamt schlechter, sind die Moleküle realistisch?") den
vollen PoseBusters-Check-Satz nachgezogen (lokal, `posebusters==0.6.5`
bereits installiert, `mol_pred`/`mol_true` bereits heruntergeladen für
12h+24h — kein neuer ARC-Trip nötig). **Einschränkung:** nur die 20
liganden-intrinsischen Checks (alles außer den 8 Protein-/Kofaktor-/
Wasser-Kontext-Checks) — dafür fehlen die 209 Rezeptor-PDBs lokal. Custom
Config `redock_noprotein.yml` (Scratchpad: `redock.yml` minus die 8
`mol_cond`-abhängigen Module), `mol_cond=None`. 6h hat kein volles Set
(nie gesampelt), daher hier nicht vertreten.

**Pass-Raten (209 Komplexe; nur Checks gezeigt, die sich zwischen
Methode/Stufe unterscheiden — restliche ~9 Checks liegen bei allen vier
Läufen bei 100%):**

| Check | SigmaFlow 12h | SigmaFlow 24h | SigmaDock 12h | SigmaDock 24h |
|---|---|---|---|---|
| `double_bond_stereochemistry` | 0.962 | 0.981 | 0.976 | 0.990 |
| `tetrahedral_chirality` | 0.756 | 0.785 | 0.856 | **0.914** |
| `bond_lengths` (strenges DG-Kriterium) | 0.120 | 0.124 | 0.148 | **0.258** |
| `bond_angles` | 0.062 | 0.081 | 0.043 | **0.187** |
| `internal_steric_clash` | **0.129** | 0.144 | 0.062 | **0.268** |
| `internal_energy` | 0.176 | 0.197 | 0.081 | **0.372** |
| `rmsd_≤_2å` | 0.000 | 0.014 | 0.000 | 0.043 |

**Einordnung — deutliche Revision der bisherigen "Trade-off, keine Methode
dominiert"-Lesart:** bei 12h war's noch gemischt (SigmaFlow sogar leicht
vorne bei `internal_steric_clash`/`bond_angles`). Bei 24h dreht sich das
fast vollständig: SigmaDock verbessert sich auf JEDEM dieser Checks
deutlich stärker als SigmaFlow zwischen 12h und 24h (z.B. `internal_energy`
0.081→0.372 vs. SigmaFlows 0.176→0.197) und liegt bei 24h überall vorne.
SigmaFlow stagniert zwischen 12h und 24h auf fast jedem Check nahezu.
**Absolute Werte bleiben niedrig bei beiden** (RMSD≤2Å <5%) — schwerer
Blind-Generalisierungstest nach nur 12-24h Training, keine
Produktionspipeline.

**Mögliche Erklärung, EXPLIZIT UNGETESTETE Hypothese (kein Ablationstest
gefahren, nur Code-Analyse):** SigmaDocks Loss hat vier Terme
(`T_score,R_score,T0,R0`, s. `tab:sigmadock-vs-sigmaflow-precise` in
`Texte/theory.tex`) — `T0`/`R0` sind DIREKTE Daten-Raum-Terme (vergleichen
die aus dem Score implizierte saubere Struktur direkt gegen die wahre
Struktur), zusätzlich zu den Score-Matching-Termen. SigmaFlow hat nur zwei
Terme (`loss_trans,loss_R`), beide im Geschwindigkeits-Raum, kein
Daten-Raum-Term. Fast jeder Check, bei dem SigmaDock stärker zulegt, ist
eine Eigenschaft der finalen 3D-Koordinaten — genau das, was `T0`/`R0`
direkt bestrafen und SigmaFlow nur indirekt (über ODE-Integrations-
Korrektheit) erreicht. **Wichtig:** dies ist eine ANDERE Hypothese als die
in Test 3 bereits GETESTETE UND VERWORFENE Zeitgewichtungs-Hypothese
(`λ(s)`-Analogon) — die schließt die Lücke nachweislich nicht; diese
Loss-STRUKTUR-Hypothese (fehlender Daten-Raum-Term) wurde noch nicht
getestet.

### ✅ Präzisierung (2026-08-08): was `T0`/`R0` genau vergleichen, und warum die Hypothese sich auf ROTATION verengt

Auf Nachfrage direkt im Code nachgesehen (nicht aus dem Gedächtnis
beantwortet), beide Punkte bestätigt:

**Was `T0`/`R0` vergleichen:** die WAHRE saubere Struktur (`T_0`/`R_0`,
Ground Truth aus dem Trainingskomplex) gegen die vom Netzwerk IMPLIZIERTE
Schätzung dieser Struktur (`T0_hat`/`R0_hat`), abgeleitet aus der
aktuellen Score-Vorhersage bei Rauschstärke `t`. **Kein Mehrschritt-
Integrieren** wie beim echten Sampling (25 Schritte in `sampling.py`):
- Translation: `T0_hat = calc_trans_0(pred_T_score, T_t, t)` — eine
  geschlossene algebraische Formel (Tweedie: `(x_t + σ_t²·score)/α_t`,
  `r3_diffuser - UMBAUEN.py:44-50`), kein Integrationsschritt.
- Rotation: `R0_hat = so3_diffuser.reverse(R_t, pred_R_score, t, dt=t,
  noise_scale=0)` — EIN einziger deterministischer Sprung direkt von `t`
  auf `0` (`dt=t` deckt das ganze Restintervall ab), dieselbe Schrittformel
  wie beim echten Sampling, aber nur einmal statt 25× angewendet
  (`r3_diffuser - UMBAUEN.py:97-119`).

**Verifizierte Algebra — Translation ist exakt redundant:** `calc_trans_0`
ist affin-linear im Score, also `T0_hat − T0 = (σ_t²/α_t)·(pred_score −
true_score)` EXAKT. Setzt man das mit den tatsächlichen Code-λ-Gewichten
(`T_score_scaling=1/σ_t`, `input_scaling=α_t`, `denoiser - UMBAUEN.py:
1231-1252`) ein, kürzt sich `λ^{p,0}_t·‖T0_hat−T0‖²` EXAKT zu
`λ^p_t·‖pred_score−true_score‖²` — Punkt für Punkt identisch zum
`T_score`-Loss, keine neue Information, nur verdoppelter Gradient.

**Das bestätigt (unabhängig neu hergeleitet) einen bereits früher in
diesem Projekt getroffenen Befund:** PAUSE-PUNKT-Bereich "Variante (b)
(x̂1-Reparametrisierung) verworfen" (Zeile ~202 oben) zeigt für SigmaFlows
Translation exakt dieselbe Beziehung, `‖x̂1_pred−x1_true‖² = (1−t)²·
‖v_pred−v_true‖²` — algebraisch redundant zum bestehenden `loss_trans`,
deshalb damals NICHT implementiert. Zwei unabhängige Herleitungen
(SigmaDock-Score-Raum jetzt, SigmaFlow-Geschwindigkeits-Raum damals)
kommen auf dieselbe Struktur — stützt die Rechnung zusätzlich.

**Rotation ist NICHT exakt redundant:** `R0_hat`/`R1_hat` entstehen durch
eine Exponentialabbildung + nichtkommutative Gruppenverknüpfung auf
SO(3), keine lineare Umrechnung. Für SigmaFlows Analogon (`R1_hat = R_t @
exp(pred_u_t_R·(1−t))` vs. wahres `R_1 = R_t @ exp(u_t_R·(1−t))`) liefert
die Baker-Campbell-Hausdorff-Formel `log(R1_hat^T R_1) = (u_t_R −
pred_u_t_R)(1−t) − (1−t)²/2·[pred_u_t_R, u_t_R] + O(\text{höhere
Ordnung})` — ein echter, nichtlinearer Korrekturterm (der Kommutator),
der nur verschwindet, wenn Vorhersage und wahres Feld um dieselbe Achse
rotieren. **Anders als bei der Translation wurde das für Rotation noch
nie geprüft** (weder in dieser noch in einer früheren Session) — eine
echte Lücke, kein bereits verworfener Pfad.

**Konsequenz für den geplanten Test:** nur ein rotations-seitiger
Daten-Raum-Term wird implementiert (nicht auch translations-seitig, das
wäre nachweislich nutzlos und würde nur Variante (b) über einen Umweg
reproduzieren).

### ⏳ Neues Experiment läuft (2026-08-08): `SigmaFlow_Variants/b_rotation_data_space_loss/`, Job 8490675

Neuer `R_1`-Daten-Raum-Term für die Rotation implementiert, mirror von
SigmaDocks `R0`-Konstruktion, aber mit SigmaFlows eigenen Bausteinen
(kein SDE-Reverse-Schritt, sondern ein einzelner Euler-Sprung mit
`so3_utils.exp`/`log`, exakt wie `so3_flow_matcher.py::euler_step`, nur
mit `dt=1−t` statt kleiner Schrittweite, da SigmaFlows Daten-Ende bei
`t=1` liegt, nicht `t=0` wie bei SigmaDock):

```python
R_1_hat = R_t @ so3_utils.exp(pred_u_t_R * (1 - t_batch)[:, None, None])
log_rel = so3_utils.log(R_1_hat.transpose(-1, -2) @ R_1)
batch_rot_data_space_loss = (log_rel * log_rel).sum(dim=(-1, -2))
```

Direkt in `loss_R` aufaddiert (nicht als eigener Dict-Key) — dadurch
braucht `trainer.py` KEINE Änderung, `rot_score_weight` wirkt automatisch
auf die Summe aus Geschwindigkeits- und Daten-Raum-Term, genau wie bei
SigmaDocks `T_score+T0`/`R_score+R0`-Gruppierung unter einem gemeinsamen
Gewicht. Kleinste nötige Änderung: zusätzlich `R_t` (bereits intern
berechnet, bisher nicht nach außen gereicht) in den `out`-Dict von
`sigma_flow_generator.py`s Forward-Pass aufgenommen.

**Von Claude implementiert, nicht vom User** (Abweichung vom sonst
gültigen Standard für dieses Projekt — siehe eigene Feedback-Memory
"SigmaFlow code ownership": User schreibt SigmaFlow-Code standardmäßig
selbst, hier auf explizite Nachfrage hin diesmal beibehalten, aber mit
vollständigem Code-Walkthrough).

**Verifiziert vor dem ARC-Lauf:** lokaler CPU-Smoke-Test auf den 10
Dummy-Komplexen (`--max_steps 5 --accelerator cpu`), Exit-Code 0, endliche
Losses (`loss_train/loss_R=12.01`, `loss_trans=4.31`), kein NaN/Crash.
`sample_posebusters_full.sh` im selben Ordner ebenfalls mit
`PYTHONPATH`-Override versehen (sonst würde die spätere Auswertung
stillschweigend den unveränderten Code laden, wie beim
`a_time_weighting`-Präzedenzfall).

**Zwei Nachfragen präzisiert, nachdem der Job schon lief:**
1. Die `T0`-Redundanz gilt für SigmaDock SELBST, nicht nur in unserem
   Vergleich — die Algebra benutzt ausschließlich SigmaDocks eigene
   Formeln. Warum die Original-Autoren den Term trotzdem eingebaut haben,
   ist unklar (kein erklärender Kommentar gefunden, `papers/` noch nicht
   gezielt durchsucht) — Hypothese: leicht zu übersehende Kürzung, ODER
   bewusst als impliziter ~2×-Gewichtsfaktor auf die Translation
   einkalkuliert.
2. Unser `R_1_hat` ist NICHT dieselbe Formel wie SigmaDocks `R0_hat`,
   trotz gleicher Grundidee (ein-Schritt-Sprung via Exponentialabbildung
   zum Datenende). Direkt in `so3_diffuser - UMBAUEN.py:325-355` +
   `so3_utils.py::expmap` nachgeprüft: SigmaDock benutzt `R_t @
   exp(R_t^T @ perturb)` (Score als Welt-Rahmen-Tangentialvektor,
   transportiert via `R_t^T`), unsere Ergänzung benutzt `R_t @
   exp(pred_u_t_R·dt)` direkt (SigmaFlows eigene, bereits etablierte
   rechts-trivialisierte/Körper-Rahmen-Konvention, exakt wie
   `so3_flow_matcher.py::euler_step`). Bereits vorher in `Texte/
   theory.tex` als echte architektonische Differenz zwischen den beiden
   Codebasen dokumentiert — kein neuer Befund, aber jetzt konkret an
   dieser Stelle bestätigt.

Hochgeladen nach `/data/stat-cadd/shug8458/SigmaFlow_Variants_JulianMueller/
b_rotation_data_space_loss/` (per `scp -r`), submittet 2026-08-08 als
**Job 8490675** (`medium`, `--time=12:00:00`, identische Hyperparameter
`trans_score_weight=2.0 rot_score_weight=0.5` wie die bisherigen Läufe).
`sigmadock loaded from`-Pfad-Check noch zu bestätigen (User prüft, sobald
der Job angelaufen ist).

**Plan nach Abschluss:** dieselbe Prozedur wie bei den anderen Stufen —
Fairness-Check, volles PoseBusters-Set sampeln
(`sample_posebusters_full.sh`, neuer `CKPT_DIR`), Bindungslängen +
die 20 liganden-intrinsischen Checks auswerten, verglichen gegen die
bereits vorhandenen SigmaFlow-12h-Zahlen oben (kein neuer Kontroll-Lauf
nötig, Baseline existiert bereits) und gegen SigmaDock-12h. Besonders
interessant: ob sich speziell die rotations-lastigen Checks
(`tetrahedral_chirality`, `bond_angles`, `internal_steric_clash`) bewegen
— das wäre das erwartete Signal, falls die Hypothese stimmt. Kein
Ablationstest im strengen Sinn (nur ein Lauf, kein Wiederholungs-Seed) —
ein erstes Signal, kein endgültiger Beweis.

### Nächste Schritte

1. **Freigabe für die Stufe danach einholen:** laut gestuftem Plan (6h→
   12h→24h, dann erst 3-/7-Tage) war 24h die letzte Stufe vor den
   Mehrtages-Läufen — vor einem 3-Tage-Lauf beim User nachfragen,
   ob das gewünscht ist, statt automatisch weiterzumachen. Noch nicht
   eingeholt (Stand 2026-08-08).
2. ✅ **ERLEDIGT (2026-08-09):** Job `8490675` ausgewertet — siehe
   "Job 8490675 ... ausgewertet" oben. Ergebnis: kein Effekt, Hypothese
   widerlegt (zweiter negativer Befund nach Test 3).
2b. **Neu offen:** wie geht es nach zwei negativen Ergebnissen
   (Zeitgewichtung UND Rotations-Daten-Raum-Term) weiter? Die
   zurückgestellte Idee "gezielter Anker-Atom-Distanz-Loss" (s.o., PAUSE-
   PUNKT #14 weiter oben, "💡 Idee (zurückgestellt)") wurde explizit auf
   Eis gelegt, bis die 6h/12h/24h-Läufe ausgewertet sind — das ist jetzt der
   Fall. Mit dem User klären, ob das als nächstes verfolgt werden soll,
   oder ob zunächst die 3-Tage-Freigabe (Punkt 1) Priorität hat.
3. Protein-Kontext-Checks (Volume-Overlap/Mindestabstand zu Protein,
   Kofaktoren, Wasser — 8 der 28 Checks) noch nicht ausgewertet, dafür
   fehlen die 209 Rezeptor-PDBs lokal. Bei Bedarf von ARC nachladen
   (`/data/stat-cadd/shug8458/data/posebusters/` bzw.
   `posebusters_paper/posebusters_benchmark_set/`, s.o.).
4. Die Sampling-Geschwindigkeits-Frage (SigmaFlow ~4x langsamer beim
   reinen Vorhersageschritt) bleibt ungeklärt — algorithmisch keine
   Erklärung gefunden, könnte an GPU-Konkurrenz oder Implementierungs-
   details liegen. Nicht dringend, betrifft nur Sampling-Laufzeit.
5. Die neue `Failed to parse pocket`-Asymmetrie (SigmaFlow ~30 vs.
   SigmaDock 0, s.o.) ist nicht weiter untersucht — falls das bei einer
   künftigen Stufe wieder auftaucht, lohnt sich ein Blick in die
   Datenlade-/Filter-Codepfade beider Bäume, ob das ein echter
   Unterschied oder nur Sampling-Zufall ist.
5. Auffällige Einzelkomplexe für eine mögliche visuelle Inspektion
   (z.B. via PyMOL), jetzt mit noch mehr Gewicht als vorher (identisch
   über 12h UND 24h): SigmaDocks `7T1D_E7K` (26/39) und `6TW5_9M2`
   (12/34).
6. Kosmetischer `val_loss=0.0000`-Dateinamen-Bug (s.o.) könnte bei
   Gelegenheit gefixt werden (`filename="checkpoint-{step:02d}-{val_loss:.4f}"`
   → passenden Metriknamen verwenden), nicht dringend.
7. Lokale Python-Umgebung (dieser Windows-Rechner) hat jetzt den vollen
   Sampling-/Trainings-Software-Stack installiert (s.o.) — nicht committet,
   aber für künftige lokale Tests wiederverwendbar.
8. Offen, nicht dringend: `set -euo pipefail` auch in den ÄLTEREN
   `slurm/*.sh`-Skripten nachrüsten (bisher nur in den drei neuen dieser
   Session).

### ✅✅✅ Job 8490675 (`b_rotation_data_space_loss`, 12h) ausgewertet (2026-08-09): Hypothese widerlegt — zweites negatives Ergebnis

**Fairness-Check bestanden** (gleiches Verfahren wie bei allen bisherigen
Stufen): `sacct` zeigt `TIMEOUT`, `ExitCode=0:0`, volle `12:00:18` gelaufen.
Log frei von NaN/Traceback (einzige Erwähnung war der Callback-Name selbst,
"Full NaN Check Callback enabled"). `sigmadock loaded from` zeigt korrekt
auf `b_rotation_data_space_loss/src/...`. Checkpoint
`experiments/sigmadock/0-08-08_13-06-22/checkpoints/last.ckpt` (00:45 Uhr
gespeichert) verifiziert als der tatsächlich von diesem Job erzeugte
Checkpoint — direkt per `sacct --format=Start,End` bestätigt: Job startete
13:02:50, Checkpoint-Ordner-Zeitstempel `13:06:22` (plausibler
Start-Overhead), Checkpoint-Datei 00:45 liegt sauber innerhalb des
Job-Zeitfensters (endet 01:03:08), ~18 Min vor Kill — passt. Der zweite,
ähnlich benannte Ordner (`0-08-08_12-44-48`, Checkpoint nur 15 Min nach
Ordner-Erstellung gespeichert) ist ein Überbleibsel eines früheren
Smoke-Tests, nicht dieses Laufs.

**Zusätzlich verifiziert, auf User-Nachfrage vor dem Festhalten des
Ergebnisses** (drei Fragen: richtiges Modell trainiert? Loss-Änderung
tatsächlich aktiv? richtiges Modell gesampelt?): der ARC-Code-Stand von
`sigma_flow_generator.py` enthält den `R_1`-Daten-Raum-Term Zeile für Zeile
identisch zur lokalen Kopie (per `grep -n "ROTATION DATA-SPACE LOSS" -A3`
bestätigt). Zusätzliches indirektes Argument: `compute_losses()` liest
`out["R_t"]`/`out["R_1"]` UNBEDINGT (kein Feature-Flag) — wäre der
Daten-Raum-Term nicht aktiv gewesen, hätte bereits der allererste
Trainingsschritt mit `KeyError` abgebrochen (exakt der Fehler, der beim
Sampling auftrat, s.u.), nicht 12h sauber durchlaufen.

**Sampling-Zwischenfall, gefunden + gefixt:** erster Sampling-Versuch (Job
8504542) brach nach 5:50 Min mit `KeyError: 'R_t'` in `compute_losses()` ab.
**Root Cause:** `compute_losses()` hat zwei Aufrufer — den Forward-Pass
(Training, liefert `R_t`/`R_1` seit der Implementierung dieser Variante) und
die Verlust-LOGGING-Stelle in `sampling.py::sampler()`/`sample_notebook()`
(baut ihr eigenes, von Hand zusammengestelltes `out`-Dict für die
Trainings-Ziel-vs-Vorhersage-Diagnose pro ODE-Schritt) — letztere wurde bei
der Implementierung der neuen Loss-Komponente übersehen. Betraf NUR die
Diagnose-Loss-Anzeige während des Samplings, NICHT die tatsächliche
ODE-Trajektorie/generierte Struktur (die nutzt `grad_T_p`/`grad_R_p` direkt,
unverändert). **Fix:** `R_t`/`R_1` (bereits als lokale Variablen im
Sampling-Loop vorhanden) in beide betroffenen Dict-Literale ergänzt
(`sampling.py`, zwei Stellen: `sample_notebook()` Zeile ~187,
`sampler()` Zeile ~425) — von Claude implementiert (gleiche
Ownership-Ausnahme wie beim ursprünglichen `R_1`-Term, s.o.). Datei erneut
per `scp` hochgeladen, Job 8504658 (neuer Sampling-Versuch, gleicher
Checkpoint) lief sauber durch (`COMPLETED`, `ExitCode=0:0`, 6:28 Min, 209
`.sdf`s erzeugt, `sigmadock loaded from` + `ckpt:`-Pfad + "Successfully
loaded EMA model." im Log bestätigt).

**Auswertung** (lokal, RDKit-Bindungslängen-Check + volle 20
liganden-intrinsische PoseBusters-Checks — beide Auswertungsskripte
diesmal NICHT im ephemeren Scratchpad, sondern dauerhaft gespeichert unter
`SigmaFlow_Variants/posebusters_full_comparison/` (`bond_length_check.py`,
`run_posebusters_ligandonly.py`, `redock_noprotein.yml`) — die
Vorgänger-Skripte aus früheren Sessions ("Scratchpad") überlebten die
Session nicht und mussten diesmal rekonstruiert werden; **Methodik anhand
der bekannten 12h-Baseline-Zahlen verifiziert** (beide Skripte reproduzieren
die bereits dokumentierten SigmaFlow-12h/SigmaDock-12h-Werte exakt, u.a. ein
Definitions-Detail korrigiert: "mean/median/max Übergangsbindung" bezieht
sich auf die tatsächliche PRÄDIZIERTE Bindungslänge der auffälligen
Bindungen, nicht auf ihre Abweichung von der wahren Länge — nach Korrektur
exakte Übereinstimmung mit den STATUS.md-Werten)):

Bindungslängen (5576 Bindungen, `TOL=0.02 Å`):

| Lauf | n Übergangsbindungen | % | mean | median | max |
|---|---|---|---|---|---|
| SigmaFlow, 12h (Baseline) | 728 | 13.1% | 2.68 Å | 2.23 Å | 15.73 Å |
| **SigmaFlow, rotdata, 12h (NEU)** | 733 | 13.1% | 2.71 Å | 2.24 Å | 15.25 Å |
| SigmaDock, 12h | 830 | 14.9% | 1.25 Å | 1.22 Å | 18.08 Å |

20 liganden-intrinsische PoseBusters-Checks (209 Komplexe, nur abweichende
Checks gezeigt):

| Check | SigmaFlow 12h | **rotdata 12h (NEU)** | SigmaFlow 24h | SigmaDock 12h | SigmaDock 24h |
|---|---|---|---|---|---|
| double_bond_stereochemistry | 0.962 | 0.957 | 0.981 | 0.976 | 0.990 |
| tetrahedral_chirality | 0.756 | 0.751 | 0.785 | 0.856 | 0.914 |
| bond_lengths (DG) | 0.120 | 0.124 | 0.124 | 0.148 | 0.258 |
| bond_angles | 0.062 | 0.057 | 0.081 | 0.043 | 0.187 |
| internal_steric_clash | 0.129 | 0.129 | 0.144 | 0.062 | 0.268 |
| internal_energy | 0.176 | 0.167 | 0.197 | 0.081 | 0.372 |
| rmsd_≤_2å | 0.000 | 0.000 | 0.014 | 0.000 | 0.043 |

**Ergebnis: kein Effekt, Hypothese widerlegt.** Alle Werte der neuen
Variante liegen innerhalb der Rauschbreite der unveränderten
SigmaFlow-12h-Baseline — auch die drei rotations-lastigen Checks
(`tetrahedral_chirality`, `bond_angles`, `internal_steric_clash`), die laut
der Hypothese (fehlender Daten-Raum-Term erklärt SigmaDocks Vorsprung bei
der chemischen Plausibilität) am ehesten hätten reagieren sollen. Keine
Bewegung in Richtung SigmaDock. Das ist der **zweite unabhängige negative
Befund** zur "SigmaFlow fehlt ein Daten-Raum-Term"-Familie von Hypothesen
(nach Test 3, Zeitgewichtung Variante a) — beide plausiblen, aus der
SigmaDock-Loss-Struktur abgeleiteten Korrekturen schließen die
24h-PoseBusters-Lücke nicht. Die eigentliche Ursache für SigmaDocks
stärkere Verbesserung zwischen 12h und 24h bleibt ungeklärt.

### ⏳ Neues Experiment implementiert + lokal verifiziert (2026-08-09): `SigmaFlow_Variants/c_anchor_atom_distance_loss/`

Auf Nachfrage ("wie schaff ich es SigmaFlow auf SigmaDock-Niveau zu bringen?")
gemeinsam die verbleibenden Optionen durchgesprochen. Anders als die beiden
vorherigen Varianten (a: Zeitgewichtung, b: Rotations-Daten-Raum-Term — beide
laut BCH-Herleitung im Kern nur verschiedene Umgewichtungen des BESTEHENDEN
Geschwindigkeitsfehlers, beide wirkungslos) wurde diesmal die
zurückgestellte **Anker-Atom-Distanz-Idee** umgesetzt: ein Loss-Term, der
direkt den Abstand zwischen den beiden Anker-Atomen einer geschnittenen
Torsionsbindung bestraft — genau die Größe, die die externe
RDKit-"Übergangsbindung"-Diagnostik misst. Kombiniert zwei UNABHÄNGIGE
Fragment-Vorhersagen zu einer neuen Größe, ist damit nicht auf dieselbe Art
redundant zum bestehenden Loss wie die ersten beiden Varianten.

**Offene Frage aus PAUSE-PUNKT #14 (oben, "💡 Idee") geklärt:** die
Anker-Atompaare pro geschnittener Bindung ÜBERLEBEN bis zum Trainingszeitpunkt
— nicht als rohe Indexliste, sondern als reguläre Graph-Kante. `chem/
processing.py::get_global_ligand_graph` fügt bereits eine Kante zwischen den
beiden Anker-Atomen jeder geschnittenen Bindung hinzu, Edge-Typ
`"ligand_torsional_bond"` (`oracle.py`, Kommentar: "Literally the torsional
bond separated by distance we know"). Kein zusätzliches Durchreichen nötig —
`batch.edge_index[:, batch.edge_entity == HPARAMS.get_edge_idx("ligand_torsional_bond")]`
liefert direkt die richtigen, bereits batch-korrekten Atom-Indexpaare.

**Implementierung** (`src/sigmadock/diff/sigma_flow_generator.py`, von Claude
umgesetzt, gleiche Ausnahme wie bei Variante b):
- neue Methode `_compute_anchor_distance_loss()`: pro Anker-Bindung
  `(pred_dist − true_dist)²`, wobei `true_dist = ‖pos_0[u]−pos_0[v]‖` (`pos_0`
  ist trotz des Namens die wahre Struktur) und `pred_dist =
  ‖pos_1_hat[u]−pos_1_hat[v]‖`. Ergebnis `[B x F]`-geshaped (Bindungsfehler
  wird in den Slot EINES der beiden beteiligten Fragmente addiert — da
  `scaled_fragmented_loss()` ohnehin über alle Fragmente eines Komplexes
  summiert, ist es egal welches der beiden).
- `forward()`: rekonstruiert `pos_1_hat` (Einzelschritt-Vorhersage der
  finalen Struktur) durch Wiederverwendung der BESTEHENDEN
  `_apply_transformations()`-Methode — kein neuer Geometrie-Code, nur mit
  `R_1_hat`/`trans_1_hat` (CondOT/Exp-Map-Einzelschritt aus der
  Netzwerk-Vorhersage) statt `R_t`/`trans_t` aufgerufen.
- `compute_losses()`: `anchor_dist_loss` wird via `out.get("anchor_dist_loss",
  0.0)` zu `loss_trans` addiert — bewusst OPTIONAL, damit `sampling.py`s von
  Hand gebaute Logging-Dicts (die dieses Feld nicht haben) nicht denselben
  `KeyError`-Fehler wiederholen, der Variante b beim ersten Sampling-Versuch
  getroffen hat.

**Lokal verifiziert (CPU, 10 Dummy-Komplexe, vor jedem ARC-Upload):**
1. `--max_steps 5 --accelerator cpu`: Exit-Code 0, endliche Losses
   (`loss_train/loss_trans=9.71` [inkl. neuem Term], `loss_R=11.89`,
   `total=15.65`), kein NaN/Crash, 2 Epochen sauber durchlaufen.
2. **Zusätzlich, auf eigene Initiative vor dem Vertrauen in Punkt 1:** ein
   temporärer Debug-Print direkt in `forward()` bestätigt, dass
   `anchor_dist_loss` über Trainings- UND Validierungsschritte hinweg
   tatsächlich überwiegend NICHT null ist (typisch 4-9 von 8-13
   Fragment-Slots pro Batch), endliche Werte in plausibler Größenordnung
   (meist < 10, vereinzelt bis ~35 im schlecht trainierten Smoke-Test-Netz —
   passt zur Größenordnung der RDKit-gemessenen Übergangsbindungsfehler aus
   Test 1). Ausgeschlossen: der Edge-Filter matcht nichts und der Term ist
   still immer 0 (hätte den KeyError-Vorfall von Variante b wiederholt, aber
   in einer Weise, die NICHT durch einen Crash aufgefallen wäre). Print
   danach wieder entfernt.

**Vorbereitet:** `slurm/train_pdbbind_general_12h.sh` (identische
Hyperparameter wie a/b, für direkte Vergleichbarkeit) und
`slurm/sample_posebusters_full.sh` (mit `PYTHONPATH`-Override diesmal von
Anfang an ergänzt, statt wie bei Variante b erst nach einem Fehlschlag
nachgerüstet).

### ✅✅✅ Variante C ausgewertet (2026-08-09): DRITTES negatives Ergebnis — und das aussagekräftigste

**Trainingslauf Job 8505487** (`medium`, 12h): Fairness-Check vollständig
bestanden — `TIMEOUT`/`ExitCode=0:0`/`12:00:27` (kein stiller Frühabbruch),
`sigmadock loaded from` korrekt auf `c_anchor_atom_distance_loss/src/...`,
kein NaN/Traceback. Checkpoint `experiments/sigmadock/0-08-09_02-38-01/
checkpoints/last.ckpt` (14:25 gespeichert, passt exakt ans Ende des
12h-Fensters). Zwei weitere, ähnlich benannte Ordner (`0-08-09_02-11-06`,
`0-08-09_02-24-08`) sind Überbleibsel der lokalen Smoke-Tests (Checkpoints
nur 7-19 Min nach Ordner-Erstellung) — nicht dieses Laufs.

**Sampling Job 8512134**: `COMPLETED`/`0:0`/11:43, `ckpt:`-Zeile zeigt den
richtigen Checkpoint, `sigmadock loaded from` den richtigen Code-Pfad,
"Successfully loaded EMA model.", 209 `.sdf`s erzeugt.

**Zusätzliche Absicherung, dass wir nicht versehentlich die Baseline
nochmal ausgewertet haben** (eigene Initiative, nachdem die Zahlen
auffällig nah an der Baseline lagen): direkter Koordinatenvergleich der
vorhergesagten Posen gegen die SigmaFlow-Baseline-Posen — max.
Atom-Abweichung 0.14-0.72 Å über 8 Stichprobenkomplexe. Es sind also
nachweislich VERSCHIEDENE Modelle, das Null-Ergebnis ist kein
Verwechslungs-Artefakt.

**Ergebnis Bindungslängen (5576 Bindungen):**

| Lauf | n Übergangsbindungen | % | mean | median | max |
|---|---|---|---|---|---|
| SigmaFlow, 12h (Baseline) | 728 | 13.1% | 2.68 Å | 2.23 Å | 15.73 Å |
| SigmaFlow, rotdata (Var. b) | 733 | 13.1% | 2.71 Å | 2.24 Å | 15.25 Å |
| **SigmaFlow, anchordist (Var. c)** | **730** | **13.1%** | **2.69 Å** | **2.26 Å** | **15.68 Å** |
| SigmaDock, 12h | 830 | 14.9% | 1.25 Å | 1.22 Å | 18.08 Å |

**Ergebnis PoseBusters (nur abweichende Checks):**

| Check | SF 12h | SF rotdata | **SF anchordist** | SF 24h | SD 12h | SD 24h |
|---|---|---|---|---|---|---|
| double_bond_stereochemistry | 0.962 | 0.957 | **0.957** | 0.981 | 0.976 | 0.990 |
| tetrahedral_chirality | 0.756 | 0.751 | **0.751** | 0.785 | 0.856 | 0.914 |
| bond_lengths (DG) | 0.120 | 0.124 | **0.124** | 0.124 | 0.148 | 0.258 |
| bond_angles | 0.062 | 0.057 | **0.067** | 0.081 | 0.043 | 0.187 |
| internal_steric_clash | 0.129 | 0.129 | **0.134** | 0.144 | 0.062 | 0.268 |
| internal_energy | 0.176 | 0.167 | **0.163** | 0.197 | 0.081 | 0.372 |
| rmsd_≤_2å | 0.000 | 0.000 | **0.000** | 0.014 | 0.000 | 0.043 |

**Kein Effekt — obwohl dieser Loss-Term GENAU die gemessene Größe direkt
bestraft.** Das ist qualitativ ein stärkerer Befund als die beiden
vorherigen Nullergebnisse: Varianten a und b konnte man mit "war im Kern
nur eine Umgewichtung des bestehenden Fehlers" wegerklären (BCH-Herleitung
oben). Variante c dagegen fügt eine echte, neue, kreuz-fragmentäre Größe
hinzu — und die vom Modell erzeugten Übergangsbindungen werden trotzdem
nicht kürzer.

### 📊 Nebenbefund mit echtem methodischem Wert: Lauf-zu-Lauf-Varianz jetzt quantifiziert

Wir haben jetzt DREI unabhängig trainierte SigmaFlow-12h-Läufe mit
identischen Hyperparametern (Baseline, Var. b, Var. c — die
Loss-Änderungen wirkten offensichtlich nicht). Ihre Bindungslängen-Zahlen:
**728 / 733 / 730** Übergangsbindungen, mean **2.68 / 2.71 / 2.69 Å**.

Die Lauf-zu-Lauf-Streuung dieser Metrik liegt also bei etwa **±3
Bindungen und ±0.03 Å**. Das ist wichtig, weil es die bisherigen
Nullergebnisse nachträglich absichert: die gemessenen "Nicht-Effekte"
(+5, +2 Bindungen) liegen genau in dieser Streuung, und SigmaDocks
Vorsprung (830 Bindungen, mean 1.25 Å) liegt um GRÖSSENORDNUNGEN
außerhalb davon. Die Metrik ist also empfindlich genug, um einen echten
Effekt zu sehen — sie sieht nur keinen.

### 💡 Neue Arbeitshypothese nach drei Nullergebnissen (ungetestet)

Die drei gescheiterten Loss-Eingriffe legen eine Umdeutung nahe, die
bisher nicht formuliert war: **die lange Übergangsbindung ist womöglich
gar kein eigenständiger Defekt, sondern nur ein Ablesewert der allgemeinen
Platzierungsgenauigkeit.** Jedes Fragment wird starr und unabhängig
platziert; die Bindungslänge an der Schnittstelle ergibt sich aus ZWEI
unabhängigen SE(3)-Vorhersagen. Wenn das Modell Fragmente generell mit
~2-3 Å Fehler platziert, MUSS die Übergangsbindung um diese
Größenordnung falsch sein — unabhängig davon, was der Loss verlangt. Ein
Loss-Term kann nichts erzwingen, was die Gesamtgenauigkeit des Modells
nicht hergibt.

Das würde erklären, warum alle drei Eingriffe folgenlos blieben, und es
verschiebt die Frage von "welcher Loss-Term fehlt SigmaFlow?" zu "warum
platziert SigmaFlows Netzwerk Fragmente generell ungenauer als
SigmaDocks?". **Konkret prüfbar** (noch nicht getan): die
Pro-Fragment-Platzierungsgenauigkeit (RMSD/Translations-/Rotationsfehler
pro Fragment) beider Methoden direkt messen und gegen den
Übergangsbindungs-Fehler auftragen. Falls die Korrelation stark ist, ist
die Hypothese gestützt und weitere Loss-Varianten sind Zeitverschwendung.

### 🚨🚨🚨 Hypothese GEPRÜFT (2026-08-09) — widerlegt, aber dabei ein weit wichtigerer Befund: BEIDE Modelle sind bei 12h faktisch untrainiert

Skript `SigmaFlow_Variants/posebusters_full_comparison/
placement_vs_bond_error.py` (dauerhaft gespeichert). Methode, ohne neuen
ARC-Lauf: da Fragmente STARR platziert werden, behält jede
Intra-Fragment-Bindung ihre wahre Länge exakt — daraus lässt sich die
Fragmentierung aus den Vorhersagen selbst rekonstruieren
(Zusammenhangskomponenten über längenerhaltende Bindungen). Pro Fragment
dann Kabsch-Ausrichtung Vorhersage↔Wahrheit → Schwerpunktfehler +
Rotationsfehler. **Methodik selbst-validiert:** Residual-RMSD nach starrer
Ausrichtung = 0.019-0.066 Å, bestätigt, dass die Fragment-Rekonstruktion
korrekt ist und die Platzierung wirklich starr erfolgt.

**Die Hypothese selbst ist WIDERLEGT:** die Korrelation zwischen
kombiniertem Fragment-Platzierungsfehler und Übergangsbindungs-Fehler ist
schwach (SigmaFlow r=0.18, SigmaDock r=0.05, anchordist r=0.16). Die lange
Übergangsbindung ist also NICHT einfach ein Ablesewert der
Platzierungsgenauigkeit.

**Der eigentliche Befund (unerwartet, viel gravierender):**

| Metrik (12h-Checkpoints, 209 Komplexe) | SigmaFlow | SigmaDock | Zufalls-Baseline |
|---|---|---|---|
| Pro-Fragment-Schwerpunktfehler (mean/median) | 5.19 / 4.57 Å | 12.92 / 6.39 Å | — |
| **Pro-Fragment-Rotationsfehler (mean)** | **122.2°** | **118.7°** | **126.5°** |
| Gesamt-Molekül-RMSD (mean/median) | 5.58 / 5.14 Å | 13.57 / 6.55 Å | — |
| Anteil RMSD < 2 Å | 0.000 | 0.000 | — |

**Die Fragment-Orientierungen beider Modelle sind praktisch zufällig.**
Für Haar-gleichverteilte Zufallsrotationen in SO(3) beträgt der mittlere
Rotationswinkel 126.5° (analytisch `(π²/2+2)/π`, empirisch über 20.000
Stichproben bestätigt: 126.6°). Gemessen: SigmaFlow 122.2°, SigmaDock
118.7° — beide nur marginal besser als reiner Zufall. Kein Modell hat bei
12h gelernt, Fragmente sinnvoll zu ORIENTIEREN.

**Frames verifiziert, bevor das geglaubt wurde:** Gesamt-Molekül-
Schwerpunktfehler SigmaFlow nur 2.13 Å (kein systematischer
Koordinatensystem-Versatz), und die berechneten RMSDs reproduzieren
unabhängig PoseBusters' eigenes `rmsd_≤_2å = 0.000` für beide Methoden.

**Was das für die gesamte bisherige Untersuchung bedeutet:** die
wochenlang verfolgten Übergangsbindungs-Unterschiede (2.68 Å vs. 1.25 Å
etc.) sind Unterschiede zwischen ZWEI FAKTISCH UNTRAINIERTEN Modellen.
Das erklärt auf einen Schlag, warum alle drei Loss-Varianten wirkungslos
blieben: ein Loss-Term kann die Feinjustierung einer Größe nicht
verbessern, solange das Modell die zugrundeliegende Aufgabe (Fragmente
überhaupt richtig orientieren) noch gar nicht gelöst hat. Die
Übergangsbindungs-Metrik ist zudem als Zielgröße irreführend — sie kann
für ein gleichmäßig schlechtes Modell "besser" aussehen.

**Compute-Einordnung (aus dem Paper, Appendix E.3, direkt nachgelesen):**
die publizierten SigmaDock-Ergebnisse (79.9% Top-1 RMSD<2Å) stammen aus
**4 Tagen auf 4 NVIDIA-A100 (DDP), Batch-Size 32, bis zu 256 Epochen, bis
zur Konvergenz** ≈ 384 GPU-Stunden. Unsere Läufe: **12h auf einer
einzelnen L40S, Batch-Size 8** ≈ 12 GPU-Stunden — etwa **1/32 der
Rechenzeit bei 1/4 der Batch-Size**. Dass beide Modelle bei 0% RMSD<2Å
liegen, ist damit vollständig erwartbar und kein Hinweis auf einen Defekt
in SigmaFlow.

### 🚨🚨🚨 KONFIGURATIONSFEHLER GEFUNDEN (2026-08-09): alle pdbbind-Läufe blieben im LR-WARMUP stecken

Beim Vorbereiten der Dummy-Langläufe entdeckt (nicht gesucht): `max_epochs`
steuert nicht nur die Laufzeit, sondern über `max_steps`
(`scripts/train.py:218-222`) auch den **Learning-Rate-Schedule**
(`trainer.py:137-157`: `LinearLR`-Warmup über `lr_warmup_frac=1/16` der
`max_steps`, danach Cosine-Annealing).

Ausgerechnet für unsere Läufe (`--max_epochs 1000`, 19.443 Komplexe,
Batch 8 → `max_steps` = 2.430.375, Warmup allein = 151.898 Steps):

| Lauf | erreichter Step | % des Schedules | LR am Ende | Phase |
|---|---|---|---|---|
| pdbbind 12h | 15.500 | 0.6% | **1.02e-5** (10% des Peaks) | **noch Warmup** |
| pdbbind 24h | 31.700 | 1.3% | **2.09e-5** (21% des Peaks) | **noch Warmup** |
| Dummy 3h (`max_epochs=700`) | 3.070 | 88% | annealing | Schedule fast komplett |

**Sämtliche pdbbind-Läufe (6h/12h/24h, SigmaFlow UND SigmaDock) haben die
Warmup-Phase nie verlassen** und trainierten durchgehend bei 1-2
Größenordnungen unter der vorgesehenen Peak-LR (`max_lr_start=1e-4`,
`config.py:140`); die Annealing-Phase wurde nie erreicht.

**Das ist ein Fehler in UNSERER Konfiguration, nicht in SigmaDock.** Das
Paper (Appendix E.3) nutzt 256 Epochen bei Batch 32 auf 4×A100 über 4 Tage
— damit ist der Warmup erreichbar. Unsere Kombination `max_epochs=1000`
(4× mehr als das Paper) MIT 1/32 der Rechenzeit und 1/4 der Batch-Size
macht den Schedule unerreichbar.

**Erklärungskraft:** passt zu allen bisherigen Beobachtungen — beide
Methoden gleichermaßen betroffen (gemeinsame `train.py`), Rotationen nahe
Zufall, drei Loss-Varianten wirkungslos (bei effektiv 1e-5 LR kann keine
Loss-Änderung etwas ausrichten), und die Dummy-Läufe (Schedule läuft
durch) schnitten mit 102° besser ab als die pdbbind-Läufe mit 122°,
obwohl sie weit weniger Daten sahen. **Noch nicht bewiesen** — ein
Korrekturlauf steht aus (s. Nächste Schritte).

**Konsequenz für alle künftigen Skripte:** `max_epochs` muss auf die
Epochenzahl gesetzt werden, die im gegebenen Zeitbudget TATSÄCHLICH
erreichbar ist, nicht "großzügig" hoch. Ein zu großer Wert verstümmelt den
LR-Schedule still. Für 12h auf pdbbind-general bei Batch 8 wären das ca.
`max_epochs=6-7` (statt 1000).

### ✅ Fünf 9h-Dummy-Overfit-Skripte vorbereitet (2026-08-09)

Auf User-Vorschlag ("die 10 Dummies sollte er ja mit mehr Zeit auswendig
lernen können") — Zweck ist die saubere Trennung von "zu wenig Zeit" vs.
"strukturell kaputt", die auf PoseBusters nicht möglich ist.

- `SigmaFlow_Development/slurm/train_dummy_overfit_9h.sh` (Baseline)
- `SigmaFlow_Variants/{a_time_weighting,b_rotation_data_space_loss,
  c_anchor_atom_distance_loss}/slurm/train_dummy_overfit_9h.sh`
- `SigmaFlow_Variants/sigmadock_control_scripts/
  train_dummy_overfit_9h_sigmadock.sh` (**SigmaDock-Kontrolle**, muss in den
  separaten ARC-SigmaDock-Klon hochgeladen werden, kein lokaler Klon
  vorhanden)

Alle mit `set -euo pipefail`, ohne das tote `--rot_vector_field_scaling`,
`bash -n`-syntaxgeprüft. **`max_epochs=1800`** bewusst auf die in 9h
erreichbare Epochenzahl kalibriert (3h → 614 Epochen ≈ 3.7 Epochen/Min),
damit der LR-Schedule diesmal durchläuft — im Skriptkommentar ausdrücklich
als "load-bearing, nicht "sicherheitshalber" erhöhen" markiert.

**Primäre Auswertungsmetrik: Rotationsfehler gegen die 126.5°-Zufallsgrenze**
(`placement_dummy.py`), NICHT Übergangsbindungslängen.

### ⚠️ Methodik-Falle gefunden und behoben: Singular/Plural der Referenz-Ligandendatei

Beim ersten Dummy-Auswertungsversuch ergaben sich RMSD-Werte, die
`RESULTS.md` widersprachen (12.53 Å statt der dokumentierten 4.31 Å).
Ursache: im Dummy-Ordner existieren BEIDE Dateien `<cid>_ligand.sdf` und
`<cid>_ligands.sdf`, und sie unterscheiden sich für manche Komplexe (z.B.
`1HWI_115`). `conf/experiments/dummy_train.yaml` nutzt
`sdf_regex: ".*_ligand\.sdf$"` — **Singular, verankert**, matcht die
Plural-Datei also NICHT. PoseBusters ist der umgekehrte Fall
(`.*ligands.*\.sdf$`, Plural). Nach Korrektur reproduziert die Messung
`RESULTS.md` **exakt** (SigmaFlow 4.31/3.67, SigmaDock 5.08/4.76) — das
validiert zugleich die gesamte Platzierungs-Analyse-Methodik unabhängig.

### ⏳ LR-Fix-Läufe gestartet (2026-08-09): Jobs 8512798 (SigmaFlow) + 8512922 (SigmaDock)

Skripte: `SigmaFlow_Development/slurm/train_pdbbind_general_6h_lrfix.sh` und
`SigmaFlow_Variants/sigmadock_control_scripts/
train_pdbbind_general_6h_lrfix_sigmadock.sh` (Letzteres muss in den
separaten ARC-SigmaDock-Klon hochgeladen werden). Beide `--max_epochs 3`
statt 1000, `short`, 6h.

**Warum 6h statt 12h** (User-Vorschlag, nach Datenprüfung bestätigt): die
bestehenden 6h/12h-Läufe zeigen bei SigmaFlow praktisch keine Verbesserung
(133.3° → 131.1° Rotationsfehler, beide SCHLECHTER als die 126.5°-
Zufallsgrenze; SigmaDock 126.3° → 113.4°). Gesucht wird also kein
2°-Effekt, sondern ein großer — der korrigierte Schedule läuft mit 20-100×
höherer LR (Peak 1e-4 statt ~5e-6 am Laufende). **Der Vergleich ist damit
sogar schärfer:** schlägt der 6h-korrigierte Lauf den bestehenden
12h-kaputten (122.2° auf allen 209 Komplexen), ist der Befund bei halber
Rechenzeit eindeutig. Auswertung auf dem vollen 209er-Set (die
6h-Zahlen oben stammen nur von 10 Komplexen, die 6h-Checkpoints existieren
nicht mehr).

**Zwei Fehlschläge unterwegs, beide behoben:**
1. Job 8512799 (erster SigmaDock-Versuch) `FAILED` nach 5:44 mit
   `ModuleNotFoundError: No module named 'sigmadock.diff.denoiser'`.
   **Ursache: Claude hatte das Kontrollskript aus dem SigmaFlow-Skript
   abgeleitet und dabei die Umgebung mitgeschleppt.** SigmaDock läuft in
   `myenv` (aktiviert per `source activate`, schlichtes `python`), NICHT in
   `sigmaflow_env` — dort ist SigmaFlow_Development in die site-packages
   installiert, sodass `import sigmadock` auf das SigmaFlow-Paket auflöst
   (das hat `sigma_flow_generator.py`, kein `denoiser.py`). Behoben durch
   wörtliche Übernahme des Umgebungs-Blocks aus dem nachweislich
   funktionierenden `train_pdbbind_general_24h.sh` desselben Klons; im
   Skript ausführlich kommentiert, damit es niemand "harmonisiert".
   Die Flags `--rot_score_method space --rot_score_scaling rms` waren
   dagegen KORREKT (im funktionierenden Skript genauso vorhanden).
2. Kurzzeitig doppelte SigmaFlow-Jobs (8512798 + 8512885), 8512885
   gecancelt.

**Lehre:** ein Skript für die jeweils ANDERE Codebasis nie aus dem
SigmaFlow-Skript ableiten, sondern immer aus einem dort nachweislich
funktionierenden Skript — Umgebung, Aktivierungsart und Interpreter
unterscheiden sich zwischen den beiden ARC-Bäumen.

**Frühe Verifikation statt Post-mortem** (Konsequenz aus den bisherigen
Vorfällen): unmittelbar nach Jobstart `grep "sigmadock loaded from"` prüfen
— muss auf den jeweils EIGENEN Baum zeigen. Danach: LR-Verlauf gegen die
Erwartung prüfen (Peak 1e-4 innerhalb der ersten ~455 Steps, danach
Annealing; beim kaputten Schedule wären es ~5e-6 am Laufende — Faktor 20,
unverwechselbar).

### ✅✅✅ LR-Fix ausgewertet (2026-08-09): Effekt real, aber klein — der eigentliche Blocker ist die Zahl der GRADIENTENSCHRITTE

Beide Läufe `COMPLETED` mit `ExitCode 0:0` **unterhalb** der 6h-Walltime
(SigmaFlow 5:52:02, SigmaDock 5:29:23) — d.h. die 3 Epochen wurden
abgeschlossen und der LR-Schedule lief **erstmals vollständig durch**,
inklusive Annealing. Anders als bei allen früheren Läufen ist `COMPLETED`
hier das ERWÜNSCHTE Ergebnis, kein Warnsignal. Beide Code-Pfade,
Checkpoints (`0-08-09_16-08-32` bzw. `0-08-09_16-24-54`), EMA-Ladung und
209 Sampling-Ergebnisse verifiziert (Jobs 8523816/8523817).

| Lauf | Rotationsfehler | RMSD | Schwerpunktfehler |
|---|---|---|---|
| SigmaFlow 12h, kaputter LR (Baseline) | 122.2° | 5.58 Å | 5.19 Å |
| SigmaFlow 12h, kaputter LR (Var. b) | 122.2° | 5.61 Å | — |
| SigmaFlow 12h, kaputter LR (Var. c) | 122.1° | 5.58 Å | — |
| **SigmaFlow 6h, LR KORRIGIERT** | **120.1°** | **5.09 Å** | **4.75 Å** |
| SigmaDock 12h, kaputter LR | 118.7° | 13.57 Å | 12.92 Å |
| **SigmaDock 6h, LR KORRIGIERT** | **117.3°** | **12.19 Å** | **11.21 Å** |
| *Zufall* | *126.5°* | — | — |

**Rauschgrenze quantifiziert** (drei unabhängige 12h-Läufe mit effektiv
identischer Konfiguration — Baseline, Var. b, Var. c): Rotationsfehler
streut über **0.1°** (σ=0.04°). Die LR-Fix-Verbesserung von −2.1° ist damit
das ~50-fache der Streuung, also eindeutig real; sie tritt bei BEIDEN
Methoden und in BEIDEN Metriken auf, bei HALBER Rechenzeit (6h schlägt
12h).

**Aber der Effekt ist klein:** 120.1° gegenüber 126.5° Zufall bleibt
praktisch Zufall. Der LR-Fehler war real und messbar schädlich, ist aber
NICHT der Hauptblocker.

### 🎯 Eigentliche Ursache identifiziert: ~5% der Gradientenschritte des Papers

Die Denkweise in EPOCHEN war irreführend — entscheidend ist die absolute
Zahl der Optimierungsschritte:

| | Schritte | Batch | gesehene Samples |
|---|---|---|---|
| Paper (4×A100, 4 Tage, Appendix E.3) | 155.544 | 32 | 4,98 Mio |
| unser 6h-pdbbind-Lauf | ~7.290 | 8 | 0,06 Mio |
| unser 3h-Dummy-Lauf | ~3.070 | 2 | 0,006 Mio |

**Entscheidendes Gegenargument gegen die Epochen-Sicht:** der
Dummy-3h-Lauf absolvierte **614 Epochen** — mehr als die 256 des Papers —
und blieb trotzdem bei 102° Rotationsfehler. Nicht die Epochenzahl ist zu
klein, sondern die absolute Schrittzahl (Faktor ~20-50).

Das erklärt rückwirkend widerspruchsfrei: drei wirkungslose Loss-Varianten,
Zufalls-Rotationen bei beiden Methoden, 0% RMSD<2Å durchgängig. Kein
Loss-Term und kein Schedule-Fix kompensiert einen Faktor 20 bei den
Gradientenschritten.

**Hochrechnung aus dem verifizierten 6h-Durchsatz (3 Epochen in 5:52):**

| Laufzeit | Epochen | Schritte | vs. Paper |
|---|---|---|---|
| 3 Tage | ~37 | ~89.000 | 57% der Schritte |
| 7 Tage | ~86 | ~209.000 | 134% der Schritte, aber nur 33% der Samples (Batch 8 statt 32) |

**Für künftige Skripte:** `max_epochs` muss jeweils neu auf das Zeitbudget
kalibriert werden (3 Epochen ≈ 6h ⇒ 3-Tage-Lauf: `max_epochs=36`,
7-Tage-Lauf: `max_epochs=85`). Ein zu großer Wert verstümmelt still den
LR-Schedule.

### 🔴🔴🔴 ROOT CAUSE GEFUNDEN (2026-08-09): Rahmen-Mismatch in der Rotationsvorhersage (WELT vs. KÖRPER)

**Der Fehler:** SigmaFlow vergleicht im Rotations-Loss eine Netzwerkausgabe
im **Welt-Rahmen** (links-trivialisiert) mit einem Ziel im
**Körper-Rahmen** (rechts-trivialisiert). Der nötige Adjoint-Transport
`u_body = R_tᵀ · ω_world · R_t` **fehlt vollständig**.

**Belege, jede Zeile im Code verifiziert:**

| Seite | Code | Rahmen |
|---|---|---|
| Netzwerk-Ausgabe | `newton_maruyama`: `omega = so3_utils.hat(dW_world)`, Kommentar: *"Transport into GLOBAL (left-invariant - WORLD) frame"* | **WELT** |
| Weitergabe | `_compute_vector_field`: `pred_u_t_R = updates["omega"]` — ohne jede Umrechnung | **WELT** |
| Trainings-Ziel | `calc_rot_vector_field`: `log(R_tᵀ @ R_1)/(1-t)`, Docstring: *"right trivialised vector field"* | **KÖRPER** |
| ODE-Schritt | `euler_step`: `R_next = R_t @ exp(v_t·dt)` — RECHTS-Multiplikation | **KÖRPER** |
| SigmaDock-Original | `dR = exp(scaled_omega); R0_hat = dR @ R_t` — LINKS-Multiplikation | **WELT, konsistent** |

`grep` über den gesamten SigmaFlow-Rotationspfad findet **keinen**
Adjoint-Transport (die vorhandenen `.transpose(-1,-2)` betreffen `delta_R`
und die Trägheits-Regularisierung).

**Schweregrad:** `get_fragment_com_and_rot` setzt `rots.append(I3)` — die
Referenz-Orientierung `R_1` ist die IDENTITÄT. Damit ist `R_t = I` nur
exakt bei `t=1`; bei gleichverteilt gezogenem `t` ist `R_t` fast immer eine
allgemeine Rotation, der Fehler also bei praktisch jedem Trainingsbeispiel
wirksam.

**Warum genau das die innere Kohärenz zerstört:** eine Links-Multiplikation
dreht ein Fragment um eine WELT-Achse — benachbarte Fragmente mit ähnlicher
Netzausgabe drehen gemeinsam, die relative Orientierung bleibt erhalten,
die Anker bleiben zusammen. Eine Rechts-Multiplikation dreht um die
KÖRPER-Achsen des jeweiligen Fragments — dieselbe Netzausgabe erzeugt je
nach aktueller Orientierung `R_t` verschiedene Weltrotationen, die relative
Orientierung wird zerstört. Das Netzwerk müsste diese
orientierungsabhängige Verzerrung selbst kompensieren, kann das über den
mechanischen Kopf (Drehmoment aus Weltkoordinaten → I⁻¹ → hat) aber
strukturell nicht ausdrücken.

**Warum nur die Rotation betroffen ist:** `pred_u_t_trans =
updates["total_force"]` ist ein R³-Vektor im Weltrahmen, das Ziel
`u_t_trans` ebenfalls — Translation ist korrekt. **Exakt das zeigen die
Messungen:** relative Schwerpunkt-Verschiebung SigmaFlow 4.44 Å vs.
SigmaDock 4.47 Å (identisch, Translation intakt), relative
Anker-Verschiebung 2.91 Å vs. 1.82 Å (Rotation defekt).

**Warum Test 1 (Oracle) den Fehler NICHT gefunden hat:** dort wurde das
wahre Vektorfeld eingesetzt, das bereits im Körper-Rahmen vorliegt — ODE
und Rekonstruktion sind in sich konsistent und rekonstruieren exakt. Der
Fehler betrifft ausschließlich das, was das NETZWERK ausgeben soll.

**Erklärt rückwirkend:** die drei wirkungslosen Loss-Varianten (alle
operierten auf einer falsch gerahmten Größe), die nahezu zufälligen
Rotationen, und dass das Anker-Defizit über 6h/12h/24h konstant bleibt
(struktureller Fehler, kein Trainingsdefizit).

**Minimaler Fix (eine Zeile, `R_t` liegt bereits vor):** in
`_compute_vector_field` statt `pred_u_t_R = updates["omega"]`:
```python
R_t = sampled["R_t"]
pred_u_t_R = R_t.transpose(-1, -2) @ updates["omega"] @ R_t
```
Herleitung: `R_t exp(u_body·dt) = exp(ω_world·dt) R_t` ⟺ `u_body = R_tᵀ ω_world R_t`.

### ⚠️ Messfehler in der vorherigen Sitzung, hiermit korrigiert

Die am selben Tag berichtete Kohärenz-Zerlegung (SigmaFlow „reißt das
Molekül auseinander") war **falsch**: sie maß Fragment-SCHWERPUNKTE, und
die sind bei beiden Methoden identisch weit auseinander (4.44 vs. 4.47 Å).
Der Bindungsfehler hängt an den ANKERATOMEN, deren Lage von der Rotation
bestimmt wird. Korrigierte Zahlen auf identischer Fragmentierung (111
Komplexe, 397 Bindungen): Anker-Paare 2.91 Å (SF) vs. 1.82 Å (SD).

**Zusätzlich entdeckte Kontamination aller historischen
Bindungslängen-Zahlen:** die aus den Vorhersagen rekonstruierte
Fragmentierung unterscheidet sich bei **39% der Komplexe** zwischen den
Läufen — es wurden teilweise VERSCHIEDENE Bindungsmengen verglichen. Auf
gematchter Teilmenge bleibt die Richtung des Befunds bestehen, die
absoluten Zahlen in den Tabellen sind aber nur eingeschränkt gültig.

### Systematische Ausschlussliste (alle einzeln gemessen, nicht angenommen)

| Hypothese | Status | Beleg |
|---|---|---|
| Atomreihenfolge/-identität vertauscht | ausgeschlossen | 0/209 Abweichungen, beide |
| SigmaDock erzwingt Bindungen nachträglich hart | ausgeschlossen | σ=0.51, nur 51% im Fenster 1.2-1.7 Å |
| `pos_0`/`pos_t`-Mismatch Training vs. Sampling | ausgeschlossen | in beiden Pfaden konsistent |
| `node_entity`-Filter verwirft Anker | ausgeschlossen | `atom=0, anchor=1`, Filter `<=1` korrekt |
| Referenz-Bindungslängen asymmetrisch genutzt | ausgeschlossen | in BEIDEN via `**kwargs` verworfen |
| Stochastisches Alignment beim Sampling | ausgeschlossen | `alignment_tries: 0` |
| Uniformer Skalierungsfehler | ausgeschlossen | Verhältnisse nicht konstant |
| `sample_conformer=true` bei SigmaDock | ausgeschlossen | 72.7% der Bindungen <0.001 Å |
| `postprocessor.py`-Unterschiede | ausgeschlossen | byte-identisch, nur Gnina, deaktiviert |
| Triangulations-Konditionierung deaktiviert | ausgeschlossen | `rel_distance=True` in beiden, explizit übergeben |
| Δd-Signal zeitlich invertiert | ausgeschlossen | Δd→0 am jeweiligen Datenende in beiden Konventionen |
| ODE-/Rekonstruktionsmechanik | ausgeschlossen | Test 1 (Oracle) rekonstruiert exakt (≤0.0002 Å) |
| Untertraining | ausgeschlossen als ALLEINIGE Ursache | Anker-Defizit konstant über 6h/12h/24h |
| **Rahmen-Mismatch Welt/Körper** | **BESTÄTIGT** | **Code-Belege oben** |

### ✅ Fix implementiert + verifiziert (2026-08-09): `SigmaFlow_Variants/d_frame_fix/`

Neue Variante (9 MB, ohne Alt-Checkpoints), Änderung ausschließlich in
`_compute_vector_field`:

```python
R_t = sampled["R_t"]
pred_u_t_R = R_t.transpose(-1, -2) @ updates["omega"] @ R_t
```
mit vollständiger Herleitung im Docstring. `R_t` war bereits im
`sampled`-Dict vorhanden — auch im Sampling-Pfad
(`sampling.py::_predict_vector_field_step` übergibt `{"R_t": ..., "trans_t": ...}`),
der Fix greift also in Training UND Inferenz.

**Mathematische Verifikation (float64, 64 Zufallsrotationen):**
- Schiefsymmetrie erhalten: 2.2e-16 → weiterhin gültiges so(3)-Element
- Transport-Identität `R_t·exp(u_body·dt) = exp(ω_world·dt)·R_t`: 7.8e-16 über dt ∈ {1, 0.1, 0.01}
- Norm erhalten (Isometrie): 4.4e-16
- **Grenzfall `R_t = I`: Abweichung exakt 0.0** → Fix degeneriert zur alten Version;
  unabhängige Bestätigung der Fehleranalyse (Fehler nur bei t=1 unsichtbar)
- **Alte Variante im selben Test: Abweichung 1.377 bei dt=1** → Fehler in der
  Größenordnung des Signals selbst, nicht numerisches Rauschen

**Schweregrad im realen Training (20.000 Stichproben, exakt via
`conditional_probability_path` mit `R_1 = I`):**

| Größe | Wert |
|---|---|
| Abstand `R_t` zur Identität | 63.3° mean / 57.7° median |
| Anteil <5° (Fix folgenlos) | **nur 4.43%** |
| Relative Änderung der Vorhersage | 0.765 mean / 0.697 median |
| Winkel altes vs. neues Vektorfeld | **47.7°** |

→ Das Netz wurde bei ~95% aller Beispiele auf ein im Mittel um 48°
verdrehtes Ziel trainiert (völlig unkorreliert wäre √2≈1.41; wir liegen bei
0.77, also etwa auf halbem Weg zwischen korrekt und zufällig).

**Smoke-Test** (`--max_steps 5 --accelerator cpu`, 10 Dummy-Komplexe):
Exit-Code 0, endliche Losses (`loss_R=11.89`, `loss_trans=7.08`), kein NaN.
**Auffälligkeit geprüft statt hingenommen:** `loss_R` ist auf fünf
Nachkommastellen identisch zur Baseline. Ursache ist `zero_init_last=True`
(`config.py:109`) — der Ausgabeblock ist nullinitialisiert, also `ω≈0`, und
`R_tᵀ·0·R_t = 0`; der Loss ist dann `‖0−Ziel‖²` unabhängig vom Rahmen. Nach
5 Schritten ist die Vorhersage noch praktisch null. Zusätzlich per
`inspect.getsource` bestätigt, dass der Adjoint-Transport in der
tatsächlich geladenen Datei steht.

**SLURM-Skripte:** `slurm/train_pdbbind_general_6h_framefix.sh`
(`--max_epochs 3`, LR-Schedule kalibriert) und `slurm/sample_posebusters_full.sh`
(`PYTHONPATH`-Override von Anfang an, Pfad auf `d_frame_fix` korrigiert).
Das Erfolgskriterium steht als Kommentar im Trainingsskript, damit die
Auswertung nicht wieder auf Bindungslängen abrutscht.

### ⏳ Job 8525523 gestartet (2026-08-09 23:54, Ende ~05:54): Frame-Fix, 6h

**Vor dem Lauf beide Ebenen verifiziert** (nicht nur der Pfad, weil ein
unvollständiger `scp -r` denselben Pfad hätte zeigen können):
1. Fix physisch im hochgeladenen Code: `grep` findet Zeile 913
   `pred_u_t_R = R_t.transpose(-1, -2) @ updates["omega"] @ R_t`
2. Richtige Kopie geladen: `sigmadock loaded from:
   .../d_frame_fix/src/sigmadock/__init__.py`

**Nebenbeobachtung für künftige Uploads:** dieser Lauf brauchte ~11 Min bis
zur `sigmadock loaded from`-Zeile (frühere Läufe deutlich schneller). Grund:
beim Anlegen des Ordners wurden alle `__pycache__`-Verzeichnisse gelöscht,
also muss Python beim ersten Import sämtliche Module neu kompilieren — auf
ARCs geteiltem Dateisystem spürbar. Harmlos, aber gut zu wissen, damit das
nicht künftig als hängender Job fehlgedeutet wird. Diagnose lief über
`sstat -j <id>.batch` (die Form OHNE `.batch` liefert auf ARC keine Zeilen)
und über das Wachsen der `.out`-Datei.

**Kontrollgruppe steht bereits:** Job 8512798 (6h, `max_epochs=3`,
unveränderter Code) — die Differenz zu 8525523 ist damit exakt der
Frame-Fix, kein weiterer Lauf nötig.

### Nächste Schritte

0. **HÖCHSTE PRIORITÄT: den Rahmen-Fix implementieren und testen.**
   Einzeiler in `_compute_vector_field` (s.o.). Danach: 6h-Lauf mit
   korrigiertem LR-Schedule (`--max_epochs 3`) und Vergleich der
   **relativen Rotationsfehler benachbarter Fragmente** gegen die jetzigen
   Werte (SigmaFlow 121.8° benachbart / 124.9° nicht benachbart;
   SigmaDock 102.9° / 126.6°). Greift der Fix, muss SigmaFlows
   Nachbar-Wert deutlich unter den Nicht-Nachbar-Wert fallen.
   **Vor dem Lauf lokal verifizieren:** Oracle-Test bleibt exakt, und ein
   Smoke-Test zeigt endliche Losses.
1. **Freigabe für die 3-Tage-Stufe beim User einholen** — jetzt erstmals
   INHALTLICH begründet statt nur "nächster Punkt im Plan": es ist die
   kleinste Stufe, die in die Größenordnung der Paper-Schrittzahl kommt
   (57%) und einen aussagekräftigen SigmaFlow-vs-SigmaDock-Vergleich
   überhaupt ermöglicht. Mit `--max_epochs 36` und `--partition=long`.
   Alternative/ergänzend zu prüfen: größere Batch-Size (Paper nutzt 32,
   wir 8) und/oder DDP über mehrere GPUs — beides erhöht den Durchsatz
   stärker als reine Laufzeitverlängerung.
1. Erledigt: LR-Schedule-Fehler verifiziert und korrigiert (s.o.).
1. ✅ ERLEDIGT: Platzierungsgenauigkeits-Hypothese geprüft (s.o.) —
   widerlegt, aber mit dem obigen, wichtigeren Befund.
2. **Keine weiteren Loss-Varianten**, bis mindestens ein Lauf existiert,
   bei dem der Rotationsfehler DEUTLICH unter der 126.5°-Zufallsgrenze
   liegt. Vorher ist jeder Feinjustierungs-Test sinnlos (drei
   Nullergebnisse belegen das bereits empirisch).
3. **Priorität ist jetzt Trainingsdauer/-durchsatz, nicht Loss-Design.**
   Beim User die 3-Tage-Freigabe einholen (laut gestuftem Plan ohnehin die
   nächste Stufe). Realistisch einordnen: selbst ein 7-Tage-Einzel-GPU-Lauf
   bleibt bei ~1/5 der Paper-Rechenzeit — DDP über mehrere GPUs und/oder
   größere Batch-Size wäre der wirksamere Hebel, falls verfügbar.
4. Der Rotationsfehler (mittlerer Winkel vs. 126.5°-Zufallsgrenze) sollte
   ab jetzt die **primäre Fortschritts-Metrik** für jede weitere Stufe
   sein — er zeigt unmittelbar, ob das Modell die Kernaufgabe überhaupt
   lernt, anders als die Übergangsbindungs-Metrik.

---

## 🔖 PAUSE-PUNKT #13 (2026-07-26) — älter, siehe #14 oben für aktuellen Stand

**Kontext:** Direkte Fortsetzung von PAUSE-PUNKT #12. Zwei Dinge diese
Session: (1) der L40S-GPU-Engpass auf ARC (`short`-Partition, 5h+ Wartezeit
für einen einzelnen `gpu:l40s:1`-Job, siehe unten) zwang zu einem
V100-Sampling-Test, der nebenbei einen sauberen Cross-GPU-Konsistenzbeweis
lieferte; (2) der PAUSE-PUNKT-#12-Cleanup wurde nachträglich verifiziert
(Diff-Review + repo-weiter Grep + echter End-to-End-Lauf), und der
Gesamtstand vor dem nächsten großen Schritt (voller Trainingslauf) wurde
präzisiert.

### ✅ V100-GPU-Fallback getestet: Cross-GPU-Konsistenz bestätigt

**Auslöser:** `sample_dummy.sh` fordert hart `--gres=gpu:l40s:1` an;
`squeue`/`sinfo` zeigten alle L40S-Nodes (`htc-g061`-`084`) als `mix`/`alloc`,
kein einziger frei — SLURMs eigene Prognose (`scontrol show job`) nannte
`StartTime` ca. 24h nach Submit für einen 20-Minuten-Job. Fix: Sampling mit
`--gres=gpu:v100:1` überschrieben (zwei genutzte, komplett `idle` V100-Nodes
in `sinfo` gefunden), Checkpoint `experiments/sigmadock/0-07-25_21-40-56/
checkpoints/last.ckpt` (der PAUSE-PUNKT-#11-Produktions-Hyperparameter-Lauf).

**Zwischenfall, gefunden+erklärt, kein Bug:** erster V100-Versuch (Job
8338069) meldete `sacct`-Status `COMPLETED, ExitCode=0:0`, aber das `.out`-Log
zeigte einen Python-Traceback (`FileExistsError`, weil `sample_dummy.sh` ohne
`OUTPUT_DIR`-Override immer denselben Default-Pfad
`sampling_output/results/dummy_train/last/seed_0/` benutzt — der existierte
noch von Job 8254111 aus PAUSE-PUNKT #9). SLURM meldete trotzdem Erfolg, weil
`sample_dummy.sh` kein `set -e` hat und die letzte Skriptzeile ein
immer-erfolgreiches `echo` ist — der fehlgeschlagene Python-Exitcode wird
überschrieben. **Lehre, noch nicht umgesetzt:** `set -euo pipefail` sollte in
allen `slurm/*.sh`-Skripten nach der Shebang ergänzt werden, damit sowas
künftig als `FAILED` auffällt statt sich als falscher Erfolg zu tarnen.

**Korrekter Re-Run** (Job 8338077, `OUTPUT_DIR=sampling_output_v100_prodhparams`,
sonst identisch) lief sauber durch. RMSD gegen wahre Pose (raw/aligned,
Kabsch, alle 10 Komplexe, lokales Skript, nicht committet):

| | raw (Mittel/Median) | aligned (Mittel/Median) |
|---|---|---|
| V100 (dieser Lauf) | 4.32 / 3.70 Å | 2.87 / 2.82 Å |
| L40S (`sampling_output_prodhparams/`, PAUSE-PUNKT #11, dokumentiert) | 4.31 / 3.67 Å | 2.86 / 2.81 Å |

Direkter Koordinatenvergleich V100 vs. L40S (identischer Checkpoint, Seed 0):
mittlere Abweichung **0.04 Å**, max **0.10 Å** über alle 10 Komplexe — reine
Fließkomma-/Kernel-Unterschiede zwischen GPU-Architekturen, keine
strukturelle Divergenz. **Ergebnis: Sampling ist GPU-architektur-unabhängig
korrekt**, nicht zufällig nur auf L40S richtig.

### ✅ PAUSE-PUNKT-#12-Cleanup nachträglich verifiziert (nicht nur behauptet)

Drei unabhängige Prüfungen, alle bestanden:
1. `git diff` der 8 geänderten Dateien Zeile für Zeile gegen die
   `STATUS.md`-Beschreibung abgeglichen — deckungsgleich, keine
   Überraschungen.
2. Repo-weiter Grep (nicht nur die 8 Dateien) nach `get_fragment_com`,
   `_reverse_step`, `cfg.diffusion.*`, `rot_score_method`/`_scaling`,
   `rot_vector_field_method`/`_scaling` — keine Treffer außer einem
   erklärenden Kommentar in `config.py`. Keine Karteileichen, auch nicht in
   den SLURM-Skripten mit Hydra-Overrides.
3. Der V100-Sampling-Lauf oben ist gleichzeitig ein echter
   End-to-End-Smoke-Test des umbenannten Codes (`cfg.ode.*`,
   `HPARAMS.general.epsilon_t`, Konstruktor ohne die toten
   Rotations-Parameter) — Checkpoint-Laden, Modellaufbau, ODE-Integration
   liefen fehlerfrei durch.

**Nebenbei geklärt (User-Frage):** `epsilon_t=0.01` als `t_min`-Untergrenze
ist **keine** Singularitäts-Fix (per Grep bestätigt: kein `/t` oder `log(t)`
irgendwo in `diff/`) — das Trainingsziel (`conditional_probability_path`,
`u_t = x_1-x_0` bzw. `log(R_0^T R_1)`) ist konstant in `t`, auch bei `t=0`.
Die einzige echte Singularität (`1/(1-t)` in `calc_trans_vector_field`/
`calc_rot_vector_field`) sitzt bei `t→1` und betrifft nur die geschlossenen
Marginalfeld-Formeln (`use_true_vector_field`-Debug-Pfad), nicht das
Trainingsziel. Der eigentliche Grund für `t_min == epsilon_t`:
Verteilungs-Konsistenz zwischen Training und Sampling (Netzwerk soll nie
außerhalb seines trainierten `t`-Bereichs abgefragt werden) — nicht
numerische Notwendigkeit. Ob `epsilon_t=0.01` selbst (statt z.B. kleiner)
gut gewählt ist, bleibt offen/unverifiziert (geerbte SigmaDock-Konvention).

### 📋 Präzisierter Status vor dem großen Trainingslauf (User-Frage beantwortet)

User-Annahme "Code ist fertig, nur noch 7 Tage trainieren" korrigiert. Was
tatsächlich noch fehlt, bevor der volle Trainingslauf überhaupt starten
kann (alles unverändert offen, hier nur nochmal geprüft/bestätigt):

1. **`slurm/train.sh` existiert nicht** (per `ls` bestätigt) — nur
   `train_dummy_*.sh` für den 10-Komplexe-Test. `conf/training/slurm.yaml`
   verweist im Kopfkommentar selbst darauf ("4-GPU DDP, 7-Tage-Lauf").
2. **ARC-Partition/Zeitlimit für Mehrtages-Jobs nie geklärt** — `short` ist
   auf Stunden gedeckelt (heute live erlebt: 5h+ Wartezeit für einen
   einzelnen 20-Minuten-L40S-Job auf `short`), für einen 7-Tage-Job
   ungeeignet. Welche Partition passt, mit User/ARC-Support zu klären.
3. **4-GPU DDP nie getestet** — jeder bisherige echte Lauf lief mit
   `--devices 1`. `devices: auto, strategy: ddp` steht in der Config, aber
   Multi-GPU-Device-Placement/Gradienten-Sync ist unverifiziertes Neuland.
4. **Datensatzpfade für den vollen Datensatz nie gegen die Config-Regex
   verifiziert** — bisher lief alles nur gegen die 10 Dummy-Komplexe.
5. **W&B-Logging-Entscheidung offen** — `--offline_run` reicht vermutlich
   nicht für einen unbeaufsichtigten 7-Tage-Lauf.
6. Bereits geprüft und **nicht mehr offen** (alter `STATUS.md`-Punkt
   gegengecheckt): der `_so3_diffuser.set_device(...)`-Bug in `sample.py`
   ist weg (per Grep bestätigt, `_so3_diffuser`/`set_device` kommt nirgends
   mehr im Code vor) — war schon in einer früheren Session gefixt.

**Und danach ist auch noch nicht "fertig":** laut `CLAUDE.md` §11 steht nach
dem großen Lauf noch die volle PoseBusters-Auswertung auf dem vollen
Datensatz aus (bisher nur der kleine Redock-Check auf den 10
Dummy-Komplexen, PAUSE-PUNKT #10/#11).

### Nächste Schritte

1. **Zuerst:** ARC-Partition/Zeitlimit für Mehrtages-DDP-Jobs klären (User
   oder ARC-Support/-Doku — nicht von hier aus beantwortbar).
2. Danach `slurm/train.sh` schreiben (nutzt `conf/training/slurm.yaml`),
   Datensatzpfade verifizieren, W&B-Entscheidung treffen.
3. Optional, unabhängig: `set -euo pipefail` in `slurm/*.sh` ergänzen, damit
   stille Fehlschläge (siehe oben) künftig als `FAILED` auffallen.
4. `.gitignore` wurde diese Session um generierte Verzeichnisse
   (`experiments/`, `sampling_output*/`, `posebusters_results/`,
   Screenshots, `rmsd_summary_all_runs.csv`, `pymol_scripts/`) ergänzt, um
   Checkpoints/Benchmark-Outputs dauerhaft aus dem Repo-Verlauf
   herauszuhalten (siehe `CLAUDE.md` §9).

---

## 🔖 PAUSE-PUNKT #12 (2026-07-26) — älter, siehe #13 oben für aktuellen Stand

**Kontext:** Direkte Fortsetzung von PAUSE-PUNKT #11 — nachdem SigmaFlow
quantitativ auf SigmaDock-Niveau validiert war, wurde aufgeräumt: toter Code
entfernt, Diffusions-Sprachreste umbenannt, `t_min`/`epsilon_t` vereinheit-
licht. **Wichtig für Checkpoint-Kompatibilität:** alle Änderungen betreffen
nur lokale Variablen, Kommentare, oder Konstruktor-Parameter, die dank
`**kwargs`-Auffangbecken in `SigmaFlowGenerator` beim Laden alter Checkpoints
folgenlos ignoriert werden — die wertvollen Checkpoints aus PAUSE-PUNKT
#10/#11 (kompletter HP-Sweep + 3h-Bestätigungslauf) laden weiterhin
unverändert.

### ✅ Entfernter toter Code (verifiziert per Grep, nicht geraten)

- **`rot_vector_field_method`/`rot_vector_field_scaling`** — komplett raus aus
  `SigmaFlowGenerator.__init__`, `config.py` (Dataclass-Feld + CLI-Flag),
  `conf/training/slurm.yaml`. Grund: wirkungslos (siehe PAUSE-PUNKT #11 —
  `_compute_vector_field` hat nie eine `alpha`-Reskalierung angewendet,
  anders als die Diffusions-Score-Berechnung). Die theoretische Frage, ob
  Flow Matching so eine Reskalierung bräuchte, bleibt offen — falls je
  gebraucht, sollte sie neu und bewusst implementiert werden, nicht als
  Plumbing-Leiche herumliegen.
- **`noise_scale`/`noise_decay` in `sampler()`** (`sampling.py`) — komplett
  raus (Parameter + `noise_scales`-Berechnung), inkl. `conf/sampling/base.yaml`.
  Nur `scripts/sample.py` und die zwei Diagnose-Skripte riefen `sampler()`
  auf, alle drei entsprechend angepasst (kein `cfg.diffusion.noise_scale`
  mehr).
  **Bei `sample_notebook()` bewusst NICHT entfernt** — alle vier
  Notebook-Aufrufstellen (`04_diffusion.ipynb`, `05_crossdock_sampling.ipynb`,
  `extensions/sampling.ipynb`) übergeben `noise_scale`/`noise_decay` explizit
  als Keyword-Argument; Entfernen hätte die Notebooks mit `TypeError`
  zerstört. Nur die intern nie wirksame `noise_scales`-Berechnung wurde dort
  entfernt, die Parameter selbst bleiben (jetzt im Docstring als inert
  dokumentiert).
- **`get_fragment_com`** (`sigma_flow_generator.py`) — komplett raus, nie
  aufgerufen (per Grep bestätigt), hatte zudem einen vergessenen Debug-`print`.
- **Toter `if False:`-Kovarianz-Rotationsblock** in `get_fragment_com_and_rot`
  (~30 Zeilen, nie ausgeführt) — auf die tatsächlich laufende Zeile
  (`rots.append(I3)`) reduziert.

**Bewusst NICHT entfernt** (kein echter toter Code):
- `sigma_min` — Docstring sagt ehrlich "reserved for future use", riesiger
  Blast-Radius (viele Aufrufer, `enforced_cfg`-Overrides in mehreren
  Skripten), zu riskant für zu wenig Nutzen.
- `reverse_rotations` — hat bei `True` echten Effekt in `newton_maruyama`,
  nur standardmäßig aus.
- `sample_notebook()` selbst — wird von drei Notebooks benutzt, ist Duplikat
  von `sampler()`, aber kein toter Code.

**Nebenbefund beim Prüfen der Notebook-Aufrufstellen (nicht behoben, nur
dokumentiert):** drei der vier Aufrufe übergeben `discretization="edm"` an
`sample_notebook()` — das ist dort aber nie implementiert (nur `sampler()`
hat den `edm`-Zweig). Diese Notebook-Zellen würden schon mit `ValueError`
abstürzen, unabhängig von dieser Session. Vorbestehend, separates Thema.

### ✅ Umbenennungen (inhaltlich sinnfreie SigmaDock-Reste)

- `sampling.py`: Docstrings ("Evaluate the SE3Diffuser by performing a
  reverse sampling process" → "Integrate the flow-matching ODE forward..."),
  Kommentare ("Reverse step" → "ODE step", "reverse kinematics" →
  "roto-translations"), lokale Funktion `_reverse_step` → `_predict_vector_field_step`
  (rein intern, keine Aufrufer außerhalb der Datei).
- `sigma_flow_generator.py`: "Output of forward_marginal" →
  "Output of conditional_probability_path", "Forward pass for the denoiser"
  → "...for the flow-matching generator", "Noise the current denoised
  states" → "Interpolate the conditional probability path...".
- `trainer.py`: "Preconditioned Denoiser" → "SigmaFlowGenerator".
- `oracle.py`: `epsilon_t`-Kommentar korrigiert (war "avoid division by zero
  in diffusion process" — sachlich falsch/irreführend für Flow Matching;
  jetzt präzise beschrieben als Trainings-Zeit-Untergrenze).
- `conf/sampling/base.yaml`: Config-Gruppe `diffusion:` → `ode:` (betraf 3
  Skripte: `scripts/sample.py`, `scripts/diagnose_vector_field.py`,
  `scripts/diagnose_step0.py`, plus 2 SLURM-Skripte mit Hydra-Overrides —
  alle mitgezogen).
- **Bewusst nicht angefasst:** `net/`-Verzeichnis (EquiformerV2-Backbone,
  z.B. `AtomDiffusionEncoder`) — laut `CLAUDE.md` §9 tabu, auch für reine
  Umbenennungen.

### ✅ `t_min`/`epsilon_t` vereinheitlicht

`sampler()`/`sample_notebook()`s `t_min`-Default zeigt jetzt direkt auf
`HPARAMS.general.epsilon_t` (`0.01`) statt einem separaten, kleineren
hartkodierten Wert (`1e-3` im Funktions-Default, `0.005` in
`conf/sampling/base.yaml` — beide jetzt `0.01`, einheitliche Quelle der
Wahrheit). Begründung: Training sampelt `t` nie unter `epsilon_t` — das
Netzwerk sollte beim Sampling nie außerhalb seines trainierten Zeitbereichs
abgefragt werden. Kostenlos: der tatsächliche Startzustand (`trans_0`, `R_0`,
reines Rauschen) ändert sich dadurch nicht, nur das `t`-Label beim ersten
Integrationsschritt — kein Retraining nötig, kein Risiko für bestehende
Checkpoints.

### Nächste Schritte

1. Offen: ob `sample_notebook()`s `edm`-Diskretisierung (von 3 Notebook-Zellen
   erwartet, aber nie implementiert) nachgerüstet werden soll.
2. Ansonsten: siehe PAUSE-PUNKT #11 — großer Trainingslauf auf vollem
   Datensatz ist der nächste inhaltliche Schritt.

---

## 🔖 PAUSE-PUNKT #11 (2026-07-26) — älter, siehe #12 oben für aktuellen Stand

**Kontext:** Direkte Fortsetzung von PAUSE-PUNKT #10 — nachdem der
`edm`-Sampling-Bug gefixt war und SigmaFlow ungefähr auf SigmaDock-Niveau
lag, wurde geprüft, ob SigmaDock nur deshalb "gewinnt", weil es mit seinen
eigenen Produktions-Hyperparametern verglichen wurde, während SigmaFlow noch
mit Ad-hoc-Werten lief. Ergebnis: **ja**, und nach Angleichung liegt
SigmaFlow jetzt mindestens gleichauf mit SigmaDock — teils sogar leicht
davor.

### ✅ Zweiter toter Hyperparameter-Pfad gefunden + gefixt: `rot_score_method`/`rot_score_scaling`

Exakt dasselbe Namens-Mismatch-Problem wie in PAUSE-PUNKT #6 für SigmaDock
dokumentiert, aber diesmal bei SigmaFlow: `config.py` nannte die Felder
`rot_score_method`/`rot_score_scaling`, der `SigmaFlowGenerator`-Konstruktor
erwartet aber `rot_vector_field_method`/`rot_vector_field_scaling` — landete
im `**kwargs`-Auffangbecken, wurde beim Training komplett ignoriert.
**Gefixt** (`config.py`: Dataclass-Feld + CLI-Flag umbenannt, inkl. der
falschen `"score"`-Option, die eigentlich `"vector_field"` heißen muss, um
zum Denoiser zu passen; `conf/training/slurm.yaml` zur Konsistenz
mitgezogen). Damit ist `--rot_vector_field_scaling` jetzt über die CLI
tatsächlich erreichbar.

**Wichtiger Nachtrag, beim genaueren Hinschauen entdeckt:** `rot_vector_field_scaling`
ist bei SigmaFlow trotz des Fixes **aktuell komplett wirkungslos** — nicht
wegen eines Namens-Bugs, sondern weil `_compute_vector_field` in
`sigma_flow_generator.py` das rohe Newton-Maruyama-`omega` **direkt** als
Vektorfeld zurückgibt, ohne die `alpha`-Reskalierung, die die
Original-Diffusion in `_compute_scores` durchführt (Kommentar im Code: "no
time dependent scaling needed, unlike the diffusion score" — eine bewusste,
plausible Design-Entscheidung aus einer früheren Session, aber sie lässt den
Parameter seitdem ins Leere laufen). Der `rms`-vs-`true`-Unterschied im
Sweep unten war folglich **Trainings-Rauschen, kein echter Effekt** — beide
Einstellungen durchlaufen exakt denselben Code. Offene Frage für eine
spätere Session: ob Flow Matching überhaupt eine analoge Reskalierung
braucht (SigmaFlows CFM-Ziel ist zeit-konstant, anders als der
Diffusions-Score — die Design-Entscheidung könnte also berechtigt sein,
wurde aber nie explizit hergeleitet/verifiziert).

### ✅ Breiter Hyperparameter-Sweep (10 Kombinationen, 1-2h/Job) + Lehre über Auswertungs-Rauschen

`trans_score_weight ∈ {1.0, 2.0}` × `rot_score_weight ∈ {0.5, 1.0, 2.0, 3.0}`
(mit `rot_vector_field_scaling=rms`), plus 2 Stichproben mit `scaling=true`
bei `trans=2.0`. **Wichtige Methodik-Lektion unterwegs:** eine einzelne
Validierungs-Auswertung pro Checkpoint ist zu verrauscht, um Kombinationen
verlässlich zu vergleichen — derselbe Checkpoint zeigte in verschiedenen
Einzelmessungen `loss_trans`-Werte von `1.27` bis `1.95` (Flow-Matching-Zeit
`t` und Rauschen werden bei jeder Validierung neu zufällig gezogen, nicht
fixiert). Gelöst durch **8-fache Wiederholung pro Checkpoint + Mittelwert**
(`eval_unweighted_losses.py`, nicht committet) — zwei unabhängige
8-fach-Durchläufe stimmten danach gut überein.

**Ergebnis (korrigiert eine frühere Annahme):** höheres `rot_score_weight`
(2 oder 3) verschlechtert `loss_trans` spürbar, **ohne** `loss_R` überhaupt
zu verbessern — reiner Nachteil. Die ursprüngliche Hypothese (aus einem
frühen, kleinen Diagnose-Test), SigmaFlow bräuchte ein höheres
Rotations-Gewicht als SigmaDock, hat sich nicht bestätigt. Beste
Kombination: **`trans_score_weight=2.0, rot_score_weight=0.5,
rot_vector_field_scaling=rms`** — deckungsgleich mit SigmaDocks eigener
Produktions-Empfehlung (`conf/training/slurm.yaml`).

Auch hier: Ordnernamens-Kollisionen bei parallel gestarteten Jobs traten
mehrfach auf (Zeitstempel nur sekundengenau, `get_exp_dir` hat inzwischen
einen `-v2`-Fallback, der aber bei >2 gleichzeitigen Kollisionen nicht
ausreicht) — betroffene Kombinationen wurden einzeln mit zeitlichem Abstand
nachgeholt.

### ✅✅ 3h-Bestätigungslauf: SigmaFlow jetzt auf/über SigmaDock-Niveau

Job mit obiger Kombination, `slurm/train_dummy_overfit_gpu_3h_prodhparams.sh`,
2:45h, dieselben 10 Komplexe. `best_model_score` (`loss_val/total`) = **5.23**
bei Epoche 614 — grob halbiert gegenüber dem ursprünglichen Lauf mit den
Ad-hoc-Gewichten (`12.69`, PAUSE-PUNKT #9). Anschließend gesampelt und
gegen alle bisherigen Läufe verglichen (RMSD + PoseBusters, alle 10
Komplexe):

| | roh (Mittel/Median) | nach Kabsch-Ausrichtung (Mittel/Median) |
|---|---|---|
| SigmaFlow, alt (Ad-hoc-Gewichte, `edm`-Bug) | 10.21 / 10.37 Å | 7.54 / 7.76 Å |
| SigmaFlow, `edm`-Fix, Ad-hoc-Gewichte | 6.04 / 5.72 Å | 3.09 / 3.06 Å |
| **SigmaFlow, `edm`-Fix + Produktions-Gewichte** | **4.31 / 3.67 Å** | **2.86 / 2.81 Å** |
| SigmaDock, eigene (Ad-hoc-)Gewichte | 6.90 / 6.85 Å | 2.85 / 2.90 Å |
| SigmaDock, Produktions-Gewichte | 5.08 / 4.76 Å | 2.59 / 2.60 Å |

**SigmaFlow liegt jetzt bei der rohen Abweichung vor SigmaDocks eigenem
Produktionslauf** (4.31 Å vs. 5.08 Å), bei den ausgerichteten Werten
praktisch gleichauf (2.86 vs. 2.59 Å). Dieselbe gemeinsame architektonische
Schwäche bei beiden (`bond_lengths`/`bond_angles`/`minimum_distance_to_protein`:
`0.0` bei **allen** fünf verglichenen Varianten, hyperparameter-unabhängig —
bestätigt erneut, dass das eine geerbte Architektur-Eigenschaft ist, keine
Methoden-Schwäche).

**PoseBusters im Detail, `sigmaflow_prodhparams` vs. `sigmadock_prodhparams`
(nachträglich präzisiert — die erste Zusammenfassung hatte das zu grob als
"SigmaFlow leicht vorne" dargestellt, was so nicht stimmt):** von 28 Checks
sind 23 exakt identisch. Bei den 5, wo es einen Unterschied gibt, gewinnt
**durchgehend SigmaDock** (nie SigmaFlow):

| Check | SigmaFlow | SigmaDock | Differenz |
|---|---|---|---|
| `double_bond_stereochemistry` | 0.9 | 1.0 | 1/10 Komplexe |
| `tetrahedral_chirality` | 0.8 | 0.9 | 1/10 Komplexe |
| `volume_overlap_with_protein` | 0.1 | 0.2 | 1/10 Komplexe |
| `volume_overlap_with_organic_cofactors` | 0.9 | 1.0 | 1/10 Komplexe |
| `internal_energy` | 0.0 | 0.2 | 2/10 Komplexe |

Vier der fünf Unterschiede sind nur 1 von 10 Komplexen — bei dieser
Stichprobengröße (und der bereits nachgewiesenen Auswertungs-Stochastizität,
siehe oben) kein verlässliches Signal. Einzige Ausnahme mit doppelt so
großem Unterschied: `internal_energy` (2/10) — passt mechanistisch zum
schon unabhängig gefundenen kleinen Rest-Unterschied bei der ausgerichteten
RMSD (SigmaDock 2.59 Å vs. SigmaFlow 2.86 Å, siehe Tabelle oben), also
vermutlich etwas konsistentere relative Fragment-Platzierung bei SigmaDock
— aber bei n=2/10 nicht als bewiesener struktureller Vorteil zu werten,
eher ein schwacher, plausibler, unbestätigter Hinweis.

**Zusammengefasst, präziser:** RMSD-Positionsgenauigkeit spricht leicht für
SigmaFlow, PoseBusters-Struktur-Plausibilität leicht für SigmaDock — in
Summe wirklich **gleichwertig**, keine Methode dominiert die andere.

**Einordnung:** die Flow-Matching-Konversion (das eigentliche Projektziel
laut `CLAUDE.md` §1) ist damit auf dem 10-Komplexe-Überanpassungstest
**validiert** — bei identischen Daten, identischer Architektur und fair
verglichenen Hyperparametern erreicht SigmaFlow mindestens die Qualität der
Original-Diffusions-Implementierung, in mehreren Metriken sogar leicht
darüber.

### Nächste Schritte

1. Offene Frage von oben: braucht Flow Matching eine eigene, theoretisch
   hergeleitete Reskalierung analog zu `rot_score_scaling`/`alpha(t)`? Aktuell
   nicht dringend, da SigmaFlow bereits auf SigmaDock-Niveau ist — aber
   könnte die verbleibende kleine Lücke schließen, falls später relevant.
2. Gemeinsame Fragment-Übergangs-Schwäche (Bindungslängen/-winkel an
   Torsionsbindungs-Schnittstellen, `PoseBusters`: `0.0` bei allen Varianten)
   — potenzieller Verbesserungspunkt für beide Methoden, nicht Teil des
   ursprünglichen Konversions-Auftrags.
3. Grünes Licht für den nächsten großen Schritt: Training auf dem vollen
   Datensatz vorbereiten (siehe PAUSE-PUNKT #8 "Wichtige Klarstellung" —
   Partition/Zeitlimit für Mehrtages-Lauf klären, `slurm/train.sh` schreiben,
   Datensatz-Pfade auf ARC verifizieren, W&B-Logging-Entscheidung).
4. `t_min`-Default (`0.005`) liegt weiterhin knapp unter `epsilon_t=0.01` —
   kleiner, nicht als kritisch bestätigter Rest-Mismatch (siehe PAUSE-PUNKT #10).
5. `noise_scale`/`noise_decay` in `sampling.py` weiterhin totes Feature
   (siehe PAUSE-PUNKT #10).

---

## 🔖 PAUSE-PUNKT #10 (2026-07-24) — älter, siehe #11 oben für aktuellen Stand

**Kontext:** Direkte Fortsetzung von PAUSE-PUNKT #9 — der dort dokumentierte
Qualitätsbefund ("Ligand komplett zerschossen") wurde in dieser Session bis
zur Ursache zurückverfolgt, ein echter Bug gefunden und gefixt, und der Fix
quantitativ über alle 10 Komplexe verifiziert (nicht nur visuell an einem
Beispiel).

### ✅ Root Cause gefunden + gefixt: `edm`-Zeitplan lief rückwärts

`sampling.py::sampler()`s `discretization="edm"`-Zweig (Karras et al. 2022,
1:1 aus dem Original-Diffusions-Code übernommen) baute den Zeitplan als
`(t_max^(1/rho) + i/(N-1)*(t_min^(1/rho)-t_max^(1/rho)))^rho` — das läuft von
`t_max` (i=0) **absteigend** zu `t_min` (i=N-1). Für echte Diffusion ist das
richtig (Rückwärtsprozess geht von maximalem Rauschen zu wenig Rauschen). Für
unsere Flow-Matching-Konvention (`0→1`, Rauschen→Daten, aufsteigend, seit dem
`dt`-Vorzeichenfix in PAUSE-PUNKT #5) ist das **rückwärts** — und `edm` ist
der tatsächliche Default in `conf/sampling/base.yaml` (nicht `power`!), lief
also in **jedem** bisherigen echten Sampling-Lauf, inklusive Job 8254111 (die
`.sdf`s aus PAUSE-PUNKT #9).

**Gefunden über:** ein neues Debug-Skript (`scripts/diagnose_step0.py`, lokal
nicht committet, siehe unten) + ein temporärer `debug_first_step`-Parameter
in `sampler()` (`sampling.py`, standardmäßig `False`, keine
Verhaltensänderung), der bei `i==0` `t` sowie NaN/Inf-Status von
vorhergesagtem und wahrem Vektorfeld ausgibt. Zeigte: `t=1.000000` bei
Schritt 0 (statt `t_min=0.005`) — daraus folgt zwingend `t_min==t_max` in der
tatsächlich benutzten Zeitfolge, was nur mit dem `edm`-Zweig zusammenpasst.

**Zwischenzeitlich verworfene Hypothesen** (der Reihe nach getestet, alle per
echtem Experiment widerlegt, nicht nur angenommen):
- `epsilon_t`-Mismatch (Training nie `t<0.01`, Sampling-Default `t_min=1e-3`
  darunter): mit `t_min=0.02` erneut getestet — identisches `inf`/`nan`,
  widerlegt.
- Stillschweigend genullte NaN-Losses während des Trainings
  (`trainer.py`-Sicherheitsnetz): `grep -c "NaN/Infs in loss"` im
  Trainings-Log von Job 8238209 → **0** Treffer, widerlegt.
- SO(3)-Log-Map-Singularität nahe 180°-Rotationen als Ursache: ein
  unabhängiger Testlauf zeigte bei `179.56°` noch endliche Werte, widerlegt
  als generelle Erklärung (die eigentliche Ursache war die simple
  `1/(1-t)`-Division bei `t=1.0` exakt, nicht ein Rotationswinkel-Sonderfall).
- Triangulations-Mechanismus (`get_triangle_equality_mapping`,
  Fragment-Anker-Kanten): `data.py`/`chem/processing.py`/`chem/fragmentation.py`
  Byte-für-Byte identisch zwischen `SigmaDock/` und `SigmaFlow_Development/`
  verglichen (`diff`, keine Unterschiede in der Logik), `ignore_triangulation:
  false` in beiden echten Trainingskonfigurationen bestätigt — kein
  Unterschied, keine Erklärung.

**Fix** (2 Zeilen, `sampling.py::sampler()`, nur der `edm`-Zweig — `power`
und `sample_notebook()` waren nicht betroffen): `t_min`/`t_max` in der Formel
vertauscht, sodass sie jetzt aufsteigend läuft (`t_min` bei `i=0`, `t_max`
bei `i=N-1`), konsistent mit `power` und der restlichen Schleife.
`conf/sampling/base.yaml`: Default zusätzlich von `edm` auf `power`
umgestellt (die einzige bisher durch PAUSE-PUNKT #5 tatsächlich verifizierte
Variante — `edm`s theoretische Rechtfertigung, ungleichmäßige Krümmung
auszugleichen, stammt aus Karras' Diffusions-SDE und wurde nie für den
linearen Flow-Matching-Pfad neu hergeleitet, siehe Diskussion in dieser
Session).

**Wichtige Klarstellung, mit User besprochen:** das Training selbst ist von
diesem Bug **nicht** betroffen — `_sample_time`/`conditional_probability_path`
sampeln `t` direkt und unabhängig von der `discretization`/`num_steps`/`rho`-
Logik, die nur beim iterativen Sampling (Inferenz) eine Rolle spielt. Der
Checkpoint aus Job 8238209 ist also unter normalen Bedingungen entstanden;
nur das **Auslesen** (Sampling) war kaputt.

### ✅ Fix quantitativ über alle 10 Komplexe verifiziert (nicht nur 1 Komplex visuell)

Neuer Sampling-Lauf mit Fix (Job 8283171, `slurm/sample_dummy.sh`,
`OUTPUT_DIR=sampling_output_after_edm_fix`, gleicher Checkpoint wie
PAUSE-PUNKT #9). Anschließend lokales Analyse-Skript (RDKit,
Rotationsbindungs-Fragmentierung wie im Training, `Chem.FragmentOnBonds`,
über alle 10 Komplexe, nicht committet, bei Bedarf aus dieser Beschreibung
rekonstruierbar) verglich rohe und nach optimaler starrer Ausrichtung
(Kabsch) verbleibende RMSD gegen die wahre Pose, für drei Sätze: SigmaFlow
alt (Job 8254111, kaputt), SigmaFlow neu (Job 8283171, gefixt), SigmaDock
(Job 8260312-Checkpoint, Original-Vergleichslauf aus PAUSE-PUNKT #9).

| | roh (Mittel/Median) | nach Kabsch-Ausrichtung (Mittel/Median) |
|---|---|---|
| SigmaFlow alt | 10.21 / 10.37 Å | 7.54 / 7.76 Å (nur 26 % Reduktion) |
| **SigmaFlow neu** | **6.04 / 5.72 Å** | **3.09 / 3.06 Å (49 % Reduktion)** |
| SigmaDock | 6.90 / 6.85 Å | 2.85 / 2.90 Å (59 % Reduktion) |

**Ergebnis:** SigmaFlow (neu) liegt über alle 10 Komplexe gemittelt sogar
minimal **besser** als SigmaDock (roh), und zeigt jetzt **dieselbe Fehlerart**
(größtenteils reine Rotations-/Translations-Abweichung, keine intern
zerrissene Struktur mehr — Reduktion durch Ausrichtung fast so groß wie bei
SigmaDock). Der ursprüngliche visuelle Eindruck aus einem einzelnen
PyMOL-Vergleich (`1G9V_RQ3`, wo SigmaDock zufällig etwas näher an der wahren
Pose lag) war **nicht repräsentativ** für den ganzen Datensatz.

Zusätzlich geprüft und **verworfen**: ein gemeinsames, für beide Methoden
"schwieriges" Fragment als Erklärung für den optischen Eindruck — nur 2 von
10 Komplexen haben dasselbe fehleranfälligste Fragment bei SigmaFlow-neu und
SigmaDock (bei ~7 Fragmenten/Komplex im Schnitt ist das nahe am
Zufallsniveau, kein Beleg für einen strukturellen gemeinsamen Schwachpunkt).

Vier Konkurrenz-Hypothesen für den (jetzt kleinen, ~3 Å) Restunterschied
durchgegangen: Vergleichbarkeit der Läufe (✅ bestätigt vergleichbar —
Trainingsskripte bis auf Pfade wortgleich, kein Pretrained-Checkpoint-Laden,
keine unterschiedlichen Hyperparameter), Dropout/Regularisierung (identisch
in beiden, `attention_dropout=0.3`/`edge_dropout=0.1`, kein Override in
beiden Trainings-Skripten — erklärt eher generell "warum überfittet keins
perfekt", nicht den Methodenunterschied), Sampling-Schrittzahl (beide `25`,
identisch), Architektur/Trainings-Loss-Gewichtung (SigmaDock gewichtet den
Score zeitabhängig `1/T_score_scaling²` etc., SigmaFlow nutzt eine rohe,
ungewichtete MSE — plausibelste verbleibende, aber **unbestätigte**
Erklärung für den kleinen Restunterschied, noch nicht getestet).

### Neue, nicht committete Hilfsskripte dieser Session (bei Bedarf rekonstruierbar)

- `scripts/diagnose_vector_field.py` + `slurm/diagnose_vector_field.sh`:
  ruft `sampler()` direkt auf einem echten Checkpoint+Batch auf, druckt
  `all_losses` pro Schritt (die `scripts/sample.py` bisher berechnet, aber
  wegwirft) sowie die finale Positions-Abweichung in Å.
- `scripts/diagnose_step0.py` + `slurm/diagnose_step0.sh`: baut Schritt 0
  manuell nach, prüft NaN/Inf getrennt an jeder Zwischenstufe (Rohe
  Netzwerk-Ausgabe / aggregierte Kraft-Drehmoment / vorhergesagtes Feld /
  wahres Feld) — führte am Ende NICHT zur eigentlichen Ursache (andere
  Zufallsziehung als der Original-Bug-Lauf, RNG-Stream divergierte leicht
  zwischen Skripten), der `debug_first_step`-Parameter direkt in `sampling.py`
  war der Weg, der tatsächlich funktionierte.
- Lokales Fragment-RMSD-Vergleichsskript (RDKit, s.o.) für die
  10-Komplexe-Auswertung.

### ✅ Chemische Plausibilität geprüft (PoseBusters, lokal, `redock`-Konfiguration, alle 10 Komplexe)

Lokal ausführbar ohne ARC (posebusters bereits lokal installiert, siehe
PAUSE-PUNKT #5). Alle drei Sätze (SigmaFlow alt/neu, SigmaDock) gegen die
wahre Pose + Protein gebustet.

**Kernergebnis:** die größten Probleme (`bond_lengths`, `bond_angles`: **0 %
Pass-Rate bei allen drei Methoden**; `minimum_distance_to_protein`,
`volume_overlap_with_protein`: ebenfalls 0 % bei allen drei) sind eine
**gemeinsame** Eigenschaft der geerbten Fragment-Architektur — jedes Fragment
wird starr unabhängig platziert, nichts erzwingt eine chemisch sinnvolle
Bindungslänge/-winkel genau an der Torsionsbindungs-Schnittstelle zwischen
zwei Fragmenten, und nichts verhindert Protein-Ligand-Überlappung. Betrifft
SigmaDock genauso wie SigmaFlow — kein Unterschied zwischen den Methoden,
sondern eine Lücke in der ursprünglichen (unverändert übernommenen)
Architektur selbst.

Bei den Checks, wo es Unterschiede gibt, liegt **SigmaFlow (neu) leicht vor**
SigmaDock, nie schlechter: `double_bond_stereochemistry` (1.0 vs. 0.9),
`tetrahedral_chirality` (0.9 vs. 0.8), `internal_energy` (0.1 vs. 0.0).
Kein Komplex besteht bei irgendeiner der drei Methoden **alle** Checks
(0/10 überall) — `rmsd_≤_2Å` (Standard-Erfolgskriterium) liegt bei 0.0 für
alle drei, konsistent mit den ~3-6 Å Restfehlern von oben.

**Fazit:** chemisch ist SigmaFlow (neu) mindestens gleichauf, tendenziell
minimal besser als SigmaDock. Die eigentlich interessantere Erkenntnis ist
die gemeinsame Fragment-Übergangs-Schwäche (Bindungslängen/-winkel,
Protein-Kollisionen) — ein potenzieller Verbesserungspunkt für **beide**
Methoden, nicht Teil des ursprünglichen Konversions-Auftrags.

Analyseskript nicht committet (RDKit-basiert, `posebusters.PoseBusters`
direkt auf die lokal vorliegenden `.sdf`s angewendet, `config="redock"`,
bei Bedarf aus dieser Beschreibung rekonstruierbar).

### Computational Cost: im Prinzip identisch bei gleicher Schrittzahl

Original-Diffusions-`sampling.py` (`SigmaDock/src_sigmadock/src_sigmadock_diff/
sampling - UMBAUEN.py`) geprüft: ruft das Netzwerk ebenfalls **genau einmal
pro Schritt** auf (`solver="euler"` Default in beiden), identische
EquiformerV2-Architektur/Parameterzahl, identischer `num_steps=25`-Default
in beiden `conf/sampling/base.yaml`. Kein struktureller Kostenunterschied
zwischen Diffusion und Flow Matching in dieser Implementierung. Gemessene
Job-Laufzeiten unterschieden sich zwar sichtbar (SigmaFlow-Sampling-Job:
4:14 gesamt/18s reines Predicting; SigmaDock-Sampling-Job: 59s gesamt),
aber ohne Aufschlüsselung für SigmaDock nicht sauber von
Umgebungs-Overhead (unterschiedliche Conda-Envs) zu trennen — eher kein
echter Algorithmus-Kostenunterschied.

### Nächste Schritte

1. **Optional, unbestätigte Hypothese testen:** zeitabhängige Loss-Gewichtung
   für Flow Matching ausprobieren (analog zu Diffusions-Score-Scaling, aber
   für Flow Matching theoretisch neu herzuleiten, nicht blind kopieren) —
   könnte den verbleibenden `~3 Å`-Unterschied schließen. Nicht dringend, da
   SigmaFlow bereits auf Augenhöhe mit SigmaDock ist.
2. `t_min`-Default (`0.005`) liegt weiterhin knapp unter `epsilon_t=0.01`
   (Trainings-Untergrenze) — kleiner, nicht als kritisch bestätigter
   Rest-Mismatch, könnte bei Gelegenheit angeglichen werden.
3. `noise_scale`/`noise_decay`-Parameter in `sampling.py` sind totes Feature
   (werden berechnet, aber nie zur Trajektorie addiert) — entweder
   implementieren oder ehrlich entfernen/dokumentieren.
4. Mit diesem Erfolg: grünes Licht, um über den nächsten großen Schritt
   nachzudenken (großer Trainingslauf auf dem vollen Datensatz, siehe
   PAUSE-PUNKT #8 "Wichtige Klarstellung" — Überanpassung auf 10 Beispielen
   ist jetzt aber tatsächlich sauber validiert, nicht mehr durch einen
   Sampling-Bug verzerrt).

---

## 🔖 PAUSE-PUNKT #9 (2026-07-22) — älter, siehe #10 oben für aktuellen Stand

**Kontext:** Job 8238209 (`train_dummy_overfit_gpu_3h.sh`, 3h, `rot_score_weight=2.0`)
lief nachts durch (bis Epoche 638, Zeitlimit erreicht). Diese Session: Sampling
zum ersten Mal end-to-end mit einem echten Checkpoint getestet (nie zuvor
gemacht) — dabei einen echten Bug gefunden+gefixt, dann die Posen in PyMOL
angeschaut und einen ernsten Qualitätsbefund gemacht, den wir gerade mit
einem SigmaDock-Original-Vergleichslauf einordnen.

### ✅ Bug gefunden + gefixt: `sigma_min` fehlte beim Checkpoint-Laden

`SigmaFlowGenerator.__init__` nahm `sigma_min` entgegen, speicherte es aber
nie als `self.sigma_min` — nur `self.flow_matcher = SE3_FlowMatcher(sigma_min)`.
`extract_init_kwargs`/`filter_cfg_for_cls` (nutzen `getattr`/`hasattr` auf die
Konstruktor-Parameternamen) haben `sigma_min` deshalb beim Training **aus jeder
bisherigen Checkpoint-`hyper_parameters`-Config stillschweigend entfernt** —
betrifft alle Checkpoints seit Projektbeginn, nie aufgefallen, weil
`scripts/sample.py` vorher nie mit einem echten Checkpoint durchlief.
`load_from_scratch` crashte beim Sampling mit `TypeError: missing 1 required
positional argument: 'sigma_min'`.

**Fix (Commit `20960ee`):**
- `sigma_flow_generator.py:66`: `self.sigma_min = sigma_min` ergänzt (behebt
  es für alle künftigen Checkpoints).
- `scripts/sample.py:529`: `enforced_cfg` erzwingt jetzt `sigma_min=0.0`
  beim Laden (nötig für bereits existierende Checkpoints, deren gespeicherte
  hparams den Key nie hatten — `0.0` ist korrekt, da nie `--sigma_min`
  gesetzt wurde). Kleiner bekannter Schönheitsfehler: dieser Eintrag hängt an
  derselben Bedingung wie `cache_path` (`if cfg.model.cached_s03_dir is not
  None`) — aktuell unkritisch, da `cached_s03_dir` default `"cache"` ist,
  aber bei Bedarf entkoppeln.

### ✅ Sampling mit echtem Checkpoint zum ersten Mal erfolgreich durchgelaufen

Checkpoint `experiments/sigmadock/0-07-22_05-38-04/checkpoints/last.ckpt`
(Job 8238209, Epoche 620, `global_step=3105`). Sampling-Job 8254111
(`slurm/sample_dummy.sh`) lief sauber durch, alle 10 Komplexe bekamen eine
`.sdf` unter `sampling_output/results/dummy_train/last/seed_0/`.

### ⚠️ Wichtiger Qualitätsbefund: Posen sehen in PyMOL stark verzerrt aus

Visueller Vergleich (wahre Pose grün, Vorhersage magenta) in PyMOL: ungefähr
richtige Bindetaschen-Region, aber der Ligand selbst wirkt "zerrissen" (lange,
gerade Bindungen quer durchs Bild). Screenshot unter
`Screenshots_3h_22.07.2026/`.

**Diagnose-Skript** (lokal, `scratchpad/fragment_diagnostic.py`, nicht
committet — bei Bedarf aus dieser Beschreibung rekonstruierbar): pro Molekül
an Torsionsbindungen fragmentiert (gleiche RDKit-Logik wie im Training,
`Chem.FragmentOnBonds` erhält Original-Atomindizes für echte Atome), dann pro
Fragment `raw_rmsd` (unaligned, da `x0_hat`/Referenz schon im selben
Weltkoordinatensystem sind) und `shape_rmsd` (nach Kabsch-Realignment,
isoliert interne Geometrie von Platzierung) gemessen.

**Ergebnis:** `shape_rmsd` ≈ `0.00 Å` fast überall (durch die starre
Fragment-Architektur erzwungen, kein Erfolgssignal). `raw_rmsd` dagegen
**8–15 Å pro Fragment UND für den ganzen Liganden**, bei allen 10 Komplexen
— nahe Zufalls-Platzierung, kein "leicht daneben". Schließt eine einfache
Atom-Indexierungs-/Bindungstopologie-Bug beim SDF-Export aus (Buchhaltung
`frag_atom_idx`/`node_entity<=1`/`FragmentOnBonds`-Verhalten geprüft, korrekt
verdrahtet) — der Fehler sitzt in der tatsächlichen SE(3)-Platzierung pro
Fragment, nicht im Export.

**Bestätigt über Checkpoint-Callback-State** (`ckpt["callbacks"]`, ohne
Download direkt auf ARC gelesen — der bekannte `val_loss=0.0000` im
Dateinamen ist nur ein Formatierungsartefakt, `current_score`/
`best_model_score` sind die echten Werte): `loss_val/total` **plateaut bei
~12.7–12.9** über die letzten ~700 Epochen (Step 2380: `12.69` (best), 2795:
`13.00`, 3105: `12.94`) — verbessert gegenüber dem 300-Epochen-Test
(`~18.8`, PAUSE-PUNKT #4), aber weit von "nahe Null" entfernt UND nicht mehr
wirklich fallend. Spricht für "Training (noch) nicht konvergiert", aber das
Plateau deutet an, dass einfach länger laufen lassen bei denselben
Hyperparametern allein evtl. nicht reicht.

### 🔄 Laufend: SigmaDock-Original-Vergleichslauf (Job 8260312, gestartet 2026-07-22)

Um zu klären, ob das obige ein generelles "10-Beispiele-Überanpassung
braucht Zeit/Tuning"-Problem ist (unabhängig vom Paradigma) oder etwas an der
Flow-Matching-Konvertierung bremst: Original-SigmaDock (Diffusion) wird auf
denselben 10 Komplexen mit denselben Hyperparametern trainiert
(`rot_score_weight=2.0`, `max_epochs=700`, `batch_size=2`, `--debug
--offline_run`, `--time=02:45:00`).

**Wichtig — läuft NICHT im lokalen `SigmaDock/`-Referenzordner dieses Repos**
(bleibt wie immer unangetastet), **sondern in einem separaten, eigenen ARC-Klon**
des Users: `/data/stat-cadd/shug8458/SigmaDock_Reproduction_JulianMueller/sigmadock/`
(eigenes Git-Repo, eigene Env `/data/stat-cadd/shug8458/myenv`, aktiviert via
`module load Mamba; source activate /data/stat-cadd/shug8458/myenv`). Struktur
dort ist NEUEUER/korrekter als der lokale Referenzordner (`src/` statt
`src_sigmadock/`, passt zu `pyproject.toml` — der lokale Referenzordner hat an
der Stelle eine Inkonsistenz, die ein `pip install -e .` vermutlich brechen
würde; deshalb bewusst NICHT den lokalen Ordner kopiert+gepusht, sondern der
ARC-Klon direkt gepatcht).

**Patch dort** (nicht in diesem Git-Repo getrackt, da separater Klon —
Diff hier dokumentiert für Reproduzierbarkeit):
- `scripts/sample.py`: identische `export_predictions_to_sdf`-Erweiterung
  wie bei SigmaFlow (Imports `write_sdf`/`get_mol_from_coords` ergänzt, Methode
  vor `save_results` eingefügt, Aufruf `self.export_predictions_to_sdf(results,
  out_dir)` vor `base_name = "predictions.pt"`). Vorher verifiziert: die Datei
  war bis auf diese Ergänzung 1:1 identisch mit dem lokalen `SigmaDock/scripts/
  sample.py` (Zeile für Zeile geprüft), also sicher zu patchen.
- `conf/experiments/dummy_train.yaml` neu angelegt (rein additiv, gleicher
  Inhalt wie `SigmaFlow_Development/conf/experiments/dummy_train.yaml` —
  fehlte dort vorher, nur `dummy_crossdock.yaml` (Cross-Docking, andere
  Aufgabe) war vorhanden).
- Neues SLURM-Skript `slurm/train_sigmadock_original_dummy_overfit_gpu_3h.sh`
  (dort, nicht in diesem Repo).

### Nächste Schritte

1. Warten bis Job 8260312 durch ist (~2:45h ab Start).
2. Checkpoint finden, mit demselben gepatchten `scripts/sample.py` sampeln,
   `.sdf`s herunterladen, in PyMOL neben die SigmaFlow-Ergebnisse legen —
   gleiches Muster wie bei PAUSE-PUNKT #8/9 oben.
3. Je nach Ergebnis: sieht SigmaDock (Diffusion) im selben Zeitbudget klar
   besser aus (→ etwas an der Flow-Matching-Konvertierung bremst Konvergenz,
   genauer hinschauen) oder ähnlich verzerrt/hoher Loss (→ generelles
   "10 Beispiele reichen nicht in 3h"-Problem, eher Hyperparameter/Zeit-Frage
   als Bug).
4. Optional, noch nicht umgesetzt (besprochen): `pytorch_lightning.loggers.
   CSVLogger` zusätzlich zu `WandbLogger` einhängen (`scripts/train.py:342`),
   damit Losses auch ohne `wandb sync` direkt lesbar sind (aktuell nur im
   Wandb-Offline-Verzeichnis vorhanden). Ebenfalls besprochen, nicht
   umgesetzt: automatische Job-Verkettung Training→Sampling via `sbatch
   --dependency=afterany:$SLURM_JOB_ID` (Haken: `CKPT_DIR` erst zur Laufzeit
   bekannt, bräuchte eine früh geschriebene Marker-Datei in `scripts/train.py`).

---

## 🔖 PAUSE-PUNKT #8 (2026-07-21, später am selben Tag) — älter, siehe #9 oben für aktuellen Stand

**Kontext:** Trainingsjob (`slurm/train_dummy_overfit_gpu_3h.sh`, ~2.75h,
`rot_score_weight=2.0`) wurde vom User auf ARC gestartet (`htc-login.arc.
ox.ac.uk`, Job läuft auf `htc-g048` o.ä. Compute-Node). Während der Lauf
lief, wurde ein vollständiger, penibler Audit über das **gesamte** Repo
gemacht (nicht nur die bisher angefassten Dateien) — Ergebnis unten. Wenn
diese Session hier weitermacht: **zuerst prüfen, ob der Trainingsjob fertig
ist und was er ergeben hat** (Checkpoint unter `experiments/sigmadock/
<timestamp>/checkpoints/`, dann `CKPT_DIR=... sbatch slurm/sample_dummy.sh`,
dann `.sdf`s per `scp` runterladen — Login-Host `htc-login.arc.ox.ac.uk`,
siehe Beispiel-Befehl weiter unten falls nötig — und in PyMOL ansehen).

### Vollständiger Repo-Audit: keine Blocker gefunden

Geprüft (komplett gelesen, nicht nur gegrept): `trainer.py`,
`core/callbacks.py`, `net/model.py`s `forward`-Signatur, `net/encoder.py`,
`data.py`, `core/data.py`, `datafronts.py`, alle `conf/*.yaml`, plus zwei
bisher nie erwähnte Dateien in `diff/`: `criterion.py`, `utils.py`.

**Bestätigt konsistent, keine funktionalen Diffusions-Reste:**
- `trainer.py`/`core/callbacks.py`: Loss-Gewichtung, NaN-Sicherung,
  LR-Scheduler, EMA — alles generisch. `SamplerDebugCallback` heißt zwar so
  als ginge es um den Flow-Matching-Sampler, debuggt aber tatsächlich nur
  PyTorchs `DistributedSampler` (Daten-Batching) — kein Bug, nur ein Name,
  der kurz verwirren könnte.
- `net/model.py`: `forward(self, data, t, **kwargs)` nimmt `t` direkt ohne
  eingebaute Rauschschema-Logik — unabhängig bestätigt, dass das Netzwerk
  Diffusion/Flow-Matching-agnostisch ist (deckt sich mit dem ursprünglichen
  Dependency-Mapping).
- `data.py`/`core/data.py`/`datafronts.py`: keine Diffusions-Spuren.
- **Neu gefunden, beide komplett toter Code (nirgends importiert/aufgerufen):**
  `diff/criterion.py::scale_rotational_score` (noch Diffusions-Wortwahl
  "rotational scores", aber folgenlos da unbenutzt), `diff/utils.py::
  autograd_gradients` (generischer MD-Gradienten-Helfer, gar nicht
  Diffusions-spezifisch, auch unbenutzt).
- `trainer.py:117`s beunruhigend klingender TODO-Kommentar ("old checkpoint
  will not load under new __class__") **geprüft und als unkritisch
  bestätigt**: Checkpoint-Laden konstruiert `SigmaFlowGenerator` immer frisch
  aus Kwargs (`SigmaFlowGenerator(model=equi, **denoiser_cfg)`), nicht aus
  dem gespeicherten Klassennamen — kein Risiko durch unsere Umbenennung.

**Rein kosmetische Funde (Docstrings/Kommentare/Klassennamen, keine Logik,
nichts davon blockierend):**
- `sampling.py` (beide Funktionen): Docstrings sagen noch "Evaluate the
  SE3Diffuser by performing a reverse sampling process" — sollte
  SE3FlowMatcher/Vorwärts-ODE heißen.
- `sigma_flow_generator.py`: vereinzelte Kommentare mit alten Begriffen
  ("Output of forward_marginal", "Forward pass for the denoiser", "Noise
  the current denoised states") — Logik selbst mehrfach diese Session real
  getestet und korrekt.
- `trainer.py`: Docstring "Preconditioned Denoiser".
- `net/encoder.py`/`net/model.py`: Klasse `AtomDiffusionEncoder` ist reine
  generische Atom-Typ-Embedding-Logik ohne Diffusions-Mathematik — Name ist
  Altlast.
- `oracle.py`: Kommentar zu `epsilon_t` ("avoid division by zero in
  diffusion process") übertreibt die Notwendigkeit leicht (unsere
  Trainings-Formel hat bei `t=0` keine Singularität), schadet aber nicht.

**Bereits bekannt, hier nur bestätigt weiterhin offen (kein neuer Fund):**
`rot_score_method`/`rot_score_scaling` totes Config-Feld in
`conf/training/slurm.yaml` (PAUSE-PUNKT #6/#7) — betrifft nur die künftige
Große-Lauf-Config, nicht den aktuell laufenden Dummy-Test.

### Wichtige Klarstellung für die weitere Einordnung (User-Frage beantwortet)

User fragte: wenn der Dummy-Überanpassungstest gut aussieht, fehlt dann
nichts mehr außer Doku/Cleanup/Kalibrierung? **Antwort: die Software-
Konversion (das eigentliche Projektziel laut `CLAUDE.md` §1) wäre damit
validiert — ein einsatzbereites, benchmarkgeprüftes Modell ist es aber
noch nicht.** Zwei Dinge fehlen dafür grundsätzlich, nicht nur kosmetisch:
1. **Überanpassung auf 10 Beispielen beweist keine Generalisierung** — nur
   der große Lauf auf dem vollen Datensatz beantwortet das.
2. Der große Lauf selbst ist kein triviales "Skript starten": `slurm/
   train.sh` existiert noch nicht (braucht ARC-Partitions-Klärung für
   Mehrtages-Jobs), die echten Datensatz-Pfade auf ARC wurden nie gegen die
   Config-Regex verifiziert, und danach steht die eigentliche PoseBusters-
   Auswertung noch komplett aus (laut `CLAUDE.md` §11 bewusst ein
   eigener, späterer Schritt, nicht Teil der Kern-Konversion).

### ARC-Zugangsdaten (neu dokumentiert, bisher nirgends festgehalten)

Login-Host: `htc-login.arc.ox.ac.uk` (`ssh -X shug8458@htc-login.arc.ox.ac.uk`).
Beispiel `scp` von einem lokalen Windows-Rechner aus (nicht von innerhalb
ARC), um Sampling-Ergebnisse runterzuladen:
```
scp shug8458@htc-login.arc.ox.ac.uk:/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow/SigmaFlow_Development/sampling_output/results/dummy_train/last/seed_0/*.sdf "<lokaler-Zielordner>"
```

### Nächste Schritte

1. Trainingsjob-Ergebnis prüfen (Loss-Kurven, Checkpoint vorhanden).
2. `slurm/sample_dummy.sh` mit dem Checkpoint laufen lassen, `.sdf`s
   runterladen, in PyMOL ansehen — **`use_true_vector_field=false` (Default,
   schon so in `conf/sampling/base.yaml`) ist Pflicht für einen echten Test**,
   nicht `true` (das würde das Netzwerk umgehen, siehe PAUSE-PUNKT #5).
3. Je nach Ergebnis: falls die Posen gut aussehen, grünes Licht fürs
   Vorbereiten des großen Laufs (siehe "Wichtige Klarstellung" oben für den
   tatsächlichen Umfang davon). Falls nicht: genauer hinschauen, bevor
   Rechenzeit für den großen Lauf investiert wird.
4. Weiterhin offen (unverändert aus PAUSE-PUNKT #6/#7): `rot_score_method`-
   Entscheidung, `rot_score_weight`-Kalibrierung für den großen Lauf,
   `slurm/train.sh` bauen, Datensatz-Pfade auf ARC verifizieren.

---

## 🔖 PAUSE-PUNKT #7 (2026-07-21, später am selben Tag) — älter, siehe #8 oben für aktuellen Stand

### `scripts/sample.py` portiert + SDF-Export gebaut — fertig, verifiziert

**`scripts/sample.py`** existiert jetzt (portiert von `SigmaDock/scripts/sample.py`,
vorher komplett fehlend in `SigmaFlow_Development`). Mechanische Fixes wie
geplant: `denoiser.diffuser._so3_diffuser.set_device(...)`-Zeile entfernt,
`use_true_scores` → `use_true_vector_field` (auch in `conf/sampling/base.yaml`),
`SigmaDockDenoiser` → `SigmaFlowGenerator`. Alle übrigen Abhängigkeiten
(`sampling_setup.py`, `SampleCycleWrapper`, `compute_gnina_score`,
`compact_posebusting`) waren schon diffusionsfrei vorhanden, keine Änderung
nötig. Importiert sauber (lokal getestet, brauchte `spyrmsd` nachinstalliert).

**`SamplingModule.export_predictions_to_sdf`** (neue Methode, User-geschrieben,
von Claude durchs Schreiben begleitet): schreibt pro Komplex+Seed die finale
gedockte Pose (`x0_hat` + `lig_ref`, schon in echten Weltkoordinaten, keine
weitere Umrechnung nötig) als eigene `.sdf`-Datei, via `get_mol_from_coords`
+ `write_sdf`. Aufgerufen aus `save_results`, direkt nach `predictions.pt`.

Zwei Bugs beim ersten Entwurf gefunden und vom User selbst gefixt:
1. Tippfehler `from sigmadock.chem.parsing import write_sdf.` (Punkt am Ende) → Syntaxfehler.
2. Fehlendes `self`: Methode wurde zunächst als eigene Klassenmethode
   geschrieben (guter Instinkt — sauberer als die ursprünglich vorgeschlagene
   verschachtelte Funktion), aber ohne `self`-Parameter, obwohl über
   `self.export_predictions_to_sdf(...)` aufgerufen. Gefixt mit `self` als
   erstem Parameter (Alternative gewesen wäre `@staticmethod`, da der
   Funktionskörper `self` nie benutzt — User hat sich für `self` entschieden).

**Verifiziert mit echten Daten** (nicht nur Syntax-Check): reales
Ligand-Molekül (25 Atome) aus dem Dummy-Datensatz geladen, `export_predictions_to_sdf`
mit synthetischen `x0_hat`-Koordinaten (bekannter Offset vom Original)
aufgerufen, geschriebene `.sdf` zurückgelesen — Atomanzahl korrekt,
Koordinaten-Rundtrip-Fehler `2e-6` (reine Text-Rundung des SDF-Formats).
Dateiname korrekt saniert (`::` → `__`, Seed-Index enthalten, kein
Windows-Pfadproblem).

**Nebenbei gefunden und bereinigt:** eine nicht getrackte `denoiser.py`
tauchte während dieser Arbeit wieder auf — bestätigt (per `git show`-Diff
gegen den Stand direkt vor der Umbenennung) **byte-identisch** zur
vor-Rename-Version, reines Editor-Artefakt (alter Tab, beim Speichern neu
angelegt), keine verlorene Arbeit. Gelöscht.

### Nächstes Vorhaben (gerade besprochen): 2-3h-Überanpassungslauf auf ARC + Sampling zur visuellen Kontrolle in PyMOL

User-Ziel: auf den 10 Dummy-Komplexen für ~2-3h trainieren (länger/gründlicher
als die bisherigen ~300-Epochen-Tests aus PAUSE-PUNKT #3/#4), dann mit dem
neuen `scripts/sample.py` eine `.sdf` der gedockten Pose erzeugen und in
PyMOL ansehen, ob es passt.

**Geprüft, was dafür bereit ist:**
- Checkpointing läuft unabhängig von `--debug`
  (`scripts/train.py:293-319`: der Kommentar "Logging & Checkpointing if not
  in debug mode" ist irreführend — nur WandB-Login und zwei Debug-Callbacks
  hängen an `--debug`, `ModelCheckpoint` wird immer hinzugefügt,
  `save_last=True` + Top-3 nach `val_loss`). Checkpoint landet unter
  `experiments/sigmadock/<timestamp>/checkpoints/`.
- `slurm/train_dummy_overfit_gpu.sh` existiert schon und funktioniert
  (2× erfolgreich gelaufen, siehe PAUSE-PUNKT #3/#4) — muss nur auf
  2-3h/mehr Epochen skaliert werden.
- `scripts/sample.py` ist jetzt fertig (s.o.), `sampling_setup.py` unterstützt
  `experiment=dummy_train` direkt (dieselbe Config wie fürs Training) — kein
  Sonderfall nötig.
- Original-SigmaDock hat ein generisches `slurm/sample.sh`-Vorbild (keine
  Diffusions-Reste, reine Hydra-CLI-Aufrufe) — als Basis für ein
  `SigmaFlow_Development/slurm/sample_dummy.sh` portierbar.

**Wichtiger Hinweis fürs Sampling, damit der Test aussagekräftig ist:**
`use_true_vector_field=False` (Default) benutzen, NICHT `True` — `True`
umgeht das Netzwerk komplett (reiner ODE-ohne-Netzwerk-Selbsttest, siehe
PAUSE-PUNKT #5) und würde unabhängig vom Trainingserfolg "funktionieren".
Nur mit `False` testet man tatsächlich, ob das Netzwerk etwas gelernt hat.

**Erwartungshaltung zum "perfekt überfitten":** mit 10 Beispielen und einem
großen Netz ist starke Überanpassung sehr plausibel, aber "perfekt" (exakt
Null-Loss) ist auf 2-3h (~500-600 Epochen laut bisheriger Rate ~3.4
Epochen/Min auf GPU L40S) nicht garantiert — Runde 2 aus PAUSE-PUNKT #4
(300 Epochen, `rot_score_weight=2.0`) kam auf `loss_R=6.76`/`loss_trans=3.5`,
deutlich unter Baseline aber nicht Null. Erwartung: sichtbar bessere,
wahrscheinlich schon ziemlich passende Posen, aber realistisch einschätzen,
nicht "garantiert perfekt" versprechen.

### ✅ Beide fehlenden SLURM-Skripte gebaut und lokal verifiziert (soweit ohne ARC/GPU möglich)

- **`slurm/train_dummy_overfit_gpu_3h.sh`** (neue Datei, `train_dummy_overfit_gpu.sh`
  bleibt als Referenz für die 300-Epochen-Läufe unverändert): `--time=02:45:00`,
  `--max_epochs 700` (Rate-Schätzung aus Job 8177699/8182812: `~3.4` Epochen/Min
  auf GPU L40S → `~600` Epochen in 2.75h; `max_epochs` bewusst etwas höher
  gesetzt, damit das `--time`-Limit die tatsächliche Grenze ist, nicht ein zu
  früh erreichtes `max_epochs` — `ModelCheckpoint` mit `save_last=True`
  speichert so oder so den zuletzt abgeschlossenen Stand). `rot_score_weight`
  bei `2.0` belassen (bisher am besten getestet), leicht editierbar falls
  gewünscht.
- **`slurm/sample_dummy.sh`** (neue Datei, portiert von `SigmaDock/slurm/sample.sh`,
  generisches Vorlagen-Skript ohne Diffusions-Reste): `experiment=dummy_train`,
  `graph.sample_conformer=false` (nutzt die Bound-Pose-Fragmentierung wie beim
  Training, direktester Test auf Auswendiglernen), `postprocessing.scoring=null
  postprocessing.bust_config=null` (Vina/PoseBusters für den reinen Sichtcheck
  nicht nötig, spart potenzielle Fehlerquellen). `CKPT_DIR` als Pflicht-Env-Var
  (Pfad zum Checkpoint aus dem Trainingslauf).

  **Lokal verifiziert** (kein ARC/GPU nötig dafür): Hydra-Overrides lösen über
  `prepare_sampling_cfg` korrekt zu `cfg.model.ckpt_dir`/`cfg.data.data_dir`/
  `cfg.experiments.name` auf; `build_sampling_datafront` findet mit
  `experiment=dummy_train` + echtem `data_dir=notebooks` tatsächlich alle 10
  Dummy-Komplexe (`Datafront pairs: 10`, "Re-Docking"-Modus wie erwartet, da
  kein `reference_sdf`). Was **nicht** lokal testbar war: der eigentliche
  Sampling-Lauf selbst (braucht einen echten trainierten Checkpoint + GPU).

**Nächster konkreter Schritt:** User führt beide Skripte auf ARC aus
(`sbatch slurm/train_dummy_overfit_gpu_3h.sh`, danach — sobald ein Checkpoint
unter `experiments/sigmadock/<timestamp>/checkpoints/` liegt —
`CKPT_DIR=... sbatch slurm/sample_dummy.sh`), dann die geschriebenen `.sdf`-
Dateien unter `sampling_output/results/dummy_train/.../` in PyMOL ansehen.
`rot_score_weight` für diesen Testlauf ist mit `2.0` vorbelegt, bei Bedarf in
`train_dummy_overfit_gpu_3h.sh` direkt editierbar (eine Zeile).

## 🔖 PAUSE-PUNKT #6 (2026-07-21, später am selben Tag) — älter, siehe #7 oben für aktuellen Stand

**Auftrag dieser Runde:** vor dem großen Trainingslauf ausführlich prüfen,
ob wirklich alles funktioniert und was noch fehlt. Ergebnis: Code-seitig
alles verifiziert lauffähig, aber der große Lauf selbst ist noch nicht
startbereit (Infrastruktur/Konfiguration fehlt, keine Code-Bugs).

### ✅ Bestätigt funktionierend

- **Trainings-Pipeline**, frisch lokal getestet mit dem Code-Stand nach
  allen PAUSE-PUNKT-#5-Fixes (`scripts/train.py`, 5 Schritte, `dummy_train`,
  CPU, `--offline_run --debug`): läuft sauber durch, endliche Loss-Werte
  (`loss_train/total=8.96`, `loss_train/loss_R=8.81`,
  `loss_train/loss_trans=4.56`), kein NaN, kein Crash. Bestätigt: die
  `denoiser.py`-Änderungen aus PAUSE-PUNKT #5 (nur `_compute_true_vector_field`
  betroffen, wird beim Training nicht aufgerufen) haben den Trainingspfad
  nicht beeinträchtigt.
- Sampling-Pipeline: siehe PAUSE-PUNKT #5 (End-to-End-Test mit echten Daten,
  weiterhin gültig).
- Repo-weite Suche (`src/`, nicht nur die 5 Kerndateien) nach
  `self.diffuser`/`_so3_diffuser`/`IGSO3`/`forward_marginal`/
  `reverse_marginal`: keine weiteren funktionalen Diffusions-Reste gefunden.
  Einzige Fundstelle: eine veraltete Docstring-Erwähnung von
  `forward_marginal` in `denoiser.py` (Zeile ~652, Kommentartext, kein Code)
  — rein kosmetisch, nicht behoben (nicht dringend).

### ⚠️ Neuer Fund: totes/fehlgeleitetes Config-Feld (`rot_score_method`/`rot_score_scaling`)

`conf/training/slurm.yaml` (Referenz-Config für den großen Lauf) setzt
`rot_score_method: space` und `rot_score_scaling: rms`. Der
`SigmaDockDenoiser`-Konstruktor erwartet aber `rot_vector_field_method`/
`rot_vector_field_scaling` (bei der Flow-Matching-Konversion umbenannt,
`config.py`s `RunConfig`-Dataclass-Felder und die `--rot_score_method`-CLI-
Flag in `config.py` wurden dabei nie mit umbenannt).

Da `scripts/train.py:200-212` die komplette Config per `**args.__dict__` in
den `SigmaDockDenoiser`-Konstruktor entpackt, landet `rot_score_method` im
`**kwargs`-Auffangbecken (`denoiser.py`s `__init__`, "ignored but can be
used for future extensions") und wird **stillschweigend ignoriert** — der
Denoiser nutzt immer seinen Default `rot_vector_field_method="vector_field"`,
unabhängig vom Config-Wert.

**Zweite, unabhängige Bestätigung, dass das ohnehin folgenlos ist:**
`self.rot_vector_field_method`/`self.rot_vector_field_scaling` werden in
`__init__` nur validiert und gespeichert, aber **nirgends sonst im ganzen
`denoiser.py` abgefragt** — die "space"-Alternative (im Original-
Diffusions-Code eine echte, andere Berechnung) wurde beim Umbau nie für
Flow Matching implementiert. Der Parameter ist komplett wirkungslos, ganz
unabhängig vom Namens-Mismatch.

**Einordnung:** nicht blockierend — das tatsächlich laufende Verhalten
(`vector_field`-Modus, einzige real implementierte Option) ist exakt das,
was in allen bisherigen Tests (Überanpassung PAUSE-PUNKT #4, heutiger
Smoke-Test) bereits erfolgreich validiert wurde. Aber die Config täuscht
eine nicht existierende Wahlmöglichkeit vor. **Entscheidung mit User noch
nicht getroffen:** Feld aus `config.py`/`conf/training/slurm.yaml`/CLI
entfernen (ehrlicher) oder `rot_vector_field_method`/`_scaling` als echten,
funktionierenden Kwarg-Pfad nachrüsten (mehr Aufwand, nur nötig falls
"space"-Modus je gebraucht wird)?

### ❌ Bestätigt: fehlt noch für den großen Lauf (Infrastruktur, keine Code-Bugs)

1. **Kein SLURM-Skript für den großen Lauf.** `conf/training/slurm.yaml`
   verweist im Kopfkommentar auf `slurm/train.sh` — diese Datei existiert
   nicht. `slurm/` enthält bisher nur `train_dummy_test.sh` und
   `train_dummy_overfit_gpu.sh` (beide klein, Dummy-Datensatz).
2. `rot_score_weight` final festlegen (weiterhin offen, siehe PAUSE-PUNKT
   #4: `0.5` Baseline, `2.0` im Diagnose-Test besser für Rotation aber
   schlechter für Translation, `1.0` als ungetesteter Mittelweg
   vorgeschlagen).
3. Partition/Zeitlimit für einen Mehrtages-Job auf ARC — weiterhin nicht
   geklärt (Partition `short` ist auf Stunden gedeckelt, User kennt die
   richtige Partition, muss nur noch besprochen werden).
4. `--offline_run` vs. echtes W&B-Online-Logging (bräuchte API-Key-Setup)
   für den "richtigen" Lauf — weiterhin nicht entschieden.
5. **Nicht lokal prüfbar, braucht ARC-Zugriff:** ob die echten
   Datensatz-Ordnerstrukturen (`/data/stat-cadd/shug8458/data/{pdbbind,
   astex,posebusters}`) tatsächlich zu den Regex-Mustern in
   `conf/experiments/pdbbind-general.yaml` (`dataset: "pdbbind/general-set/"`,
   `pdb_regex: ".*pocket\.pdb$"`, `sdf_regex: ".*ligand.*\.sdf$"`) und den
   Geschwister-Configs (`pdbbind-refined`, `pdbbind-core`, `posebusters`,
   `astex`) passen. Configs selbst wurden gelesen und sehen intern
   konsistent aus, aber ohne Zugriff auf die echten ARC-Ordner nicht
   verifizierbar.

### ✅ Nachtrag (noch selbe Session): `SigmaDockDenoiser` → `SigmaFlowGenerator` umbenannt

User-Frage: "denoiser.py" beinhaltet ja gar kein Denoising mehr, passender
Name? Klasse tut heute zweierlei: (a) generische Starrkörper-/Graph-Infra
(Fragment-Masse/Trägheitstensor, COM/Rotation, Kraft→Drehmoment via
`linear_mechanics`/`newton_maruyama`, Graph-Kanten pflegen), (b) die
eigentliche Flow-Matching-Prozess-Orchestrierung (Zeit sampeln, vom
`flow_matcher` interpolieren lassen, Netzwerk aufrufen, Kraft/Drehmoment in
Vektorfeld übersetzen, Loss berechnen). "Denoiser" trifft nur noch den
historischen Diffusions-Teil.

Checkpoint-Kompatibilität vorher geprüft und bestätigt unkritisch: Laden
läuft über `state_dict` (Parameter-Namen), nicht über volles
Objekt-Pickling — ein reiner Rename bricht keine bestehenden/künftigen
Checkpoints.

**Neuer Name: `SigmaFlowGenerator`** (passt zur bestehenden `Sigma*`-
Konvention im Rest des Codes — `SigmaDataset`, `SigmaDataModule`,
`SigmaLightningModule` —, kollidiert nicht mit `R3_FlowMatcher`/
`SO3_FlowMatcher`/`SE3_FlowMatcher`, da die Klasse deren Objekt über
`self.flow_matcher` benutzt statt selbst eine zu sein).

**Umgesetzt (7 Fundstellen, alle geprüft — kein `SigmaDockDenoiser`/
`diff.denoiser`-Rest mehr im ganzen `SigmaFlow_Development`-Baum):**
- `src/sigmadock/diff/denoiser.py` → `sigma_flow_generator.py` (`git mv`,
  Historie erhalten), Klasse + Docstrings/Kommentare/`__repr__` angepasst.
- Aufrufer aktualisiert: `sampling.py`, `trainer.py`, `utils.py`,
  `scripts/train.py`, `04_diffusion.ipynb`, `05_crossdock_sampling.ipynb`.
- Bewusst NICHT angefasst: lokale Variablen-/Parameternamen wie `denoiser`
  (z.B. `def sample_notebook(denoiser: SigmaFlowGenerator, ...)`) — nur der
  Klassen-/Dateiname war irreführend, nicht jeder Variablenname, der eine
  Instanz davon hält. Kleinstmögliche Änderung, kein Scope-Creep.

**Verifiziert:** Imports laden sauber, CPU-Dummy-Trainingslauf (5 Schritte)
läuft nach dem Rename bit-identisch zu vorher (`loss_train/total=8.96053`,
gleicher Seed) — kein Crash, keine Verhaltensänderung.

### 🔜 Nächstes großes Vorhaben (besprochen, noch nicht begonnen): `scripts/sample.py` + SDF-Export

User will: auf den 10 Dummy-Proteinen trainieren, dann eine `.sdf`-Datei der
gedockten Pose bekommen (zum Anschauen in PyMOL). Rechercheergebnis:

- Original-SigmaDock speichert beim Sampling **keine** SDF, sondern
  `predictions.pt` (rohe Tensoren: `x0`, `x0_hat`, `trajectory`, `lig_ref`-
  RDKit-Mol) — `SamplingModule.save_results` in `SigmaDock/scripts/sample.py`.
  Die Bausteine für einen echten SDF-Export existieren schon, sind aber nie
  verbunden: `sigmadock.chem.statistics.get_mol_from_coords(coords, ref_mol)`
  (baut ein RDKit-Mol mit den vorhergesagten Koordinaten) +
  `sigmadock.chem.parsing.write_sdf(mols, path)` (fertiger `Chem.SDWriter`-
  Wrapper).
- `SigmaFlow_Development/scripts/sample.py` **existiert komplett noch
  nicht** (nur `train.py` liegt in `scripts/`). Alle Abhängigkeiten dafür
  sind aber schon vorhanden und diffusionsfrei: `sampling_setup.py` (rein
  generischer Hydra-/Datafront-Code, geprüft), `SampleCycleWrapper`
  (`core/data.py`), `compute_gnina_score` (`chem/postprocessor.py`),
  `compact_posebusting` (`chem/statistics.py`), `conf/sampling/base.yaml`.
- **Entscheidung getroffen:** volles `scripts/sample.py` von SigmaDock
  portieren (nicht nur ein Mini-Skript), da für PoseBusters im großen Stil
  ohnehin gebraucht wird. Zwei mechanische Fixes nötig beim Portieren
  (gleiches Muster wie die Notebook-Fixes): `denoiser.diffuser._so3_diffuser
  .set_device(...)` entfernen, `use_true_scores` → `use_true_vector_field`
  (auch in `conf/sampling/base.yaml`: `use_true_scores: false` →
  `use_true_vector_field: false`).
- **SDF-Export-Design (besprochen, User zugestimmt):** in
  `SamplingModule.save_results` integrieren (dort, wo `predictions.pt`
  entsteht, gleiche Stelle wie künftiges Vina-Scoring/PoseBusters), nur die
  finale Pose (`x0_hat` + `lig_ref`), keine ganze Trajektorie (reicht für
  "passt die Pose" in PyMOL, Trajektorie wäre optionale spätere Erweiterung).
- **Noch nicht begonnen:** weder das mechanische Portieren von `sample.py`
  noch die neue SDF-Export-Funktion selbst.

### Nächste Schritte (mit User zu klären, Priorität von oben nach unten)

1. `scripts/sample.py` portieren + SDF-Export-Funktion bauen (s.o., als
   Nächstes geplant).
2. Entscheidung zum toten `rot_score_method`-Feld (PAUSE-PUNKT #6 oben).
3. `rot_score_weight` final festlegen.
4. Partition/Zeitlimit klären, dann `slurm/train.sh` für den großen Lauf
   schreiben (Basis: `conf/training/slurm.yaml`, analog zu den
   Dummy-Skripten aber mit `--strategy ddp --devices 4` o.ä. und
   Mehrtages-`--time`).
5. W&B-Logging-Entscheidung.
6. Datensatz-Pfade/Regex auf ARC verifizieren (User oder nächste Session
   mit ARC-Zugriff).

---

## 🔖 PAUSE-PUNKT #5 (2026-07-21) — älter, siehe #6 oben für aktuellen Stand

**Der Sampling-Pfad (kritischer Fund aus PAUSE-PUNKT #3, "komplett kaputt")
ist in dieser Session vollständig repariert und Schritt für Schritt
verifiziert worden.** User hat jede Einheit selbst geschrieben (Lehr-Workflow
aus CLAUDE.md §6 durchgezogen), Claude hat jeweils Interface/Mathematik erklärt,
Review gemacht und danach lokal getestet. Reihenfolge und Ergebnisse:

### 1. `calc_trans_vector_field` (`R3_FlowMatcher`, `src/sigmadock/diff/r3_flow_matcher.py`)

Neue Methode: `u_t = (x_1 - x_t) / (1-t)` — hergeleitet aus der bereits
verifizierten `conditional_probability_path`-Interpolation (auflösen nach
`x_0`, einsetzen in `u_t = x_1 - x_0`). Liefert auf dem exakten Pfad dasselbe
Feld wie das Trainingsziel, korrigiert aber zusätzlich selbstständig in
Richtung `x_1`, falls `x_t` (z.B. durch Integrationsfehler) nicht exakt auf
der Geraden liegt — genau die Eigenschaft, die für ODE-Sampling gebraucht
wird. Singularität bei `t=1`, deshalb iteriert `sampling.py` bewusst nie über
`t=t_max=1.0` selbst.

Lokal getestet (4 Fälle, alle bestanden): Pfad-Konsistenz (`3e-7`
Abweichung), Selbstkorrektur von gestörtem `x_t` aus (ein Euler-Schritt mit
`dt=1-t` trifft `x_1` exakt), Endlichkeit nahe `t=1` (`t=0.9999`, groß aber
kein `NaN`/`Inf`), Shape-Check.

### 2. `calc_rot_vector_field` (`SO3_FlowMatcher`, `so3_flow_matcher.py`)

Analog auf SO(3): `ω = log(R_t^T R_1) / (1-t)`, rechts-trivialisiert
(konsistent mit der bestehenden `conditional_probability_path`-Konvention).
Herleitung: `R_1 = R_t · exp((1-t)·ω)` nach `ω` aufgelöst.

Getestet (5 Fälle): Pfad-Konsistenz (Restfehler bis `~7e-4` bei Rotationspaaren
nahe `ω≈172-173°` — **kein neuer Bug**, sondern der bereits in PAUSE-PUNKT #3
dokumentierte, akzeptierte Präzisionsrest von `so3_utils` nahe `ω=π`;
verifiziert durch Diagnose: Fehler korreliert exakt mit dem Rotationswinkel
pro Sample), Selbstkorrektur (`8e-7`), Endlichkeit nahe `t=1`, Schiefsymmetrie
von `u_t_R` (exakt `0`), Shape-Check.

### 3. `calc_vector_field` (`SE3_FlowMatcher`, `se3_flow_matcher.py`)

Komponiert die beiden obigen zu `{"u_t_trans": ..., "u_t_R": ...}` (gleiche
Schlüssel wie `conditional_probability_path`, damit `denoiser.py` sie direkt
weiterverwenden kann). Erster Entwurf gab zusätzlich `trans_t`/`R_t` unverändert
zurück (Kopie der Eingabe, kein Mehrwert) — auf Review hin entfernt.

Getestet: Pfad-Konsistenz beider Komponenten, Shape/Keys, **volle 20-Schritt-
ODE-Trajektorie** von zufälligem Startzustand bis `t=1` ausschließlich mit
`calc_vector_field`+`euler_step` — landet praktisch exakt bei `trans_1`/`R_1`
(`trans_error=0.0`, `R_error=1.4e-6`). Stärkster Beleg, dass Formel und
Verdrahtung stimmen.

### 4. `denoiser.py::_compute_true_vector_field` (Zeile ~995-1016)

Der ursprüngliche Crash-Grund aus PAUSE-PUNKT #3: `self.diffuser` (existiert
nicht mehr, umbenannt zu `self.flow_matcher`) und die nie implementierten
`calc_trans_vector_field`/`calc_rot_vector_field`. Jetzt ein einziger Aufruf:
`return self.flow_matcher.calc_vector_field(Tt, Rt, trans_1, R_1, t_batch)`.

**Wichtige Klarstellung, die im Zuge dessen mit dem User besprochen wurde:**
Diese Methode wird **nicht** für echte, blinde Posen-Generierung benutzt (dafür
kennt `_compute_vector_field`, unverändert, nur Netzwerk-Output, nie `R_1`).
`_compute_true_vector_field` dient nur (a) dem Redocking-Diagnose-Logging
(`compute_losses` pro Schritt, `R_1` aus bekannter Testpose) und (b) dem
expliziten `use_true_vector_field=True`-Debug-Modus (Netzwerk komplett
umgangen, reiner ODE-Selbsttest). Im Normalfall (`use_true_vector_field=False`,
Default) fließt das wahre Feld nirgends in die generierte Trajektorie ein —
kein Datenleck.

Getestet mit einer echten `SigmaDockDenoiser`-Instanz (Stub-Module für
`torch_geometric`/`rdkit`-abhängige Importe, da lokal nicht installiert, s.u.):
Rückgabe-Keys, Pfad-Konsistenz beider Feld-Komponenten (`3.6e-7` /
`3.1e-4`), Shapes — alle bestanden.

### 5. `sampling.py`: `dt`-Vorzeichenfehler behoben (NEUER, unabhängiger Fund)

Beim Testen der obigen Fixes fiel auf: `dt = timesteps[i] - timesteps[i+1]`
(beide Sampling-Funktionen, `sample_notebook` Zeile ~162, `sampler` Zeile
~391) ist bei der jetzt **steigenden** Zeitfolge (`t_min→t_max`, Flow-Matching-
Konvention `0→1`) **negativ** — ein Überbleibsel der alten Diffusions-
Konvention (dort lief die Zeit `1→0`, da war dieselbe Formel korrekt).
Empirisch bestätigt: mit der alten Formel divergiert eine 18-Schritt-
Integration mit dem *wahren* Feld komplett (`trans_error=48.7`,
`R_error=1.8`, praktisch eine zufällige Endrotation) statt bei `trans_1`/`R_1`
anzukommen. Fix: `dt = timesteps[i+1] - timesteps[i]`. Nach dem Fix:
`trans_error=0.0`, `R_error=1e-6` (Formel direkt aus der Datei extrahiert und
verifiziert, nicht nur eine Kopie getestet).

Dieser Bug war in PAUSE-PUNKT #3 **nicht** dokumentiert (die dortige
Verifikation "keine Diffusions-Konventions-Leckage gefunden" bezog sich nur
auf den Trainingspfad, nicht auf `sampling.py`) — reiner Zufallsfund beim
End-to-End-Testen der neuen `calc_vector_field`-Kette.

### 6. Notebook-Bugs behoben (`04_diffusion.ipynb`, `05_crossdock_sampling.ipynb`, `extensions/sampling.ipynb`)

Der in PAUSE-PUNKT #3 als "zweiter, unabhängiger Crash-Punkt" notierte
`denoiser.diffuser._so3_diffuser.set_device(device)`-Aufruf war beim
genaueren Hinsehen nur einer von **zwei** Blockern pro Notebook:

- `denoiser.diffuser._so3_diffuser.set_device(device)` /
  `ema_model.model.diffuser._so3_diffuser.set_device(device)`: obsoleter
  Diffusions-Device-Cache-Aufruf, ersatzlos entfernt (`SE3_FlowMatcher`/
  `SO3_FlowMatcher` cachen kein Device, jeder Aufruf bekommt es als Parameter
  oder erbt es vom Input-Tensor — kein Ersatz nötig).
- `sample_notebook(..., use_true_scores=...)` / `sampler(..., use_true_scores=...)`:
  falscher Parametername (Diffusions-Rest), aktuelle Signatur heißt
  `use_true_vector_field`. Hätte selbst nach Fix des ersten Bugs sofort mit
  `TypeError: unexpected keyword argument` gecrasht. In allen drei Notebooks
  umbenannt (4 Fundstellen: `04` ×2, `05` ×1, `extensions/sampling.ipynb` ×1).

**Zusätzlicher, bisher nicht dokumentierter dritter Fund** (nur in `04` und
`05`, im jeweiligen "kein Checkpoint gefunden"-Fallback-Zweig): der manuelle
`SigmaDockDenoiser(model=..., include_interactions=False, cache_path=CACHE)`-
Aufruf übergibt kein `sigma_min` — Pflichtparameter ohne Default
(`TypeError: missing 1 required positional argument`). Betrifft nur den Pfad
ohne trainiertes Checkpoint (aktuell der einzig mögliche Zustand, da der große
Trainingslauf noch nicht stattgefunden hat) — behoben mit `sigma_min=0.0`
(konsistent mit der Konvention aus `R3_FlowMatcher`/`SE3_FlowMatcher` in
dieser Session).

**Bewusst NICHT angefasst:** Notebook `04` enthält zusätzlich einen älteren,
in sich abgeschlossenen Demo-Abschnitt (erste ~6 Zellen), der die alten
Diffusions-Klassen (`R3Diffuser`, `SE3Diffuser`, `SO3Diffuser`) direkt
importiert und deren *Vorwärts*-Prozess visualisiert (Noise-Schedules,
`forward_marginal`) — komplett unabhängig von `sample_notebook`/`denoiser.py`,
keine Fehlerquelle für den reparierten Sampling-Pfad. Diesen Abschnitt auf
Flow-Matching umzustellen (oder zu entfernen) wäre eine separate,
größere Entscheidung (Inhalt/Umfang), nicht Teil des "Bug ausmerzen"-Auftrags
dieser Session — bei Bedarf gesondert besprechen.

### Methodische Notiz: lokales Testen ohne vollen Stack

Wichtig für künftige Sessions: die in PAUSE-PUNKT #3 gemachte Notiz "volle
sigmadock-Umgebung ist auch lokal installiert" **trifft aktuell nicht mehr
zu** — auf dieser Maschine gibt es nur eine Anaconda-Base-Umgebung
(`C:\Users\julia\anaconda3`, `torch==2.10.0+cpu`), **ohne** `rdkit`,
`torch_geometric`, `pytorch_lightning`. Systemweite Suche bestätigt: keine
andere Konfiguration mit diesen Paketen lokal vorhanden. `KMP_DUPLICATE_LIB_OK=TRUE`
muss gesetzt sein, sonst crasht schon der `torch`-Import (OMP-Konflikt).

Alle Tests dieser Session liefen trotzdem lokal (ohne ARC), indem gezielt nur
die tatsächlich benötigten `sigmadock.diff.*`-Module per `importlib` direkt
geladen wurden (unter Umgehung von `sigmadock/__init__.py`, das eager alle
Submodule inkl. `rdkit`-Abhängigkeiten importiert) — für den
`denoiser.py`-Test zusätzlich mit Stub-Modulen für `torch_geometric`/
`sigmadock.chem.processing`/`sigmadock.oracle` (diese werden nur innerhalb
von Methodenkörpern gebraucht, nicht bei `__init__`/Klassendefinition, daher
funktioniert das sauber). Für einen echten End-to-End-Test von
`sample_notebook()` selbst mit einem echten `Batch`-Objekt (reale
Molekül-/Graphdaten) bräuchte man dagegen den vollen Stack — bisher nicht
gemacht, siehe "Nächste Schritte".

### ✅ Nachtrag (noch selbe Session): End-to-End-Test mit echten Daten erfolgreich

Fehlende Pakete lokal nachinstalliert (`rdkit`, `torch-geometric`,
`pytorch-lightning`, `biopython`, `e3nn`, `omegaconf`, `hydra-core`,
`posebusters` — alle in die Anaconda-Base-Umgebung, `pip install`, kein
Versions-Pinning in `pyproject.toml` vorgegeben). Hinweis: lokale
Python-Version ist `3.13.9`, `pyproject.toml` deklariert
`requires-python = ">=3.9, <3.13"` — die einzelnen Pakete liegen trotzdem
alle sauber, `import sigmadock` (das komplette Package, nicht mehr nur
gestubbte `diff`-Submodule) funktioniert jetzt ohne Workarounds.

Danach echten Dummy-Datensatz geladen (`SigmaDataset` über
`conf/experiments/dummy_train.yaml`, `notebooks/dummy_data/`, 10 Beispiele)
und `sample_notebook(..., use_true_vector_field=True)` — der eingebaute
ODE-Selbsttest, Netzwerk komplett umgangen — auf zwei echten Molekülen
laufen lassen:

- Sample 0 (`1G9V_RQ3`): Abweichung Liganden-Atome vs. Referenzpose
  `5.97 Å → 0.06 Å` nach 25 Schritten. Kein Crash.
- Sample 6 (anderes Molekül): `5.69 Å → 0.06 Å`. Kein Crash.

Wichtige Lektion beim ersten Versuch: `pos_t`/`all_pos` liegen in einem
skalierten, taschen-zentrierten internen Koordinatensystem — für einen
fairen Vergleich mit `ref_pos` muss man erst zurücktransformieren
(`pos_t * HPARAMS.general.dimensional_scale + pocket_com`, exakt wie am Ende
von `sample_notebook` selbst schon vorgerechnet, aber nicht zurückgegeben).
Ohne diese Rücktransformation sieht es fälschlich nach Nicht-Konvergenz aus.

`use_true_vector_field=False` (echter, netzwerkgetriebener Pfad) ließ sich
mit einem einfachen `DummyModel`-Stub nicht sauber durchtesten (Stub hat
nicht die exakte Aufruf-Signatur/Rückgabeform des echten EquiformerV2) —
bewusst nicht weiterverfolgt, da dieser Pfad (`_compute_forces` →
Modell → `_compute_vector_field`) in dieser Session unverändert blieb und
bereits durch die erfolgreichen Trainingsläufe (PAUSE-PUNKT #3/#4, echtes
Modell, echte Backward-Pässe über 300 Epochen) unabhängig bestätigt ist.

**Damit ist der Sampling-Pfad nicht nur fachlich, sondern auch praktisch auf
echten Daten verifiziert.** Punkt 1 der ursprünglichen "Nächste Schritte"
unten ist erledigt.

### Nächste Schritte (noch nicht final entschieden, mit User zu klären)

1. `rot_score_weight` für den großen Lauf final festlegen (siehe
   PAUSE-PUNKT #4, weiterhin offen).
2. Echtes SLURM-Skript für den großen (Mehrtages-)Trainingslauf schreiben
   (siehe PAUSE-PUNKT #2, weiterhin offen).
3. Optional, nicht dringend: den älteren Diffusions-Demo-Abschnitt in
   `04_diffusion.ipynb` (erste ~6 Zellen) auf Flow-Matching umstellen oder
   entfernen — bewusst in dieser Session nicht angefasst (s.o.).
4. Optional: lokale Umgebung jetzt vollständiger (siehe oben) — könnte
   künftige Sessions erleichtern, falls weiter lokal getestet werden soll,
   statt jedes Mal auf ARC zu warten.

---

## 🔖 PAUSE-PUNKT #4 (2026-07-17, später am selben Tag) — älter, siehe #5 oben für aktuellen Stand

**Zwei Dinge in dieser Sitzung erledigt, lokal committet/gepusht(?) noch zu
prüfen — siehe "Nächster Schritt" unten:**

### 1. Sample-4-Bug: echte Ursache gefunden (widerlegt alte Vermutung unten) + gefixt

Die Vermutung in PAUSE-PUNKT #3 ("RDKit ETKDGv3-Ligand-Embedding ist
stochastisch, `None` propagiert bis `.GetAtoms()`") war **falsch** — lokal
reproduziert (volle `sigmadock`-Umgebung ist auch lokal installiert, kein
ARC-Zugriff nötig) und dabei widerlegt. Repro-Skript: Dummy-Datensatz über
`SigmaDataset.__getitem__` in einer Schleife über alle 10 Beispiele × 40
Wiederholungen aufgerufen, mit `traceback.format_exc()` statt der (vorher
teils unterdrückten) Kurzmeldung.

**Echte Ursache:** Betrifft die **Proteintasche**, nicht den Liganden.
`read_pdb_from_string()` (`src/sigmadock/chem/parsing.py`) gibt bei einem
RDKit-Sanitisierungs-/Valenzfehler beim Parsen des Pocket-PDB-Blocks
absichtlich `None` zurück (Kommentar im Code: nicht-sanitisiertes Parsing
würde Bindungs-/Atom-Features zerstören — das war schon immer so gewollt).
Aber `parse_complex()` (`src/sigmadock/data.py`) hat dieses `None` nie
geprüft — es lief unbemerkt 3 Funktionsaufrufe weiter bis
`get_global_protein_graph()` mit `protein.GetAtoms()` abstürzte
(`AttributeError: 'NoneType' object has no attribute 'GetAtoms'`).

Betrifft **ausschließlich** `1R1H_BIR` (Dummy-Index 4), 15/40 Versuchen
schlugen fehl, alle anderen 9 Beispiele nie. Erklärt auch die beobachtete
Intermittenz: `pocket_distance_noise` verschiebt den Taschen-Cutoff
zufällig bei jedem Aufruf; je nachdem ob eine bestimmte valenz-brechende
Randresidue mit reinfällt oder nicht, schlägt die Sanitisierung fehl oder
nicht.

**Fix (3 Stellen, alle in `SigmaFlow_Development/src/sigmadock/`):**
- `chem/parsing.py::read_pdb_from_string`: totes, unerreichbares
  Fallback-`print`/Retry nach dem bewusst gesetzten `return None` entfernt
  (Aufräumen, kein Verhaltenswechsel — der Kommentar erklärt jetzt auch
  *warum* kein Fallback existiert).
- `data.py::parse_complex`: neue explizite Prüfung `if pocket_mol is None:
  raise ValueError(...)` direkt nach der Konvertierung, mit Dateipfad in
  der Fehlermeldung — Fehler wird jetzt an der eigentlichen Fehlerquelle
  sauber gemeldet statt drei Ebenen tiefer als kryptischer `AttributeError`.
- `data.py::__getitem__`s Exception-Handler: druckt jetzt **immer** die
  Kurzmeldung `[WARN] Sample {idx} failed: {e}. Skipping...` (vorher wurden
  `ValueError`s komplett stillschweigend verschluckt, siehe Fund in
  PAUSE-PUNKT #3); vollständiger Traceback zusätzlich nur bei
  `verbose=True` (Log-Rauschen bei langen Läufen vermeiden).

**Verhalten für Training unverändert:** Sample 4 wird weiterhin
übersprungen (wie vorher, `force_retry` default `False` in unseren
Skripten) — der Unterschied ist nur: sauberer, verständlicher Log-Eintrag
statt Absturz-Traceback aus einem völlig unrelated Modul. Verifiziert:
360 erfolgreiche `__getitem__`-Aufrufe über die anderen 9 Beispiele im
Test, keine neuen Fehler eingeführt.

### 2. `rot_score_weight` für Testlauf Runde 2 erhöht: 0.5 → 2.0

Kontext: Runde-1-Ergebnis (Job `8177699`, siehe PAUSE-PUNKT #3) zeigte
`loss_R` praktisch auf Zufalls-Baseline-Niveau (`≈10.75` vs. `≈10.6`
"sag-Null-voraus"), während `loss_trans` deutlich sank. `0.5` kompensiert
rechnerisch bereits exakt den strukturellen Faktor 2 aus der
schiefsymmetrischen Matrixnorm (siehe PAUSE-PUNKT #3, "Finding 1") — pro
Freiheitsgrad war Rotation also schon gleichgewichtet mit Translation im
Loss, hat aber trotzdem kaum gelernt. Mit dem User besprochen: bewusst
aggressiver Diagnose-Test (nicht die finale Kalibrierung fürs große
Training) — `2.0` (4× relativ zur bisherigen Parität) statt vorsichtigerem
`1.0`, um in einem kurzen 1.5–2h-Testlauf ein klares Signal zu bekommen, ob
mehr Gewicht überhaupt hilft.

**Umsetzung:** bewusst NICHT der globale Default in `trainer.py`/
`config.py` geändert (bleibt `0.5`, weiterhin die kalibrierte Ausgangslage
für zukünftige Läufe) — stattdessen `--rot_score_weight 2.0` explizit als
CLI-Flag in `SigmaFlow_Development/slurm/train_dummy_overfit_gpu.sh`
ergänzt (Job umbenannt zu `sigmaflow-overfit-gpu-test-rotw2`,
`--time=01:45:00` unverändert, passt bereits zur gewünschten 1.5–2h-Dauer,
war in Runde 1 kalibriert für exakt 300 Epochen auf GPU).

### ✅ Runde 2 gelaufen (2026-07-17, später): Job `8182812`, COMPLETED

Commit `f321dc9` gepusht (User hat auf ARC gepullt + `sbatch`t, kein
SSH-Zugriff für mich in dieser Umgebung, s.o.). Alle 300 Epochen sauber
durchgelaufen, kein Crash.

**Sample-4-Fix in echter ARC-Umgebung bestätigt:** `[WARN] Sample 4
failed: Failed to parse pocket into a valid RDKit Mol from .../1R1H_BIR_
protein.pdb (likely a sanitization/valence error in the extracted pocket
block).. Skipping...` erscheint sauber mehrfach im Log, kein Absturz, Job
läuft bis `max_epochs=300 reached` durch. Fix funktioniert wie erwartet.

**`rot_score_weight`-Vergleich, Runde 1 (`0.5`, Job `8177699`) vs. Runde 2
(`2.0`, Job `8182812`), Endwerte Epoche 299:**

| Metrik | Runde 1 (0.5) | Runde 2 (2.0) |
|---|---|---|
| `loss_train/loss_R` | 10.75 | **6.76** |
| `loss_val/loss_R` | 8.76 | **7.81** |
| `loss_train/loss_trans` | 2.12 | 3.50 |
| `loss_val/loss_trans` | 2.49 | 3.16 |

Sanity-Check bestanden: gewichtete Summe stimmt exakt
(`1.0×3.49711 + 2.0×6.76139 = 17.0199 == loss_train/total`).

**Interpretation:** Klarer Trade-off, kein Freifahrtschein. Rotation
lernt jetzt nachweislich (`loss_R` deutlich unter der Zufalls-Baseline
`≈10.6`, nicht mehr nur knapp darüber wie in Runde 1) — bestätigt, dass
das Problem in Runde 1 in erster Linie **Untergewichtung** war, keine
strukturelle Blockade der SO(3)-Flow-Matching-Implementierung.
Translation ist dafür bei fixem Trainingsbudget (300 Epochen) sichtbar
schlechter geworden (mehr Gradienten-Budget ging in Rotation). `2.0` ist
vermutlich zu aggressiv für den finalen Lauf — ein Mittelweg (z.B. `1.0`)
ist für den großen Lauf noch nicht getestet, aber naheliegender nächster
Schritt falls weiter kalibriert werden soll. Bewusst nicht weiter
automatisch durchprobiert (Zeit-/Kostenaufwand pro Testlauf, User-Wunsch
abwarten).

### Nächste Schritte (noch nicht final entschieden, mit User zu klären)

1. `rot_score_weight` für den großen Lauf final festlegen (z.B. `1.0` als
   Mittelweg testen, oder direkt mit `2.0`/`0.5` in den großen Lauf gehen
   und dort weiter beobachten) — Trade-off Rotation-vs-Translation im
   Hinterkopf behalten.
2. Sampling-Pfad reparieren (`_compute_true_vector_field` in
   `denoiser.py`/`sampling.py`, siehe PAUSE-PUNKT #3 für die hergeleiteten
   Formeln) — weiterhin nicht blockierend fürs Training, aber Pflicht vor
   jeglicher Posen-Generierung/PoseBusters-Arbeit.
3. Echtes SLURM-Skript für den großen (Mehrtages-)Trainingslauf schreiben
   (Partition/Zeitlimit für Mehrtages-Jobs noch mit User zu klären,
   `conf/training/slurm.yaml` als Basis, `--offline_run` vs. echtes
   W&B-Logging überdenken).

---

## 🔖 PAUSE-PUNKT #3 (2026-07-17) — älter, siehe #4 oben für aktuellen Stand

**Überanpassungs-Testlauf: BESTANDEN.** Job `8177699` (`sigmaflow-overfit-
gpu-test`, resubmittiert mit `--time=01:45:00`), Status `COMPLETED`,
Laufzeit `01:28:03` (kein Timeout, sauber zu Ende gelaufen, alle 300
Epochen). Log: `slurm_logs/8177699.out`.

**Verlustkurven-Befund (wandb Run-summary/Sparklines am Ende des Logs):**
- `loss_val/total`, `loss_val/loss_trans`, `loss_val/loss_R` zeigen alle
  einen **klaren Abwärtstrend** über die 300 Epochen (Sparklines gehen von
  `█`/`▇` am Anfang zu `▁`/`▂` am Ende). Das ist das gesuchte
  Überanpassungs-Signal: **das Modell kann lernen**, keine strukturelle
  Blockade in der Flow-Matching-Implementierung. **Grünes Licht für den
  großen Trainingslauf.**
- `loss_train/*` ist deutlich verrauschter (erwartbar bei nur 5
  Batches/Epoche auf 10 Beispielen, hohe Varianz pro Schritt).
  `loss_train/loss_trans` zeigt trotzdem einen erkennbaren Abwärtstrend
  gegen Ende; `loss_train/loss_R` ist die verrauschteste Kurve von allen,
  ohne klaren Trend.
- Finale Werte (Epoche 299): `loss_train/loss_R=10.75`,
  `loss_train/loss_trans=2.12`, `loss_train/total=7.50`,
  `loss_val/loss_R=8.76`, `loss_val/loss_trans=2.49`.

**Rigorose Ende-zu-Ende-Codeprüfung: ABGESCHLOSSEN (2026-07-17).** Komplett
unabhängig durchgeführt (nicht auf frühere Zusammenfassungen in dieser Datei
verlassen, sondern Code frisch gelesen und nachvollzogen). Ergebnis:

### ⚠️ WICHTIG: Datei-Pfade in dieser Datei sind veraltet

Die fünf Kerndateien liegen **nicht mehr** unter
`SigmaFlow_Development/src_sigmadock_diff/*_adapted.py` (wie in den
"Datei X/5"-Abschnitten weiter unten dokumentiert), sondern wurden zu einem
echten installierbaren Package umstrukturiert (bestätigt über den
Trainings-Log: `sigmadock loaded from: .../SigmaFlow_Development/src/
sigmadock/__init__.py`). **Aktueller, tatsächlich importierter Ort:**
- `src/sigmadock/diff/r3_flow_matcher.py`, `so3_flow_matcher.py`,
  `se3_flow_matcher.py`, `so3_utils.py` (die gepatchte Version, nicht das
  Original — Patch bestätigt vorhanden)
- `src/sigmadock/diff/denoiser.py` (der "adaptierte" Denoiser, ohne Suffix)
- `src/sigmadock/diff/sampling.py`
- `src/sigmadock/trainer.py`

Kein Bug, nur Doku-Rückstand (die Ordnerstruktur-Angleichung, die weiter
unten im Abschnitt "Infrastruktur-Lücke" noch als offen vermerkt ist, wurde
also irgendwann zwischen den Sitzungen erledigt, ohne dass diese Datei
nachgezogen wurde). Die Design-Entscheidungen/Formeln in den historischen
"Datei X/5"-Abschnitten weiter unten bleiben inhaltlich gültig (gegen den
aktuellen Code geprüft), nur die genannten Pfade/Dateinamen sind zu
korrigieren, falls dort referenziert.

### Trainingspfad: verifiziert korrekt

`t=0→t=1`-Konvention konsistent überall bestätigt, keine
Diffusions-Konventions-Leckage gefunden. Exakter Cross-Check gegen die
echten wandb-Zahlen: `trainer.py` gewichtet `total_loss = trans_score_weight
(1.0)·loss_trans + rot_score_weight (0.5)·loss_R`. Eingesetzt:
`1.0×2.12487 + 0.5×10.74963 = 7.499685` — trifft `loss_train/total=7.49969`
exakt. Trainingsschleife/Loss-Aggregation sind also nachweislich korrekt
verdrahtet, nicht nur "es sieht plausibel aus".

### 🔴 Kritischer, neuer Fund: Sampling-Pfad ist komplett kaputt

Ernster als der bisherige Vermerk unten ("scripts/sample.py hat denselben
Bug, nicht blockierend") — es gibt kein `scripts/sample.py`, der eigentliche
Blocker sitzt direkt in `denoiser.py`/`sampling.py`:
- `sampling.py` (`sample_notebook()` und `sampler()`) ruft in **jedem**
  ODE-Schritt unconditional `denoiser._compute_true_vector_field(...)` auf.
- Diese Methode ruft `self.diffuser.calc_trans_vector_field(...)`/
  `calc_rot_vector_field(...)` auf. `self.diffuser` wurde zu
  `self.flow_matcher` umbenannt (per Grep bestätigt: `self.diffuser`
  kommt nirgends sonst mehr vor) — UND `calc_trans_vector_field`/
  `calc_rot_vector_field` wurden **nie implementiert**, in keiner Datei.
- **Jeder** Aufruf von `sample_notebook()`/`sampler()` crasht sofort mit
  `AttributeError`. Zusätzlich haben alle drei Sampling-Notebooks
  (`04_diffusion.ipynb`, `05_crossdock_sampling.ipynb`,
  `extensions/sampling.ipynb`) einen zweiten, unabhängigen Crash-Punkt:
  `denoiser.diffuser._so3_diffuser.set_device(device)` (obsoleter
  Device-Cache-Aufruf).

**Fix braucht echte neue Mathematik, keine Umbenennung.** Hergeleitete
Formeln (aus der bereits verifizierten `conditional_probability_path`-Logik):
```
u_t_trans = (trans_1 - trans_t) / (1 - t)
u_t_R = so3_utils.log(R_tᵀ @ R_1) / (1 - t)
```
**Nicht blockierend fürs große Training**, aber muss vor jeglicher
Posen-Generierung/PoseBusters-Arbeit erledigt werden — sollte explizit
eingeplant werden, nicht als triviales Aufräumen behandelt.

### Finding 1 (Größenordnung loss_R vs. loss_trans) — GEKLÄRT

Drei sich überlagernde, alle bestätigte Ursachen:
1. **Struktur-Faktor 2:** `u_t_R`/`pred_u_t_R` sind schiefsymmetrische
   3×3-Matrizen; für `hat(v)` gilt `‖hat(v)‖²_F = 2‖v‖²` — Summe über alle 9
   Matrixeinträge zählt jeden der 3 echten Freiheitsgrade doppelt.
   Translation summiert sauber über 3 Komponenten. Reine Buchhaltung, kein
   Bug.
2. **Baseline-Vergleich:** Da `R_1` immer Identität ist (toter PCA-Code,
   siehe Datei-4/5-Abschnitt unten), ist `u_t_R = log(R_0ᵀ)`. Analytisch
   berechnete erwartete quadrierte Norm für Haar-uniforme Rotation:
   `E[ω²] = π²/3+2 ≈ 5.29`; mit Faktor 2 ergibt die "sag-Null-voraus"-
   Baseline `≈10.6` — nahezu identisch mit dem beobachteten `loss_R=10.75`
   nach 300 Epochen. **Das Modell hat sich bei Rotation kaum von der
   trivialen Baseline wegbewegt**, während Translation deutlich darunter
   liegt — Rotation lernt hier spürbar langsamer als Translation.
   Beobachtenswert für den großen Lauf, kein Blocker.
3. `rot_score_weight=0.5` (`trainer.py:27-29`, **byte-identisch** zum
   Original-Diffusions-Trainer übernommen) war gegen den alten, intern
   reskalierten Score-Loss kalibriert; diese Reskalierung wurde beim
   Flow-Matching-Umbau korrekt entfernt. Dass `0.5` den Faktor-2-Artefakt
   aus Punkt 1 ungefähr kompensiert, ist vermutlich Zufall, keine bewusste
   Neukalibrierung — bei Bedarf später neu austarieren.

### Finding 2 (Sample-4-Warnung) — Ursache mit hoher Konfidenz identifiziert

RDKits `ETKDGv3`-Konformer-Embedding in `ligalign.py` ist stochastisch
(Zufalls-Seed pro Versuch); für das spezifische Molekül in Sample 4
schlagen manchmal alle Embedding-Versuche fehl → `None` propagiert
ungeprüft weiter bis zu einem `.GetAtoms()`-Aufruf, der crasht. Erklärt
sowohl warum es **immer** Sample 4 ist (spezifische Molekülgeometrie) als
auch warum es **intermittierend** ist (abhängig vom Zufalls-Seed-Zustand
der jeweiligen Epoche). Datenpipeline fängt das sicher ab (`None` wird vor
dem Batching gefiltert, keine Korruption erreicht den Loss).

**Zusätzlicher neuer Fund dabei:** `data.py:556-558` — `ValueError`-Fehler
in derselben Pipeline werden komplett stillschweigend verschluckt, **ohne
jede Log-Ausgabe**. Die sichtbaren Sample-4-Warnungen könnten also nur ein
Teil der tatsächlichen Fehlerrate sein. Empfehlung: einmal
`traceback.format_exc()` statt `str(e)` loggen, um die exakte Zeile zu
fixieren (noch nicht gemacht).

### Kleinere Funde (nicht dringend)
- Toter Code in `ligalign.py` (`TorsionalOptimizer.optimize` ruft eine
  nicht-existente Methode `_calculate_rmsd` auf; `best_rmsd`-Vergleich in
  derselben Klasse ist durch Init-Bug quasi wirkungslos) — auf keinem
  aktiven Pfad (`data.py` nutzt `ConformerOptimizer.optimize_torsions`
  direkt, nie `TorsionalOptimizer.optimize()`).
- Vergessener `print(idx, pos_sel[atom_idx_sel].shape)`-Debug-Aufruf in
  `denoiser.py`s `get_fragment_com`, feuert bei jedem Forward-Pass —
  harmlos, aber Log-Rauschen.

### Gesamturteil des Audits

Trainingspfad: solide, verifiziert korrekt, grünes Licht für den großen
Lauf bestätigt. Sampling-Pfad: komplett kaputt, ernster als bisher
dokumentiert, muss vor jeglicher Posen-Generierung neu implementiert
werden (echte Mathematik, s.o.), aber nicht blockierend fürs Training
selbst.

### Nächste konkrete Schritte (mit User besprochen, noch nicht final entschieden welches zuerst)

1. `_compute_true_vector_field` in `denoiser.py`/`sampling.py` reparieren
   (echte neue Implementierung nötig, s.o. Formeln) — jetzt höher
   priorisiert als vorher angenommen.
2. Echtes SLURM-Skript für den großen Trainingslauf schreiben (blockiert
   nicht durch Punkt 1) — siehe Pfade/Fakten-Abschnitt unten.
3. Optional, nicht dringend: `rot_score_weight` neu kalibrieren, Sample-4/
   `ValueError`-Logging verbessern (`traceback.format_exc()`), Debug-`print`
   in `get_fragment_com` entfernen, toten Code in `ligalign.py` aufräumen.

---

## 🔖 PAUSE-PUNKT #2 (2026-07-16, später am selben Tag) — älter, siehe #3 oben für aktuellen Stand

**Gerade laufend/zu prüfen beim Wiedereinstieg:** Ein GPU+Überanpassungs-
Testlauf (`slurm/train_dummy_overfit_gpu.sh`, 300 Epochen auf den 10 Dummy-
Beispielen, Early Stopping deaktiviert) wurde auf ARC eingereicht, kurz
bevor die Sitzung pausiert wurde. **Job-Nummer wurde in der Sitzung nicht
mitgeteilt** — als erstes prüfen:
```bash
sacct -u shug8458 --format=JobID,JobName,State,Elapsed,Start,End -S today
```
Zeile mit `sigmaflo` (Jobname `sigmaflow-overfit-gpu-test`) suchen, `State`
prüfen (`COMPLETED`/`TIMEOUT`/`FAILED`/noch `RUNNING`/`PENDING`).

**Vorgeschichte dieses Testlaufs:** Ein erster Versuch (Job `8176776`) lief
mit `--time=01:00:00`, schaffte 208 von 300 Epochen (~15s/Epoche auf GPU),
wurde dann vom Zeitlimit gekillt (`TIMEOUT`, kein Absturz). Zeitlimit
daraufhin auf `01:45:00` erhöht (Hochrechnung: 300 Epochen × ~15s ≈ 86 Min
+ Anlaufzeit-Puffer) und neu eingereicht — **das ist der Job, dessen
Ergebnis jetzt geprüft werden muss.**

**Bereits bestätigt aus dem ersten (unvollständigen) Versuch — nicht
nochmal prüfen, nur den fehlenden Rest (vollständige Verlustkurve) holen:**
- ✅ GPU funktioniert: `CUDA available: True`, `device: NVIDIA L40S`,
  `sigmadock` korrekt aus `SigmaFlow_Development` geladen (nicht aus dem
  alten SigmaDock-Repo). Erster GPU-Testlauf unseres Flow-Matching-Codes
  war also erfolgreich.
- ⚠️ **Ungeklärter Fund, noch zu untersuchen (nicht blockierend):**
  Wiederholt (Epochen 18, 79, 168, 188 im ersten Versuch) tauchte
  `[WARN] Sample 4 failed: 'NoneType' object has no attribute 'GetAtoms'.
  Skipping...` auf — ein bestimmtes Dummy-Beispiel (Index 4) lässt sich
  gelegentlich nicht parsen (RDKit bekommt `None` statt Molekül-Objekt).
  Wird abgefangen (kein Absturz, Sample wird übersprungen), aber
  intermittierend (nicht bei jedem Zugriff), was auf eine echte, noch nicht
  verstandene Ursache hindeutet (Datenqualität in einer der `.sdf`-Dateien,
  oder eine Racebedingung/Nichtdeterminismus im Parsing-Code). **Vor dem
  großen Lauf anschauen, aber nicht zwingend blockierend**, da der Code
  robust genug ist, es zu überspringen.
- Vollständige Verlustkurve über alle 300 Epochen: noch nicht eingesehen
  (erster Versuch wurde vor Abschluss gekillt, `wandb`-Zusammenfassung
  erscheint nur bei sauberem Lauf-Ende; zweiter Versuch mit mehr Zeit sollte
  das liefern — **das als erstes beim Wiedereinstieg prüfen**, per `cat
  slurm_logs/<jobnummer>.out` am Ende, Abschnitt "Run summary"/Sparklines
  wie beim allerersten CPU-Erfolgslauf).

**Was als nächstes zu tun ist, sobald der Testlauf-Befund vorliegt:**
1. Verlustkurve über 300 Epochen einsehen — sinkt `loss_train`/`loss_val`
   jetzt klar (Überanpassung, wie erhofft)? Falls ja: Modell kann lernen,
   grünes Licht für den großen Lauf. Falls nein: genauer hinschauen, bevor
   eine Woche Rechenzeit investiert wird.
2. `Sample 4`-Parsing-Warnung untersuchen (siehe oben).
3. **Echtes SLURM-Skript für den großen Trainingslauf schreiben:**
   - Datensatz liegt bereits vor: `/data/stat-cadd/shug8458/data/{pdbbind,
     astex,posebusters}` (User hat das per `find` bestätigt) — `--data_dir
     /data/stat-cadd/shug8458/data` sollte funktionieren (Unterordner-Namen
     in `conf/experiments/pdbbind-core.yaml` etc. noch gegenchecken, ob sie
     zur tatsächlichen Ordnerstruktur unter `data/pdbbind/` passen — nicht
     abschließend verifiziert).
   - `conf/training/slurm.yaml` (existiert bereits, unverändert vom
     Original) als `--config`-Datei nutzen statt manueller CLI-Flags.
   - Reale Config sagt selbst "4-GPU DDP, 7-Tage-Lauf" — braucht andere
     Partition/Zeitlimit als `short` (das ist auf Stunden gedeckelt, nicht
     Tage). **User kennt Partitionsnamen, muss noch geklärt werden, welche
     für Mehrtages-Jobs passt — noch nicht besprochen.**
   - `--offline_run` vs. echtes W&B-Online-Logging für den "richtigen" Lauf
     überdenken (bräuchte W&B-API-Key-Setup, noch nicht besprochen).
   - Environment/Pfade sind bereits bekannt und funktionieren (siehe unten,
     "Wichtige Pfade/Fakten für ARC").
4. `scripts/sample.py`-Fix (später, nicht blockierend fürs Training) — hat
   noch denselben `.diffuser._so3_diffuser.set_device(...)`-Bug wie
   `trainer.py` vor dessen Fix.

**Wichtige Pfade/Fakten für ARC (damit nichts neu erfragt werden muss):**
- Projekt-Ordner: `/data/stat-cadd/shug8458/SigmaFlow_Development_JulianMueller/SigmaFlow/SigmaFlow_Development`
- Conda-Umgebung: `/data/stat-cadd/shug8458/sigmaflow_env` (separate Umgebung,
  Python 3.11.15, komplett getrennt vom alten `myenv`/SigmaDock — Grund:
  gemeinsame Umgebung hätte `sigmadock`-Namenskollision riskiert, siehe
  Meilenstein-Abschnitt weiter unten für die Details des dabei gefundenen
  Bugs).
- Python-Interpreter **immer über absoluten Pfad** aufrufen
  (`/data/stat-cadd/shug8458/sigmaflow_env/bin/python`), nicht über `PATH`/
  `python` nach `conda activate` — in einem `#!/bin/bash -l`-Batch-Skript
  hat sich `conda activate` als unzuverlässig erwiesen (siehe Meilenstein-
  Abschnitt).
- Echter Datensatz: `/data/stat-cadd/shug8458/data/{pdbbind,astex,
  posebusters}`.
- `slurm_logs/` muss vor jedem `sbatch`-Aufruf manuell existieren (`mkdir -p
  slurm_logs`), sonst schlägt der Job sofort fehl (SLURM öffnet die
  `--output`/`--error`-Dateien vor Skriptausführung).
- Partition `short`, GPU-Typ `l40s` (z.B. `--gres=gpu:l40s:1`).
- Nützliche Befehle: `squeue -u shug8458` (laufend/wartend), `sacct -u
  shug8458 --format=JobID,JobName,State,Elapsed,Start,End -S today`
  (Historie/Status nach Jobende), `cat slurm_logs/<jobnummer>.{out,err}`
  (Ergebnisse).

---

## 🔖 PAUSE-PUNKT #1 (2026-07-16, Ende der Sitzung) — älter, siehe #2 oben für aktuellen Stand

**Wo wir stehen:** Der komplette SigmaFlow-Trainingsloop läuft nachweislich
Ende-zu-Ende auf ARC (siehe Meilenstein-Abschnitt weiter unten für Details).
Wir haben gerade besprochen, dass der Validierungsverlust im Smoke-Test
**gestiegen** ist (nicht Überanpassung) — bei nur 15 Optimierungsschritten
auf 15M Parametern absolut erwartbar, keine Sorge, das war nie das Ziel
dieses Tests (siehe "Was der Smoke-Test NICHT zeigt" weiter unten).

**Nächste Schritte, in dieser Reihenfolge vorgeschlagen (mit User noch nicht
final vereinbart, nur besprochen):**

1. **Offene Schlüsselfrage, zuerst klären:** Liegt der große Datensatz
   (PDBbind general/refined/core, PoseBusters, Astex) auf ARC schon fertig
   vorbereitet (z.B. unter `/data/stat-cadd/shug8458/
   SigmaDock_Reproduction_JulianMueller/`, da User dort schon 2 Benchmarks
   mit dem Original-SigmaDock gefahren hat)? Falls ja: einfach `--data_dir`
   dorthin zeigen lassen (Datenlade-Pipeline ist unverändert von uns). Falls
   nein: eigene, potenziell große Aufgabe außerhalb unserer Kontrolle
   (Download/Registrierung bei PDBbind etc.) — **noch nicht mit User
   geklärt, unbedingt zuerst fragen.**
2. **GPU-Rauchtest** (empfohlen vor echtem Training): bisher NUR CPU
   getestet (lokal und auf ARC). Ob `R3_FlowMatcher`/`SO3_FlowMatcher`/
   `SE3_FlowMatcher`/`denoiser.py` auf einer echten GPU (Tensor-Geräte-
   Platzierung!) fehlerfrei laufen, ist unverifiziert. Vorschlag: Kopie von
   `slurm/train_dummy_test.sh` mit `--accelerator gpu --devices 1` und
   `#SBATCH --gres=gpu:l40s:1`, sonst identisch (weiterhin Dummy-Datensatz,
   winzig, günstig).
3. **Überanpassungs-Test** (User-Idee, sinnvoll): viele Schritte (Hunderte
   statt 15) auf den 10 Dummy-Beispielen, Early Stopping deaktivieren oder
   `--early_stopping_patience` groß setzen (aktuell `1/4`, ein *Verhältnis*
   zu `max_steps`, keine absolute Epochenzahl — Vorsicht bei der Anpassung).
   Zeigt, ob das Modell überhaupt lernen *kann* — bisher nicht getestet.
4. **Echtes SLURM-Skript fürs große Training**: nutzt `conf/training/
   slurm.yaml` (existiert unverändert, siehe `SigmaFlow_Development/conf/
   training/slurm.yaml`) statt manueller CLI-Flags wie im Dummy-Test.
   Referenz-Config selbst sagt "4-GPU DDP, 7-Tage-Lauf" — braucht andere
   Partition/Zeitlimit als `short` (typischerweise auf Stunden gedeckelt,
   nicht Tage) — **User kennt Partitionsnamen, muss zusammen geklärt
   werden, noch nicht besprochen welche Partition für Mehrtages-Jobs
   passt.** Auch: `--offline_run` vs. echtes W&B-Logging (bräuchte
   API-Key-Setup, noch nicht besprochen) für einen "richtigen" Lauf
   überdenken.
5. **`scripts/sample.py`** hat denselben `.diffuser._so3_diffuser.
   set_device(...)`-Bug wie `trainer.py` vor dessen Fix (siehe
   Infrastruktur-Lücke-Abschnitt weiter oben) — nicht blockierend fürs
   Training selbst, aber nötig bevor später Posen generiert/PoseBusters-
   Benchmarks gefahren werden sollen. Noch nicht angefasst.

**Zeitschätzung (mit User geteilt, Vorbehalt: hängt stark an Punkt 1):**
Falls Datensatz schon vorbereitet vorliegt: Schritte 2-4 zusammen vermutlich
**3-4 Stunden aktive Sitzungszeit** (SLURM/Environment/Config-Mechanik ist
inzwischen Routine, kein Neuland mehr). Die eigentliche Trainingslaufzeit
danach (Punkt 4) ist separat und unbeaufsichtigt — Referenz-Config nennt
**~7 Tage auf 4 GPUs** als vorgesehene volle Trainingsdauer, kein aktiver
Arbeitsaufwand für uns während dieser Zeit.

**Was der Smoke-Test NICHT zeigt (wichtig, nicht verwechseln):** Der
bisherige Erfolg beweist "Pipeline stürzt nicht ab, Shapes/NaN/Vorwärts-
Rückwärtspass korrekt" — NICHT "Modell lernt gut". `loss_val/total` ist im
Smoke-Test über 3 Epochen gestiegen (`▁▅█`), `loss_train/total` erst
gestiegen dann leicht gefallen (`▁█▅`) — bei nur 15 Optimierungsschritten
auf 15M Parametern und Hyperparametern, die für den großen Datensatz
kalibriert sind (nicht für 10 Beispiele), völlig erwartbar und kein
Alarmsignal. Lernfähigkeit wurde bewusst noch nicht getestet (siehe
Überanpassungs-Test-Vorschlag oben).

---

Letztes Update: 2026-07-16 (**Alle 6 Dateien strukturell fertig** —
`R3_FlowMatcher.py`, `SO3_FlowMatcher.py`, `SE3_FlowMatcher.py`,
`denoiser_adapted.py`, `sampling_adapted.py`, und neu **`trainer_adapted.py`**
(vormals `trainer.py`, als Datei 6 nachträglich in den Fahrplan aufgenommen,
siehe eigener Abschnitt weiter unten). Komplette Kern-Konversion von
SigmaDock zu SigmaFlow syntaktisch sauber und konsistent durchgetraced.
Fehlende Assets (`Jd.pt`, Dummy-Datensatz `.pdb`/`.sdf`) wurden aus dem
echten Repo nachgeladen. Nächster Schritt: aufgeschobene
Namensaufräumrunden in Dateien 4+5 (klein, nicht dringend), UND — deutlich
wichtiger — ein erster echter Lauf in einer Umgebung mit vollem
`torch_geometric`/`pytorch-lightning`-Stack, sobald verfügbar. Bisher wurde
NICHTS davon tatsächlich mit echten Tensoren ausgeführt, nur Syntax + Logik
manuell verifiziert. Ordnerstruktur-Angleichung (`src_sigmadock/` vs. echtes
`src/sigmadock/`) ist weiterhin NICHT gemacht — bleibt der letzte bekannte
Blocker vor einem echten Lauf, siehe Infrastruktur-Lücke-Abschnitt).

### Konkreter erster Schritt für die nächste Session

**Alle 5 Kern-Dateien sind strukturell fertig** (Details siehe die
einzelnen "Datei X/5"-Abschnitte weiter unten — jede Datei hat dort ihre
vollständige Bug-/Entscheidungs-Historie). Es gibt keine offene
Design-Entscheidung mehr, die vor dem Weitermachen geklärt werden müsste.
Drei mögliche Richtungen für die nächste Session, unentschieden, mit dem
User zu Beginn zu klären:

1. **Aufgeschobene Namensaufräumrunden** in `denoiser_adapted.py` und
   `sampling_adapted.py` — vollständige Liste jeweils am Ende der
   "Datei 4/5"- und "Datei 5/5"-Abschnitte ("Verbleibend, niedrige
   Priorität"). Rein mechanisch, kein Design-Risiko, kleinster möglicher
   nächster Schritt.
2. **Vorbereitung für einen echten Testlauf** — siehe Abschnitt
   "Infrastruktur-Lücke" weiter unten: Ordnerstruktur-Mismatch
   (`src_sigmadock/src_sigmadock_diff/` vs. echtes `src/sigmadock/diff/`)
   angleichen, Dummy-Datensatz-Strukturdateien (`.pdb`/`.sdf`) besorgen,
   eine Umgebung mit `torch_geometric`/`pytorch-lightning` finden oder
   aufsetzen. Das ist der einzige Weg, tatsächlich zu verifizieren, dass
   der bisher nur manuell durchgetracte Code wirklich funktioniert.
3. **Nichts von beidem, sondern gezielt nochmal mit frischem Kopf über den
   gesamten Trainings-/Sampling-Pfad drüberlesen** (Zeile für Zeile, wie
   bisher, aber als eigene Verifikations-Runde statt im Rahmen eines
   Bausteins) — zusätzliche Sicherheit vor einem echten Testlauf, falls der
   User das lieber zuerst hätte als Option 2 direkt anzugehen.

**Empfehlung:** Falls eine Umgebung mit vollem Stack ohnehin bald verfügbar
wird, lohnt sich Option 2 zuerst (das eigentliche Ziel — ein Testlauf —
deckt auch alle Reste aus Option 1/3 indirekt mit auf, sobald echte Fehler
beim Ausführen auftreten). Falls nicht, ist Option 1 der pragmatischste
Zeitvertreib. Keine davon ist dringend oder blockierend.

**Noch nicht geklärt (wichtig, sobald Fragment-Aggregation über einzelne
Punkte hinausgeht):** Unsere `R3_FlowMatcher`/`SO3_FlowMatcher`/
`SE3_FlowMatcher`-Bausteine sind für beliebiges `n` (flache Punktwolken)
gebaut — `denoiser_adapted.py` nutzt sie bereits erfolgreich mit `n=B×F`
(alle Fragmente aller Batch-Elemente geflattet), das passt also schon
nahtlos. Was noch offen ist: `euler_step` mit **Pro-Fragment-`dt`-Tensor**
(nicht nur skalares `dt`) — kommt spätestens in Datei 5 (Sampling-Loop mit
variablem `dt` pro Schritt) oder falls die Datenraum-Rekonstruktion
(`T0_hat`/`R0_hat`) doch nachgerüstet wird.

---

## Gesamt-Fahrplan (5 Dateien, aus dem initialen Dependency-Mapping)

1. **`r3_diffuser.py` → `R3_FlowMatcher.py`** ✅ FERTIG (einfachster Fall, siehe
   unten — Fragment-Aggregation kommt erst in `denoiser.py`)
2. **`so3_diffuser.py` → `SO3_FlowMatcher.py`** ✅ FERTIG (einfachster Fall,
   siehe unten)
3. **`se3_diffuser.py` → `SE3_FlowMatcher.py`** ✅ FERTIG (einfachster Fall,
   siehe unten)
4. **`denoiser.py` → `denoiser_adapted.py`** ✅ Trainingspfad fertig (siehe
   unten — kleine Aufräumreste, nicht blockierend)
5. **`sampling.py` → `sampling_adapted.py`** ✅ FERTIG — deterministische
   Euler-ODE-Integration ersetzt Reverse-SDE (siehe unten)

Bestätigt beim Dependency-Mapping (nicht erneut prüfen, außer der Code hat sich
seither geändert): **nur diese 5 Dateien referenzieren irgendwo im Repo die
Diffusions-Klassen** (`grep` über ganz `SigmaDock/` bestätigt). Kein anderes
Modul braucht einen strukturellen Umbau.

---

## Grundlegende Design-Entscheidungen (gelten für alle 5 Dateien)

1. **Namenskonvention:** Bewusst neue, Flow-Matching-eigene Namen statt
   Wiederverwendung der Diffusions-Begriffe (`R3FlowMatcher`/
   `R3_FlowMatcher` statt `R3Diffuser`, `conditional_probability_path` statt
   `forward_marginal`, kein `score` mehr sondern Vektorfeld `u_t`/`v_θ`).
2. **Zeitkonvention:** `t=0` = Quellverteilung/Rauschen, `t=1` =
   Zielverteilung/Daten. Inferenz-ODE integriert `0→1`. Das ist die
   Flow-Matching-Standardkonvention (Lipman et al.) und **umgekehrt** zur
   ursprünglichen SigmaDock-Diffusionskonvention (dort: `t=0` Daten, `t=1`
   Rauschen, Reverse-SDE integriert `1→0`). Wichtig: das muss konsistent in
   alle 5 Dateien durchgezogen werden, sobald wir dort ankommen.
3. **Gewählter Pfad:** lineare/conditional-OT-Interpolation,
   `x_t = (1-t)·x₀ + t·x₁`, Zielvektorfeld `u_t = x₁ - x₀` (konstant in t).
   `sigma_min` wird im Konstruktor gespeichert, aber **noch nicht benutzt**
   (offen: brauchen wir überhaupt eine "verrauschte" OT-Variante, oder bleibt
   es bei der reinen linearen Interpolation mit `sigma_min=0`?).

## Wichtige Architektur-Erkenntnisse aus dem Mapping (Kontext für später)

- `denoiser.py` (Original) mischt diffusionsspezifische Logik mit generischer
  Graph-/Starrkörper-Infrastruktur, die **erhalten bleibt**: Fragment-COM/
  Rotation, Massen/Trägheitstensor, `linear_mechanics` (Kraft→Drehmoment),
  Graph-Update/Edge-Pruning. Ersetzt werden nur `_sample_diffusion`,
  `_compute_scores`, `compute_losses` (die SDE/Score-spezifischen Teile).
- `SE3Diffuser` greift von `denoiser.py` aus direkt auf `_r3_diffuser`/
  `_so3_diffuser` zu (Underscore-Attribute, aber de facto Teil der
  Schnittstelle) — dieses Kompositionsmuster beim Neubau von
  `se3_diffuser.py` beibehalten.
- `so3_diffuser.py`'s `d_log_f_d_omega` (IGSO3-Dichte-Ableitung) ist zu 100%
  diffusionsspezifisch und wird für Flow Matching **nicht** gebraucht.
- `so3_utils.py` (hat/vee/log/exp/expmap/Omega/regularize) ist generische
  Lie-Theorie-Infrastruktur, bleibt unverändert wiederverwendbar.
- `model#Der Equiformer.py` und `timestep_embedder.py` nehmen bereits
  kontinuierliches `t ∈ [0,1]` entgegen — keine Änderung absehbar.
- `sigmadock.oracle.HPARAMS` und ein eventuelles Lightning-Trainer-Modul
  liegen **nicht** in diesem Repo — Checkpoint-/Config-Kompatibilität kann
  nicht verifiziert werden, das bleibt eine offene Unsicherheit.

---

## Datei 1/5: `SigmaFlow_Development/src_sigmadock_diff/R3_FlowMatcher.py` ✅

### Fertig und verifiziert (Code wurde tatsächlich ausgeführt, nicht nur gelesen)

```python
class R3_FlowMatcher:
    def __init__(self, sigma_min: float):
        """
        sigma_min: reserved for future noised OT path, currently unused —
        the linear path below is deterministic.
        """
        self.sigma_min = sigma_min

    def sample_init(self, n: int, device: str) -> torch.Tensor:
        return torch.randn(n, 3, device=device)

    def conditional_probability_path(self, x_1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = x_1.shape[0]
        device = x_1.device
        x_0 = self.sample_init(n, device)
        x_t = (1 - t[:, None]) * x_0 + t[:, None] * x_1
        u_t = x_1 - x_0
        return x_t, u_t

    def euler_step(self, x_t: torch.Tensor, v_t: torch.Tensor, dt: float) -> torch.Tensor:
        x_next = x_t + v_t * dt
        return x_next
```

Verifiziert:
- Shapes korrekt (`[n,3]`), Randzeit-Check `t=1 → x_t == x_1` (`torch.allclose`
  True).
- `euler_step`: `dt=1` ab `x_0` mit `v_t=u_t` liefert exakt `x_1` (kein
  Diskretisierungsfehler, weil `u_t` konstant in `t` ist — Spezialfall der
  linearen Interpolation, gilt NICHT mehr sobald `v_θ` vom Netz kommt).
  `dt=0` ist Identität. Zwei Halbschritte == ein Vollschritt (bestätigt
  Fehlerfreiheit bei konstantem Feld). Shape/dtype/device werden von `x_t`
  geerbt, kein eigener `device`-Parameter nötig (anders als `sample_init`,
  das einen neuen Tensor aus `torch.randn` erzeugt).
- `sigma_min`: Entscheidung getroffen — **bleibt als dokumentierter,
  aktuell ungenutzter Platzhalter** für einen möglichen späteren verrauschten
  OT-Pfad (nicht entfernt, siehe Docstring oben).

Design-Entscheidung protokolliert: `euler_step` ist bewusst eine normale
Methode (nicht `@staticmethod`), obwohl sie `self` nicht benutzt — User
bevorzugt Konsistenz im Aufrufstil mit den anderen beiden Methoden.

Datei 1 ist fachlich vollständig für den einfachsten Fall (unkonditioniertes
R3-Flow-Matching auf einzelnen Punkten, ohne Fragment-Aggregation — die kommt
erst in `denoiser.py`, Schritt 6 im Fahrplan).

---

## Datei 2/5: `SigmaFlow_Development/src_sigmadock_diff/SO3_FlowMatcher.py` ✅

### Design-Entscheidungen für SO(3)-Flow-Matching (mathematischer Rahmen,
bereits mit dem User besprochen, gilt für die ganze Datei)

- **Quellverteilung `p_0`:** Haar-Gleichverteilung auf SO(3), via
  `so3_utils.sample_uniform` (exakt, kein Umweg über IGSO3 bei `t=1` wie im
  Original nötig).
- **Pfad:** geodätische Interpolation `R_t = R_0 · exp(t · log(R_0^T R_1))`
  (Analogon zu `x_t=(1-t)x_0+t x_1`, da SO(3) keine konvexe/lineare Struktur
  hat).
- **Trivialisierung:** rechts-trivialisiert (`Δ = R_0^T R_1`, Tangentialvektor
  im Körpersystem von `R_0`) — bewusst konsistent zur bereits vorhandenen
  Konvention in `so3_utils.expmap` (Zeile 85: `R^T · tangent`), damit spätere
  Kombination mit `expmap` in `euler_step` keine Vorzeichen-/Richtungsfehler
  produziert.
- **Konditionales Vektorfeld:** `u_t = log(R_0^T R_1)`, konstant in `t` (bi-
  invariante Metrik auf SO(3) ⇒ konstante körperfeste Winkelgeschwindigkeit
  entlang der Geodäte) — strukturelles Echo des R3-Falls.
- **Theoretische Basis:** Chen & Lipman, "Flow Matching on General
  Geometries" (allgemeine Riemannsche Konstruktion) + Yim et al. 2023, "Fast
  protein backbone generation with SE(3) flow matching" (SO(3)-spezifisch,
  direkter Flow-Matching-Nachfolger von `jasonkyuyim/se3_diffusion`, dem
  Codebase von dem `SigmaDock`s `so3_diffuser.py` adaptiert wurde). Exakte
  Gleichungsnummern noch nicht verifiziert (PDF-Tooling/poppler war zum
  Zeitpunkt der Diskussion nicht nutzbar) — bei Bedarf mit Paper
  gegenchecken.
- **Singularität bei `ω=π`:** wird bereits von `so3_utils.rotation_vector_
  from_matrix` behandelt (Fallunterscheidung `mask_pi`), keine eigene Lösung
  nötig, aber im Hinterkopf behalten falls NaNs bei Winkeln nahe `π` auftreten.

### WICHTIG: eigene Kopie `so3_utils_adapted.py` statt `so3_utils.py`

Beim Testen von `conditional_probability_path` (Randzeit-Check `t=1`) einen
**echten numerischen Bug** in der ursprünglich als "unverändert
wiederverwendbar" deklarierten `so3_utils.py` gefunden: `Omega()` klemmt das
`arccos`-Argument auf `[-0.99, 0.99]` → jede Rotation mit wahrem Winkel
zwischen `~172°` und `180°` bekommt einen falschen (zu kleinen) Winkel, und
der Log-Map-Rückweg (`rotation_vector_from_matrix`) bricht dadurch **lautlos**
(kein NaN, kein Crash, einfach falsches Ergebnis) — betraf empirisch **~9%**
zufälliger Rotationspaare. Der Bug steckt unverändert auch in
`SigmaDock/so3_utils.py` (identische Datei, dort nie angefasst, wie
vorgeschrieben), fällt dort aber vermutlich nicht auf, weil die Diffusion nie
einen exakten `exp(log(R))==R`-Rundweg braucht — bei uns ist das aber
strukturell notwendig für die geodätische Interpolation.

**Fix (User-Entscheidung, mit Begründung):** Kopie unter neuem Namen
`SigmaFlow_Development/src_sigmadock_diff/so3_utils_adapted.py` angelegt
(Namensänderung macht sichtbar, dass dies *nicht* mehr die unangetastete
generische Version ist), darin `Omega()`s Clamp von `[-0.99,0.99]` auf
`[-1+1e-7, 1-1e-7]` verengt. Begründung: `Omega()` castet intern ohnehin nach
`float64` und hat schon einen `eps`-Shrink-Mechanismus — der harte `0.99`-
Clamp war unnötig konservativ dazu. Zusätzlich: `u_t` ist bei uns ein reines
Regressionsziel (kein Gradientenfluss nötig), daher ist die ursprüngliche
Sorge um `arccos`-Gradienten-Explosion nahe `±1` hier nicht relevant.

Ergebnis nach Fix: katastrophale Fehler (`0.5`–`2.0`) verschwunden, `t=1`-
Randzeit-Test jetzt bei `max diff ≈ 2e-3` über 200 Samples. Verbleibender
kleiner Restfehler (`~1e-3`) exakt nahe `ω=π` liegt an einer separaten,
harmloseren Präzisionsgrenze im "mask_pi"-Sonderzweig von
`rotation_vector_from_matrix` (Wiki-Formel-Vorzeichenauswahl) — bewusst nicht
weiter optimiert (Maß-Null-Ereignis, exakt-antipodale Rotationen), siehe
CLAUDE.md §9 (nicht vorzeitig optimieren).

**Alle künftigen SO(3)-Bausteine müssen `so3_utils_adapted` importieren, nicht
`so3_utils`.**

### Fertig und verifiziert

```python
class SO3_FlowMatcher:
    def __init__(self):
        pass  # kein Konstruktor-Parameter aktuell nötig

    def sample_init(self, n: int, device: str) -> torch.Tensor:
        R_0 = so3_utils_adapted.sample_uniform(n).to(device, dtype=torch.float32)
        return R_0

    def conditional_probability_path(self, R_1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = R_1.shape[0]
        device = R_1.device
        R_0 = self.sample_init(n, device)
        Delta = R_0.transpose(-1, -2) @ R_1
        log_Delta = so3_utils_adapted.log(Delta)
        u_t = log_Delta
        R_t = R_0 @ so3_utils_adapted.exp(t[:, None, None] * log_Delta)
        return R_t, u_t

    def euler_step(self, R_t: torch.Tensor, v_t: torch.Tensor, dt: float) -> torch.Tensor:
        R_next = R_t @ so3_utils_adapted.exp(v_t * dt)
        return R_next
```

Verifiziert: Shape `[n,3,3]`/`[n,3,3]`, `dtype=float32`, Device korrekt.
`sample_init`: Orthogonalität + `det=1` bei Maschinenpräzision.
`conditional_probability_path`: `R_t` gültige Rotation (orthogonal, `det=1`),
`u_t` schiefsymmetrisch, `t=1`-Randzeit-Test bestanden (nach so3_utils-Fix
oben), 2000-Sample-Statistik-Check für den `exp(log)`-Rundweg durchgeführt.
`euler_step`: Shape/Orthogonalität/`det=1` bestätigt, exakter Einzelschritt-
Test (`dt=1`, `v_t=u_t` ab `R_0` → `R_1`) mit Median-Fehler `~9e-7` über 500
Samples, `dt=0`-Identität, Halbschritt-Konsistenz (`~2e-6`) — alles bestanden.

Wichtige Korrektur während der Review (mein eigener Fehler in der
ursprünglich vorgeschlagenen Spezifikation, nicht im User-Code): erster
Vorschlag war `so3_utils_adapted.expmap(R_t, v_t·dt)`, aber `expmap` erwartet
den Tangentialvektor in **eingebetteter** Form (`R_t @ ξ`), nicht das rohe
körperfeste `ξ` selbst — sonst ist `R_t^T @ tangent` nicht schiefsymmetrisch
(Warnung `"must be skew symmetric"` bestätigte das). Korrigiert zu direkter
Formel `R_t @ so3_utils_adapted.exp(v_t*dt)` (mathematisch identisch zu
`expmap(R_t, R_t @ (v_t*dt))`, aber einfacher, spart eine unnötige
`R^T@R`-Rechnung, konsistent mit der bereits in `conditional_probability_path`
verwendeten Formel).

Zwei Bugs gefunden und korrigiert während der Review (beide Male dasselbe
Muster): `self` fehlte zunächst in beiden Methodensignaturen (Instanz-
Methoden brauchen `self` als ersten Parameter, sonst wird beim Aufruf über
eine Instanz die Instanz selbst fälschlich in den ersten "echten" Parameter
einsortiert → `TypeError`). Außerdem ein Tippfehler `so_utils` → `so3_utils`
(vor dem Umbenennen zu `so3_utils_adapted`).

Konzeptklärung während der Review: Frage, warum `u_t = log_Delta` direkt und
nicht `log_Delta/(1-t)` — beide Formen numerisch äquivalent verifiziert
(`log(R_t^T R_1)/(1-t) == log(R_0^T R_1)`, Rundungsfehler `~1e-5`), aber
`log_Delta` direkt zu teilen wäre falsch (falsches Argument, Explosion nahe
`t=1`). Wir nutzen die einfachere `log_Delta`-Form, weil `R_0` während des
Trainings ohnehin direkt verfügbar ist (kein Grund für die `R_t`-basierte
"Endpunkt-Parametrisierung").

Datei 2 ist fachlich vollständig für den einfachsten Fall (analog zu Datei 1)
— unkonditioniertes SO(3)-Flow-Matching auf einzelnen Rotationen, ohne
Fragment-Aggregation (kommt erst in `denoiser.py`).

---

## Datei 3/5: `SigmaFlow_Development/src_sigmadock_diff/SE3_FlowMatcher.py` ✅

### Design-Entscheidungen (mit User besprochen)

- **SE(3) als direktes Produkt, nicht als echte semidirekte Produktgruppe:**
  bestätigt beim Lesen von `se3_diffuser.py` — das Original koppelt Translation
  und Rotation nie (immer unabhängige `trans_score`/`rot_score`), behandelt
  SE(3) faktisch als `R³ × SO(3)`. Wir übernehmen das bewusst identisch
  (Kompatibilität, keine neue Kopplung einführen) — explizit benannt statt
  "SE(3)" locker zu verwenden, wenn eigentlich nur getrennte Operationen
  laufen (CLAUDE.md §5).
- **Kompositionsmuster beibehalten:** `self._r3_flow_matcher` /
  `self._so3_flow_matcher` als Unterstrich-Attribute, analog zu
  `_r3_diffuser`/`_so3_diffuser` im Original.
- **Rückgabetyp: `dict[str, torch.Tensor]`** statt Tupel (neu eingeführtes
  Konzept, siehe Teaching-Kontext unten) — selbstdokumentierende Keys statt
  positionsabhängiger Tupel-Entpackung, analog zu `forward_marginal` im
  Original. Keys für `sample_init`: `"trans_0"`, `"R_0"`.
- Keine neue Mathematik in dieser Datei — reine Komposition der bereits
  fertigen R3-/SO(3)-Bausteine.

### Fertig und verifiziert

```python
class SE3_FlowMatcher:
    def __init__(self, sigma_min: float):
        self._r3_flow_matcher = R3_FlowMatcher(sigma_min=sigma_min)
        self._so3_flow_matcher = SO3_FlowMatcher()

    def sample_init(self, n: int, device: str) -> dict[str, torch.Tensor]:
        trans_0 = self._r3_flow_matcher.sample_init(n, device)
        R_0 = self._so3_flow_matcher.sample_init(n, device)
        return {"trans_0": trans_0, "R_0": R_0}
```

Verifiziert: Shapes (`[n,3]`, `[n,3,3]`), `dtype=float32`, Dict-Keys korrekt.

Review-Fund (vom User korrigiert): `sigma_min` war zunächst im Konstruktor
hart codiert (`R3_FlowMatcher(sigma_min=0)`) statt als eigener `__init__`-
Parameter durchgereicht — Hyperparameter sollen bis nach außen (Config/
Trainings-Skript) konfigurierbar bleiben, nicht mitten in der
Klassenhierarchie fest verdrahtet werden. Korrigiert:
`__init__(self, sigma_min: float)`.

### `conditional_probability_path` und `euler_step` — fertig

```python
def conditional_probability_path(self, trans_1, R_1, t) -> dict[str, torch.Tensor]:
    trans_t, u_t_trans = self._r3_flow_matcher.conditional_probability_path(trans_1, t)
    R_t, u_t_R = self._so3_flow_matcher.conditional_probability_path(R_1, t)
    return {"trans_t": trans_t, "R_t": R_t, "u_t_trans": u_t_trans, "u_t_R": u_t_R}

def euler_step(self, trans_t, R_t, v_t_trans, v_t_R, dt) -> dict[str, torch.Tensor]:
    trans_new = self._r3_flow_matcher.euler_step(trans_t, v_t_trans, dt)
    R_new = self._so3_flow_matcher.euler_step(R_t, v_t_R, dt)
    return {"trans_new": trans_new, "R_new": R_new}
```

Namenskonvention geklärt: bewusst `R`/`trans` als durchgängiges Präfix/Suffix
(`R_t`, `u_t_R`, `R_new` / `trans_t`, `u_t_trans`, `trans_new`) statt eines
dritten Begriffs wie `rot` — Konsistenz-Entscheidung des Users.

Verifiziert: Dict-Keys/Shapes/`dtype` korrekt, `R_new` bleibt gültige Rotation
(Orthogonalität, `det=1`), exakter Vollschritt-Test (`dt=1` ab `trans_0,R_0`
mit `v=u_t` → `trans_1,R_1`): Translation quasi exakt (`~2e-7`), Rotation im
bekannten, bereits akzeptierten Toleranzbereich nahe `ω≈π` (`~1e-2`, ein
einzelner Ausreißer von 300 Samples), `dt=0`-Identität für beide Komponenten
bestätigt.

Zwei Bugs gefunden und korrigiert während der Review:
- `conditional_probability_path` gab zunächst `(result, trans_t, R_t,
  u_t_trans, u_t_R)` zurück statt nur `result` — redundante doppelte
  Rückgabe der bereits im Dict enthaltenen Werte, widersprach der
  vereinbarten Schnittstelle. Dabei auch bemerkt: die (fehlerhafte) Typ-
  Annotation mit undefinierten Namen als "Typen" führte in Python 3.14 zu
  keinem sofortigen Fehler, weil PEP 649 (verzögerte Annotation-Auswertung)
  seit dieser Version Standard ist — in jeder älteren Python-Version wäre das
  sofort beim Import gecrasht.
- `euler_step` baute das Ergebnis-Dict, hatte aber kein `return result` —
  klassischer Fall von "Funktion ohne erreichtes `return` gibt automatisch
  `None` zurück", kein Crash, einfach ein stiller falscher Rückgabewert.

Fahrplan-Schritt 3/5 abgeschlossen — der komplette SE(3)-Flow-Matching-Kern
(Quellverteilung, Pfad+Vektorfeld, Euler-Integration, für Translation UND
Rotation gemeinsam) ist fachlich fertig, für den einfachsten Fall
(unkonditionierte Einzelpunkte/-rotationen, keine Fragment-Aggregation).

---

## Datei 4/5: `SigmaFlow_Development/src_sigmadock_diff/denoiser_adapted.py` ✅

### WICHTIG: andere Strategie als bei Dateien 1-3

Diese Datei wird **nicht** von Null neu geschrieben (anders als die letzten
drei) — sie mischt ~70% generische Graph-/Starrkörper-Infrastruktur (bleibt
unverändert) mit diffusionsspezifischer Logik. Datei wurde als Kopie des
Originals angelegt und wird **chirurgisch editiert** (CLAUDE.md §1
"surgical intervention"), Umbenennung zu `denoiser_adapted.py` analog zu
`so3_utils_adapted.py`.

### Architektur-Analyse des Originals (Volltext gelesen, nicht nur `grep`)

**Zentraler Fund — was das Netz tatsächlich vorhersagt (CLAUDE.md §5):** Das
Netz sagt **Rauschen** voraus (`epsilon`-Parametrisierung, Kommentar im
Original Zeile 813: *"pseudo forces ~ epsilon ∈ N(0,1)"*), **keine echten
physikalischen Kräfte**. Die "Physik" ist eine Umdeutung:
```
pro-Atom "Kraft" (=Rauschen)
  → linear_mechanics: Summe zu Gesamtkraft + Drehmoment pro Fragment (r×F)
  → newton_maruyama: dT=F/m, dW=I⁻¹·τ  (Masse-/Trägheits-Normalisierung,
    KEINE Zeit-Integration, KEIN Rauschterm trotz des Namens — der Name ist
    ein Überbleibsel der SDE-Denkweise)
  → _compute_scores: multipliziert mit score_scaling(t)=1/σ(t)-artig →
    ERST hier entsteht der eigentliche Score
```
`linear_mechanics`/`newton_maruyama` sind im Kern eine
**äquivarianz-erhaltende Pooling-Operation** (viele Pro-Atom-Vektoren →
ein Translationsvektor + ein Rotations-Generator pro Fragment), eine
Architektur-Entscheidung, keine Diffusions-Theorie — bleibt für Flow
Matching unverändert nutzbar.

**Randnotiz:** `get_fragment_com_and_rot` hat toten Code (`if False:`) für
eine PCA-basierte initiale Fragment-Rotation — `R0` ist im Original **immer
Identität** (Kommentar: *"should be independent in the score"*). Keine
Diffusions-Theorie, bewusste Implementierungsentscheidung, übernehmen wir so.

**Ansatzpunkt gefunden:** `_compute_scores` hat zwei Modi
(`rot_score_method`): `"space"` sagt bereits ein **direktes Rotations-Update**
voraus (`R0_hat = exp(scaled_omega) @ R_t`), der Score wird erst danach
draus abgeleitet — strukturell näher an dem, was wir brauchen, als der
`"score"`-Modus.

### Diffusionsspezifisch (wird ersetzt) vs. generisch (bleibt)

**Bleibt unverändert:** `get_flat_fragment_index`, `get_fragment_com`,
`get_fragment_com_and_rot`, `get_fragment_mass_inertia`,
`get_transformations_from_rototranslations`, `_apply_transformations`,
`_update_batch`, `_compute_interaction_edges`, `_prune_local_edges`,
`get_local_graph`, `merge_and_process_edges`, `_compute_forces` (Netzaufruf
selbst — nur Interpretation der Ausgabe ändert sich später),
`linear_mechanics`, `newton_maruyama`, `_compute_fragment_dynamics`,
`_prepare_batch`, `sample_time`.

**Wird ersetzt (in dieser Reihenfolge):**
1. Import + Konstruktor ✅ FERTIG (siehe unten)
2. `_sample_diffusion` → neue Methode, ruft `SE3_FlowMatcher.
   conditional_probability_path` — NÄCHSTER SCHRITT
3. `_get_scalings` → fällt komplett weg (keine Score-Skalierung nötig)
4. `_compute_scores` → vereinfacht sich stark (kein Score mehr)
5. `compute_losses` → Score-MSE wird zu Vektorfeld-MSE

### Baustein 1: Import + Konstruktor — fertig

Geändert: Import `from sigmadock.diff.se3_diffuser import SE3Diffuser` →
`from SE3_FlowMatcher import SE3_FlowMatcher` (+ `SO3_FlowMatcher`, aktuell
ungenutzt, evtl. später gebraucht). Import `from sigmadock.diff import
so3_utils` → flacher Import `import so3_utils_adapted` (bewusst **überall**
in der Datei umbenannt, auch in noch nicht angefassten Methoden — reiner
Bugfix, keine Diffusions-/Flow-Matching-Semantik betroffen). Konstruktor:
alle Diffusions-Schedule-Parameter (`min_beta`, `max_beta`, `schedule`,
`min_sigma`, `max_sigma`, `num_sigma`, `num_omega`, `cache_path`,
`use_cached_score`, `L`) ersetzt durch einzelnes `sigma_min: float`.
`self.diffuser = SE3Diffuser(...)` → `self.flow_matcher =
SE3_FlowMatcher(sigma_min)`.

Bugs gefunden und korrigiert während der Review:
- `SyntaxError: parameter without a default follows parameter with a
  default` — `sigma_min` (ohne Default) stand hinter `include_interactions`
  (mit Default `=True`). Python verlangt alle Nicht-Default-Parameter vor
  allen Default-Parametern. Behoben durch Umsortieren.
- Import-Pfad `from sigmadock.diff import so3_utils_adapted` — falscher
  Modulpfad (kein installiertes `sigmadock`-Package hier), korrigiert zu
  flachem `import so3_utils_adapted`.
- Docstring nannte fälschlich `min_sigma` (alter Parametername) statt
  `sigma_min`, korrigiert.

**Verifikations-Einschränkung:** `torch_geometric` ist in dieser Umgebung
NICHT installiert — echter `import`-Test wie bei Dateien 1-3 nicht möglich.
Nur `python -m py_compile` (Syntax-Check) + manuelles Durchtracen jeder
Variable/jedes Methodenaufrufs in `forward()` durchgeführt. Laufzeitverhalten
mit echten Batches bleibt bis zu einer Umgebung mit vollem Stack ungetestet.

`self.diffuser` wurde konsequent zu `self.flow_matcher` umbenannt.

### Baustein 2: `_sample_flow` (vormals `_sample_diffusion`) — fertig

Ruft `self.flow_matcher.conditional_probability_path(trans_1=..., R_1=...,
t=...)`, gibt deren Dict (`trans_t`, `R_t`, `u_t_trans`, `u_t_R`) direkt
zurück (volles Commitment zur neuen Namensvokabular, keine Kompatibilitäts-
Übersetzungsschicht — User-Entscheidung, sauberer als mein ursprünglicher
Vorschlag).

**Wichtige Lektion aus dieser Runde:** Eine große, manuelle Umbenennung quer
über viele Aufrufstellen (`T0`→`trans_1`, `R0`→`R_1`, `T_t`→`trans_t`,
`sampled_diffusion`→`sampled_flow`) hat mehrfach einzelne Stellen
übersprungen (Methodenkörper nicht mitgezogen, Parameter-Keyword an
Aufrufstelle nicht angepasst, ein Tippfehler `u_t_trains`). Jedes Mal per
`grep` nach dem *alten* Namen über die ganze Datei gesucht, um lückenlos zu
verifizieren — diese Technik hat sich als zuverlässig erwiesen und sollte
bei künftigen großen Umbenennungen zuerst angewendet werden, bevor man auf
"sieht fertig aus" vertraut.

### Baustein 3: `_get_scalings` entfernt, `_compute_vector_field` (vormals
`_compute_scores`) neu geschrieben — fertig

**Zentrale Design-Entscheidung:** `updates["total_force"]` (Translation) und
`updates["omega"]` (Rotation, schon eine `[...,3,3]`-schiefsymmetrische
Matrix — exakt derselbe Objekttyp wie unser `u_t_R`) werden **direkt** als
`pred_u_t_trans`/`pred_u_t_R` übernommen — keine Skalierung, keine
Score-Ableitung nötig, weil unser `u_t` (anders als der Diffusions-Score)
keine zeitabhängige Skala hat. Die komplette `rot_score_method`
(`"space"`/`"score"`)-Verzweigung fällt weg, beide Zweige existierten nur,
um einen Score zu konstruieren.

**Bewusst weggelassen (dokumentierte Vereinfachung, keine vergessene
Funktion):** Die Datenraum-Rekonstruktion (`T0_hat`/`R0_hat`) — bräuchte
einen Euler-Schritt mit Pro-Fragment-`dt=(1-t)`-Tensor, den unsere
`euler_step`-Methoden in Dateien 1-3 aktuell nicht unterstützen (nur
skalares `dt` getestet). Der Standard-Flow-Matching-Trainingsverlust
(Lipman et al.) braucht diesen Term ohnehin nicht — reine
Vektorfeld-Regression reicht für Korrektheit. Kann später als dokumentierte
Erweiterung nachgerüstet werden (CLAUDE.md §1).

**Architektur-Prinzip geklärt (User-Frage, wichtig für weitere Bausteine):**
`newton_maruyama`/`linear_mechanics` bleiben bei ihrer physikalischen
Begrifflichkeit (`force`, `torque`, `omega`) — sie sind bewusst generische,
wiederverwendbare Infrastruktur, die nichts von Flow Matching weiß.
`_compute_vector_field` ist genau die Übersetzungsstelle zwischen
generischem Physik-Output und Flow-Matching-spezifischer Benennung — das ist
der eigentliche Zweck dieser Methode, kein Zufall.

### Baustein 4: `compute_losses` — fertig

Nur noch reine Vektorfeld-MSE (Translation + Rotation), keine
score-scaling-abhängigen Gewichte (`lambda_*`, `alpha_trans_t`) mehr, keine
Datenraum-Verluste (konsistent zu Baustein 3). Rückgabe-Keys von User
proaktiv zu `"loss_trans"`/`"loss_R"` benannt (vermeidet Kollision mit den
gleichnamigen Tensor-Keys in `out`).

### Baustein 5: `_sample_time_and_sigma` → `_sample_time` — fertig

`sigma`-Berechnung komplett entfernt (wurde nirgends mehr konsumiert, nachdem
Baustein 4 alle scaling-abhängigen Gewichte entfernt hat) — damit auch der
letzte `self.diffuser`-Aufruf im Trainingspfad weg. Gibt jetzt nur noch `t`
zurück (kein Tupel mehr) — Docstring/Type-Annotation der Methode ist noch
nicht nachgezogen (fällt unter spätere Namensaufräumrunde).

### Status des Trainings-Vorwärtspfads: komplett durchgetraced, konsistent

`forward()` komplett Zeile für Zeile verifiziert (jede Variable/jeder
Methodenaufruf existiert und passt zur jeweiligen Signatur), `compute_losses`
liest nur Keys, die `forward()` tatsächlich liefert. Kein `self.diffuser`
mehr im Trainingspfad.

**Verbleibend in Datei 4:**
- `_compute_true_vector_field` (vormals `_compute_true_scores`, Zeile
  1016/1018) — ruft noch `self.diffuser.calc_trans_vector_field`/
  `calc_rot_vector_field` (nicht-existente Methoden, reines Textumbenennen
  ohne Logik-Ersetzung, genau wie bei `_get_scalings` vorher). Gehört aber
  laut Dependency-Mapping nicht zu `forward()`, sondern zu `sampling.py`
  (Datei 5) — kann bis dahin warten.
- Aufgeschobene Namensaufräumrunde (User-Entscheidung): veraltete Docstrings
  (`_sample_flow`, `_apply_transformations`, `_compute_fragment_dynamics`,
  `_sample_time` erwähnen noch alte Namen/falsche Tupel-Rückgabetypen),
  toter Code (`_get_scalings`-Methode selbst noch vorhanden, nur ihre
  Aufrufe entfernt; `rot_vector_field_method`/`rot_vector_field_scaling`-
  Konstruktor-Parameter jetzt ungenutzt), `_compute_vector_field`s
  ungenutzte `sampled`/`t_batch`-Parameter.

---

## Infrastruktur-Lücke für einen echten Testlauf (nicht Teil der 5-Dateien-
Konversion, aber relevant sobald getestet werden soll)

Nutzer hat 2026-07-15 das vollständigere lokale `SigmaDock/`-Repo bereitgestellt
(`scripts/train.py`, `scripts/sample.py`, `conf/` Hydra-Configs,
`pyproject.toml`, `slurm/`). Gegen das echte öffentliche Repo abgeglichen
(`github.com/alvaroprat97/sigmadock`, per `gh api` geprüft, nicht geraten):

- **Bestätigt:** `SigmaDockDenoiser` ist tatsächlich die vom echten
  `scripts/train.py` verwendete Klasse (`from sigmadock.diff.denoiser import
  SigmaDockDenoiser`) — wir arbeiten an der richtigen Stelle.
- **Fehlt lokal, existiert im echten Repo unter `src/sigmadock/`:**
  `oracle.py` (Quelle von `HPARAMS`, 324 Zeilen, simple `@dataclass`-
  Definitionen), `trainer.py` (`SigmaLightningModule`, der eigentliche
  PyTorch-Lightning-Trainer), `config.py`, `data.py` (Achtung: anderer
  Import-Pfad als unser lokales `src_sigmadock_core/data.py`, das ist eine
  andere Datei), `datafronts.py`, `sampling_setup.py`, `utils.py`,
  `src/sigmadock/__init__.py`.
- **Struktur-Mismatch:** Lokaler Ordner heißt `src_sigmadock/
  src_sigmadock_diff/...`, echtes Repo strukturiert `src/sigmadock/diff/...`
  (ohne doppeltes Präfix) — deshalb hat `import sigmadock` in dieser Umgebung
  nie funktioniert, nicht nur weil nichts pip-installiert ist. Muss vor einem
  echten Trainingslauf angeglichen werden.
- **Dummy-Datensatz (`notebooks/dummy_data/`):** lokal nur CSV-Manifest +
  Setup-Skripte vorhanden, die eigentlichen `.pdb`/`.sdf`-Strukturdateien
  (mehrere echte kleine Komplexe, z. B. `1G9V_RQ3`, `1HWI_115`, ...) fehlen
  noch. Es gibt eine eigene Experiment-Config dafür (`conf/experiments/
  dummy_crossdock.yaml`) — das ist der vom Original-Repo vorgesehene
  kleinste Testdatensatz, nicht optional für einen echten Testlauf, auch wenn
  nicht blockierend für die laufende Dateien-Konversion.

**Einordnung:** Diese Lücke ist reine Infrastruktur ("SigmaDock zum Laufen
bringen"), hat nichts mit der Diffusion→Flow-Matching-Konversion zu tun.
Separat von der 5-Dateien-Arbeit zu behandeln, nicht vermischen.

**Update 2026-07-15, nach User-Entscheidung (explizit per Nachfrage, nicht
eigenmächtig):** Die 8 fehlenden Dateien wurden 1:1 aus dem echten Repo
(`github.com/alvaroprat97/sigmadock`, `main`-Branch) nachgeladen und unter
`SigmaDock/src_sigmadock/{oracle,trainer,config,data,datafronts,
sampling_setup,utils,__init__}.py` abgelegt (Geschwister-Ebene zu
`src_sigmadock_chem/`, `src_sigmadock_core/`, etc. — passend zur
bestehenden lokalen Umbenennungs-Konvention, ohne etwas Bestehendes
anzufassen). Syntax aller 8 Dateien mit `py_compile` bestätigt.
**Bewusst NICHT gemacht** (wäre eine größere, separate Entscheidung):
Ordnerstruktur-Angleichung (`src_sigmadock/src_sigmadock_diff/` vs. echtes
`src/sigmadock/diff/`) — die neuen Dateien nutzen daher weiterhin
kanonische `from sigmadock.xxx import ...`-Imports, die in dieser lokalen
Struktur nicht auflösen, bis die Struktur irgendwann angeglichen wird (nur
relevant für einen echten Trainingslauf, nicht für unsere Konversionsarbeit).

**Update 2026-07-16, vollständiger Konsistenz-Audit (auf User-Anfrage):**
Kompletter Abgleich `SigmaDock/` gegen `github.com/alvaroprat97/sigmadock@main`
via `gh api` (Tree-Listing + Byte-Diff einzelner Dateien, nicht geraten).
Ergebnis: **Inhalt zu 100% identisch** (`r3_diffuser.py`, `so3_utils.py`,
`denoiser.py`, `pyproject.toml`, `conf/config.yaml` stichprobenartig
gegengeprüft, keine Abweichung) — nur die Ordner-/Dateinamens-Konvention
weicht ab, wie oben dokumentiert. Alle 6 SigmaFlow-Dateien erneut gelesen
(nicht nur Log vertraut) und mit `py_compile` kompiliert — alle fehlerfrei.

Dabei zwei bisher unentdeckte Lücken gefunden und nachgeladen (per
`git clone --depth 1` vom echten Repo, Dateien 1:1 kopiert, Stichproben-Diff
+ `torch.load`-Integritätscheck bestanden):
- **`Jd.pt`** (Wigner-D-Vorberechnung, 12 Tensoren) fehlte komplett unter
  `src_sigmadock_net/` — `wigner.py` lädt sie fest beim Modul-Import
  (`torch.load` in Zeile 8), ohne sie crasht jeder Import des
  Equiformer-Netzwerks sofort, unabhängig von der Flow-Matching-Konversion.
  Jetzt vorhanden unter `SigmaDock/src_sigmadock/src_sigmadock_net/Jd.pt`.
- **Dummy-Datensatz-Strukturdateien** (`.pdb`/`.sdf`, 10 Komplexe, 134
  Dateien) fehlten unter `notebooks/dummy_data/` (nur CSV-Manifest +
  Setup-Skripte waren vorhanden). Jetzt nachgeladen unter
  `SigmaDock/notebooks/dummy_data/{1G9V_RQ3,1HWI_115,1MZC_BNE,1OWE_675,
  1R1H_BIR,1S3V_TQD,1U1C_BAU,1V4S_MRK,1YQY_915,2BSM_BSM}/`, nichts
  Bestehendes überschrieben.

**Wichtigster neuer Fund des Audits — noch NICHT behoben, echter Blocker
für einen Trainingslauf:** `trainer.py`, `scripts/train.py` und
`scripts/sample.py` (alle drei Teil der 8 kürzlich nachgeladenen
Infrastruktur-Dateien, aber bisher nie gegen den neuen Denoiser
gegengeprüft) sind noch komplett auf der alten Diffusions-Schnittstelle:
1. `trainer.py:373` und `scripts/sample.py:104`: beide rufen
   `self.model.diffuser._so3_diffuser.set_device(device)` — `.diffuser`
   heißt jetzt `.flow_matcher`, und `SO3_FlowMatcher` hat gar keine
   `set_device`-Methode → `AttributeError` beim ersten Device-Move.
2. `trainer.py::_shared_step` (der tatsächliche Lightning-Trainingsschritt):
   erwartet vom Denoiser-`forward()`-Output Keys wie `pred_T_score`,
   `true_R_score`, `pseudoforces`, `force_per_fragment` — unser `forward()`
   liefert `pred_u_t_trans`/`u_t_R`/etc. Und die Loss-Kombination
   (`self.trans_score_weight * losses["T_score"] + ... +
   self.rot_data_weight * losses["R0"]`) erwartet Keys `T_score`/`R_score`/
   `T0`/`R0` — unser `compute_losses()` liefert nur `loss_trans`/`loss_R`
   (Datenraum-Terme `T0`/`R0` wurden in Baustein 3 bewusst weggelassen) →
   **`KeyError` im allerersten Trainingsschritt.**
3. `scripts/train.py` baut `SigmaDockDenoiser(...)` ohne `sigma_min` zu
   übergeben (kein CLI-Argument dafür vorhanden), `sigma_min` ist aber ein
   Pflichtparameter ohne Default → **`TypeError` schon bei der
   Konstruktion**, noch vor dem ersten Batch.

**Einordnung:** Das ist kein Stilproblem, sondern ein echter, bisher nicht
eingeplanter Arbeitsschritt — vorgeschlagen als **"Datei 6/6: `trainer.py`
anpassen"**, analog zum bisherigen Muster (chirurgisch, Datei lesen, Keys
durchtracen, mit User zusammen umbauen).

---

## Datei 6/6: `SigmaFlow_Development/trainer_adapted.py` ✅

### Architektur-Analyse (vor Beginn der Arbeit)

Weit weniger diffusionsspezifisch als die anderen 5 Dateien: das meiste ist
generische PyTorch-Lightning-Infrastruktur (Optimizer/Scheduler-Konfiguration,
DDP-Debugging, Logging-Boilerplate, Epoch-Hooks), die unverändert bleibt.
Nur drei Stellen hängen tatsächlich an der alten Diffusions-Schnittstelle.

**Bleibt unverändert:** `configure_optimizers`, `forward()` (die generische
Lightning-Wrapper-Methode, ruft `denoiser.compute_losses()` +
`denoiser.scaled_fragmented_loss()` auf — beide bereits generisch, arbeiten
mit beliebigen Dict-Keys), `compute_grad_norm`, `on_after_backward`,
`training_step`, `validation_step`, alle Epoch-Hooks.

### Design-Entscheidung (mit User geklärt)

`trans_data_weight`/`rot_data_weight` (Konstruktor-Parameter, steuerten im
Original zwei Datenraum-Rekonstruktionsverluste `T0`/`R0`) — **komplett
entfernt statt als ungenutzter Platzhalter behalten** (anders als
`sigma_min` in `R3_FlowMatcher`). Begründung: kein aktueller Bedarf
absehbar genug, um den toten Parameter zu rechtfertigen (CLAUDE.md §9,
"kein Code für etwas, das nicht passieren kann") — falls der Datenraum-Term
später nachgerüstet wird, kommen die Parameter mit echter Bedeutung zurück.

### Drei Bausteine, alle vom User umgesetzt und verifiziert

1. **`on_fit_start`** (ursprünglich: `self.model.diffuser._so3_diffuser.
   set_device(device)`, verschob den IGSO3-Dichte-Cache des alten
   `SO3Diffuser` auf die GPU) → auf ersten Versuch korrekt zu `pass`
   reduziert. `SO3_FlowMatcher` hat keinen Device-abhängigen Cache, es gibt
   nichts zu verschieben.
2. **`__init__`**: `trans_data_weight`/`rot_data_weight` samt Docstring-
   Einträgen und `self.*`-Zuweisungen sauber entfernt.
3. **`_shared_step`**: Verlust-Kombination von vier (`T_score`/`R_score`/
   `T0`/`R0`) auf zwei Terme (`losses["loss_trans"]`/`["loss_R"]`) reduziert;
   `log_dict` entsprechend auf zwei Einträge angepasst; tote Entpackung
   (`p_trans_score`/`p_rot_score`/`t_trans_score`/`t_rot_score`, las
   nicht mehr existierende `pred_T_score`/`pred_R_score`/`true_T_score`/
   `true_R_score`-Keys, wurde im Rest der Funktion ohnehin nirgends
   benutzt) ersatzlos gelöscht. `force_per_atom`/`force_per_fragment`/
   `torque_per_fragment`-Zeilen unverändert gelassen (Keys existieren
   unverändert in unserem `forward()`).

Finaler Sweep (`grep` nach `.diffuser`, `T_score`, `R_score`, `"T0"`, `"R0"`,
`trans_data_weight`, `rot_data_weight`, `pred_T_score`, `pred_R_score`,
`true_T_score`, `true_R_score`): keine Treffer. `py_compile`: fehlerfrei.
Alles im ersten Anlauf korrekt umgesetzt, keine Bugs in der Review gefunden
(anders als bei Dateien 1-5, wo fast jede Runde mindestens einen Bug ergab).

Datei 6/6 fachlich fertig. Damit ist der komplette Pfad von
`scripts/train.py` (Konstruktion) über `SigmaLightningModule` (Trainingsloop)
bis zum `SigmaDockDenoiser` (Vektorfeld-Vorhersage + Verlust) intern
konsistent durchgetraced.

---

## Ordnerstruktur-Angleichung ✅ (2026-07-16)

`SigmaFlow_Development/` hat jetzt eine echte, installierbare Package-Struktur
unter `SigmaFlow_Development/src/sigmadock/`, die 1:1 dem echten Repo
(`src/sigmadock/{chem,core,diff,geo,net,torch_utils}/...`) entspricht.
`SigmaDock/` bleibt unangetastet (flache lokale Konvention als Referenz).

**Vorgehen:**
1. Fehlende `__init__.py` in `SigmaDock/` nachgeladen (`core/`, `diff/`,
   `geo/` hatten keine — alle drei leer, 0 Bytes, unproblematisch). `SigmaDock/`
   ist damit jetzt ein lückenloses Abbild des echten Repos.
2. **Phase A** (mechanisch, von Claude ausgeführt): ~35 unveränderte Module
   aus `SigmaDock/` mit ihren echten Namen (z.B.
   `src_sigmadock_chem/fragmentation#macht die Aufteilung in Parts.py` →
   `chem/fragmentation.py`) nach `SigmaFlow_Development/src/sigmadock/`
   kopiert. Byte-Diff-Stichproben bestätigen Identität.
3. **Phase B** (Importe fixen, von Claude ausgeführt nach expliziter
   Nutzeranfrage): die 6 Flow-Matching-Dateien an ihre kanonischen Orte
   verschoben:
   - `R3_FlowMatcher.py` → `diff/r3_flow_matcher.py` (keine Import-Änderung,
     nutzte nur `torch`)
   - `so3_utils_adapted.py` → `diff/so3_utils.py` (keine Import-Änderung;
     jetzt DIE einzige `so3_utils.py` in diesem Package, `_adapted`-Suffix
     entfällt, keine Verwechslungsgefahr mehr mit dem Original, das hier gar
     nicht existiert)
   - `SO3_FlowMatcher.py` → `diff/so3_flow_matcher.py` (`import
     so3_utils_adapted` → `from sigmadock.diff import so3_utils`, 4
     Aufrufstellen umbenannt, veralteter Kommentar zum `so3_utils`-Bugfix
     inhaltlich aktualisiert)
   - `SE3_FlowMatcher.py` → `diff/se3_flow_matcher.py` (2 Importe auf
     `sigmadock.diff.r3_flow_matcher`/`so3_flow_matcher` umgestellt)
   - `sampling_adapted.py` → `diff/sampling.py` (2 Importe umgestellt)
   - `denoiser_adapted.py` → `diff/denoiser.py` (3 Importe umgestellt, 2
     Aufrufstellen `so3_utils_adapted.hat` → `so3_utils.hat`)
   - `trainer_adapted.py` → `trainer.py` (keine Import-Änderung, nutzte
     bereits durchgehend kanonische `sigmadock.xxx`-Importe)

   Konvention: **absolute Importe** (`from sigmadock.diff.xxx import Yyy`),
   nicht relative (`from .xxx import Yyy`) — konsistent mit dem Rest des
   Codebases (alle kopierten Original-Module nutzen ausschließlich absolute
   Importe).

4. **Aufräumen:** die jetzt doppelte alte flache Struktur (`src_sigmadock_
   {chem,core,diff,geo,net,torch_utils}/`, plus vorbestehende, bisher nicht
   dokumentierte Top-Level-Duplikate `__init__.py`/`config.py`/`data.py`/
   `datafronts.py`/`oracle.py`/`sampling_setup.py`/`utils.py`/
   `trainer_adapted.py` direkt unter `SigmaFlow_Development/`) wurde entfernt
   — aber **nur nach vollständiger Diff-Verifikation** jeder einzelnen Datei
   gegen ihren neuen Ort (reine Kopien: Byte-identisch; die 4 editierten
   Dateien: Diff zeigt exakt nur die dokumentierten Import-Änderungen, sonst
   nichts). Alle betroffenen Dateien waren git-`??` (nie committet) — daher
   vorab besonders sorgfältig geprüft, bevor gelöscht wurde.

**Verifikation:** `py_compile` über die komplette neue Struktur (61 Dateien)
fehlerfrei. `grep`-Sweep nach alten flachen Namen (`so3_utils_adapted`,
`denoiser_adapted`, unqualifizierte `_FlowMatcher`-Importe) über die gesamte
neue Struktur: keine Treffer. Echter Import-Test (`PYTHONPATH` auf
`src/`, `import sigmadock.diff.se3_flow_matcher`) **schlägt fehl** — aber aus
einem strukturell erwarteten, nicht selbst verursachten Grund: `sigmadock/
__init__.py` (unverändert aus dem echten Repo) importiert beim Package-Import
eager **alle** Submodule (`chem, core, diff, geo, net, torch_utils`), und
`core/data.py` braucht `torch_geometric` — das ist in dieser Umgebung nicht
installiert (nur `torch` 2.13.0+cpu und `numpy` 2.4.1 vorhanden,
`torch_geometric`/`pytorch_lightning` fehlen). Das ist Original-Verhalten des
echten Repos, keine Konsequenz unserer Restrukturierung, und wurde bewusst
NICHT "repariert" (CLAUDE.md §9: keine unrelated Refactors) — bleibt der
letzte Blocker: ohne vollen Dependency-Stack lässt sich `sigmadock` gar nicht
importieren, egal wie sauber die Struktur ist.

**Verbleibend für einen echten Testlauf:**
1. ~~`torch_geometric`/`pytorch_lightning` installieren~~ ✅ erledigt, siehe
   unten.
2. `scripts/train.py` übergibt beim Konstruieren des Denoisers kein
   `sigma_min` (kein CLI-Argument dafür) → `TypeError` bei Konstruktion,
   noch nicht behoben.
3. Phase C (optional, noch nicht angegangen): eigene `pyproject.toml` für
   `SigmaFlow_Development/` + `pip install -e .`, damit `import sigmadock`
   auch ohne manuelles `PYTHONPATH`-Setzen von überall funktioniert.

---

## Dependency-Stack installiert ✅ (2026-07-16) — erster echter Ausführungserfolg

**Umgebung:** Python 3.14.2 (einzige lokal verfügbare Version — kein
conda/venv/pyenv vorhanden). `pyproject.toml` verlangt offiziell `<3.13`,
aber alle benötigten Pakete haben inzwischen (neuere Releases als beim
Schreiben des Original-Repos) Wheels für 3.14 — Dry-Run vorab per `pip
install --dry-run` verifiziert, bevor wirklich installiert wurde.

**Installiert:** `numpy`, `torch` (2.13.0+cpu, war schon da), `pytorch-lightning`
2.6.5, `e3nn` 0.6.0, `torch-geometric` 2.8.0, `rdkit` 2026.3.3, `biopython`
1.87, `tqdm`, `matplotlib`, `scipy`, plus die `train`-Extras `wandb`,
`torchsummary`, `hydra-core`, `omegaconf`, `posebusters` (letztere werden von
`config.py`/`sampling_setup.py`/`scripts/sample.py` gebraucht, per `grep`
bestätigt).

**Bewusst NICHT installiert: `esm`.** Erster Versuch (`esm` mit im Batch)
scheiterte an dessen transitiver Abhängigkeit `biotraj`, die eine C-Extension
aus Quellcode bauen muss (`error: Microsoft Visual C++ 14.0 or greater is
required` — kein Build-Toolchain auf diesem Windows-Rechner installiert).
Vor einem Workaround erst geprüft, ob `esm` überhaupt auf dem kritischen Pfad
liegt: `extract_esm_embeddings.py` wird in `processing.py` nur **lazy,
innerhalb einer Funktion** importiert (`from sigmadock.chem.
extract_esm_embeddings import (...)`, Zeile 785), und nur falls der
Parameter `esm_embeddings` (Default `None`) tatsächlich gesetzt wird — für
einen ersten Testlauf mit dem Dummy-Datensatz nicht nötig. Deshalb `esm`
aus der Install-Liste entfernt, Rest erfolgreich installiert. Falls später
echte ESM3-Proteinembeddings gebraucht werden: entweder Visual C++ Build
Tools nachinstallieren, oder eine ältere/alternative `esm`-Version ohne
`biotraj`-Abhängigkeit suchen, oder conda nutzen (bringt vorgebaute Pakete
mit).

**Verifikation (echte Imports, nicht nur `py_compile`):**
```
PYTHONPATH=SigmaFlow_Development/src python -c "
import sigmadock                                          # OK
from sigmadock.diff.se3_flow_matcher import SE3_FlowMatcher  # OK
from sigmadock.diff.denoiser import SigmaDockDenoiser        # OK
from sigmadock.trainer import SigmaLightningModule            # OK
from sigmadock.net.model import EquiformerV2                  # OK (Jd.pt laedt)
"
```
Zusätzlich `SE3_FlowMatcher` (sample_init → conditional_probability_path →
euler_step) mit echten Tensoren innerhalb des voll geladenen Packages
ausgeführt — funktioniert, korrekte Shapes. **Das ist der erste Moment im
gesamten Projekt, in dem irgendein Teil des Codes tatsächlich lief statt nur
gelesen/`py_compile`-geprüft zu werden.**

Einzige Randnotiz: `EquiformerV2`-Import wirft vier harmlose
`FutureWarning`s (`torch.cuda.amp.autocast` ist in der installierten
`torch`-Version 2.13 deprecated zugunsten von `torch.amp.autocast('cuda',
...)`, betrifft `net/layer_norm.py`) — reine Versions-Drift zwischen dem
Alter des Original-Repos und der hier installierten, deutlich neueren
`torch`-Version, keine funktionale Auswirkung, nicht behoben (kein Teil der
Diffusion→Flow-Matching-Konversion, würde `net/` betreffen, das laut
CLAUDE.md §9 nicht angefasst werden soll).

**Verbleibend für einen echten Trainingslauf:** tatsächlich einen
Trainingslauf mit dem Dummy-Datensatz versuchen (bisher nur Imports +
Konstruktion getestet, noch kein `forward()`/`training_step()` mit echten
Daten durchlaufen), plus das ARC-SLURM-Skript (siehe unten, in Arbeit).

---

## `conf/experiments/dummy_train.yaml` ✅ (2026-07-16) — erste Trainings-Experiment-Config

`conf/experiments/dummy_crossdock.yaml` (Original) ist für Cross-Dock-
**Sampling** gedacht (`sdf_regex` matcht `query_*.sdf`, nicht die echte
gebundene Pose) — für **Training** ungeeignet, da keine gesicherte
Ground-Truth-Pose als Ziel. Neue Datei vom User selbst geschrieben (nach
ausführlicher Regex-Einführung von Grund auf, da erstes Python-Projekt und
Regex komplett neu):

```yaml
_target_: sigmadock.experiments.ExperimentConfig
name: "dummy_train"
dataset: "dummy_data"
pdb_regex: ".*_protein\\.pdb$"
sdf_regex: ".*_ligand\\.sdf$"
```

Verifiziert auf zwei Arten: (1) eigenes Testskript über alle 10
`notebooks/dummy_data/`-Ordner — jeder Ordner liefert genau 1 pdb- und 1
sdf-Treffer; (2) echter Aufruf von `sigmadock.config.get_experiment_config
("dummy_train", root_dir=...)` — Pfad löst korrekt auf, existiert,
Regex-Strings kommen nach YAML-Parsing korrekt als einfache `\.`-Escapes an.

Auch `conf/` und `notebooks/dummy_data/` wurden dafür nach
`SigmaFlow_Development/{conf,notebooks}/` gespiegelt (mechanisch, aus
`SigmaDock/` kopiert — `get_experiment_config` sucht `conf/` relativ zum
Ort von `config.py`, drei Ebenen hoch, muss also Geschwister von `src/`
sein).

**Nächster Schritt (in Arbeit, nicht blockierend):** SLURM-Batch-Skript für
einen ersten Testlauf auf dem Oxford-ARC-Cluster, referenziert
`--train_exps dummy_train`. Nutzer hat SSH-Zugang, Partition ("short"),
Modul/Conda-Setup (`module load Mamba`) bereits bekannt aus einer früheren,
erfolgreichen SigmaDock-Reproduktion (`/data/stat-cadd/shug8458/
SigmaDock_Reproduction_JulianMueller/sigmadock`, mit eigener Conda-Umgebung
`myenv`, dort SigmaDock per `pip install -e .` eingebunden).

**Wichtige Design-Entscheidung (mit User besprochen):** Da `myenv` das
*originale* SigmaDock bereits editable-installiert hat, würde eine geteilte
Umgebung `sigmadock` (alt) und `sigmadock` (neu, unseres) verwechseln können
— stiller Bug, kein Crash. Zwei Optionen besprochen: (A) `PYTHONPATH`
manuell voranstellen in derselben Umgebung (spart Platz, aber fragil, leicht
vergessen), (B) **komplett getrennte, neue Conda-Umgebung nur für
SigmaFlow**, symmetrisch zur bestehenden Struktur (`SigmaDock_Reproduction_
JulianMueller/sigmadock` + `myenv` ↔ `SigmaFlow_Development_JulianMueller/
SigmaFlow/SigmaFlow_Development` + neue Umgebung). **User hat sich für B
entschieden** — sauberer, keine Verwechslungsgefahr, nutzt dasselbe
`install.sh`-Muster, das der User schon vom Original kennt.

**Für Weg B fehlte noch die Möglichkeit, `pip install -e .` für
SigmaFlow_Development durchzuführen (Phase C, vorher als "optional"
zurückgestellt) — jetzt nachgeholt:**
- `SigmaFlow_Development/pyproject.toml` ✅ (vom User selbst geschrieben,
  nach ausführlicher Erklärung was `pyproject.toml`/`pip install -e .`
  bedeuten — erstes Mal, dass der User dieses Konzept sieht). Struktur
  identisch zu `SigmaDock/pyproject.toml` übernommen, `name`/`description`/
  `authors` anpasst zu SigmaFlow. `esm` bewusst **drinbehalten** (anders als
  hier lokal unter Windows) — auf Linux/ARC ist ein C-Compiler
  wahrscheinlich vorhanden, der Windows-spezifische `biotraj`-Bau-Fehler
  betrifft ARC vermutlich nicht; wird beim tatsächlichen Install-Log dort
  sichtbar, falls doch.
- `SigmaFlow_Development/LICENSE` ✅ (von Claude erstellt, nach Rückfrage) —
  BSD-3-Clause-Text von `SigmaDock/LICENSE.txt` übernommen (Original-
  Copyright Alvaro Prat Balasch bleibt erhalten, wie die Lizenz es
  vorschreibt für Ableitungen), mit einem Zusatzhinweis, dass SigmaFlow eine
  Ableitung ist.
- `SigmaFlow_Development/README.md` ✅ (von Claude erstellt) — neu
  geschrieben, nicht von SigmaDock kopiert.
- `SigmaFlow_Development/install.sh` ✅ (von Claude kopiert, unverändert —
  rein generisches Skript, installiert nur `torch` + `pip install -e .`,
  nichts Diffusions-Spezifisches).

`SigmaFlow_Development/` ist damit jetzt eine vollständige, für sich
installierbare Projektstruktur (`LICENSE`, `README.md`, `conf/`,
`install.sh`, `notebooks/`, `pyproject.toml`, `scripts/`, `src/`) — auch
`conf/` und `notebooks/dummy_data/` wurden dafür aus `SigmaDock/` gespiegelt
(mechanisch).

**Noch offen:** neue Conda-Umgebung auf ARC anlegen (Pfad-Konvention analog
zu `myenv`, noch zu klären), `bash install.sh` dort ausführen, dann erst das
eigentliche SLURM-Batch-Skript schreiben (Grundlagen — Warteschlangen-
Konzept, `#SBATCH`-Direktiven, Aufbau der Datei — bereits sehr ausführlich
und langsam erklärt, siehe Konversation; konkrete Werte für unseren
Testlauf noch nicht final zusammengesetzt).

---

## ✅✅✅ MEILENSTEIN (2026-07-16): Erster erfolgreicher End-zu-Ende-Trainingslauf auf ARC

Nach Umgebungs-Setup (`sigmaflow_env`, separate Conda-Umgebung auf ARC,
`/data/stat-cadd/shug8458/sigmaflow_env`, Python 3.11, `pip install -e
".[train]"` inkl. `esm`/`biotraj` — baute auf Linux problemlos, anders als
hier lokal unter Windows) und `SigmaFlow_Development/slurm/
train_dummy_test.sh` (minimaler Smoke-Test: 10 Dummy-Komplexe,
`batch_size=2`, CPU, `--offline_run`, `--debug`) lief der komplette
Trainingsloop **zum ersten Mal überhaupt mit echten Tensoren, echten
Daten, Ende-zu-Ende, ohne Absturz.**

**Zwei echte Bugs unterwegs gefunden und gefixt** (beide nur beim
tatsächlichen Ausführen entdeckbar, nicht durch Codelesen):
1. `#SBATCH --output`/`--error` verweisen auf `slurm_logs/`, aber SLURM
   öffnet diese Dateien, bevor irgendeine Zeile des Skripts läuft — ein
   `mkdir -p slurm_logs` *innerhalb* des Skripts kommt zu spät. Fix: Kommentar
   im Skript, Ordner muss vor `sbatch` manuell angelegt werden.
2. `source activate <pfad>` (ohne vorheriges Sourcen von `conda.sh`)
   aktivierte die Umgebung in einer nicht-interaktiven Login-Shell nur
   unvollständig: `$CONDA_PREFIX` zeigte korrekt auf `sigmaflow_env`, aber
   `PATH`/`which python` lösten trotzdem auf die Basis-Mamba-Modul-Installation
   (Python 3.10) auf — wodurch `sigmadock` **lautlos aus dem alten
   SigmaDock-Repo** (`SigmaDock_Reproduction_JulianMueller`) geladen wurde,
   nicht aus unserem. Genau das Risiko, vor dem beim "Weg A vs. Weg B"-
   Vergleich gewarnt wurde — trat trotz Weg B auf, wegen der Aktivierungs-
   Mechanik selbst, nicht wegen geteilter Umgebungen. Fix: Python-Interpreter
   der Umgebung über absoluten Pfad aufrufen (`/data/.../sigmaflow_env/bin/
   python`), PATH-Auflösung komplett umgangen. Zusätzlich Diagnose-Zeilen
   im Skript ergänzt (`which python`, Version, `sigmadock.__file__`), die
   das beim nächsten Mal sofort sichtbar machen würden.

**Verifizierte Ergebnisse aus dem echten Lauf:**
- `sigmadock` korrekt aus `SigmaFlow_Development/src/sigmadock/__init__.py`
  geladen (Diagnose-Zeile bestätigt), Python 3.11.15.
- `MetaFront` lud korrekt `total_pairs=10` für train/val/test (unsere
  `dummy_train.yaml` funktioniert im echten Trainingslauf, nicht nur im
  isolierten Test).
- `SigmaDockDenoiser`/`EquiformerV2` erfolgreich konstruiert (15.0M
  Parameter, vollständige Layer-Zusammenfassung im Log).
- 3 Trainings-Epochen (5 Batches/Epoche) liefen durch, Verluste mit
  **unseren** Flow-Matching-Namen geloggt (`loss_train/loss_R`,
  `loss_train/loss_trans`, `loss_val/loss_R`, `loss_val/loss_trans`,
  `loss_*/total`) — bestätigt, dass die komplette Kette
  `SE3_FlowMatcher.conditional_probability_path` → `denoiser._compute_
  vector_field` → `denoiser.compute_losses` → `trainer._shared_step` →
  PyTorch-Lightning-Backward-Pass in der Praxis funktioniert.
- Kein NaN/Inf (weder unser eigener Check in `compute_losses` noch
  `FullNaNCheckCallback` schlugen an).
- Lauf endete nach 3 Epochen durch **Early Stopping** (nicht Absturz, nicht
  Zeitlimit) — Ursache identifiziert: `--max_steps 5` fließt im
  *originalen*, unveränderten `scripts/train.py` in die
  Early-Stopping-Geduld-Berechnung ein (`patience = max_steps × early_
  stopping_patience_ratio ≈ 1.25`), unabhängig von der tatsächlichen
  Trainingsschritt-Begrenzung (die stattdessen über `max_epochs` läuft,
  welches `--max_steps` NICHT überschreibt, da `Trainer(...)` in
  `train.py` nur `max_epochs`, nie `max_steps` übergeben bekommt — eine
  Eigenart des Original-Codes, keine Folge unserer Konversion). Validierungs-
  verlust stieg über die 3 Epochen (erwartet bei nur 10 Beispielen/3
  Epochen, keine sinnvolle Lernkurve zu erwarten, für den Zweck dieses
  Smoke-Tests irrelevant).

**Einordnung:** Das ist der zentrale Meilenstein des gesamten bisherigen
Projekts — zum ersten Mal lief die komplette Diffusion→Flow-Matching-
Konversion (alle 6 Dateien) nicht nur syntaktisch/isoliert getestet,
sondern als vollständiger End-zu-Ende-Trainingslauf mit echten
Proteindaten. Verbleibende Arbeit ab hier ist Verfeinerung
(Hyperparameter, echte PoseBusters-Benchmarks, `scripts/sample.py`-Fix
analog zu `trainer.py`, GPU-Testlauf), nicht mehr grundlegende
Korrektheit der Konversion.

---

## `sigma_min`-Lücke geschlossen ✅ (2026-07-16)

**Ort:** `SigmaFlow_Development/src/sigmadock/config.py` (`RunConfig`-
Dataclass + `parse_args()`), `SigmaFlow_Development/scripts/train.py` (neu
angelegt, Kopie aus `SigmaDock/scripts/train.py`).

**Fix:** `sigma_min: float = 0.0` als neues Feld in `RunConfig` ergänzt
(Sektion "Flow matching", direkt vor "Rotation components"), passendes
`--sigma_min`-CLI-Argument ergänzt (gleiches Muster wie die anderen
`float`-Argumente: `default=None`, damit ein nicht gesetztes CLI-Flag den
Dataclass-Default nicht überschreibt). Dadurch landet `sigma_min`
automatisch in `args.__dict__`, das `train.py` unverändert per `**args.
__dict__` an `SigmaDockDenoiser(...)` durchreicht — **keine Änderung an
`train.py` selbst nötig** für diesen Teil.

**Zweiter, beim Testen entdeckter Bug (nicht vorher bekannt):**
`scripts/train.py`s `SigmaLightningModule(...)`-Aufruf übergab noch explizit
`trans_data_weight=args.trans_data_weight, rot_data_weight=args.
rot_data_weight` — diese landen (da `SigmaLightningModule.__init__` sie
nicht mehr kennt) in dessen `**kwargs`, das am Ende an `pl.LightningModule.
__init__(**kwargs)` weitergereicht wird → `TypeError: _DeviceDtypeModuleMixin.
__init__() got an unexpected keyword argument 'trans_data_weight'`. Nur
gefunden, weil tatsächlich konstruiert wurde, nicht durch Codelesen. Fix:
die beiden Zeilen aus dem `SigmaLightningModule(...)`-Aufruf entfernt.
Konsistent dazu auch `trans_data_weight`/`rot_data_weight` komplett aus
`RunConfig` + `parse_args()` entfernt (analog zur bereits getroffenen
Entscheidung bei der `trainer.py`-Anpassung) — sonst hätte man sie per CLI
setzen können, ohne dass sie irgendetwas bewirken (stiller Blindgänger).

**Verifikation (echte Konstruktion, nicht nur Import):**
```python
cfg = RunConfig()
denoiser = SigmaDockDenoiser(dummy_model, cache_path=..., cutoff_*=-1, **cfg.__dict__)
lightning_model = SigmaLightningModule(denoiser=denoiser, ..., fragment_scaling=cfg.fragment_scaling,
                                        trans_score_weight=cfg.trans_score_weight,
                                        rot_score_weight=cfg.rot_score_weight, ...)
lightning_model.on_fit_start()
```
Beide Konstruktionen + `on_fit_start()` laufen jetzt fehlerfrei durch —
exakt der Aufrufpfad, den `scripts/train.py` tatsächlich nutzt (nur
`equimodel` durch `nn.Linear(1,1)` als Platzhalter ersetzt, um nicht extra
ein volles `EquiformerV2` mit allen Args konstruieren zu müssen).

`scripts/train.py` liegt jetzt auch unter
`SigmaFlow_Development/scripts/train.py` (unverändert kopiert). Damit sind
alle bisher bekannten strukturellen/Interface-Lücken für einen Testlauf
geschlossen. Bekannte, separate, noch offene Baustelle (nicht Teil dieser
Anfrage, nicht angefasst): `scripts/sample.py` hat denselben `.diffuser.
_so3_diffuser.set_device(...)`-Bug wie `trainer.py` vor dessen Fix — noch
nicht kopiert/adaptiert, betrifft nur Sampling/Inferenz, nicht Training.

---

## Datei 5/5: `SigmaFlow_Development/src_sigmadock_diff/sampling_adapted.py` ✅

### Design-Entscheidungen

- **Duplikation `sample_notebook`/`sampler` bewusst NICHT aufgelöst** (User-
  Entscheidung) — beide Funktionen sind ~95% identisch im Original
  (`sample_notebook` trackt zusätzlich `all_edges`, `sampler` ist die von
  `scripts/sample.py` tatsächlich genutzte Produktions-Variante). Beide
  werden separat, parallel nach demselben Muster umgebaut — näher am
  Original, mehr Aufwand, aber CLAUDE.md-konform (minimal nötige Änderung
  statt Refactoring).
- **Zeit-Diskretisierung radikal vereinfacht:** `rho`-Potenzgesetz (Karras/
  EDM-Stil) durch gleichmäßig verteilte Zeitschritte ersetzt
  (`torch.linspace(t_min, t_max, num_steps)`, aufsteigend, kein `rho`-
  Exponent) — passt zu CLAUDE.md §5 ("simplest correct method first"),
  spätere Verfeinerung möglich (analog zur bereits im Original als TODO
  vermerkten Heun's-Method-Erweiterung, `solver="heun"` war im Original
  ohnehin nie implementiert, nur als Parameter vorhanden).
- **Zeitrichtung umgedreht:** Original iteriert absteigend `t_max→t_min`
  (deren Konvention: `t=1`=Rauschen, `t=0`=Daten). Unsere ODE integriert
  `0→1`, Schleife läuft jetzt aufsteigend — erste praktische Konsequenz der
  ganz am Anfang getroffenen Zeitkonventions-Entscheidung.
- **`noise_scale`/`noise_decay`/quadratische Rausch-Skalierung**: komplett
  entfernt, deterministische ODE braucht das nicht.
- **`_compute_true_scores`/`use_true_scores`-Diagnosezweig (ReDocking-
  Szenario mit bekannter Pose) bewusst NICHT entfernt** (User-Entscheidung,
  abweichend von meinem ursprünglichen Vorschlag "erstmal weglassen") —
  wird mitgepflegt, korrekt an `_compute_true_vector_field` angebunden.
  Spannender Zusammenhang zu einer früheren Diskussion (SO3_FlowMatcher,
  Datei 2): genau hier wäre die alternative `u_t`-Formel
  (`log(R_t^T R_1)/(1-t)`) relevant, aktuell nutzt `_compute_true_vector_
  field` aber die bereits vorhandene, direkte Formel — nicht weiter vertieft.

### `sample_notebook` — fertig (Kernpfad, ohne `sampler`)

Umbau: `denoiser.diffuser.sample_ref`→`denoiser.flow_matcher.sample_init`
(Dict statt Tupel!), `denoiser._compute_scores`→`_compute_vector_field`,
`denoiser._compute_true_scores`→`_compute_true_vector_field`,
`denoiser._get_scalings`-Aufruf komplett entfernt (Methode existiert in
Datei 4 nicht mehr), `denoiser.diffuser.reverse`→`denoiser.flow_matcher.
euler_step` (kein `noise_scale`/`t` mehr nötig — nur Position, Geschwindig-
keit, `dt`). **Wichtige Bestätigung:** `dt` ist im Sampling-Loop ein
**Skalar pro Schritt** (`dt = timesteps[i+1]-timesteps[i]`, für den ganzen
Batch gleich), kein Pro-Fragment-Tensor — die in `STATUS.md` offen notierte
Frage zu `euler_step` mit Pro-Fragment-`dt` stellt sich hier NICHT, unsere
bestehenden `euler_step`-Methoden (Dateien 1-3) passen unverändert.

Mehrere Bugs gefunden und korrigiert während der Review (alle aus derselben
Fehlerfamilie: Namens-Drift zwischen alter Diffusions- und neuer Flow-
Matching-Konvention beim manuellen Umbenennen):
- Dict direkt in zwei Variablen entpackt (`a, b = irgendein_dict_mit_2_keys`)
  statt über `dict["key"]` zuzugreifen — entpackt lautlos die **Schlüssel**
  (Strings), nicht die Werte. Kein Crash, falsche Werte, erst später
  bemerkbar — wichtiges Python-Konzept, neu erklärt.
- `R_0`-Variable doppelt belegt: `_get_initial_states` lieferte ursprünglich
  `T_0`/`R_0` (Original-Konvention: Index 0 = Daten), wir nennen das jetzt
  `trans_1`/`R_1` (unsere Konvention: Index 1 = Daten) — aber an mehreren
  Stellen wurde aus Gewohnheit trotzdem wieder `R_0`/`T_0` verwendet, wo
  eigentlich `R_1`/`trans_1` gemeint war (bzw. `T_0` war schlicht nicht mehr
  definiert). Trat mehrfach unabhängig auf, gleiches Muster.
- Closure-Bug: `_reverse_step`-Parameter `R_0` wurde im Funktionskörper
  nicht benutzt, stattdessen wurde durch Python-Scoping automatisch die
  gleichnamige äußere Variable `R_1` gelesen (Parameter komplett wirkungslos,
  kein Fehler, stiller Bug). Gefixt: `R_1=R_0` (Schlüsselwort bleibt `R_1`
  wegen `_compute_fragment_dynamics`s Parametername, Wert kommt aus dem
  lokalen Parameter `R_0`).
- `T_next`/`R_next` nie aus dem `euler_step`-Rückgabe-Dict entpackt (`step_
  result = ...` berechnet, aber nie `["trans_new"]`/`["R_new"]` draus
  gelesen) → `NameError`.
- Danach: `trans_next` (beim Entpacken neu gewählter Name) vs. `T_next`
  (in zwei nachfolgenden Zeilen weiterhin benutzt) — Inkonsistenz,
  behoben durch einheitlich `trans_next`.

**Verbleibend, niedrige Priorität (für Aufräumrunde vorgemerkt):** Beim
`_reverse_step`-Aufruf wird `R_0` (Rausch-/Quellrotation) übergeben, wo
konzeptionell `R_1` (echte Daten, passend zum `pos_0`-Bezugssystem)
reingehören würde — wirkt sich nur aus, wenn `denoiser.verbose=True`
(reiner Trägheitstensor-Sanity-Check, keine Auswirkung auf die Trajektorie).

### `sampler` — fertig ✅

Exakt dasselbe Umbau-Muster wie `sample_notebook`, dieselbe Fehlerfamilie
tauchte unabhängig nochmal auf (bestätigt: kein Zufall, sondern ein
systematisches Risiko beim manuellen Parallel-Umbauen zweier ähnlicher
Funktionen):
- Type-Hint `SE3Diffuser`→`SigmaDockDenoiser` (gleicher Fund wie bei
  `sample_notebook`).
- `_get_initial_states`-Rückgabe erneut fälschlich `R_0` statt `R_1`
  genannt → Überschreiben durch die gesampelte Quellrotation + `NameError`
  an mehreren Folgestellen (identisches Muster wie bei `sample_notebook`,
  unabhängig wieder aufgetreten).
- Zeit-Diskretisierung vereinfacht (`power`-Zweig jetzt aufsteigend/ohne
  `rho`; `edm`-Zweig bewusst NICHT entfernt, bleibt als toter, per Default
  nicht erreichter Code stehen — geringe Priorität).
- `_reverse_step`-Closure-Bug (`R_1=R_0` statt `R_1=R_1`) — hier aber
  **richtig gelöst auf ersten Versuch** (aus `sample_notebook` gelernt).
- **Eigenständiger neuer Fund:** `_compute_true_vector_field`-Aufruf in der
  Hauptschleife hatte `R_1=R_0` (Rauschquelle statt echte Daten) — anders
  als der niedrigprioritäre `_reverse_step`-Fall wird dieser Wert **bei
  jedem Schritt unconditional berechnet** und fließt sowohl in den
  `use_true_vector_field=True`-Pfad als auch ins Verlust-Logging ein — real
  wirksamer Bug, kein reiner Diagnose-Nebeneffekt. Behoben zu `R_1=R_1`.

Finaler Sweep über die komplette Datei (`grep` nach `.diffuser`,
`_compute_scores`, `_compute_true_scores`, `_get_scalings`, `T_next`, `T_0`,
`use_true_scores`): keine Treffer mehr. `py_compile`: fehlerfrei.

### Fahrplan-Schritt 5/5 abgeschlossen

Datei 5 fachlich vollständig für den Kernpfad (deterministische Euler-ODE-
Sampling-Schleife, beide Funktionen `sample_notebook`/`sampler`). Damit ist
die **komplette 5-Dateien-Konversion SigmaDock→SigmaFlow strukturell
fertig** — vorbehaltlich eines echten Laufs (siehe unten) und der
aufgeschobenen Namensaufräumrunden.

### Verbleibend, niedrige Priorität, für spätere Aufräumrunde gesammelt

- Beide `_reverse_step`-Aufrufe übergeben `R_0` (Rauschquelle) statt `R_1`
  (echte Daten) an den nur `verbose`-relevanten Trägheitstensor-Sanity-
  Check — keine Auswirkung auf die Trajektorie.
- `noise_scales`/`noise_scale`-Berechnung ist toter Code (unser `euler_step`
  kennt kein `noise_scale`) — in beiden Funktionen.
- `edm`-Diskretisierungs-Zweig in `sampler` nicht entfernt (nur in
  `sample_notebook`), aktuell nicht erreichbar (Default ist `"power"`).
- Veraltete Docstrings ("Instance of SE3Diffuser" etc.) in beiden Funktionen.
- `_get_scalings`-Methode selbst existiert noch als toter Code in
  `denoiser_adapted.py` (nur ihre Aufrufe wurden entfernt).
- `_compute_vector_field`s ungenutzte `sampled`/`t_batch`-Parameter
  (Datei 4).

---

## Teaching-Kontext (siehe auch CLAUDE.md §3a)

- Nutzer hat quasi keine Python-Vorerfahrung (dies ist im Wesentlichen das
  erste große Python-Projekt). Jedes neue Sprachkonzept muss erklärt werden,
  wenn es zum ersten Mal auftaucht.
- Bereits erklärte Konzepte (müssen nicht erneut von null erklärt werden,
  aber bei Bedarf kurz auffrischen): `class`/`__init__`/`self`, Attribut-
  Zuweisung vs. Parameter-Shadowing, einrückungsbasierte Blockstruktur,
  Tensor-Broadcasting mit `[:, None]`, Funktionen mit `tuple`-Rückgabe +
  Unpacking, `torch.randn`, Funktionsaufruf `()` vs. Indizierung `[]`,
  Docstrings, `dict` (Schlüssel-Wert-Container, `{"key": value}`, Zugriff über
  `d["key"]`, Typ-Annotation `dict[str, T]`, Vorteil ggü. Tupel: selbst-
  dokumentierender Zugriff statt Positions-Merken), `from modul import Name`
  (direkter Klassen-/Funktionsimport ohne Präfix) vs. `import modul` (mit
  Präfix, `modul.funktion()` — üblich bei "Werkzeugkasten"-Modulen mit vielen
  Funktionen wie `so3_utils_adapted`).
- Workflow, den der Nutzer bevorzugt: Nutzer implementiert einen Baustein,
  antwortet "okay" (oder beschreibt das Problem). **Ich prüfe den Code danach
  immer selbst** (Datei lesen, wo sinnvoll auch tatsächlich per Bash
  ausführen/testen) — "okay" wird nicht blind akzeptiert. Bugs werden mit
  Erklärung des zugrundeliegenden Python/PyTorch-Konzepts zurückgemeldet,
  nicht nur als Korrektur.

---

# 📚 Theory-Summary-Strang (Sessions bis 2026-08-15)

Chronologisch, neueste zuletzt. Alle Zahlen sind gemessen, nicht geschätzt.

## Runde: Output-Head-Audit (Commit `fe61d7a`, 166 → 174 S.)

**Strukturbefund, der die Fragestellung korrigierte.** `EquiformerV2.forward`
hat genau **einen** geometrischen Output: `forces [N,3]`, ein 3-Vektor je
Atom aus dem ℓ=1-Block via `narrow(1,1,3)`. Translation und Rotation sind
**keine Heads**, sondern werden außerhalb des Netzes durch klassische
Starrkörpermechanik abgeleitet. **Es gibt keinen Torsion-Head** —
`torsion`/`dihedral` kommen in `diff/` null mal vor; Torsionsfreiheit steckt
in der Fragmentierung.

**Zentraler Befund — die Gruppenwirkung ist Konjugation, nicht Linksmult.**
`R_t` ist eine Delta-Rotation gegen `pos_0`; `pos_0` ist Netzinput und dreht
mit; `R_1` ist hart die Identität. Aus `_apply_transformations` gemessen:

| Substitution | erzeugt Q·pos_t |
|---|---|
| `R_t → Q R_t` | 1.06 ✗ |
| `R_t → Q R_t Qᵀ` | **3.7e-16** ✓ |

Folge: `pred_u_t_R` **ist** body-frame im Sinn `Ṙ = R ω̂` (fixiert durch
`euler_step`: `R_t @ exp(v dt)`), aber **nicht invariant**:

| Hypothese | Residuum |
|---|---|
| invariant | 1.20 ✗ |
| `u → Q u Qᵀ` | **2.3e-15** ✓ |

Ziel transformiert identisch (8.3e-15) → Loss invariant. **Frame Fix korrekt.**

> **Merksatz, im Dokument als Caution:** „body-frame" fixiert die Beziehung zu
> `Ṙ`, **nicht** das Transformationsgesetz — das hängt auch daran, wie die
> Gruppe auf `R` wirkt. Eine frühere Dokumentaussage („body-frame Output
> bleibt unverändert") war dadurch falsch und wurde korrigiert.

Volle Kette (float64, 12 Haar-Rotationen), alles 1e-15: `total_force` 4.5e-15,
`torque` 1.3e-15, Trägheit `I→QIQᵀ` 8.6e-16, `delta_W` 2.4e-15,
`omega` 2.6e-15.

**Widerlegt:** die yzx-Permutationshypothese (1.49). Grund, der eine offene
Frage schloss: `init_edge_rot_mat` richtet auf `e_2` aus, und in der
e3nn-Ordnung (y,z,x) ist der zweite Slot genau m=0. Die beiden seltsam
wirkenden Konventionen heben sich auf.

**Nebenbefund, `PENDING ARC VALIDATION`:** Die S²-Aktivierung sitzt zwischen
`_rotate` (Z.250) und `_rotate_inv` (Z.324), also **im Kantenframe**, und ihr
Gitter ist nicht invariant unter Drehung um die Kantenachse.
`init_edge_rot_mat` zieht diese Achse per `torch.rand_like` bei jedem Forward
neu. Bei Produktionsgröße (128 Kanäle, 6 Blöcke, 15 M Parameter):

| | S² (Default) | Gate |
|---|---|---|
| Nicht-Determinismus f(x) vs f(x) | **1.06e-01** | 1.4e-15 |
| Äquivarianz f(Qx) vs Q f(x) | **9.4e-02** | 5.3e-15 |

Der Forward ist im eval-Modus nicht deterministisch, ~10 % relativ. **Gemessen
bei zufälliger Initialisierung** — ob Training das unterdrückt, ist offen und
braucht einen Checkpoint. Betrifft SigmaDock und SigmaFlow gleich, erklärt
also **keinen Unterschied** zwischen ihnen. Nicht vorschnell als Erklärung für
Performance-Unterschiede verwenden (User-Vorgabe).

## Runde: Engine-Swap-Audit (Commit `826d2c5`, 174 → 181 S.)

Dreischichtige Trennung: **A** geometrischer Backbone, **B** mechanische
Reduktion, **C** generative Semantik.

**Layer A ist byte-identisch.** `diff` über alle 17 Dateien in `net/`
inklusive `model.py`: **0 abweichende Zeilen**. Parameterzahl damit
konstruktionsbedingt identisch. Außerhalb: `oracle.py` 5 Zeilen (nur
Kommentar), `data.py` 13 (nur Fehlerausgabe + `None`-Check),
`chem/processing.py` 0, `diff/so3_utils.py` **2 Zeilen (ein Clamp)**.

**Layer B** formelgleich. Zwei Abweichungen vom Lehrbuch in *beiden* Modellen:
Einheitsmassen (`m_i = 1`) und zwei Regularisierungen.

**Layer C** vollständig ersetzt, konsistent:
- **SigmaDock lernt nicht den Score**, sondern den varianznormierten:
  `pred = F·(1/σ)`, `λ = σ²`, kürzt sich zu `L = ‖F − σ·s_true‖²`.
- SigmaFlow hat kein λ — korrekt.
- Beide trainieren **zwei** aktive Terme: SigmaDocks `T0`/`R0` haben
  Default-Gewicht 0.0.
- Zeitkonvention gespiegelt (SigmaDock t=0 Daten, SigmaFlow t=1 Daten), beide
  füttern demselben Encoder das rohe `t` — kein σ, kein log σ. Sauber.

Numerisch (float64): `u_t` ist wirklich `ẋ_t` ≤ 7e-11; `u_t^R` ist wirklich
`R_tᵀṘ_t` ≤ 3e-10; Endpunkte exakt; **Euler trifft den Endpunkt bei jeder
Schrittweite** ≤ 3e-16 → der Integrator implementiert genau `x + dt·v` bzw.
`R @ exp(dt·v)`, **kein geerbter σ-Faktor hat den Port überlebt**.

**Zwei Confounder (für die Thesis wichtig):**
1. **Rotations-Quellverteilung stimmt nicht überein.** IGSO(3) bei σ_max=1.5
   hat mittleren Winkel **115.0°**, Haar **126.5°**, Totalvariation **0.130**.
   Translation dagegen praktisch identisch (VP-Terminalvarianz 0.99996).
2. **Gewichte 1.0 : 0.5 übernommen, ihre Rechtfertigung nicht.** In SigmaDock
   saßen sie auf λ = σ²; in SigmaFlow multiplizieren sie zwei rohe MSEs in
   verschiedenen Einheiten. Gemessen `E‖u_R‖²_F = 10.6` (datenunabhängig),
   `E‖u_trans‖²` je nach COM-Streuung 3.8–15.1.

**Weiterer Befund — die Log-Abbildung ist nicht exakt.** Das FM-Rotationsziel
*ist* `log(R_0ᵀR_1)`. Über 40 000 Haar-Rotationen in float64: Median 5e-7,
99 % 2.8e-3, Max 2e-2. Zwei Ursachen:
- `Omega()` multipliziert die Spur mit `(1 − 1e-6)` vor dem `arccos` →
  verzerrt **jeden** Winkel; bei 90° exakt 5.00e-07 rad.
- π-Zweig schaltet erst bei `|θ−π| < 1e-2` zu; dazwischen (178–179.4°) wird
  `θ/(2 sin θ)` benutzt → bei 179°: 4.3e-3.

Der Kommentar in `so3_flow_matcher.py`, der Clamp sei *„fixed"*, ist zu stark.
**Einordnung: erklärt den Rotationsausfall NICHT** (138° vs 132° Zufall ist
O(1), das hier ist relativ 2e-7 typisch).

`sigma_min` endgültig tot: `SO3_FlowMatcher` nimmt das Argument nicht
entgegen. Terminologie korrigiert: `forces` ist ein Name, keine Physik —
korrekt ist *atomweises ℓ=1-äquivariantes Vektorfeld*.

**Urteil: „Mostly, with the following confounders."**

## Runde: Diffusion didaktisch neu aufgebaut (Commit `cd16b4c`, 181 → 196 S.)

Erste Runde unter der neuen Theory-first-Priorität. **Kein
Implementierungsdetail angefasst.**

Neues Kapitel **„From Random Walks to Stochastic Differential Equations"**,
vor `chap:diffusion` eingefügt (`\label{chap:sde-toolbox}`):
- Diskrete Rauschkette als Ausgangspunkt.
- **Warum √Δt**: vollständige Skalenrechnung `Var(S_n) = σ²T·Δt^(2α−1)`, nur
  α = ½ überlebt. Tabelle aller drei Fälle, numerisch bestätigt.
- Brownsche Bewegung aus den drei Eigenschaften der Teilsummen hergeleitet.
- Nirgends differenzierbar in zwei Zeilen → Caution: `dW_t` ist kein
  Differential; SDE = Integralgleichung.
- Itô-Integral auf benötigtem Niveau (linker Randpunkt, Itô-Isometrie mit
  Beweis). **Keine** Martingaltheorie.
- Euler–Maruyama in **beiden** Richtungen gelesen.
- OU-Prozess vollständig durchgerechnet.
- Drift vs. Diffusion, mit Auflösung des Scheinwiderspruchs (kohärente vs.
  inkohärente Summation).
- Pfad vs. Marginal als eigener Abschnitt.
- Fokker–Planck motiviert (1D-Massenerhaltung, dann Laplace-Term; ½ aus dem
  Taylorkoeffizienten einer Faltung).
- **DDPM → VP-SDE** und **SMLD → VE-SDE** vollständig hergeleitet.

Drei Lücken im bestehenden Kapitel geschlossen:
- `sec:why-score-appears`: Gegenbeispiel `dX = dW` (unter Zeitumkehr dieselbe
  Gleichung, streut weiter) → Umkehr kann nicht pfadweise sein → der Score ist
  die weggeworfene Information. Mit Vorzeichenprobe.
- `sec:pf-ode-motivation`: Staubwolken-Bild; Faktor ½ aus
  `Δp = ∇·(p ∇log p)`.
- `sec:score-intuition`: Score als geometrisches Objekt vor jedem Algorithmus,
  1D-Gauß mit drei Ablesungen. Die dritte (explodiert für σ→0) erklärt direkt
  SigmaDocks 1/σ-Skalierung.
- `sec:sde-vs-pfode-table`: Vergleichstabelle + Caution „deterministisch ist
  nicht besser, sondern ein anderer Handel".

## Runde: Flow Matching didaktisch neu aufgebaut (2026-08-16, 196 → 230 S.)

Phase 2 der Theory-first-Priorität. **Kein Implementierungsdetail angefasst,
kein Label gelöscht** (alle 19 extern zitierten Part-VI-Labels erhalten).

Kapitel 11 („Continuous Normalizing Flows and Flow Matching in R^d")
vollständig neu aufgebaut: **7 → 39 Seiten**, 4 → **20 Sections + 21
Subsections**. Reihenfolge jetzt Theorie-zuerst: Transportproblem →
Probability Path → Flow → Continuity Equation → Nichteindeutigkeit →
marginales Feld unerreichbar → Conditional Paths → Marginalisierungssatz →
L²-Projektion → CFM-Loss → linearer Pfad → drei Worked Examples →
Gauß-Pfade → Score-vs-Velocity → **Diffusionsbrücke** → Parameterizations →
Sampling → Pfad-Design/Kopplungen → Synthese → Missverständnisse →
numerische Verifikation → Übergang zu Riemannian.

**Neu hergeleitet (nicht zitiert):**
- Continuity Equation zuerst in Integralform über ein Intervall
  (Ein-/Ausfluss an den Rändern), dann punktweise, dann `R^d`.
- **Nichteindeutigkeit von `u_t`**: `p_t ≡ N(0,I_2)` wird sowohl von `u=0` als
  auch vom Rotationsfeld `(-x_2,x_1)` erzeugt. Räumt das Missverständnis
  „zu jedem `p_t` gehört genau ein `u_t`" aus.
- **L²-Projektionslemma** mit vollem Beweis (Kreuzterm verschwindet per
  Tower-Property), daraus CFM als Korollar. Ist jetzt die *primäre*
  Beweisführung; der alte Expansionsbeweis steht als Remark daneben.
- **Loss-Floor**: `L_CFM(θ*) = E[Var(u_t(X_t|Z)|X_t,t)] > 0`. Ein Plateau
  über null ist erwartetes Verhalten, kein Underfitting. Numerisch belegt
  (1.5821 vs. exakt 0 für `L_FM`).
- Allgemeiner Gauß-Pfad `X_t = α_t z + σ_t ε` samt Feld
  `u = α̇z + (σ̇/σ)(x-αz)`; linearer Pfad als Spezialfall.
- **Score↔Velocity**: `u = a_t·score + b_t·x` mit
  `a_t = σ²α̇/α - σ̇σ`, `b_t = α̇/α` — *dieselben* Koeffizienten conditional
  wie marginal. Plus Tweedie/Denoiser in beiden Formen.
- **Diffusionsbrücke, der wichtigste Teil**: die Rand-Marginalen eines
  Diffusionsmodells *sind* rückwärts gelesen ein Gauß-Pfad,
  `α_t = α_diff(1-t)`, `σ_t = σ_diff(1-t)`. Bewiesen, dass die PF-ODE-
  Geschwindigkeit exakt `a_t·score + b_t·x` ist — für VP *und* VE. Der ½
  aus Fokker–Planck fällt dabei aus `α²+σ²=1` heraus, ganz ohne Itô.
- **Interkonvertierbarkeit** aller fünf Parameterisierungen ⟺
  `Δ_t = α̇σ - ασ̇ = σ²·d/dt(α/σ) ≠ 0` ⟺ **SNR streng monoton**.

**Namenskollision `v` sauber aufgelöst:** auf dem trigonometrischen
VP-Pfad gilt `u_t = φ̇_t · v` (Skalar, **negativ**, weil `v` in
Rausch-Richtung definiert ist). Auf dem linearen Pfad sind `u` und `v`
*nicht* parallel — außer bei genau `t = ½`, wo `u = -2v`. Beides bewiesen
und numerisch belegt.

**Zwei eigene Fehler beim Nachrechnen gefunden und korrigiert**
(vor dem Commit, siehe Report): (a) Behauptung „`u`,`v` nirgends parallel
auf dem linearen Pfad" war falsch — `2t-1=0` bei `t=½`; (b) Vorzeichen von
`C` in `thm:cfm-equivalence-scratch` war so formuliert, dass `C` negativ
sein konnte, im Widerspruch zur Floor-Aussage. Jetzt `L_CFM = L_FM + C`,
`C ≥ 0`.

**Fehlende Primärquellen dokumentiert** (in `rem:fm-sources` und
`caution:v-source`): Lipman et al. 2023 (Original-FM) und Salimans & Ho
2022 (`v`-Parameterisierung) liegen **nicht** in `papers/`. Die
`v`-Definition ist die einzige importierte, hier nicht verifizierbare
Definition; alles daraus Abgeleitete ist eigenständige Algebra.

**Verwendet:** Holderrieth & Erives §3–§4 (tatsächlich gelesen, S. 1–8,
14–33), Song et al. 2021 für die Brücke. Deren Prop. 1 (`a_t,b_t`)
unabhängig nachgerechnet und bestätigt.

## Runde: Geometry Revisit + Riemannian FM (2026-08-16, 230 → 263 S.)

Phase 3. **Kein Implementierungsdetail angefasst, kein Label gelöscht**
(alle 17 extern zitierten Kapitel-12-Labels erhalten, 0 undefinierte
Referenzen, 0 doppelte Labels).

### Kapitel 3 (Geometrie): gezielt erweitert, nicht neu geschrieben

Vor der Arbeit gegen die Frage geprüft „Was muss ein Leser verstanden haben,
bevor er RFM lernt?". Sechs echte Lücken gefunden und geschlossen:

1. **Kovariante Ableitung fehlte komplett.** `def:geodesic-scratch` benutzte
   `∇_γ̇ γ̇ = 0`, ohne dass `∇` je erklärt wurde — die Geodätendefinition war
   faktisch unlesbar. Jetzt konkret für eingebettete Mannigfaltigkeiten:
   `DV/dτ = P_{T_xM}(dV/dτ)`, Tangentialprojektion der gewöhnlichen
   Ableitung. Daraus fällt heraus: Geodäte ⟺ `γ̈ ⊥ T_γM`, und **konstante
   Geschwindigkeit ist eine Folgerung, keine Zusatzannahme**.
2. **Tangentialbündel und Vektorfeld auf einer Mannigfaltigkeit** waren nie
   definiert. Jetzt `def:tangent-bundle`, `def:vector-field-manifold`, plus
   `prop:flow-stays-on-manifold` (ein tangentiales Feld verlässt die
   Mannigfaltigkeit nie — Beweis über `DF(X_t)u_t(X_t)=0`).
3. **Cut Locus** war nie benannt (nur „injectivity radius", einmal).
   Jetzt `def:cut-locus` + `ex:cut-locus-three` (S¹, S², SO(3)) +
   `lem:subarc-minimising`.
4. **Riemannscher Gradient, Divergenz, Volumenmaß** fehlten ganz. Neue
   Section `sec:manifold-calculus`. Volumenmaß über die Gram-Determinante
   motiviert (`√|g|` ist erzwungen, nicht Konvention), Divergenz erst
   intuitiv (lokale Volumenexpansion), dann Koordinatenformel Symbol für
   Symbol, dann euklidischer Spezialfall, dann Polarkoordinaten als Check.
5. **Skalierungseigenschaft der Exp-Map** (`γ_v(t) = Exp_x(tv)`) war nie
   ausgesprochen — sie ist aber genau das, was `Exp_{x_0}(t Log_{x_0}(x_1))`
   überhaupt sinnvoll macht. Jetzt `lem:geodesic-reparam` +
   `prop:exp-scaling` + `cor:geodesic-interp-general`.
6. **S²-Beispiel für Exp/Log** fehlte (nur S¹ war da). Jetzt geschlossene
   Formeln mit Herleitung, Zahlen, und der Stelle wo es bricht (Antipode).

**Bonus: eine geerbte Zitat-Lücke geschlossen.** `prop:geodesic-oneparam`
(Einparameter-Untergruppen sind Geodäten) war mit „do Carmo Ch. 11"
zitiert. Mit der Projektionsdefinition der kovarianten Ableitung ist der
Beweis elementar: die induzierte Metrik von `⟨A,B⟩=½tr(AᵀB)` **ist** die
bi-invariante Metrik des Dokuments, und `R̈ = RΞ²` ist orthogonal zu
`T_RSO(3)`, weil `Ξ²` symmetrisch und `Ω` schiefsymmetrisch ist. Jetzt als
`prop:so3-geodesic-proof` vollständig bewiesen.

**Produktmetrik-Gewicht λ** ausgebaut (`prop:lambda-effect`): λ ändert
(i) die Geodäten **nicht**, (ii) das Volumenmaß nur um `λ^{3/2}`,
(iii) den Loss **doch** — ein Loss mit Rotationsgewicht λ *ist* die
quadrierte Norm von `g^λ` —, (iv) den exakten Minimierer **nicht**.
Damit ist die User-Frage „ist Loss-Weighting eine Metrikwahl?" präzise
beantwortet: **ja und gleichzeitig egal am Populationsoptimum**; die ganze
Wirkung sitzt in der endlichen Approximation.

### Kapitel 12 (Riemannian FM): von Grund auf neu

12 → **45 Seiten**, 4 → **16 Sections**. Reihenfolge: was bricht →
tangentialwertige Dynamik → Riemannsche Probability Paths → Riemannsche
Continuity Equation → konditionale Geodäten + Geschwindigkeit →
Marginalisierung + L²-Projektion → Vergleichstabelle → S¹ → S² → SO(3) →
Produktmannigfaltigkeit → CNF/Likelihood → Diffusion vs. RFM → ein Bild →
Missverständnisse → numerische Verifikation → Brücke zu SigmaFlow.

**Die ausstehende Lücke aus STATUS ist geschlossen.** `prop:geodesic-vf-general`
(allgemeiner Beweis, dass `u_t(X_t) = Log_{X_t}(x_1)/(1-t)`) war bisher nur
für SO(3) ausgeführt, mit der Begründung, der allgemeine Fall brauche
Jacobi-Felder. **Das stimmte nicht.** Der Beweis braucht genau zwei Lemmata,
beide elementar: affine Umparametrisierung erhält Geodäten, und ein
Teilbogen einer minimierenden Geodäte ist minimierend. Damit:
`σ(s) := γ(t+s(1-t))` ist die minimierende Geodäte von `X_t` nach `x_1` mit
`σ̇(0) = (1-t)Ẋ_t`, also `x_1 = Exp_{X_t}((1-t)Ẋ_t)`, also
`Log_{X_t}(x_1) = (1-t)Ẋ_t`. Fünf Zeilen. Als `rem:no-jacobi` ausdrücklich
festgehalten, dass die frühere Begründung die Voraussetzung überschätzt hat.

**Riemannsche Continuity Equation** (zweite offene Lücke) jetzt zweifach
hergeleitet: einmal über Massenerhaltung + Riemannschen Divergenzsatz,
einmal in Koordinaten — dort zeigt sich, dass sie **die flache
Continuity Equation für die Koordinatendichte `p√|g|`** ist und die beiden
`√|g|` in `div_g` genau Hin- und Rückumrechnung sind.

**Der eine Punkt, an dem der euklidische Beweis wirklich Geometrie benutzte,
ist benannt** (`sec:rfm-averaging`): `E[u_t(x|Z) | X_t = x]` mittelt
Tangentialvektoren — erlaubt, weil die Bedingung den Punkt `x` fixiert und
alle gemittelten Vektoren damit in *demselben* `T_xM` liegen. Mit
`caution:rfm-averaging-trap`: eine unbedingte Erwartung `E[Ẋ_t]` wäre
**nicht** definiert.

**Parallel Transport** wird nicht entwickelt, aber `rem:no-parallel-transport`
erklärt warum: jeder Vergleich von Tangentialvektoren fixiert vorher den
Punkt, und die Body-Frame-Trivialisierung liefert die Identifikation ohnehin.

**Cut Locus theoretisch vs. numerisch sauber getrennt**
(`sec:rfm-so3-cut-locus`): bei θ=π exakt ist der Pfad nicht eindeutig, aber
Maß null → für einen Erwartungswert irrelevant. *Nahe* π ist die Formel
schlecht konditioniert, und das ist auf SO(3) der **meistbesuchte** Bereich
(`p(θ) ∝ 1−cos θ` maximal bei π) — 9 % der Haar-Paare. Als eigene Tabelle.

**Trennung Theorie / SigmaFlow-Konvention eingehalten** (§60/§61): der
Haupttext benutzt durchgehend die Standard-Linkswirkung `R ↦ QR`; die
Konjugationswirkung `R ↦ QRQᵀ` steht ausschließlich in
`caution:rfm-conjugation-preview` am Kapitelende, als Eigenschaft der
SigmaFlow-*Zustandsparametrisierung*, nicht der Methode. Ebenso der
`Omega()`-Clamp: Theorie beschreibt die **exakte** Principal-Log-Map, der
Implementierungsbefund steht separat in `rem:rfm-log-implementation-note`.

**Verwendet:** Chen & Lipman 2024 (ICLR), §2–§3.2 vollständig gelesen. Deren
Prämetrik-Konstruktion (Gl. 13) mit `d = d_g` und `κ(t)=1−t` liefert
`u_t(x|x_1) = Log_x(x_1)/(1−t)` — Vorzeichen **positiv** in unserer
Zeitkonvention; unabhängig nachgerechnet und bestätigt. do Carmo und Lee für
die importierten Standardfakten (Levi-Civita = Tangentialprojektion,
Riemannscher Divergenzsatz).

## Runde: globaler Konsolidierungs-Audit (2026-08-16, 263 → 277 S.)

Phase 4. **Keine neue Theorie**, sondern Kohärenz des Gesamtdokuments.
0 Fehler, 0 undefinierte Referenzen, 0 doppelte Labels.

**Dependency-Audit** (Erstverwendung vs. Definition, 30 Kernbegriffe
maschinell geprüft): Vorwärtsreferenzen sind durchgehend ausgeschildert
(„from \Cref{chap:groups} onward"), also sauber — **eine** echte Lücke:
**Bedingte Erwartung war nirgends definiert.** `def:conditional-density`
gab es, `\EE[Y|X]`, Tower Property und „Funktionen von X gehen durch" nicht
— obwohl `lem:l2-projection`, der zentrale Beweis des ganzen Dokuments,
genau darauf beruht. Neue Section `sec:conditional-expectation` in Kap. 2
mit Definition, beiden Rechenregeln samt Beweis und einem
Zwei-Punkt-Beispiel, das auch die *falsche* Manipulation zeigt.

**Notations-Audit** (maschinelle Kollisionssuche): sechs echte
Mehrfachbelegungen gefunden, jetzt in `caution:symbol-reuse` im Frontmatter
global aufgelistet — „score" (statistisch vs. Docking-Score), `s`
(Rauschzeit vs. `s_θ`), `λ` (Metrikgewicht vs. DSM-Gewichtung `λ(s)`),
`φ` (Fragment-Rekonstruktionsabbildung vs. Winkel `φ_t`), `v` (Flow-Velocity
vs. Diffusions-`v`), `d` (Dimension vs. `d_g`). Plus die Liste der Symbole,
die *nicht* mehrfach belegt sind. Lokal behoben: `σ` wurde in
`ex:two-point-2d` als Logistik-Funktion benutzt — entfernt.

**Dichte-Audit** (Zeilen/Beispiele/Cautions je Kapitel gezählt): klarer
Ausreißer ist **Kap. 7 (Diffusion auf SO(3)): 433 Zeilen, 0 Beispiele,
0 Cautions** — bei schwerem Stoff (Heat Kernel, IGSO(3), Peter–Weyl).
Behoben mit `ex:igso3-limits`: beide Grenzfälle der Reihe, mit Zahlen.
Groß-σ liefert `prop:so3-brownian-stationary` in einer Zeile, **und** den
quantitativen Befund, dass `σ_max = 1.5` noch *nicht* uniform ist
(`max|f−1| = 0.98` gegen `1.3e-10` bei σ=5) — das ist die **Ursache** des in
`sec:engine-sources` gemessenen Quellverteilungs-Confounders. Klein-σ liefert
das Bild „IGSO(3)-Rauschen = gaußscher Rotationsvektor" und räumt das
`ω²`-Missverständnis aus (Haar-Jacobi, nicht Rauschmodell).

**Redundanz-Audit**: geprüft und **freigegeben** — die drei Herleitungen der
Continuity Equation (Kap. 5 1D, Kap. 11 Integralform, Kap. 12 Riemannsch)
sind ausgeschildert und bauen aufeinander auf; die zwei
Äquivarianz-Definitionen (Kap. 4 linear, Kap. 8 allgemeine Gruppenwirkung)
sind explizit als Verallgemeinerung markiert. Keine Bad Duplication
gefunden. Auch **keine neue Synthesis-Kapitel** angelegt: die vom User
skizzierte Diffusion-vs-FM-Synthese existiert bereits dreifach
(`sec:fm-diffusion-synthesis`, `sec:rfm-vs-diffusion`, `chap:engine-swap`)
plus „Closing Remarks" — eine vierte wäre genau die Duplikation, die der
Audit vermeiden soll.

**Navigation** (das eigentliche Problem bei 277 Seiten):
- **8 Part-Roadmaps** („What you will learn / Why it is needed / What
  depends on it") am Anfang jedes Parts.
- **12 Kapitel-Lernziele** („What you should now be able to do", 4–6
  Punkte) am Ende jedes Theoriekapitels. Kap. 5 hatte bereits eine
  hervorragende Q&A-Tabelle und blieb unangetastet.
- Damit ist die Story jetzt aus dem Inhaltsverzeichnis allein lesbar.

**Übergang SigmaDock → Flow Matching** war der einzige abrupte: Kap. 10
endete auf Ranking-Heuristiken. Jetzt schließt es mit einer expliziten
Keep/Replace-Aussage und einem Verweis auf
`tab:sigmadock-vs-sigmaflow-precise`.

## Runde: die drei letzten Blocker (2026-08-16, 277 → 286 S.)

Phase 5. Abschluss des Theoriestrangs. 0 Fehler, 0 undefinierte Referenzen,
0 doppelte Labels.

**1. `cleveref` repariert.** Ursache: alle Theorem-Umgebungen teilten sich
per `\newtheorem{X}[theorem]{...}` den `theorem`-Zähler, und cleveref
bestimmt den Typ über den *Zähler*, nicht über die Umgebung → jeder
`\Cref` schrieb „Theorem". Lösung: `aliascnt` (aus `oberdiek`, per `tlmgr`
nachinstalliert). `\newaliascnt{definition}{theorem}` legt einen Zähler
`definition` an, der ein Alias von `theorem` ist; `\aliascntresetthe` macht
`\thedefinition` = `\thetheorem`. **Nummerierung damit byte-identisch**,
kein Label angefasst, kein `\Cref`-Aufruf geändert. Plus explizite
`\crefname`/`\Crefname` für alle neun Umgebungen.

Regressionstest: neues Skript `audits/cleveref_type_check.py` liest die
`.aux`, extrahiert für jedes Label den von cleveref gespeicherten Typ und
vergleicht ihn mit dem Label-Präfix. 349 theoremartige Labels geprüft, alle
korrekt. Typverteilung jetzt: definition 81, proposition 69, caution 64,
remark 56, example 42, theorem 23, lemma 4, notation 4, corollary 2.

Nebenbefund dabei: 5 eigene Labels waren *falsch benannt* (vier
`ex:`-Labels saßen auf Subsections, `rem:trivialisation-terminology` auf
einer Caution). Umbenannt, nicht auf die Whitelist gesetzt.

**2. EquiformerV2-Worked-Example** (Kapitel 9, neue §9.13, ~7 Seiten,
14 Schritte). Drei Atome, zwei Kanten, `l_max=1`, ein Head. Vollständig
durchgerechnet: Kantengeometrie → Radialfeatures → Kantenframe →
m-Sektoren → SO(2)-Faltung → m=0-Invariante → Attention-Logits → Softmax →
Werte → Rückrotation → Aggregation → Residual. Alle Zahlen kommen aus
`audits/equiformer_toy_block.py`, keine von Hand.

Tragender Trick: **bei `l=1` ist die Wigner-D-Matrix in Cartesischer Basis
die Rotationsmatrix selbst** — deshalb ist das Beispiel exakt und trotzdem
von Hand nachrechenbar. `caution:toy-block-scope` trennt explizit fünf
exakte von fünf didaktisch vereinfachten Punkten.

Zwei numerische Checks, beide bestanden: Äquivarianz
`F(Qx,Qh) = Q F(x,h)` (1.6e-15 über 200 Rotationen) und
**Eichunabhängigkeit** — das Ergebnis hängt nicht von der willkürlich
gewählten Transversalachse des Kantenframes ab (1.7e-16 über 200
Referenzvektoren). Dazu ein **Gegentest**: bricht man die SO(2)-Bedingung
absichtlich (verschiedene Gewichte auf die beiden Transversalpartner),
springt die Streuung auf 0.85 — der Test kann also scheitern.

**3. Part VIII bereinigt.** Vorher: Part hieß „Synthesis", enthielt aber nur
den Numerik-Index, und die eigentlichen „Closing Remarks" standen als
unnummeriertes `\chapter*` dahinter. Jetzt:
`Part VIII — Implementation Reference and Synthesis` mit
Kap. 16 „Numerical and Implementation Reference" und Kap. 17
„Closing Synthesis" (§17.1 eine Ein-Seiten-Vergleichstabelle der beiden
Modelle, ausschließlich aus vorhandenem Material mit Querverweisen;
§17.2 der bestehende Closing-Remarks-Text). **Keine neue Theorie.**

**Eigener Fehler, gefunden und behoben:** `STATUS.md` war durch einen
früheren PowerShell-Befehl doppelt kodiert
(`Get-Content -Raw | Set-Content -Encoding utf8` — PS 5.1 liest UTF-8 ohne
BOM als Windows-1252). Repariert per cp1252-Rücktransformation, verifiziert
gegen `git show HEAD:STATUS.md`. **Lehre:** in PowerShell nicht nur beim
*Schreiben* `-Encoding utf8` setzen, sondern auch beim *Lesen*
(`Get-Content -Encoding utf8`) — sonst ist der Round-Trip destruktiv. Besser:
solche Ersetzungen mit Python statt PowerShell machen.

## Audit-Skripte (`audits/`, alle lokal, ohne GPU/Checkpoint)

| Skript | Zweck |
|---|---|
| `frame_audit.py` | Frame jedes Outputs + Gruppenwirkung auf `R_t` |
| `s2_gauge_audit.py` | S²-Äquivarianzbruch bei Produktionsgröße |
| `engine_swap_audit.py` | FM-Ziele, Quellverteilungen, Loss-Skalen, Log-Map |
| `sde_limit_check.py` | √Δt, DDPM→VP, SMLD→VE, Euler–Maruyama |
| `flow_matching_theory_audit.py` | **neu (Phase 2):** 27 Checks über 3 Schedules — Gauß-Feld, Score↔Velocity (cond. + marg.), Continuity Equation, Marginalisierungssatz (Monte Carlo), tanh-Beispiel, PF-ODE↔FM für VP/VE, `v` vs. `u`, SNR-Bedingung, Nichteindeutigkeit, L²-Projektion + Loss-Floor. Alle grün. |
| `riemannian_fm_theory_audit.py` | **neu (Phase 3), erweitert in Phase 4:** 38 Checks. SO(3): Exp/Log-Rundlauf, ½-Metrik-Isometrie, Bi-Invarianz, `d_g = θ`, `R̈ ⊥ T_RSO(3)`, konstante Geschwindigkeit, Interpolant-Endpunkte, `R_tᵀṘ_t = Ξ` per finiter Differenz, Current-Point-Log, 90°-Beispiel, Haar-Invarianz (KS-Test). S²: Exp/Log, Tangentialität, `prop:geodesic-vf-general` numerisch. S¹: Wrap. Produkt: Endpunkte, Geschwindigkeiten, λ-Unabhängigkeit der Geodäte. Flacher Grenzfall → Kapitel 11. Exp/Log aus Rodrigues selbst implementiert, **nicht** aus dem Repo importiert. Alle grün. |
| `cleveref_type_check.py` | **neu (Phase 5):** liest `Texte/theory.aux`, extrahiert für jedes Label den von cleveref gespeicherten Referenztyp und vergleicht ihn mit dem Label-Präfix. Regressionstest für den `aliascnt`-Fix. 349 theoremartige Labels, alle korrekt. |
| `equiformer_toy_block.py` | **neu (Phase 5):** der Toy-EquiformerV2-Block aus §9.13. Erzeugt alle im Text zitierten Zahlen und testet zwei Strukturaussagen: SO(3)-Äquivarianz (1.6e-15) und Eichunabhängigkeit vom Kantenframe (1.7e-16), plus einen Gegentest, der zeigt dass der zweite Check scheitern *kann*. |

**Zwei Methodikfallen, beide selbst hineingetappt und behoben** — bei neuen
Skripten beachten:
1. `zero_init_last` muss auf `False`. Der Default `True` nullt die letzte
   Schicht → Output exakt 0 → jeder Äquivarianztest leer erfüllt.
2. **Monte Carlo verdeckt Konvergenzordnungen.** Bei 40 000 Samples ist der
   Stichprobenfehler 5e-3; die behauptete 1/N-Konvergenz war unsichtbar. Wo
   die Marginale gaußisch sind: geschlossen rechnen, nicht simulieren.
3. **(2026-08-16, Phase 2)** Ein Monte-Carlo-Test braucht eine Toleranz *in
   Einheiten seines eigenen Standardfehlers* — eine absolute Toleranz unter
   dem Sampling-Rauschen lässt eine korrekte Identität durchfallen.
4. **(2026-08-16, Phase 2)** Posterior-Gewichte immer im Log-Raum rechnen.
   Bei kleinem `σ_t` unterlaufen alle Mischungskomponenten auf 0 → `0/0` →
   `NaN`, und ein Max-Residuum-Test meldet `NaN <= tol` als bestanden.
   Ein Check muss auf seine *Fähigkeit zu scheitern* geprüft werden,
   bevor sein Bestehen als Beleg zählt.
5. **(2026-08-16, Phase 3) — dieselbe Falle, zum zweiten Mal.** Ein
   λ-Check im RFM-Audit verglich zunächst einen Ausdruck mit sich selbst
   und hätte für jede Theorie bestanden. Ersetzt durch einen, der scheitern
   kann (ändert λ die Geodäte, ist die Geschwindigkeit nicht mehr konstant
   in t). Lehre Nr. 4 war notiert und half trotzdem nicht — der Check auf
   Fallierbarkeit muss zur **Routine** werden, nicht zur Notiz.

Weitere Stolpersteine: `timestep_embedder.py` pinnt hart `timesteps.float()`
(Zeitpfad bleibt float32, speist nur ℓ=0 → für Äquivarianz irrelevant);
`model.py` legt Puffer mit `torch.zeros(...)` ohne dtype an → global
`torch.set_default_dtype(torch.float64)` setzen statt Code ändern.

## Laufende Regeln für den Theory-Strang

- **Nach jeder größeren LaTeX-Änderung kompilieren**, nicht erst am Ende.
  `cd Texte && pdflatex -interaction=nonstopmode theory.tex` (3×), dann auf
  `^!` und undefinierte Referenzen prüfen.
- Keinem Kommentar, Variablennamen oder bestehenden Dokumentsatz ungeprüft
  glauben. Bei Widerspruch Paper/Kommentar/Code entscheidet **der Code**.
- Trennen: *Mathematical requirement* / *What SigmaDock actually implements* /
  *Numerical verification*.
- Nicht vorschnell „bricht Äquivarianz" sagen — erst analysieren, wo genau die
  Approximation entsteht (die `m_max`-Trunkierung war exakt äquivariant, ich
  hatte sie fälschlich als approximativ geführt).
- `SigmaDock/` und `SigmaFlow_Minimal/` sind eingefroren.

---

# ARC-Vorbereitung — Stand 2026-08-16

`theory.tex` ist eingefroren. Diese Phase ist ausschliesslich Code, Jobs,
Configs, Experimentdesign und Validierung.

## Fertig und getestet (alle Suiten gruen)

| Artefakt | Zweck | Test |
|---|---|---|
| `arc/throughput_sweep.slurm` | misst Batch/Precision/`--debug`/`val_interval` | — (Messjob) |
| `arc/final_config.sh` | einzige Stelle fuer die finalen Werte | Validator |
| `arc/train_final_72h.slurm` | ein Skript, vier Varianten, Snapshots + USR1-Trap | `bash -n`, Validator |
| `arc/submit_final.sh` | setzt Partition/Zeit/GPU, die `#SBATCH` nicht lesen kann | `bash -n` |
| `arc/write_manifest.py` | Reproduzierbarkeits-Manifest je Lauf | Selbsttest |
| `arc/variant_diff.py` | semantischer Diff Minimal -> Variante | 2 Selbsttests |
| `arc/exp101_source_audit.slurm` | Gate fuer den Source-Strang, **ohne GPU** | `bash -n` |
| `SigmaFlow_FM_Specific/EXP-101_.../source_distance_audit.py` | die Messung | 20 Checks |
| `SigmaFlow_Evaluation/evaluate_run.py` | RMSD/TFD/Oracle@K/Top-1@K in einem Durchgang | 22 Checks |

## Zwei Befunde, die den Plan geaendert haben

1. **Die Koordinaten sind taschenzentriert.** `sigma_flow_generator.py:499`
   zieht `pocket_com` ab, `:501` teilt durch `DIMENSIONAL_SCALE = 2.7 A`.
   `trans_0 ~ N(0, I)` sitzt also im Taschenzentrum. Die im Compute-Audit
   offen gelassene Sorge, die Quelle koenne systematisch zig Angstroem
   danebenliegen, ist damit **ausgeraeumt** — aus dem Code, nicht aus einer
   Messung.

2. **`R_1 ≡ I` in SigmaFlow_Minimal.** `get_fragment_com_and_rot` gibt die
   Zielrotation unbedingt als Identitaet zurueck (`rots.append(I3)`). Ein
   Rotations-Audit gegen Minimal haette deshalb einen scheinbar perfekten
   Strukturgewinn von 132 Grad gemeldet und mit voller Zuversicht
   "EXP-102 bauen" empfohlen — ein reines Artefakt der Parametrisierung.
   `analyse()` hat jetzt eine Sperre, und die belastbare Messung laeuft gegen
   **EXP-100**, wo `R_1` eine echte inferenzsaubere Zielrotation ist.

   Das ist Lehre Nr. 6 (siehe unten) und derselbe Fehlertyp, den
   `feedback_audit_against_code` schon beschreibt: ich hatte die Zentrierung
   im Code geprueft, die Rotation aber aus der Roadmap-Formulierung
   uebernommen.

## Bewusst NICHT gebaut

- **EXP-102 (informierte Quelle)** — das Gate (EXP-101) ist noch nicht
  gelaufen. Die Roadmap sagt ausdruecklich: faellt es negativ aus,
  "wird EXP-102 nicht gebaut". Eine Heuristik jetzt zu waehlen hiesse, die
  Entscheidung vorwegzunehmen, die die Messung treffen soll.
- **EXP-105 (Confidence)** — die Roadmap definiert sie als
  `C_psi(z_1, P, L) ≈ P(RMSD < 2 A)`, also einen Klassifikator auf FERTIGEN
  Posen. Der braucht generierte Daten aus einem trainierten Checkpoint; die
  gibt es offline nicht. Vorbereitet ist stattdessen die **Nahtstelle**:
  `evaluate_run.py --scores` nimmt jeden Ranker entgegen, und der bestehende
  Baseline-Ranker (`chem/statistics.py::compute_heuristic`, Vinardo mal
  Mittel der PB-Checks) ist damit ohne neuen Code anschliessbar.

## Naechster konkreter Schritt

Sobald ARC laeuft: Runbook Abschnitt 0 und 0a, Schritte 1-5 in dieser
Reihenfolge. Schritt 3 (`exp101_source_audit.slurm`) braucht weder GPU noch
Checkpoint und kann sofort parallel laufen.

**Offen und noch nicht committet:** 8 neue Dateien sind `untracked`. Der
Validator meldet sie als Fehler, weil ARC den Code per `git pull` bekommt.
Committen ist eine Entscheidung des Nutzers, keine autonome Aktion.


---

# ARC-Plan konkretisiert — 2026-08-16 (zweite Runde)

Vollstaendiger Bericht: **`ARC_PLAN_2026-08-16.md`**.

## Verifizierter Laufstand (nicht aus Erinnerung)

Der letzte methodisch gueltige Vergleich war **12h vs. 12h**, nicht 6h:
- SigmaFlow Frame-Fix **8541310**, 11:05:40, 13.750 Steps, max_epochs=6
- SigmaDock **8541439**, 10:52:16, 13.200 Steps, max_epochs=6
- Sampling **8554147/8554149**, je 209 Komplexe x 10 Seeds

Ein gueltiges 6h-Paar existiert ebenfalls (8530243 / 8512922, max_epochs=3,
je ~7.050 Steps), wurde aber vom 12h-Paar abgeloest. Die 24h-Laeufe
(8465054/8465055) bleiben INVALID.

## Drei Zahlen, die den Plan bestimmen

1. **72 h liefern bei aktuellem Durchsatz ~37 Epochen, nicht 128.**
   Aus dem 12h-Lauf: 110.000 Beispiele in 11,1 h = 2,75/s, also 713.000 in
   72 h = 14 % des Paperbudgets. `FINAL_MAX_EPOCHS=128` verlangt eine
   3,5-fache Beschleunigung, die der Throughput-Sweep erst nachweisen muss.

2. **SigmaFlows Orientierungsfehler (145,9 Grad) liegt UEBER der
   Zufallsbaseline (132,3 Grad).** Nicht nur auf Zufallsniveau, sondern
   systematisch schlechter als raten. SigmaDock: 124,4 Grad.

3. **Die Oracle-Luecke ist riesig:** ein Zug 1,9 % (SF) / 9,1 % (SD),
   best-of-10 29,2 % / 45,9 %.

## SigmaDock-Confidence: verifiziert

**SigmaDock hat KEIN trainiertes Confidence-Modell** (Paper S. 8 und
Appendix F.2, S. 33). Stattdessen `s_i = -b_i * p_i^4` mit Vinardo-Energie
und PoseBusters-Checks. Null gelernte Parameter, null Trainings-Compute.
Bereits implementiert in `chem/statistics.py::compute_heuristic`.

**Alle Paper-Hauptzahlen sind NACH Ranking** (N_seeds = 40). Das Ranking
traegt rund 13 Prozentpunkte (Ablation I* 79,9 vs. D 66,1). Unsere Zahlen
sind Einzelziehungen und mit 79,9 % NICHT vergleichbar.

**Risiko:** `gnina` ist ein externes Binary und auf ARC ungeprueft.

## Bewusst zurueckgestellt

**Integrierte Confidence kann waehrend des generativen Trainings nicht fair
gescreent werden.** Das Target `P(RMSD < 2 A)` braucht generierte Posen mit
variierender Qualitaet; das Netz sieht waehrend des Trainings aber nur `x_t`
auf dem Wahrscheinlichkeitspfad, bei t=1 exakt die Kristallpose. Labels dafuer
zu erfinden waere unsauber. Reihenfolge stattdessen: 72h-Generatoren -> K
Samples -> Labels -> C_ext -> optional integrierter Head auf denselben Labels.
Die frei werdende GPU geht an einen zweiten Seed.

## Zwei Fehler in der eigenen Vorarbeit

1. **Snapshot-Zeitplan war kumulativ.** `sleep h*3600` je Eintrag haette bei
   "6 12 24 48" die Snapshots auf 6, 18, 42 und 90 h gelegt -- der letzte waere
   nie gefallen, ohne jede Fehlermeldung. Behoben, `arc/test_snapshot_schedule.sh`
   prueft es mit Gegenprobe.
2. **EXP-101 doppelt gebaut.** `arc/exp101_distance_audit.py` existierte
   bereits und war vollstaendiger (Hauptachsen-Heuristik samt
   Vorzeichenfixierung). Nicht gesucht. Konsolidiert, Duplikat geloescht.

## Naechster Schritt

`ARC_PLAN_2026-08-16.md` Abschnitt L, Schritte 0-8. Schritt 2 muss um
`which gnina` ergaenzt werden.

---

# ARC zurück, vier Blocker, EXP-110 — 2026-08-18 bis 2026-08-20

## Vier ARC-Blocker, alle durch Ausführen gefunden

Keiner davon war beim Lesen sichtbar. Das GREEN-Verdikt vom 2026-08-17 war
insofern zu selbstsicher: es beruhte auf Codeanalyse, nicht auf Ausführung.

| # | Blocker | Ursache | Behoben durch |
|---|---|---|---|
| 1 | `ValueError: Experiment config not found` | `.gitignore` schluckte `conf/experiments/` über `**/experiments/` | Negationsmuster `!**/conf/experiments/**` (Commit `0ddfe39`) |
| 2 | `pdbbind-core` leer auf ARC | Datensatz nie befüllt | Stage 1 auf `astex`, `precheck_dataset()` |
| 3 | `exp101_distance_audit.py`: Pfad **und** beide Regexes falsch | Pfade konstruiert statt aufgelöst | `get_experiment_config` |
| 4 | `true_dir` zeigte auf leeres Verzeichnis | `data/posebusters` hat nur leeres `raw/` | `ARC_TRUE_DIR` einmal in `arc/_common.sh` + verschachtelter Fallback |

Blocker 1 hätte den 72h-Lauf nach Sekunden getötet.

**Gemeinsame Lehre, jetzt Regel:** Datensatzorte werden **aufgelöst**, nie
konstruiert. Vorabprüfung auf nichtleer, und `COMPLETED` von SLURM ist kein
Beleg für ein Ergebnis.

## Gemessene Zahlen

- `n_train` = **19.037** statt der Paper-Zahl 19.443 (−2,09 %)
- Fragmentzahl je Ligand (Job 8607523, 208 von 209 gemessen):
  Mittel **4,51** (SD 2,11), Median 4, Modus 5, IQR 3–6, q90 8, Spanne 1–11
  → Zustandsdimension `D = 6F`, Mittel 27,1
- `F` ist je Ligand **deterministisch** — obwohl das Training zufällig aus
  den minimalen Cut-Sets zieht, schwankt die Fragmentzahl bei keinem der
  208 Liganden.
- ⚠️ Nicht mit **4,56** verwechseln: die stammt aus IS-1 unter
  `fragmentation_strategy="canonical"`. Trainingsrelevant ist **4,51**.

## Drei Befunde zur Schwierigkeit

1. **Fragmentzahl ist die dominante Schwierigkeitsachse.** Jedes zusätzliche
   Fragment multipliziert die Erfolgs-Odds mit **0,53** (`p < 1e-4`), für
   beide Verfahren. Ab `F = 7` trifft SigmaFlow keinen Komplex mehr (0/39).
2. **SigmaFlow ist gleichmäßig schlechter, nicht überproportional.** Der
   Interaktionsterm `Arm × F` ist mit **`p = 0,85`** nicht signifikant. Die
   Hypothese „die Rotationsbehandlung skaliert schlechter mit der Zahl
   starrer Körper" wird von diesen Daten **nicht** gestützt.
3. **„Leere Schnittmenge" widerlegt:** 48/15/51/94 statt der behaupteten
   Disjunktheit.

Abbildungen und Datensatz in `Thesis Visualisierungen/`.

## IS-1 geschlossen, negativ

Die informative-Quelle-Frage (CORE 4) ist **negativ** beantwortet.
EXP-102/IS-2 ist gestrichen und wird nicht wieder vorgeschlagen.

## Trainingsbudget eingeordnet

Paper: 256 Epochen, 384 GPU-h auf 4×A100. Unsere 12h = **5,8 Epochen** =
**2,3 %** davon. Der 72h-Lauf landet bei ~36 Epochen (Erwartung 50–90 je
nach Durchsatz). Beim Beurteilen von PyMOL-Posen mitdenken.

**Vorregistriert vor den Läufen:** Oracle@10/Oracle@1 (aktuell SigmaFlow
6,8, SigmaDock 4,4), mit allen drei Ausgängen und ihrer Deutung. Nicht
nachträglich umdeuten.

---

# EXP-110 — Zwei-Kopf-Vektorfeld (2026-08-20)

Vollständiger Stand in
`SigmaFlow_FM_Specific/EXP-110_two_head_vector_field/STATUS.md`.
Hier nur, was für den Gesamtfaden zählt.

## Die eine Änderung

`force_block` → `trans_block` + `rot_block` auf geteiltem Rumpf; Mittel-
Pooling je Fragment statt Newton-Euler; Rahmenwechsel `R_t^T hat(ω) R_t`
bleibt zwingend, weil das Ziel `log(R_t^T R_1)` invariant, jede äquivariante
Netzausgabe aber nicht invariant ist.

Alles andere ist byte-identisch zu Minimal @ `16a069a`.

## Commits

| Commit | Inhalt |
|---|---|
| `8473a0b` | Variante gebaut |
| `eb2d924` | Audit, 37 Checks |
| `4885963` | gematchtes 12h-Skript, Variantenstand |
| `129e910` | Abbruchsicherheit geprüft und markiert |

## 110 Audit-Checks grün

37 (Vollaudit) + 35 (Checkpoint-Round-Trip) + 22 (Geometrie) + 16
(Gradientenzuordnung). Der stärkste Befund: `L_trans` erreicht `rot_block`
mit Gradient **exakt 0** und umgekehrt — strukturelle Entkopplung, die bei
einem Kopf gar nicht prüfbar war.

## Abbruchsicherheit: nichts zu bauen

Aus dem Code belegt statt angenommen: `val_check_interval=50` bei `accum=1`
und `check_val_every_n_epoch=None` heißt Checkpoint alle **50
Trainingsbatches**, **nicht** an Epochengrenzen. Maximaler Verlust bei einem
Walltime-Kill: **50 Schritte ≈ 2,4 Minuten**. Lightning schreibt über
`_atomic_save`, ein Abbruch mitten im Schreiben lässt die vorherige Datei
heil.

`--signal=SIGUSR1` geprüft und **verworfen**: Lightnings
`_slurm_sigusr_handler_fn` ruft nach dem Speichern `scontrol requeue` und
startet den Job neu — für ein Zeitbudget-Experiment genau falsch.

Ergänzt wurde nur eine `trap` auf Shell-Ebene, die `RUN_STATUS.json`
schreibt (`COMPLETED` / `WALLTIME_ODER_SIGTERM` / `ABGEBROCHEN_rcN`).

## Fünf Fehler in der eigenen Vorarbeit, alle beim Ausführen gefunden

1. Sanity Gate als Blacklist gebaut — die ARC-Repowurzel heißt
   `SigmaFlow_Development_JulianMueller` und hätte das Verbot auf
   `"SigmaFlow_Development"` bei **jedem** Start ausgelöst. Jetzt exakte
   Suffixprüfung mit Negativkontrolle.
2. Parameterzahl in `EXP-110_README.md` war falsch (24.466.418 / +12,99 %)
   — aus einer handgesetzten Instanziierung, keine echte Konfiguration
   reproduziert sie. Korrekt aus `RunConfig`: **16.899.826 / +12,82 %**.
3. `RUN_STATUS.json` per Heredoc gebaut → ungültiges JSON, sobald ein Pfad
   einen Backslash enthält. Jetzt schreibt Python es.
4. Round-Trip-Test benutzte zunächst den falschen Loader.
5. Begründung für Mittel-Pooling war gegenstandslos (`assert M.min() >= 2`
   im Basispfad); das tragfähige Argument sind lineare 2-Atom-Fragmente mit
   `cond(I_reg) ≈ 1.5e8`.

## Nebenbei gefunden

`load_from_checkpoint(load_ema=False)` ist **kaputt** — streift genau ein
`"model."` ab, `ckpt["state_dict"]` trägt aber zwei Ebenen. Gilt in Minimal
genauso, fällt nicht auf, weil Sampling `load_from_scratch` benutzt und
`ema_state_dict` eine Ebene weniger hat. Kein Blocker, dokumentiert.

## Bereitschaft

12h-Lauf **vorbereitet, nicht abgeschickt**. Walltime 12:00:00 endgültig.
**72h für EXP-110: NEIN** — kein Ergebnis auf echten Daten rechtfertigt
bisher 72 GPU-Stunden. Die eingefrorene 72h-Konfiguration ist unberührt.
