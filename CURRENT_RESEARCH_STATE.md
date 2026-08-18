# SigmaFlow — Current Research State

**Stand: 2026-08-18.** Eingefrorener Vor-ARC-Zustand.

Dieses Dokument ist die eine autoritative Quelle für den Projektstand. Eine
neue Sitzung soll damit auskommen, ohne die Historie zu rekonstruieren.
`STATUS.md` bleibt das ausführliche Arbeitstagebuch; dieses Dokument ist die
Kurzfassung dessen, was gilt.

**Der Forschungsplan ist auf vier Kerne verengt und wird nicht erweitert:**

| | Kern | Priorität |
|---|---|---|
| CORE 1 | Finaler 72h-Vergleich SigmaFlow gegen SigmaDock | höchste |
| CORE 2 | Lernkurven über die Trainingszeit aus Snapshots | hoch |
| CORE 3 | PyMOL-Trajektorien je Integrationsschritt | mittel |
| CORE 4 | Informative Quellverteilung | nachrangig, darf CORE 1 nie verzögern |

---

## A. Wissenschaftlich gültige Basisergebnisse

Alle Läufe: 1× L40S, Batch 8, PoseBusters mit **209** Komplexen (das Paper
nutzt 308).

| Lauf | Job | Walltime | Epochen | Steps | Status |
|---|---|---|---|---|---|
| SigmaFlow Frame-Fix 6h | 8530243 | 5:44:59 | 3 | ~7 050 | **VALID** |
| SigmaDock 6h | 8512922 | 5:29:23 | 3 | ~7 050 | **VALID** |
| **SigmaFlow Frame-Fix 12h** | **8541310** | 11:05:40 | 6 | 13 750 | **VALID ← Referenz** |
| **SigmaDock 12h** | **8541439** | 10:52:16 | 6 | 13 200 | **VALID ← Referenz** |
| SigmaFlow ohne Frame-Fix 6h | 8512798 | 5:52:02 | 3 | ~7 050 | HISTORICAL ONLY (Kontrolle) |
| SF Varianten a/b/c | 8540758 / 8534746 / 8534747 | je ~5:45 | 3 | ~7 050 | Nullergebnis / verworfen / No-Op |
| SF/SD 24h | 8465054 / 8465055 | TIMEOUT | — | ~28 500 | **INVALID** (`max_epochs=1000`) |

**Kopfzahlen (12h, 209 Komplexe, 1 Seed, kein Ranking):** SigmaFlow **6/209**,
SigmaDock **17/209** unter 2 Å. Die Schnittmenge der gelösten Komplexe ist
**leer** — beide Modelle lösen disjunkte Fälle.

**Rotationsdiagnostik (Median-Winkeländerung gegen die Quelle):** SigmaFlow 12h
−15.0° [−21.6, −8.1]; SigmaDock 12h −20.4° [−26.0, −14.5]; SigmaFlow ohne
Frame-Fix +1.1° [−4.2, +6.6]. Der Frame-Fix ist damit klar wirksam, SigmaDock
transportiert Rotation aber weiterhin stärker.

**Frame-Fix:** in allen gültigen SigmaFlow-Läufen aktiv. Das Jobskript
verifiziert ihn vor jedem Training per `inspect.getsource` gegen
`R_t.transpose(-1,-2) @ updates["omega"] @ R_t`.

**EMA-Provenienz:** `use_ema=True`, `ema_halflife=2`, `ema_rampup_ratio=1/8` —
in beiden Armen byte-identisch. Training, Validierung, Sampling und Report
laufen durchgehend auf EMA-Gewichten. Deckt sich mit Appendix E.3 des Papers.

**Vergleichsregel zum Paper:** die 79.9 % sind Top-1 aus **40 Seeds** mit
Ranking auf **308** Komplexen bei **384 GPU-h** und Batch 32 — und es ist die
Konjunktion (RMSD < 2 Å **und** PB-valid). Unsere Zahlen nie dagegenstellen,
ohne alle vier Unterschiede im selben Satz zu nennen.

⚠️ `posebusters_ligandonly_SigmaDock_12h.csv` ist veraltet (0 Treffer). Immer
`..._12h_lrfix.csv` benutzen (17 Treffer).

---

## B. 72h-Bereitschaft

### Eingefroren

- **Modellcode.** `SigmaFlow_Minimal/` und der SigmaDock-Baum sind
  wissenschaftlich eingefroren. Änderungen nur bei einem echten
  Korrektheitsfehler, und dann erst nach Rückmeldung. Infrastruktur gehört
  ausschließlich nach `arc/`, `visualization/`, `audits/` und in die
  Varianten-Verzeichnisse.
- **Gradient-Erreichbarkeit: GREEN.** `audits/test_rotation_gradient_reachability.py`,
  15/15. `‖∂L_rot/∂f‖ = 24.1` gegen `‖∂L_trans/∂f‖ = 7.85`, Cosinus exakt 0.
- **Rotations- und Frame-Audit: GREEN.** 11 Angriffsvektoren, 16 Checks;
  adjungierter Transport exakt bis 6.7e-16.
- **Scheduler-Härtung: abgeschlossen.** `--max_steps` wird explizit übergeben.
  `train.py:221` rechnet den Horizont ohne Division durch `accum`; bei
  `accum=2/4` wäre er exakt 2×/4× zu groß gewesen. Numerisch belegt in
  `arc/test_scheduler_horizon.py` (16/16).
- **Konfigurationsgleichheit: GREEN.** `arc/compare_final_configs.py` vergleicht
  115 Parameter per AST: 110 identisch, 3 verfahrensbedingt
  (`rot_score_method`, `rot_score_scaling`, `sigma_min`), **0 Confounder**.

### Noch offen

- **Stufe-2-Durchsatz ist erforderlich.** Ohne ihn gibt es keinen gültigen
  Horizont. Stufe 2 misst per Zweipunktdifferenz, damit der konstante
  Datafront-Aufbau herausfällt — ein Einzellauf hätte den Durchsatz um
  Faktor 2–4 unterschätzt.
- **`arc/final_horizon.env` muss aus echten Messungen entstehen.** Die Datei
  ist in `.gitignore`, existiert absichtlich nicht, und wird von
  `arc/calculate_final_epochs.py --write-env` erzeugt.
- **Kein synthetischer Horizont ist zulässig.** `resolve_horizon_for_model()`
  rechnet `max_steps` gegen `floor(n_train/B_eff)·max_epochs` nach und lehnt
  von Hand editierte Dateien ab. `submit_final.sh` und
  `arc/final_72h_preflight.sh` verweigern ohne die echte Datei.
- **`FINAL_MAX_EPOCHS` hat keinen Default mehr.** Weder 36 noch 128 dürfen
  zurück ins Repository — beide stammten aus dem Batch-8-Regime mit `--debug`
  und ohne TF32.

### Epochenpolitik (automatisch, wird protokolliert)

Durchsatzunterschied **≤ 5 %** → gleiche Epochenzahl für beide Arme, genommen
aus dem langsameren. Identische Beispielzahl und Gradientenschritte, ein
Confounder weniger. **> 5 %** → getrennte Horizonte bei gleichem Zeitbudget
(compute-gematchter Vergleich). Aus den 12h-Läufen beträgt der Unterschied
**2 %**, es läuft also voraussichtlich auf die gemeinsame Zahl hinaus. Die
Entscheidung steht im Klartext in `final_horizon.env` und im Manifest.

### Snapshot-System

`FINAL_SNAPSHOT_HOURS = 6 12 18 24 30 36 42 48 54 60 66 72` — ein Obersatz der
geforderten acht Zeitpunkte. Ein Hintergrundprozess kopiert `last.ckpt` an
absoluten Stundenmarken (`cp` + atomares `mv`), ohne Lightning oder den
Modellcode anzufassen. Jeder Snapshot ist voll resumierbar: rohe Gewichte, EMA,
Optimizer- und Schedulerzustand. Daneben liegt eine `.meta.txt` mit Walltime,
Epoche, Step, gesehenen Beispielen, Lernrate und `schedule_progress`. Das
Manifest ergänzt SHA256 und Größe.

**Benennung ist wissenschaftlich bindend:** `sched<E>ep_at_<hhh>h` markiert
einen **Zwischenstand eines langen Schedules**. Er sitzt auf hoher Lernrate und
ist etwas anderes als der alte, eigenständig ausannealte 6h/12h-Lauf. Die
Aggregation hält beide in getrennten Serien
(`trajectory_snapshot_of_72h` gegen `annealed_endpoint`); der Test dafür ist
`arc/test_aggregate_learning_curve.py` (14/14).

---

## C. Visualisierungsstand (CORE 3)

Implementiert und lokal getestet, in `visualization/`:

| Modul | Aufgabe |
|---|---|
| `trajectory.py` | kanonische Zwischendarstellung `TrajectoryState` |
| `reconstruct.py` | Kabsch-Rekonstruktion der Fragment-Rototranslationen aus Atomkoordinaten — **ohne** den Sampler zu instrumentieren |
| `writers.py` | Mehrzustands-PDB, Metrik-CSVs |
| `pymol_scripts.py` | `view_trajectory.pml`, `view_static.pml` |
| `extract_from_sampling.py` | ARC-Seite: aus einem Sampling-Lauf die `.npz` erzeugen |
| `build_case.py` | komplettes Paket je Komplex |
| `select_cases.py` | Auswahl hochfragmentierter Komplexe |
| `plots.py` | Thesis-Abbildungen, inkl. Transport-Asymmetrie |

Ausgabe je Komplex und Checkpoint: `trajectory.pdb`, `trajectory_state.npz`,
`trajectory_metrics.csv`, `final_pose.pdb`, `view_trajectory.pml`.

**Offener Schritt:** die Kette wurde bisher nur gegen eine **synthetische**
Demo geprüft (`visualisations/SYNTH_DEMO/`, jetzt in `.gitignore`, weil
konstruiert). Es fehlt die Validierung gegen **echtes ARC-Sampling** aus dem
12h-Checkpoint. Das ist der einzige verbleibende Punkt für CORE 3 auf der
SigmaFlow-Seite.

**Fragmentzahlen:** bekannt ist `D = 6·F`, Median D 24, q90 42, max 66 →
Median **4**, q90 **7**, max **11** Fragmente. Liganden mit 15+ Fragmenten
**existieren in diesem Benchmark nicht**; die Fallauswahl muss sich an 8–11
orientieren. Die exakten Zahlen je Komplex brauchen ARC-Daten.

**SigmaDock-Parität:** SigmaFlow sammelt bereits `all_pos`
(`sampling.py:380/464/478`), `sample.py` gibt `trajectory: [T, N_lig, 3]`
heraus. Auf der SigmaDock-Seite ist noch ungeprüft, ob Zwischenpositionen
zugänglich sind. Falls Instrumentierung nötig wird: nur append-only, außerhalb
der wissenschaftlichen Modelllogik, und mit Nachweis, dass die finalen Posen
unverändert bleiben. **Darf den 72h-Start nicht verzögern.**

---

## D. Informative Quellverteilung (CORE 4)

Leiter aus `INFORMATIVE_SOURCE_ROADMAP.md`:

```
IS-0  Haar-Kontrolle              vorhanden, ist die aktuelle Quelle
IS-1  Geometrie-Audit             READY, CPU-Gate  <- als nächstes
IS-2  einfache informierte        vorbereitet, gesperrt durch IS-1
      Rotationsquelle
IS-3  Konzentrations-Sweep        dokumentiert
IS-4  Mischverteilung             dokumentiert
IS-5  gelernte Quelle             nur bei Erfolg von IS-1 und IS-2
```

**IS-1** (`arc/exp101_distance_audit.py`, Slurm: `arc/exp101_distance_audit.slurm`,
CPU-only) fragt: enthält inferenzverfügbare Geometrie genug Orientierungssignal,
um Haar zu schlagen — **vor** jedem Flow-Training? Liefert Verteilung des
Quell-Rotationsfehlers, Haar-Vergleich (Median 132.3°), Median-Rotationsdistanz,
Anreicherung unter 30°/60°/90° und Bootstrap-Unsicherheit. Eigener Test:
`arc/test_exp101_distance_audit.py`, alle Checks bestanden, inklusive
Äquivarianz auf 1.6e-15.

**Vorregistrierte Go/No-Go-Regel:** zeigt IS-1 keine belastbare Anreicherung
gegenüber Haar, wird **IS-2 nicht gebaut** und keine GPU-Zeit dafür verwendet.

**Wichtige konzeptionelle Einschränkung, im Skriptkopf festgehalten:** eine
Quelle wird bei `t=0` gezogen, wenn die Fragmente noch keine Position haben.
Konditioniert werden kann daher nur auf (Fragmentidentität, Ligandtopologie,
Tasche als Ganzes) — nicht auf die Fragmentumgebung. IS-1 misst deshalb eine
**globale** Ausrichtung, das Stärkste, was ohne Positionswissen sauber
definierbar ist.

**IS-2** würde ausschließlich die **Rotationsquelle** ändert
(`p₀ᵀ` bleibt minimal, `p₀ᴿ ≠ Haar`), alles andere fix — der sauberste Test der
Rotationshypothese. Gerüst liegt in
`SigmaFlow_FM_Specific/EXP-102_heuristic_conditional_source/`; der Arm
`sigmaflow_source` ist im Jobskript bereits vorgesehen, mit einer Warnung,
falls `SOURCE_MEDIAN_DEG` noch auf dem Haar-Default 132.3 steht statt aus
IS-1 zu kommen.

**Gelernte Quelle bleibt nachgelagert.** Kein Netz jetzt. Nur wenn IS-1 Signal
zeigt, IS-2 einen echten Nutzen belegt und Thesis-Zeit bleibt.

---

## E. ARC-Jobs

**In der Warteschlange (Stand 2026-08-18):**

| Job | Was | Bewertung |
|---|---|---|
| 8583394 | Throughput Stufe 1 | gültig, nicht abbrechen |
| 8583395 | Throughput Stufe 1 | gültig, nicht abbrechen |

Beide sind Stufe 1 auf `short`, ein Job je Arm; welcher zu welchem Arm gehört,
steht nach dem Lauf im Dateinamen `throughput_<model>.csv`. SLURM kopiert das
Batch-Skript beim Submit — sie laufen also die Vorfassung. Meine Änderungen an
Stufe 1 waren nur Kommentare und zwei CSV-Spalten, die Messung selbst ist
unverändert. **Stufe 2 muss aus der neuen Fassung nachgereicht werden.**

**Abgebrochen am 2026-08-18:** 8562617 (`sigmaflow-nfe`) und 8562618
(`sigmadock-nfe`), NFE-Sweeps, `--array=0-20%1`. Fertig sind SigmaFlow Seed 0
über alle 7 Schrittzahlen und SigmaDock Seed 0 bis 25 Schritte. **Es fehlen
SigmaDock bei 50 und 100 Schritten** (`--array=5-6`), ohne die es keine
vollständige methodenübergreifende NFE-Kurve gibt. Reine Thesis-Analyse, für
das 72h-Experiment nicht nötig.

⚠️ **Die Wall-Clock dieser Sweeps ist unbrauchbar.** Gemessene Zeiten bilden
eine U-Form mit Minimum bei 25 Schritten (1 Schritt: 11:51, 25 Schritte: 4:40,
100 Schritte: 11:25) — die Signatur eines kalten Dateisystem-Caches beim ersten
Task. NFE selbst ist davon unberührt und bleibt verwertbar; nur die
Sekundenachse müsste wiederholt werden.

**Frühere Jobs:** vollständige Laufliste in `ARC_PLAN_2026-08-16.md`,
Ergebnisse in `RESULTS.md`.

---

## F. Statusklassifikation

| Komponente | Status |
|---|---|
| 72h-Modellcode (`SigmaFlow_Minimal/`, SigmaDock) | **FROZEN / COMPLETE** |
| Scheduler-Härtung, `--max_steps`, Resume-Guard | **FROZEN / COMPLETE** |
| Snapshot-System | **FROZEN / COMPLETE** |
| Konfigurationsgleichheit beider Arme | **FROZEN / COMPLETE** |
| Gradient-, Rotations-, Frame-Audits | **FROZEN / COMPLETE** |
| Stufe-1-Throughput | **READY, REQUIRES ARC** (8583394/95 in der Queue) |
| Stufe-2-Throughput | **READY, REQUIRES ARC** |
| Finaler Horizont / `final_horizon.env` | **BLOCKED BY STAGE-2 MEASUREMENT** |
| Snapshot-Evaluation (`evaluate_snapshot.py`, `eval_snapshots.slurm`) | **READY** |
| Lernkurven-Aggregation | **READY** (14/14 gegen Fixtures) |
| `arc/eval_subset.txt` | **BLOCKED BY ARC DATA** (`make_eval_subset.py`) |
| SigmaFlow-PyMOL-Infrastruktur | **READY, REQUIRES REAL ARC INFERENCE VALIDATION** |
| SigmaDock-PyMOL-Parität | **PARTIALLY BLOCKED** — Zugriff auf Zwischenpositionen ungeprüft |
| Fragmentzahlen je Komplex | **BLOCKED BY ARC DATA** |
| IS-1 (EXP-101) | **READY, REQUIRES ARC DATA** |
| IS-2 (EXP-102) | **PREPARED, GATED BY IS-1** |
| IS-3 bis IS-5 | **OPTIONAL EXTENSION** |
| NFE-Sweep-Vervollständigung | **OPTIONAL EXTENSION** (2 Tasks) |

---

## G. Werkzeuge auf einen Blick

| Datei | Zweck |
|---|---|
| `arc/probe_datafront_size.py` | echte `n_train` messen (CPU, Verzeichnis-Scan) |
| `arc/throughput_sweep.slurm` | Stufe 1 (relative Faktoren) und Stufe 2 (Zweipunktmessung) |
| `arc/calculate_final_epochs.py` | Horizont ableiten, `final_horizon.env` schreiben |
| `arc/final_config.sh` | einzige Konfigurationsquelle, `resolve_horizon_for_model()` |
| `arc/final_72h_preflight.sh` | GREEN/AMBER/RED, lokal und ARC getrennt |
| `arc/submit_final.sh` | Submit-Wrapper, verweigert ohne gültigen Horizont |
| `arc/train_final_72h.slurm` | ein Skript für alle Arme |
| `arc/eval_snapshots.slurm` | billige Kurve über alle Snapshots eines Laufs |
| `arc/evaluate_snapshot.py` | Einzelsnapshot: Provenienz plus Metriken |
| `arc/aggregate_learning_curve.py` | Lernkurve beider Arme, Serien getrennt |
| `arc/compare_final_configs.py` | 115-Parameter-Diff der Arme |
| `arc/exp101_distance_audit.py` | IS-1-Gate |
| `visualization/build_case.py` | PyMOL-Paket je Komplex |

**Tests, die lokal ohne ARC laufen:** `arc/test_scheduler_horizon.py` (16),
`arc/test_train_rc.sh` (11), `arc/test_aggregate_learning_curve.py` (14),
`arc/test_exp101_distance_audit.py`, `audits/test_rotation_gradient_reachability.py` (15),
`visualization/tests/test_visualization.py`, `arc/compare_final_configs.py`.
