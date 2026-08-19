# SigmaFlow — Current Research State

**Stand: 2026-08-19.** ARC ist wieder online; IS-1 ist entschieden.

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

**Fragmentzahlen (Stand 2026-08-19).** Drei unabhängige Quellen stimmen überein:

| Quelle | n | Mittel | Median | q90 | Max |
|---|---:|---:|---:|---:|---:|
| `EXPERIMENT_REGISTRY.md` (D-Statistik) | 209 | — | 4 | 7 | 11 |
| lokal aus den Ligand-SDFs gerechnet | 99 | 4.37 | 4 | 8 | 9 |
| IS-1-Lauf 8606965 (953 Rotationen / 209 Komplexe) | 209 | **4.56** | — | — | — |

Der Modus liegt bei 5 Fragmenten, die Verteilung ist rechtsschief. Rund ein
Viertel der Liganden hat höchstens 2 Fragmente; 4 von 99 haben **gar keine**
Torsion und sind damit ein einziger starrer Körper.

`arc/count_fragments.slurm` (CPU-only) erzeugt die vollständige Liste je Ligand
plus Histogramm und prüft dabei zusätzlich Verzeichnisse gegen erkannte Paare.
Verifiziert: `Fragmente = min_cuts + 1`, und die Zahl **schwankt nicht**
zwischen den minimalen Cut-Sets — trotz `fragmentation_strategy="random"` ist
`D` je Ligand deterministisch.

⚠️ `enumerate_valid_fragmentations` ist kombinatorisch: ein Ligand mit 23
Torsionen lief lokal über 20 min ohne Ergebnis. Das Zählskript hat deshalb ein
hartes Zeitlimit je Molekül.

Liganden mit 15+ Fragmenten
**existieren in diesem Benchmark nicht**; die Fallauswahl muss sich an 8–11
orientieren. Die exakten Zahlen je Komplex brauchen ARC-Daten.

**SigmaDock-Parität:** SigmaFlow sammelt bereits `all_pos`
(`sampling.py:380/464/478`), `sample.py` gibt `trajectory: [T, N_lig, 3]`
heraus. Auf der SigmaDock-Seite ist noch ungeprüft, ob Zwischenpositionen
zugänglich sind. Falls Instrumentierung nötig wird: nur append-only, außerhalb
der wissenschaftlichen Modelllogik, und mit Nachweis, dass die finalen Posen
unverändert bleiben. **Darf den 72h-Start nicht verzögern.**

---

## D. Informative Quellverteilung (CORE 4) — ABGESCHLOSSEN, NEGATIV

**IS-1 ist am 2026-08-19 gelaufen (ARC-Job 8606965) und durchgefallen. Nach der
vorregistrierten Regel wird IS-2 nicht gebaut und bekommt keine GPU-Zeit.**

Gemessen über 209 PoseBusters-Komplexe, 953 Fragmentrotationen:

| Quelle | Median d_SO(3) | Mittel | gegen Haar | |
|---|---:|---:|---:|---|
| H0 Haar (uninformiert) | 131.3° | 126.2° | — | Referenz |
| **H1 Hauptachsen Konformer→Tasche** | **138.1°** | 128.4° | **−6.7°** | **TRÄGT NICHT** |
| Hid `R_0 = I` (Kontrolle) | 130.7° | 126.4° | +0.6° | trägt nicht |
| Hx Hauptachsen **mit Leakage** | 135.4° | 128.0° | −4.1° | trägt auch dann nicht |

**Die Messung ist belastbar:** H0 trifft die Haar-Erwartung (Median ~132.3,
Mittel ~126.5) auf ~1° genau. Das Urteil betrifft also die Heuristik, nicht die
Messung.

**Der entscheidende Befund ist die Leakage-Kontrolle.** Selbst *mit* geleakter
Kristallgeometrie bleibt die Heuristik mit 135.4° schlechter als Haar. Die
Obergrenze dieser Familie liegt unterhalb der uninformierten Quelle — das ist
keine fehlende Information, sondern eine falsche Konstruktion. Hauptachsen von
Konformer und Tasche sind nicht die Größen, die die gebundene Orientierung
bestimmen. Auch `R_0 = I` ist mit +0.6° ununterscheidbar von Haar: es gibt
keinen trivialen Orientierungsprior.

**Die Asymmetrie, um die es ging, ist damit quantifiziert:**

```
Translation:  N(0,I) Median 2.1  ->  Taschenmitte Median 1.5   (Prior wirkt)
Rotation:     Haar   Median 131.3 ->  beste zulässige Heuristik 138.1  (schlechter)
```

Die Translation hat bereits einen informativen Prior, weil `pocket_com` im
Ursprung liegt. Für die Rotation gibt es kein Gegenstück — Haar hat keinen
Mittelwert, und der Versuch, einen zu konstruieren, verschlechtert die Lage.

**Reichweite der Aussage.** Geprüft ist **eine** Heuristikfamilie
(Hauptachsen). Dass *keine* informative Rotationsquelle helfen könnte, folgt
daraus nicht. Für die Thesis ist die zulässige Formulierung: „Hauptachsen-
Konditionierung hilft nicht", nicht „informative Quellen helfen nicht".

Die Leiter IS-3 bis IS-5 bleibt dokumentiert (`INFORMATIVE_SOURCE_ROADMAP.md`),
ist aber ohne bestandenes IS-1 gegenstandslos. Die gelernte Quelle (IS-5) wird
nicht implementiert.

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
| Fragmentzahlen je Komplex | **READY** (`arc/count_fragments.slurm`, CPU-only); Mittel 4.56 bereits aus IS-1 bekannt |
| IS-1 (EXP-101) | **DONE — NEGATIV** (Job 8606965, Heuristik 6.7° schlechter als Haar) |
| IS-2 (EXP-102) | **CANCELLED** — vorregistrierte Regel greift, keine GPU-Zeit |
| IS-3 bis IS-5 | **GEGENSTANDSLOS** ohne bestandenes IS-1; dokumentiert, nicht geplant |
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
| `arc/exp101_distance_audit.py` | IS-1-Gate (gelaufen, negativ) |
| `arc/count_fragments.slurm` | Fragmentzahl je Ligand + Verteilung, CPU-only |
| `visualization/build_case.py` | PyMOL-Paket je Komplex |

**Tests, die lokal ohne ARC laufen:** `arc/test_scheduler_horizon.py` (16),
`arc/test_train_rc.sh` (11), `arc/test_aggregate_learning_curve.py` (14),
`arc/test_exp101_distance_audit.py`, `audits/test_rotation_gradient_reachability.py` (15),
`visualization/tests/test_visualization.py`, `arc/compare_final_configs.py`.
