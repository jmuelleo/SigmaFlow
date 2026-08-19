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

Alle Läufe: 1× L40S, Batch 8, PoseBusters mit **209** Komplexen.

**Woher die 209 kommen (geprüft 2026-08-19).**
`posebusters_paper/posebusters_benchmark_set/` enthält **209 Unterverzeichnisse
und 209 erkannte Paare** — die Zahlen stimmen überein, es wird also nichts
stillschweigend verworfen. Unsere ARC-Kopie ist eine **bereits reduzierte**
Fassung des Benchmarks. Zum Vergleich: das Original (Buttenschoen et al. 2024)
hat 428 Komplexe, das SigmaDock-Paper wertet auf 308 aus. Wie die Reduktion auf
209 zustande kam, ist im Repository nicht dokumentiert und war nicht mehr
rekonstruierbar.

Für die Thesis heißt das: **209 ist eine Eigenschaft unserer Datenkopie, keine
Auswahlentscheidung dieses Projekts** — und jeder Vergleich mit dem Paper muss
den Unterschied im Benchmarkumfang nennen.

| Lauf | Job | Walltime | Epochen | Steps | Status |
|---|---|---|---|---|---|
| SigmaFlow Frame-Fix 6h | 8530243 | 5:44:59 | 3 | ~7 050 | **VALID** |
| SigmaDock 6h | 8512922 | 5:29:23 | 3 | ~7 050 | **VALID** |
| **SigmaFlow Frame-Fix 12h** | **8541310** | 11:05:40 | 6 | 13 750 | **VALID ← Referenz** |
| **SigmaDock 12h** | **8541439** | 10:52:16 | 6 | 13 200 | **VALID ← Referenz** |
| SigmaFlow ohne Frame-Fix 6h | 8512798 | 5:52:02 | 3 | ~7 050 | HISTORICAL ONLY (Kontrolle) |
| SF Varianten a/b/c | 8540758 / 8534746 / 8534747 | je ~5:45 | 3 | ~7 050 | Nullergebnis / verworfen / No-Op |
| SF/SD 24h | 8465054 / 8465055 | TIMEOUT | — | ~28 500 | **INVALID** (`max_epochs=1000`) |

**Kopfzahlen (12h-Checkpoints, 209 Komplexe, 10 Seeds, ausgewertet 2026-08-19
auf CPU aus den vorhandenen Sampling-Ausgaben 8554147/8554149):**

| | SigmaFlow | SigmaDock |
|---|---:|---:|
| Einzelzug (Seed 0) < 2 Å | 2.4 % | 10.0 % |
| Oracle@1 | 4.4 % | 10.9 % |
| Oracle@5 | 19.1 % | 34.0 % |
| **Oracle@10** | **30.1 %** | **47.4 %** |
| RMSD Median (2090 Posen) | 4.90 Å | 4.38 Å |
| Oracle@10 Median-RMSD | 2.55 Å | 2.05 Å |

**Ranking ist der bislang größte gemessene Hebel:** von Seed 0 zu Oracle@10
gewinnt SigmaFlow Faktor 12.5, SigmaDock Faktor 4.7. Der Abstand zum Paper
(79.9 %, 40 Seeds **mit** Ranking, 308 Komplexe, 384 GPU-h) besteht damit
überwiegend aus Seeds und Ranking, nicht aus Generatorqualität. Details und
Konsistenzprüfung gegen die Seed-Varianz-Analyse in `RESULTS.md`.

🚨 **KORRIGIERT 2026-08-19: die Schnittmenge ist NICHT leer.** Gepaart über
208 Liganden bei Oracle@10 lösen **48 Komplexe beide** Modelle; 15 nur
SigmaFlow, 51 nur SigmaDock, 94 keiner. Die alte Aussage stammte aus
Einzelziehungen (6/209 gegen 17/209) und war ein Kleinstichprobenartefakt bei
einem Ziehungsrauschen, das etwa zehnmal so gross ist wie der Methodeneffekt.
McNemar über die diskordanten Paare: `p = 1.0e-05` zugunsten SigmaDock — auf
demselben Komplexsatz, nicht auf einem anderen. **Diese Korrektur muss in den
Thesis-Entwurf.**

**Fragmentzahl ist die dominante Schwierigkeitsachse:** jedes zusätzliche
Fragment multipliziert die Erfolgs-Odds mit **0.53** (`p < 1e-4`), für beide
Arme. Ab 7 Fragmenten trifft SigmaFlow keinen Komplex mehr (0/39). SigmaFlow
ist dabei **gleichmässig** schlechter (Odds-Faktor 0.43), nicht
überproportional — der Interaktionsterm ist mit `p = 0.85` nicht signifikant.
Volle Auswertung in `RESULTS.md`.

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

- **`n_train` ist GEMESSEN: 19 037** (Job 8607507, `pdbbind-general`).
  Das sind **2.09 % unter** der Paper-Zahl 19 443. Daraus folgt
  `steps_per_epoch = floor(19037/32) = 594` statt 607.

  **Warum das jetzt zwingend war:** vor der Härtung rechnete `train.py:221`
  den Horizont selbst aus `len(train_datafront)` — also automatisch mit der
  echten Zahl. Der explizite `--max_steps`, mit dem der accum-Bug umgangen
  wird, verlagert diese Verantwortung nach aussen. Hätte man dort die
  Paper-Zahl eingesetzt, wäre der Horizont bei 70 Epochen um **840 Steps zu
  gross** gewesen und der Anneal nur zu 98 % durchlaufen.
  Zur Einordnung: bei 98 % steht der Cosinusfaktor bei 0.00096, die Lernrate
  also praktisch schon am Minimum — der Schaden wäre klein gewesen, anders
  als beim accum-Bug mit Faktor 2–4. Der Punkt ist nicht die Grössenordnung,
  sondern dass die Zahl jetzt gemessen statt geerbt ist.

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

**Fragmentzahlen — GEMESSEN, vollständig** (Job 8607523, 208 von 209 Liganden;
ein Ligand lief in das 120s-Zeitlimit der Cut-Set-Enumeration).

| Fragmente | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Liganden | 5 | 34 | 41 | 29 | **42** | 18 | 16 | 15 | 5 | 1 | 2 |
| Anteil | 2.4% | 16.3% | 19.7% | 13.9% | **20.2%** | 8.7% | 7.7% | 7.2% | 2.4% | 0.5% | 1.0% |

**Mittel 4.51 · Median 4 · q25 3 · q75 6 · q90 8 · Min 1 · Max 11**
Zustandsdimension `D = 6F`: Median **24**, q90 **48**, Max **66**.

Bestätigt: `Fragmente = min_cuts + 1`, und die Zahl schwankt bei **null**
Liganden zwischen den minimalen Cut-Sets — trotz
`fragmentation_strategy="random"` ist `D` je Ligand **deterministisch**.

**Korrektur gegenüber `EXPERIMENT_REGISTRY.md`:** Median D 24 und Max D 66
stimmen exakt, **q90 ist aber 48 und nicht 42** (8 statt 7 Fragmente). Die
Registry-Zahl ist damit überholt.

**Zwei Beobachtungen für CORE 3:**
- **Hochfragmentiert heißt 8–11.** Nur **23** Liganden haben ≥ 8 Fragmente,
  nur **3** haben ≥ 10. Liganden mit 15+ Fragmenten existieren nicht; die
  Fallauswahl muss sich an 8–11 orientieren.
- **5 Liganden haben genau 1 Fragment**, also gar keine Torsion. Für sie ist
  die gesamte Generierung eine einzige globale Rototranslation. Das sind die
  saubersten Testfälle für das Rotationsverhalten, weil kein
  Fragment-Fragment-Zusammenspiel überlagert.

**Nicht dasselbe wie die IS-1-Zahl.** IS-1 meldete 953 Rotationen über 209
Komplexe (4.56 im Mittel), benutzt aber `fragmentation_strategy="canonical"`.
Hier wird über die **minimalen Cut-Sets** gezählt — das ist, was das Training
mit `"random"` tut. 4.51 gegen 4.56 sind zwei nah beieinander liegende, aber
**verschiedene** Grössen. Für alle trainingsbezogenen Rechnungen gilt 4.51.

⚠️ `enumerate_valid_fragmentations` ist kombinatorisch; ein Ligand mit vielen
Torsionen sprengt jedes Zeitbudget. Der Zähler hat deshalb ein hartes Limit je
Molekül und schätzt nichts.

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

---

## H. Sitzungsprotokoll 2026-08-19 — ARC zurück, fünf Fehler gefunden

ARC kam nach dem Temperaturausfall vom 13.08. zurück. Der Tag hat mehr
Fehler zutage gefördert als jeder Audit davor — **alle erst beim tatsächlichen
Ausführen, keiner durch Codelektüre.**

| # | Fehler | Wirkung, wäre er geblieben |
|---|---|---|
| 1 | `.gitignore` verschluckte `conf/experiments/` (Datensatzdefinitionen nie versioniert) | 72h-Lauf stirbt in den ersten Sekunden |
| 2 | Stufe 1 des Sweeps lief gegen `pdbbind-core`, auf ARC **leer** | Sweep unbrauchbar, kein Horizont |
| 3 | EXP-101 baute Pfad und Regexe selbst, widersprach der Config in allen drei Angaben | IS-1 hätte am falschen Komplexsatz gemessen |
| 4 | Alle Auswertungsskripte zeigten auf `data/posebusters` — dort liegt **keine** Referenzdatei | Lernkurven wären lautlos leer geblieben |
| 5 | `count_fragments` benutzte `DataFront` statt `MetaFront` (relative statt absolute Pfade) | alle 209 Moleküle fielen aus |

Gemeinsame Ursache: **Skripte konstruierten Datensatz-Speicherorte, statt sie
aufzulösen.** Gegenmaßnahmen sind eingebaut — `get_experiment_config` überall,
`ARC_TRUE_DIR` einmal in `_common.sh`, `precheck_dataset()` im Sweep,
Abschnitt A9 im Preflight, `--expect-in-path` gegen den falschen Codebaum.

**Was das über den GREEN-Stand vom 17.08. sagt:** er war zu selbstsicher. Die
Audits waren korrekt in dem, was sie prüften, konnten diese Fehlerklasse aber
strukturell nicht sehen. Ausführen schlägt Auditieren.

### Positiv abgeschlossen

- **Partition `long` verifiziert**: existiert, 30 Tage Limit, `gpu:l40s:4`. Der
  letzte offene ARC-Laufzeitpunkt.
- **PoseBusters-Umfang geklärt**: 209 Verzeichnisse = 209 Paare, keine stille
  Teilmenge. Unsere Kopie ist eine bereits reduzierte Fassung (Original 428,
  Paper 308).
- **IS-1 entschieden** (negativ, siehe Abschnitt D).
- **Auswertungskette validiert** an 2×2090 realen Posen, 100 % Abdeckung.
- **Oracle@K gemessen** — Ranking ist der größte belegte Hebel (Abschnitt A).

### Offen am Ende des Tages

| Job | Stand |
|---|---|
| 8606626 / 8606627 | Throughput Stufe 1 — `PD (Priority)`, GPU knapp |
| 8607507 | `probe_sizes` (n_train) — eingereicht |
| 8607523 | Fragmentzähler mit Pfad-Fix — eingereicht |

**`arc/train_final_72h.slurm` ist bis heute nie ausgeführt worden.** Alle
Änderungen daran (expliziter `--max_steps`, `TRAIN_RC`-Klassifikation,
Resume-Guard, Snapshot-Benennung, Manifest) sind syntaxgeprüft und
unit-getestet, aber nicht auf ARC gelaufen. Das ist das größte verbleibende
Ausführungsrisiko. Entschärft wird es dadurch, dass alles, was fehlschlagen
kann, **früh** fehlschlägt: Sanity-Gate, Horizont-Konsistenzprüfung und
Datensatzauflösung laufen alle vor dem ersten Trainingsschritt.

---

## I. Trainingsbudget im Verhältnis zum Original

Beide Seiten rechnen bei effektiver Batch 32 mit praktisch gleicher
Epochenlänge: das Paper auf 19 443 Beispielen (607 Schritte/Epoche), wir auf
gemessenen 19 037 (594 Schritte/Epoche). Epochen sind damit direkt
vergleichbar.

| | Epochen | Optimizer-Schritte | Anteil am Original |
|---|---:|---:|---:|
| **Paper** (Appendix E.3, 4×A100, 4 Tage ≈ 384 GPU-h) | **256** | ~155 400 | 100 % |
| **Unsere 12h-Referenzläufe** (Batch 8) | **5.8** | 13 750 | **2.3 %** |
| Unser 72h-Lauf bei 2.75 Beispielen/s | 36 | 21 400 | 14 % |
| Unser 72h-Lauf bei 4.0 | 52 | 30 900 | 20 % |
| **Unser 72h-Lauf bei 5.5 (erwartet)** | **72** | **42 800** | **28 %** |
| Unser 72h-Lauf bei 8.25 | 109 | 64 700 | 43 % |

**Erwartungsbereich: 50–90 Epochen.** Batch 32 statt 8, TF32 an und kein
`--debug` sind die drei Hebel, die Stufe 1 misst. Der Sprung von 2.3 % auf
grob 20–35 % ist ein Faktor 8–15 gegenüber dem heutigen Stand.

**Wichtige Abgrenzung:** unser Lauf ist ein *vollständiges, kürzeres*
Trainingsprogramm, kein abgeschnittenes langes. Der Cosine-Anneal wird auf
genau diese Epochenzahl kalibriert und läuft durch; das Modell endet auf der
minimalen Lernrate. Ein auf 256 Epochen kalibrierter, nach 72 abgebrochener
Schedule wäre etwas anderes und wissenschaftlich wertlos — genau dagegen ist
die gesamte Horizont-Absicherung gebaut.

**Für jede Zahl in der Thesis gilt der Kontextsatz:** ~28 % der Epochen,
1 GPU statt 4, 209 statt 308 Komplexe, ein Seed ohne Ranking gegen 40 mit.

---

## J. Vorregistrierte Kenngröße für die Lernkurven

**Verhältnis `Oracle@10 / Oracle@1`** — ein Maß für die Schärfe der erzeugten
Verteilung, unabhängig vom absoluten Niveau.

Stand 12 h: **SigmaFlow 6.8**, **SigmaDock 4.4** (30.1/4.4 bzw. 47.4/10.9).

Zur Einordnung: bei uns bringt die Verdopplung von 5 auf 10 Seeds noch +11 bis
+13 Prozentpunkte — die Kurve steigt ungebremst. Beim Paper bringen *zwei*
Verdopplungen (10 → 40 Seeds) zusammen +7.7 Punkte (72.2 % → 79.9 %). Zwei
völlig verschiedene Regime. *(Vorsicht: Paper-Zahlen sind Top-k mit Ranking,
unsere Oracle ohne — die Richtung stimmt, die Faktoren sind nicht direkt
vergleichbar.)*

**Vorhersage, vor dem Start festgehalten:**

| Beobachtung über die Snapshots | Deutung |
|---|---|
| Verhältnis **fällt** deutlich | Verteilung konzentriert sich; die Streuung war Budget |
| Verhältnis **bleibt konstant**, Niveau steigt | Modell wird besser, aber nicht sicherer → Ranking ist der Hebel |
| Verhältnis **steigt** | strukturelles Problem |

Beide Größen fallen bei der Snapshot-Auswertung ohnehin an. Als Test taugt das
nur, solange es *vorher* notiert ist — deshalb steht es hier.

**Offen:** `arc/aggregate_learning_curve.py` gibt das Verhältnis noch nicht als
eigene Spalte aus. Kleine Ergänzung, noch nicht gemacht.

---

## K. Qualitative Einzelfallbetrachtung (PyMOL)

`visualization/view_complex.pml` lädt für einen Komplex das Protein als
Kontext, die Kristallpose und **alle zehn Sampling-Seeds beider Arme**. Der
Komplexname wird aus der vorhandenen `<CID>_protein.pdb` abgeleitet, das
Skript ist also unverändert in jeden `vis_<CID>`-Ordner kopierbar. Vier
Szenen, vier Hilfsbefehle (`best`, `seed N`, `spread`, `frag N`).

Alle zehn Seeds statt einer, weil das Ziehungsrauschen rund zehnmal so gross
ist wie der Methodenunterschied — eine Einzelpose bildet Rauschen ab.

**Zwei Fälle als Gegensatzpaar angesehen (2026-08-19):**

| | 6YRV_PJ8 | 7ORW_7WA |
|---|---:|---:|
| Fragmente / Torsionen | 8 / 14 | **1 / 0** |
| Zustandsdimension D | 48 | 6 |
| SigmaFlow bester RMSD | 2.38 Å ✗ | **1.24 Å ✓** |
| SigmaDock bester RMSD | 3.01 Å ✗ | **1.89 Å ✓** |

`7ORW_7WA` ist ein starrer Körper ohne jede Torsion — dort ist die gesamte
Generierung eine einzige globale Rototranslation, und Rotationsfehler sind
nicht durch Konformation verdeckt. Der sauberste verfügbare Testfall für
SO(3)-Transport, und einer von nur fünf im Datensatz.

Beobachtung am Bildschirm: einzelne Posen sitzen gut, viele sind klar
fehlrotiert oder versetzt. Das ist bei 2.3 % des Originaltrainings zu
erwarten und zeigt sich in **beiden** Armen, auch in der unveränderten
Referenzimplementierung. Belastbar wird die Frage erst im Vergleich mit
demselben Komplex aus dem 72h-Snapshot.
