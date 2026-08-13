# Lokale Vorbereitung, bis ARC zurückkommt

> Stand 2026-08-13. Zweck: alles zu erledigen, was ohne Cluster erledigt werden
> kann, damit nach der Wartung nur noch Rechenzeit fehlt.
>
> Verwandte Dokumente:
> [`ARC_RUNBOOK_AND_CURRENT_STATE.md`](ARC_RUNBOOK_AND_CURRENT_STATE.md) (Befehle),
> [`SIGMAFLOW_RESEARCH_ROADMAP.md`](SIGMAFLOW_RESEARCH_ROADMAP.md) (Priorisierung),
> [`SigmaFlow_FM_Specific/DESIGN_NOTES.md`](SigmaFlow_FM_Specific/DESIGN_NOTES.md) (Entwürfe),
> [`PREREGISTRATION_AND_RESULT_TEMPLATE.md`](PREREGISTRATION_AND_RESULT_TEMPLATE.md) (Vorregistrierung).

---

## 1. Aufgabenklassifikation

| Aufgabe | Lokal? | ARC? | Jetzt sinnvoll? | Begründung / Ergebnis |
|---|:---:|:---:|:---:|---|
| Git-Zustand prüfen, Push vorbereiten | ✅ | – | ✅ | **erledigt** — 9 Commits + Tag unpushed, Blocker dokumentiert |
| Statische Prüfung aller ARC-Skripte | ✅ | – | ✅ | **erledigt** — `arc/validate_scripts.py`, 0 Fehler |
| Zeilenenden / Shebang / Backticks | ✅ | – | ✅ | **erledigt** — `.gitattributes`, ein echter Backtick-Bug behoben |
| Budget-Kalibrierung aus dem Code belegen | ✅ | – | ✅ | **erledigt** — Kette `max_epochs → max_steps → Scheduler` in beiden Codebasen identisch |
| Checkpoint-Validierung | ✅ | – | ✅ | **erledigt** — `arc/validate_checkpoint.py`, PASS/FAIL + JSON |
| Oracle@K-Methodik + Unit-Tests | ✅ | – | ✅ | **erledigt** — 22 Tests, gegen bekannte Zahl validiert |
| Generischer Ranking-Evaluator | ✅ | – | ✅ | **erledigt** — `metrics/ranking.py`, modellunabhängig |
| TFD-Robustheit | ✅ | – | ✅ | **erledigt** — Abdeckung 61.2 % → **100 %** |
| Interpreter der Rotationsdiagnose | ✅ | – | ✅ | **erledigt** — inkl. Selbsttest der Einordnungslogik |
| Frisch-Checkout-Reproduzierbarkeit | ✅ | – | ✅ | **erledigt** — alle Tests laufen im frischen Klon |
| Ausgabekollisionen | ✅ | – | ✅ | **erledigt** — echte Lücke gefunden (raw/ranked), behoben |
| Entwurfsnotizen D1–D7 | ✅ | – | ✅ | **erledigt** — `DESIGN_NOTES.md` |
| Vorregistrierung + Ergebnisvorlage | ✅ | – | ✅ | **erledigt** |
| Symmetrie-Negativergebnis sichern | ✅ | – | ✅ | **erledigt** — Roadmap §2.5, Skript im Repo |
| EXP-101 Skript + Smoke-Test | ✅ | – | ✅ | **erledigt** — läuft auf Dummy-Daten |
| Umgebungsmanifest lokal | ✅ | – | ✅ | **erledigt** — siehe §4 |
| — | | | | |
| Rotationsdiagnose ausführen | – | ✅ | – | braucht trainierten Checkpoint + GPU |
| 24h-Trainings | – | ✅ | – | GPU, 24 h |
| EXP-100 Sanity auf GPU | – | ✅ | – | GPU + echte Datensätze |
| Sampling | – | ✅ | – | GPU |
| EXP-101 auf PoseBusters | – | ✅ | – | Datensatz liegt auf ARC |
| Partition/GPU-Klasse verifizieren | – | ✅ | – | `sinfo` nur auf ARC |
| Conda-Umgebungen prüfen | – | ✅ | – | nur auf ARC |
| Durchsatz EXP-100 vs. Minimal | – | ✅ | – | GPU-Messung |
| — | | | | |
| Conditional Source implementieren | – | – | ❌ | **NOT WORTH DOING YET** — EXP-101 entscheidet |
| Gelernte Quelle `q_φ` | – | – | ❌ | hängt an D1 |
| Direkter Rotationskopf | – | – | ❌ | hängt an der Diagnose |
| Endpunkt-Parametrisierung | – | – | ❌ | hängt an der Diagnose |
| Staged Path | – | – | ❌ | niedrige Priorität, Präzedenzfall negativ |
| Confidence trainieren | – | – | ❌ | braucht Multi-Sample-Daten vom 24h-Modell |
| Early Pruning | – | – | ❌ | hängt an Confidence |
| Symmetry-aware Training | – | – | ❌ | **gemessen widerlegt** |
| Experimentkombinationen | – | – | ❌ | verletzt „eine Änderung je Variante" |

---

## 2. Lokal abgeschlossen

### 2.1 Gefundene Fehler und Risiken

| Fund | Schwere | Behebung |
|---|---|---|
| **Backtick in `00_preflight.sh`** hätte `git pull` **ausgeführt** statt zitiert | hoch | Anführungszeichen; `bash -n` fängt das nicht, der Validator schon |
| **Raw- und Ranked-Sampling** hätten mit gleicher NFE in denselben Ordner geschrieben | hoch | `OUT_ROOT` trägt jetzt Modell, Vergleichsart, Ranking und NFE |
| **CRLF** hätte `bad interpreter: /bin/bash^M` erzeugt | hoch | `.gitattributes`; Validator prüft den **Index**, nicht die Arbeitskopie |
| **Oracle@K mittelte RMSDs und schwellte danach** — K=1 als 0.0 % statt 3.9 % | hoch | korrekte Definition + 22 Unit-Tests |
| **TFD fiel bei 38.8 % der Moleküle aus** | mittel | Sanitisierung + Koordinatentransplantation → 100 % |
| **Seed steckt im Verzeichnis, nicht im Dateinamen** — alle Dateien heißen `_seed0.sdf` | mittel | dokumentiert; falsches Globen liefert zehnmal Seed 0 |
| Validator prüfte die Arbeitskopie statt des Git-Index | mittel | behoben |
| Validator rief WSL-`bash` statt Git-Bash | niedrig | gezielte Interpretersuche |

### 2.2 TFD, mit Zahlen

| Vorgehen | Abdeckung |
|---|---|
| `sanitize=False` (bisheriger Ladeweg der Auswertungsskripte) | **61.2 %** |
| `sanitize=True` | 93.3 % |
| zusätzlich Koordinaten auf die wahre Topologie transplantiert | **100.0 %** |

Starrkörperinvarianz bleibt erhalten (max 2.3e-14). Der Median ändert sich
durch die Behebung praktisch nicht (0.251 → 0.253) — die Behebung *ergänzt*
also fehlende Moleküle, sie verschiebt nicht die Metrik.

Die Transplantation ist zulässig, weil das Modell Atome nur **bewegt**: wahres
und vorhergesagtes Molekül sind per Konstruktion dasselbe Molekül in derselben
Atomreihenfolge. Atomzahl und Ordnungszahlen werden trotzdem geprüft.

> **`TFD READY FOR SECONDARY BENCHMARK`** — Abdeckung wird bei jeder Nennung
> mitberichtet.

### 2.3 Budget-Kalibrierung, aus dem Code belegt

```
scripts/train.py:218   max_steps = max_epochs * len(train_datafront) // (batch_size * world_size)
scripts/train.py:238   SigmaLightningModule(max_steps=max_steps, …)
core/misc.py:41-45     k_min, k_max, cycle_length ← max_steps
```

Beide Codebasen sind an dieser Stelle **identisch** (`core/misc.py` byteweise
gleich, `train.py:218` gleiche Zeile). Die Reihe 6 h → 3, 12 h → 6, 24 h → 12
ist damit bestätigt.

**Eine Asymmetrie, die vorher nicht benannt war:** `max_steps` ist zugleich eine
**harte Obergrenze** im `Trainer`. Ist eine Methode pro Schritt schneller, kann
sie die Grenze vor Ablauf der 24 h erreichen und hört früher auf. Der Vergleich
wird dann *schritt*-fair statt *zeit*-fair. Bei 12 h wurden ~14 100 von 14 250
möglichen Schritten erreicht — die Kalibrierung sitzt also knapp. Für 24 h ist
das mitzumessen; `summarize_arc_results.py` gibt beide Zahlen aus.

### 2.4 Frisch-Checkout

Ein frischer `git clone` des aktuellen Commits enthält alles Nötige. Dort
laufen: `test_ranking` (22 Tests), `validate_scripts`,
`interpret_rotation_diagnostic --self-test`, `test_exp100`, `validate_mapping`,
`test_exp100_pipeline` — alle grün. **Keine versteckten Abhängigkeiten auf
Dateien im aktuellen Arbeitsbaum.**

---

## 3. Vorbereitet, aber nicht ausgeführt

| Datei | Zweck |
|---|---|
| `arc/00_preflight.sh` | Pfade, Envs, Partitionen, GPU-Klasse, Frame-Fix — Login-Knoten |
| `arc/01_inventory_existing_runs.sh` | prüft das 24h-Verdikt auf ARC nach |
| `arc/train_sigmadock_24h.slurm` | RUN-B24-SD |
| `arc/train_sigmaflow_minimal_24h.slurm` | RUN-B24-SF |
| `arc/train_exp100_sanity.slurm` | EXP-100 auf GPU + echten Daten absichern |
| `arc/train_exp100_24h.slurm` | RUN-B24-E100 |
| `arc/sample_pb_seeds.slurm` | Sampling, `COMPARISON` × `RANKING` × NFE |
| `arc/post_sampling_audits.slurm` | Oracle@K + Symmetrie |
| `arc/exp101_distance_audit.{py,slurm}` | Quellendistanz gegen Haar |
| `arc/validate_checkpoint.py` | PASS/FAIL je Lauf |
| `arc/summarize_arc_results.py` | Sammelübersicht |
| `arc/interpret_rotation_diagnostic.py` | klassifiziert die Weichenfrage |
| `arc/validate_scripts.py` | statische Prüfung, lokal ausführbar |
| `SigmaFlow_Evaluation/metrics/ranking.py` | Oracle@K, Top-1@K, Regret, Kalibrierung |
| `SigmaFlow_Evaluation/metrics/tfd.py` | robuste TFD |
| `SigmaFlow_Evaluation/tests/test_ranking.py` | 22 Tests |

---

## 4. Umgebung

**Lokal** (nur für CPU-Tests und Auswertung relevant):

| | |
|---|---|
| Python | 3.12.10 |
| torch | 2.13.0+cpu |
| torch_geometric | 2.8.0.post1 |
| rdkit | 2026.03.5 |
| numpy / pandas / scipy | 2.5.1 / 2.2.3 / 1.18.0 |
| pytorch_lightning | 2.6.5 |
| posebusters | 0.6.5 |

**ARC** — `PENDING ARC ENVIRONMENT CHECK`. Zwei getrennte Umgebungen, das ist
zwingend:

| Umgebung | Für | Aktivierung |
|---|---|---|
| `/data/stat-cadd/shug8458/sigmaflow_env` | SigmaFlow, EXP-100 | `conda activate` |
| `/data/stat-cadd/shug8458/myenv` | SigmaDock | `source activate` |

Jeder Lauf schreibt `pip_freeze.txt` in sein Laufverzeichnis. Nach der Wartung
könnten sich Modulversionen geändert haben — deshalb wird das Manifest je Lauf
erfasst, nicht einmal zentral.

---

## 5. Datenpfade

| Pfad | Art | Von wem benutzt |
|---|---|---|
| `SigmaFlow_Minimal/notebooks/dummy_data` | repo-relativ | alle lokalen Tests |
| `SigmaFlow_Variants/posebusters_full_comparison/true_ligands` | repo-relativ | Symmetrie-, Oracle@K-, TFD-Auswertung |
| `SigmaFlow_Variants/posebusters_full_comparison/seeds10_*` | repo-relativ | Referenzzahlen 12 h |
| `/data/stat-cadd/shug8458/data` | ARC absolut | Training, Sampling, EXP-101 |
| `/data/stat-cadd/shug8458/arc_runs` | ARC absolut | alle Laufausgaben |
| `/data/stat-cadd/shug8458/{sigmaflow_env,myenv}` | ARC absolut | Umgebungen |

Alle ARC-Pfade sind an **einer** Stelle definiert (`arc/_common.sh`). Ändert
sich der Cluster, ist genau diese Datei anzupassen.

---

## 6. Fehlerbild-Checkliste für den ARC-Neustart

| Symptom | Prüfbefehl |
|---|---|
| falscher Commit | `git -C $ARC_REPO rev-parse --short HEAD` |
| SigmaFlow_Minimal / EXP-100 fehlen | `bash arc/00_preflight.sh` |
| falsche Python-Umgebung | im Log: `sigmadock geladen aus:` |
| falscher Import-Root | Sanity-Gate bricht ab — das ist der gewünschte Fall |
| GPU nicht verfügbar | `nvidia-smi`; im Log `CUDA: True` |
| `l40s` umbenannt | `sinfo -h -o "%G" \| tr ',' '\n' \| sort -u` |
| Partition weg | `sinfo -o "%20P %10a %12l"` |
| Datensatzpfad fehlt | `ls /data/stat-cadd/shug8458/data` |
| Checkpointpfad falsch | `python arc/validate_checkpoint.py --all` |
| CRLF | `file arc/*.slurm` — muss „ASCII text" ohne „CRLF" zeigen |
| Resume-Mismatch | `experiment_dir.txt` gegen den `--resume`-Pfad prüfen |
| Speicherüberlauf | `sacct -j <ID> --format=JobID,MaxRSS,State` |
| NaN | `python arc/validate_checkpoint.py --run_dir …` zählt sie |
| Plattenkontingent | `du -sh /data/stat-cadd/shug8458/arc_runs`, `quota -s` |
| `ReqNodeNotAvail` | normal während der Wartung; `--time` senken oder warten |

---

## 7. Forschungsentscheidungsbaum

```
ARC kommt zurück
│
├─ git push (LOKAL)  →  git pull (ARC)          ← Blocker, siehe §8
│
├─ arc/00_preflight.sh                          Minuten, kein Job
├─ arc/01_inventory_existing_runs.sh            bestätigt das 24h-Verdikt
│
├─ ROTATIONSDIAGNOSE  (Minuten)  ◀── die Weiche
│   │   arc/interpret_rotation_diagnostic.py klassifiziert automatisch
│   │
│   ├─ SHRINKAGE-LIKE        → EXP-101, dann D1 (heuristische Quelle)
│   │                          alternativ D4 (Endpunkt-Parametrisierung)
│   ├─ DIRECTION-NOISE-LIKE  → D3 (direkter Rotationskopf)
│   │                          Quelle wäre hier FALSCH: sie kaschierte den Kanal
│   ├─ BOTH                  → zuerst D3, dann erst D1
│   └─ INCONCLUSIVE          → mehr Komplexe, oder auf das 24h-Modell warten
│
├─ EXP-100 Sanity (40 min)
│   └─ grün → EXP-100 24h optional
│
├─ 24h: SigmaDock + SigmaFlow Minimal (parallel)
│
├─ Sampling: 10 Seeds, controlled + default, raw + ranked
│
├─ Oracle@K
│   ├─ Oracle@10 ≥ 3× Single → Confidence (D6) lohnt sich
│   └─ sonst                 → Ranking ist nicht der Engpass
│
├─ EXP-101 Distance Audit
│   ├─ Gewinn > 10°  → D1 bauen
│   └─ sonst         → D1 verwerfen, Engpass ist der Kanal
│
└─ genau EIN isoliertes Folgeexperiment
```

**Der Baum hat bewusst nur eine Weiche pro Ebene.** Zwei Äste gleichzeitig zu
verfolgen würde die Zuschreibung zerstören.

---

## 8. Der eine verbleibende lokale Blocker

### `LOCAL BLOCKER BEFORE ARC: GIT PUSH REQUIRED`

`origin/main` steht auf `f032b83`, lokal sind **9 Commits** und ein **Tag**
weiter. ARC hat damit weder `SigmaFlow_Minimal` noch EXP-100, weder die
Roadmap noch `arc/`.

```bash
cd /c/Users/julia/Documents/SigmaFlow
git push origin main
git push origin --tags          # sigflow-minimal-baseline-v1 fehlt sonst
git log --oneline origin/main..HEAD    # muss danach leer sein
```

Das Fernarchiv ist erreichbar (geprüft). Der Push wurde **nicht** ausgeführt —
das ist eine Veröffentlichung und bleibt deine Entscheidung.

---

## 9. Verbleibende lokale Aufgaben

Keine.

Die nächste benötigte Information — ob der Rotationskanal schrumpft oder
rauscht — ist ohne GPU nicht zu bekommen. Alle davon abhängigen Entwürfe sind
bis an die Implementierungsgrenze ausgearbeitet und warten auf genau dieses
eine Ergebnis.
