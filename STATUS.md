# SigmaFlow — Development Status

Diese Datei ist das "Lesezeichen" für den Projektfortschritt. Am Anfang jeder
neuen Session: diese Datei zuerst lesen, dann nahtlos weitermachen. Am Ende
jeder Session (oder vor einer Pause): diese Datei aktualisieren.

## 🔖 PAUSE-PUNKT #2 (2026-07-16, später am selben Tag) — AKTUELL, zuerst lesen

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
