# EXP-110 — Zwei-Kopf-Vektorfeld · Stand

Stand **2026-08-20**. Diese Datei beschreibt *diese Variante*. Der
Projektstand insgesamt steht in `STATUS.md` in der Repowurzel.

---

## 1. Identität des Experiments

| | |
|---|---|
| Herkunft | `SigmaFlow_Minimal` @ **16a069a** (`COPIED_FROM.txt`) |
| Implementierung | **8473a0b** — „EXP-110: Zwei-Kopf-Vektorfeld als eigene Variante" |
| Audit | **eb2d924** — „EXP-110 Audit: 37 Checks, drei Befunde, eine Selbstkorrektur" |
| Vergleichslauf | Job **8541310**, SigmaFlow Minimal Changes, 12 h, 11:05:40, 13 750 Steps |

Verifiziert: `SigmaFlow_Minimal/src` und `SigmaFlow_Variants/d_frame_fix/src`
sind byte-identisch. Der 12h-Referenzlauf lief zwar aus dem `d_frame_fix`-Baum,
kopiert wurde aus `SigmaFlow_Minimal` — der Vergleich ist trotzdem sauber.

---

## 2. Was rechnet die Variante

Das Netz gibt **zwei** äquivariante Vektorfelder je Atom aus, aus getrennten
Köpfen auf einem geteilten Rumpf:

```
f_i^T, f_i^R  ∈ R^3          (trans_block, rot_block)
```

Pooling je Fragment `f`, **Mittel**, nicht Summe:

```
v_f = mean_{i ∈ f} f_i^T                  ∈ R^3
w_f = mean_{i ∈ f} f_i^R                  ∈ R^3
```

Translationsvektorfeld direkt, Rotationsvektorfeld über die adjungierte
Konjugation vom Welt- in den Körperrahmen:

```
û_t^trans = v_f
û_t^R     = R_t^T · hat(w_f) · R_t        ∈ so(3),  [M,3,3]
```

**Rahmenkonvention:** Ziel und Vorhersage leben im **Körperrahmen**
(rechtstrivialisiert). Das Trainingsziel `log(R_t^T R_1)` ist unter globaler
Rotation **invariant** (`(QR_t)^T(QR_1) = R_t^T R_1`), jede äquivariante
Netzausgabe ist es **nicht**. Die Konjugation schließt genau diese Lücke —
für *jede* Konstruktion, auch für den alten Drehmomentweg. Der Frame-Fix war
kein Artefakt von Newton-Euler.

Der Loss ist unverändert der aus Minimal: gewichtete MSE gegen
`u_t^trans` und `u_t^R = log(R_t^T R_1)`, Gewichte 2.0 / 0.5.

Nicht mehr im aktiven Pfad: `_compute_fragment_dynamics`,
`_predict_fragment_updates`, `linear_mechanics`, `newton_maruyama`. Der Code
bleibt liegen, wird aber weder im Training noch beim Sampling aufgerufen —
im Audit nachgewiesen.

---

## 3. Parameter

Gemessen an dem Modell, das `scripts/train.py` aus `RunConfig()` baut:

| | Parameter |
|---|---:|
| Rumpf + 1 Kopf (= Minimal) | 14 979 377 |
| zweiter Kopf | 1 920 449 |
| gesamt | **16 899 826** |
| Zuwachs | **+12.82 %** |

Beide Köpfe baugleich. `zero_init_last` greift über `self.apply(...)` auf
`out_features == 1` und damit strukturell auf beide — keine Asymmetrie.

> Die früher hier und in `EXP-110_README.md` genannten 24 466 418 / +12.99 %
> waren falsch: sie stammten aus einer handgesetzten Instanziierung, nicht
> aus `RunConfig`. Korrigiert am 2026-08-20.

---

## 4. Was bewiesen ist

Alle fünf Audits am 2026-08-21 frisch grün, **135 Checks**:

```bash
# aus dem Variantenordner
PYTHONPATH=src python audits/audit_two_head_full.py          # 37/37
PYTHONPATH=src python audits/test_checkpoint_roundtrip.py    # 35/35
PYTHONPATH=src python audits/test_real_training_step.py      # 25/25
# aus der Repowurzel
PYTHONPATH=SigmaFlow_FM_Specific/EXP-110_two_head_vector_field/src \
    python audits/test_two_head_vector_field.py              # 22/22
PYTHONPATH=SigmaFlow_FM_Specific/EXP-110_two_head_vector_field/src \
    python audits/test_two_head_gradients.py                 # 16/16
```

Belegt sind unter anderem:

- `û^R` ist exakt schiefsymmetrisch;
- die Konjugation stellt die Invarianz her, ohne sie wird sie verletzt;
- `L_trans` erreicht `rot_block` mit Gradient **exakt 0** und umgekehrt —
  die Köpfe sind strukturell entkoppelt, nicht nur empirisch orthogonal;
- beide Losse erreichen den geteilten Rumpf;
- der Gradient fließt durch `R_t^T (·) R_t`;
- Training und Sampling benutzen dieselbe Kette;
- ein Checkpoint überlebt Schreiben und Zurückladen über den **echten**
  Pfad (`trainer.save_checkpoint` → `load_from_scratch`), beide Köpfe
  bitgleich, und `SigmaFlow_Minimal` kann ihn **nicht** laden.

**Selbstkorrektur.** Die ursprüngliche Begründung, Mittel-Pooling sei nötig,
weil 1-Atom-Fragmente `τ ≡ 0` hätten, war gegenstandslos: der Basispfad
enthält `assert M.min() >= 2`, solche Fragmente kamen also nie vor. Das
tragfähige Argument ist ein anderes — 2-Atom-Fragmente sind linear, und ihr
regularisierter Trägheitstensor hat `cond(I_reg) ≈ 1.5e8`.

**Echter Trainingsschritt, seit 2026-08-21 geprüft.**
`audits/test_real_training_step.py` lädt echte Komplexe aus
`notebooks/dummy_data`, ruft den echten EquiformerV2 auf und geht die volle
Kette: Vorwärtslauf, Loss, Rückwärtslauf, Optimiererschritt. Gemessen:
`loss_R` = 8.35 gegen die theoretische Haar-Erwartung
`2·E[θ²] ≈ 8.4` für `‖Log(R₀ᵀR₁)‖_F²`, und ein Verhältnis von Loss zu
Zielnorm von exakt 1.000, was bestätigt, dass `zero_init_last` greift.
Beide Köpfe bekommen Gradient, 100 % der Parameter.

Zusätzlich lief ein vollständiger `scripts/train.py`-Lauf auf CPU durch
Sanity-Validierung, Trainingsschritte, volle Validierung und
Checkpoint-Schreibung.

**Drei Blocker, die genau diese Lücke durchgelassen hatte** (alle behoben):

1. **`conf/experiments/` fehlte komplett.** Alle sieben YAMLs gingen beim
   Kopieren verloren, weil die `.gitignore`-Regel `**/experiments/` greift.
   `train.py` löst `--train_exps` und `--val_exps` darüber auf, der Job wäre
   nach Sekunden mit `Experiment config not found` gestorben. Wiederhergestellt,
   byte-identisch zu Minimal, und das Sanity Gate prüft es jetzt.
2. **`forward()` gab `force_per_fragment` und `torque_per_fragment` zurück**,
   Restschlüssel des alten Newton-Euler-Pfads, die es nicht mehr gibt.
   `NameError` in der Sanity-Validierung.
3. **`trainer.py` entpackte `score_terms["pseudoforces"]`**, ebenfalls nicht
   mehr vorhanden. War in Minimal wie hier toter Code, zugewiesen und nie
   benutzt. Entfernt.

Keiner der drei war durch Codeinspektion oder die vorherigen Audits zu finden.
Alle drei fielen beim ersten echten Lauf sofort auf.

---

## 5. Persistenz und Abbruchsicherheit

### Der 12h-Rahmen ist Absicht

Die Walltime bleibt **12:00:00** und wird **nicht** erhöht. Das ist ein
**rechenzeitgematchter** Vergleich gegen den 12h-Minimal-Lauf 8541310:

```
SigmaFlow Minimal 12 h   vs.   SigmaFlow Zwei-Kopf 12 h
```

Das Zwei-Kopf-Modell hat **+12.82 %** Parameter und schafft in denselben
12 Stunden voraussichtlich **weniger** Optimierungsschritte. Das ist Teil
des Vergleichs, kein Mangel. **Ziel sind nicht sechs abgeschlossene
Epochen — das Budget sind 12 Stunden.** Ein Lauf, der nach 5,x Epochen
endet, ist ein gültiges Ergebnis dieses Designs.

### Wie oft Zustand geschrieben wird

Aus dem Trainingscode, nicht aus einer Zutat von uns:

| | |
|---|---|
| `val_check_interval` | 50 |
| `accum_grad_batches` | 1 |
| `check_val_every_n_epoch` | `None` |

Validiert wird also **alle 50 Trainingsbatches**, nicht an Epochengrenzen,
und `ModelCheckpoint(monitor="loss_val/total", save_top_k=3, save_last=True)`
schreibt bei **jedem** dieser Ereignisse `last.ckpt` neu.

**Maximaler Verlust bei einem Walltime-Kill: 50 Schritte.** Beim Tempo des
Referenzlaufs (13 750 Schritte in 11:05:40 = 0.344 Schritte/s) sind das
**rund 2,4 Minuten**; bei 14 % mehr Rechenzeit je Schritt rund 2,8 Minuten.

Lightning schreibt über `_atomic_save` in einer fsspec-Transaktion. Ein
Abbruch mitten im Schreiben lässt die **vorherige** Datei heil; zusätzlich
liegen die drei besten Checkpoints daneben.

### Was ein Abbruch übrig lässt

`last.ckpt` enthält vollständig: Gewichte, `optimizer_states`,
`lr_schedulers`, `epoch`, `global_step`, `hyper_parameters` (inklusive
`equiformer_config`, `denoiser_config`, `max_steps`) und `ema_state_dict`.
Metriken liegen inkrementell im W&B-Offline-Log unter
`<EXP_DIR>/wandb_logs/`; `config.json` mit vollständiger Konfiguration,
Seed und Git-Commit schreibt `train.py` **vor** dem Training.

Auf SIGTERM setzt Lightning `received_sigterm`, wirft beim nächsten Batch
`SIGTERMException`, und `train.py` flusht im `finally`-Zweig `wandb`.

### Kein Vorab-Signal — bewusst

`--signal=SIGUSR1@…` wurde geprüft und **verworfen**. Lightnings
`_slurm_sigusr_handler_fn` schreibt zwar einen Checkpoint, ruft danach aber
`scontrol requeue` und startet den Job neu. Für ein zeitbudgetiertes
Experiment ist das genau das Falsche. Bei 2,4 Minuten maximalem Verlust gibt
es auch keinen Anlass.

### Teil- von Volllauf unterscheiden

Ergänzt wurde nur eine Statusmarkierung auf Shell-Ebene — kein Eingriff in
Optimierung, Modell, Loss, Daten, Seed oder Zeitplan. Eine `trap` auf
`EXIT`/`TERM` schreibt `<EXP_DIR>/RUN_STATUS.json`:

| `status` | bedeutet |
|---|---|
| `COMPLETED` | alle 6 Epochen normal durchgelaufen |
| `WALLTIME_ODER_SIGTERM` | von SLURM abgeräumt |
| `ABGEBROCHEN_rc<N>` | Absturz mit Exitcode N |

Dazu `vollstaendig` (bool), `laufzeit_sekunden`, `slurm_job_id`,
`git_commit`, und aus dem Checkpoint `epoch`, `global_step`,
`max_steps_ziel`, `fortschritt_prozent` sowie ob `trans_block`/`rot_block`
vorhanden sind. **Fehlt die Datei, wurde der Lauf hart getötet** — auch das
ist eine Aussage.

In allen drei Szenarien trocken getestet; jedes schreibt genau eine gültige
Datei.

### Wo alles landet

`ROOT_DIR` ist paketrelativ (`src/sigmadock/oracle.py` → Variantenordner),
also liegt alles **innerhalb** von EXP-110:

```
SigmaFlow_FM_Specific/EXP-110_two_head_vector_field/
└── experiments/sigmadock/0-<MM-DD_HH-MM-SS>/
    ├── checkpoints/last.ckpt          + top-3
    ├── wandb_logs/                    Metriken (offline)
    ├── config.json                    Konfiguration, Seed, Git-Commit
    └── RUN_STATUS.json                Abschlussstatus
```

Ein Überschreiben von SigmaDock-, Minimal- oder früheren EXP-110-Ergebnissen
ist strukturell ausgeschlossen: anderer Baum, und `get_exp_dir` hängt Seed
und Zeitstempel an (bei Kollision zusätzlich `-v2`). `**/experiments/` steht
in `.gitignore`.

---

## 6. Bekannte Mängel

1. **Omega-Bias.** `so3_utils.py:75` klemmt die Spur mit `* (1 - eps)`,
   `eps = 1e-6`. Folge: `Omega(I) = 0.0992°` statt 0 — ein systematischer
   Bias an der Identität. Betrifft Minimal genauso, ist also **kein**
   Unterschied zwischen den Armen, aber er ist da.
2. **`load_from_checkpoint(load_ema=False)` ist kaputt.** Die Funktion
   streift genau ein `"model."` ab, `ckpt["state_dict"]` trägt aber zwei
   Ebenen (LightningModule → Denoiser → Equiformer). Gilt in Minimal
   genauso; fällt nicht auf, weil Sampling `load_from_scratch` benutzt und
   `ema_state_dict` eine Ebene weniger hat. **Kein Blocker.**
3. **`scripts/diagnose_step0.py` ist kaputt.** Entpackt zwei Rückgabewerte
   aus `_compute_forces`, das jetzt drei liefert. Diagnoseskript, nicht
   Trainingspfad.
4. **`_compute_true_vector_field` nicht mitgezogen.** Gleiche Ursache.
5. **`assert M.min() >= 2` entfällt.** Für Mittel-Pooling ist ein
   1-Atom-Fragment unkritisch (Nenner 1), aber die Invariante wird nicht
   mehr geprüft.
6. **Kein Durchsatzwert.** Wie viel der zweite Kopf tatsächlich kostet, ist
   ungemessen. Die 10–15 % sind eine Schätzung — der Lauf misst sie.
7. **Laufname in W&B.** Mit `--debug` heißt der Run `SigmaDock-Trial`, in
   beiden Armen. Zur Unterscheidung dienen Experimentordner, `config.json`
   und `RUN_STATUS.json`, nicht der W&B-Name.

---

## 7. Bereitschaft

| | |
|---|---|
| 12h-Lauf vorbereitet | **ja** — `slurm/train_two_head_12h.slurm` |
| 12h-Lauf abgeschickt | **nein** |
| Walltime | **12:00:00, endgültig** — wird nicht erhöht |
| 72h-Produktionslauf | **NEIN.** Kein Ergebnis auf echten Daten rechtfertigt bisher 72 GPU-Stunden. Die eingefrorene 72h-Konfiguration bleibt unberührt. |

Das 12h-Skript ist bis auf den Auslesekopf zeichengleich mit dem
Referenzlauf. Statisch geprüft: `bash -n`, SBATCH-Ressourcen identisch,
`train.py`-Flagblock identisch, alle 15 Flags akzeptiert, Sanity Gate gegen
den echten Baum grün und gegen `SigmaFlow_Minimal` korrekt abweisend.

---

## 8. Was verglichen wird

Gegen Job 8541310, gleicher Datensatz, gleiches Zeitbudget:

| Größe | Referenz 8541310 |
|---|---|
| Trainings-/Validierungsloss (trans, rot getrennt) | aus dem Log |
| Fragment-Rotationsfehler, absolut | 128.1° bei 6 h; Zufallsniveau 126.5° |
| Lokalitätslücke (gebunden − ungebunden) | −12.8° [−18.3, −7.2] bei 6 h; SigmaDock −22.0° |
| Oracle@10 auf 209 Komplexen | vorhanden |
| **Oracle@10 / Oracle@1** (vorregistriert) | SigmaFlow 6.8, SigmaDock 4.4 |
| erreichte Schritte / Epochen | 13 750 / 6 |

Weil die Rechenzeit gematcht ist und nicht die Schrittzahl, gehört die
tatsächlich erreichte Schrittzahl **mit in jeden Vergleich**. Sie steht in
`RUN_STATUS.json`.

Die Oracle-Kennzahl und ihre drei Ausgänge wurden **vor** den Läufen
festgelegt (Repowurzel-`STATUS.md`). Nicht nachträglich umdeuten.

Die eigentliche Frage an EXP-110: sinkt der absolute Rotationsfehler endlich
unter das Zufallsniveau? Das ist bei 6 h **nicht** passiert und ist die
offene Hauptfrage, an der der getrennte Rotationskopf etwas ändern soll.
