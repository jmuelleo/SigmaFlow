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

Alle drei Audits am 2026-08-20 frisch grün, **75 Checks**:

```bash
PYTHONPATH=src python audits/audit_two_head_full.py    # 37/37
PYTHONPATH=SigmaFlow_FM_Specific/EXP-110_two_head_vector_field/src \
    python audits/test_two_head_vector_field.py        # 22/22   (aus der Repowurzel)
PYTHONPATH=SigmaFlow_FM_Specific/EXP-110_two_head_vector_field/src \
    python audits/test_two_head_gradients.py           # 16/16
```

Belegt sind unter anderem:

- `û^R` ist exakt schiefsymmetrisch;
- die Konjugation stellt die Invarianz her, ohne sie wird sie verletzt;
- `L_trans` erreicht `rot_block` mit Gradient **exakt 0** und umgekehrt —
  die Köpfe sind strukturell entkoppelt, nicht nur empirisch orthogonal;
- beide Losse erreichen den geteilten Rumpf;
- der Gradient fließt durch `R_t^T (·) R_t`;
- Training und Sampling benutzen dieselbe Kette.

**Selbstkorrektur.** Die ursprüngliche Begründung, Mittel-Pooling sei nötig,
weil 1-Atom-Fragmente `τ ≡ 0` hätten, war gegenstandslos: der Basispfad
enthält `assert M.min() >= 2`, solche Fragmente kamen also nie vor. Das
tragfähige Argument ist ein anderes — 2-Atom-Fragmente sind linear, und ihr
regularisierter Trägheitstensor hat `cond(I_reg) ≈ 1.5e8`.

**Was die Audits nicht zeigen.** Sie arbeiten mit Stellvertreterrümpfen, nicht
mit einem echten EquiformerV2-Vorwärtslauf. Geprüft ist die Verdrahtung der
Kopf-, Pooling- und Rahmenlogik, nicht das Netz auf echten Daten. Es gibt
**keinen** Lauf auf echten Daten.

---

## 5. Bekannte Mängel

1. **Omega-Bias.** `so3_utils.py:75` klemmt die Spur mit `* (1 - eps)`,
   `eps = 1e-6`. Folge: `Omega(I) = 0.0992°` statt 0 — ein systematischer
   Bias an der Identität. Betrifft Minimal genauso, ist also **kein**
   Unterschied zwischen den Armen, aber er ist da.
2. **`scripts/diagnose_step0.py` ist kaputt.** Entpackt zwei Rückgabewerte aus
   `_compute_forces`, das jetzt drei liefert. Diagnoseskript, nicht
   Trainingspfad — vor Benutzung reparieren.
3. **`_compute_true_vector_field` nicht mitgezogen.** Gleiche Ursache.
4. **`assert M.min() >= 2` entfällt.** Für Mittel-Pooling ist ein
   1-Atom-Fragment unkritisch (Nenner 1), aber die Invariante wird nicht
   mehr geprüft.
5. **Kein Durchsatzwert.** Wie viel der zweite Kopf tatsächlich kostet, ist
   ungemessen. Die 10–15 % im Skriptkopf sind eine Schätzung.

---

## 6. Bereitschaft

| | |
|---|---|
| 12h-Lauf vorbereitet | **ja** — `slurm/train_two_head_12h.slurm` |
| 12h-Lauf abgeschickt | **nein** |
| 72h-Produktionslauf | **NEIN.** Kein Ergebnis auf echten Daten rechtfertigt bisher 72 GPU-Stunden. Die eingefrorene 72h-Konfiguration bleibt unberührt. |

Das 12h-Skript ist bis auf den Auslesekopf zeichengleich mit dem
Referenzlauf. Statisch geprüft: `bash -n`, SBATCH-Ressourcen identisch,
`train.py`-Flagblock identisch, alle 15 Flags von `train.py` akzeptiert,
Sanity Gate gegen den echten Baum grün und gegen `SigmaFlow_Minimal`
korrekt abweisend.

**Vor dem Abschicken lesen:** der Referenzlauf brauchte 11:05:40 von 12:00:00.
Bei 10–15 % Aufschlag reicht das nicht mehr für 6 Epochen. Details und die
Ein-Stunden-Kontrolle stehen im Skriptkopf.

---

## 7. Was verglichen wird

Gegen Job 8541310, gleicher Datensatz, gleicher Fahrplan:

| Größe | Referenz 8541310 |
|---|---|
| Trainings-/Validierungsloss (trans, rot getrennt) | aus dem Log |
| Fragment-Rotationsfehler, absolut | 128.1° bei 6 h; Zufallsniveau 126.5° |
| Lokalitätslücke (gebunden − ungebunden) | −12.8° [−18.3, −7.2] bei 6 h; SigmaDock −22.0° |
| Oracle@10 auf 209 Komplexen | vorhanden |
| **Oracle@10 / Oracle@1** (vorregistriert) | SigmaFlow 6.8, SigmaDock 4.4 |

Die Oracle-Kennzahl und ihre drei Ausgänge wurden **vor** den Läufen
festgelegt (Repowurzel-`STATUS.md`). Nicht nachträglich umdeuten.

Die eigentliche Frage an EXP-110: sinkt der absolute Rotationsfehler endlich
unter das Zufallsniveau? Das ist bei 6 h **nicht** passiert und ist die
offene Hauptfrage, an der der getrennte Rotationskopf etwas ändern soll.
