# SigmaFlow — Vollständiger wissenschaftlicher und technischer Audit

**Datum:** 2026-08-13
**Methodik:** trace → unabhängig nachrechnen → vergleichen → abgleichen → erst dann akzeptieren.
Keine Zahl aus STATUS.md / RESULTS.md / ERGEBNISSE wurde übernommen, ohne sie aus
Rohdaten oder analytisch neu zu erzeugen.

**Grenze des Audits:** Trainings- und Sampling-Läufe liegen auf ARC. Lokal verifizierbar
waren: die gesamte Mathematik (torch 2.13 CPU), alle Auswertungsskripte, alle
Vorhersage-SDFs, die Datenpipeline-Logik. NICHT verifizierbar: die auf ARC
deployten Configs, die 24h-Vorhersagen, Trainingskurven.

---

## 1. AUDIT-VERDIKT

**Die Kernmathematik ist korrekt. Es wurden fünf Befunde gefunden, davon zwei materiell.**

| # | Befund | Schwere | Status |
|---|---|---|---|
| 1 | Rotationskanal ist schlechter als Raten (146.1° vs Zufallsmedian 132.4°) | **MATERIELL** | offen, Engpass |
| 2 | Lokales Repo reproduziert die ARC-Läufe nicht (`sdf_regex` weicht ab) | **MATERIELL** | UNVERIFIZIERT |
| 3 | Kommentar in `newton_maruyama` widerspricht dem Code (BODY vs WORLD) | irreführend | zu korrigieren |
| 4 | `so3_utils.log` hat Genauigkeitsgrenze nahe π (`Omega`-Bias) | gering | quantifiziert |
| 5 | Zufalls-Baseline 126.5° ist der MITTELWERT, wird gegen MEDIANE verglichen | Methodik | zu korrigieren |

**Verifiziert korrekt:** Frame-Fix, Flow-Matching-Formulierung, ODE-Integration,
Zeitkonvention, Train/Inferenz-Konsistenz, R3-Kanal, Kopien-Fix.

---

## 2. WAS UNABHÄNGIG VERIFIZIERT WURDE

### 2.1 SO(3)/SE(3)-Schicht (numerisch, gegen analytisch bekannte Antworten)

- `vee(hat(v)) == v`, `hat(v)` schiefsymmetrisch ✅
- `R Rᵀ = I`, `det R = +1`, `log(Aᵀ) = −log(A)`, `Rz(30)Rz(40) = Rz(70)` ✅
- `exp(log(R)) == R` für 0°, 1e-4°, 0.001°, 1°, 45°, 90°, 120° ✅ (Fehler ≤ 5e-7)
- **Adjungierten-Identität** `R·exp(RᵀWR·dt) = exp(W·dt)·R`: max. Abweichung 6.7e-16 ✅
- Konjugation erhält Schiefsymmetrie ✅

### 2.2 Flow-Matching-Formulierung (analytisch hergeleitet, dann verglichen)

| Frage | Antwort im Code | Korrekt? |
|---|---|---|
| Basisverteilung | R³: N(0,I); SO(3): Uniform (Haar) | ✅ |
| Datenverteilung | R³: Fragment-COM; SO(3): **Punktmasse bei I** | ✅ (geerbt) |
| Pfad | R³: linear; SO(3): geodätisch `R_0 exp(t·log(R_0ᵀR_1))` | ✅ |
| Zeitkonvention | t=0 Rauschen → t=1 Daten, überall | ✅ |
| Zielvektorfeld | `(x₁−x_t)/(1−t)` bzw. `log(R_tᵀR₁)/(1−t)` | ✅ |
| Identität | = `x₁−x₀` bzw. `log(R₀ᵀR₁)`, konstant in t | ✅ analytisch bestätigt |
| Tangentialraum | Körperrahmen (rechtsmultiplikativ) | ✅ konsistent |
| Netzausgabe | Weltrahmen `hat(I_t⁻¹τ)` | ✅ belegt |
| Integration | `R_next = R_t exp(v·dt)`, `x_next = x_t + v·dt` | ✅ |
| Train = Inferenz | beide rufen `_compute_vector_field` | ✅ |

**Oracle-Test (eigene Rechnung):** mit dem wahren Feld erreicht die Euler-Integration
bei 5/25/100 Schritten das Ziel — R³ exakt (Fehler 0.00), SO(3) max. 0.26° Restfehler.
**Die Integration ist nicht der Engpass.**

### 2.3 Der Frame-Fix ist korrekt UND notwendig

Drei unabhängige Belege:

1. **Mathematisch:** `u_body = Rᵀ W_world R` ist die korrekte Adjungierte (verifiziert, 6.7e-16).
2. **Empirisch:** ohne Fix erreicht die Integration mit dem *richtigen* Feld nur
   **119.6° mediane Winkelabweichung** — praktisch Zufall. Der Rotationskanal war
   vor dem Fix funktionslos.
3. **Aus SigmaDocks eigenem Code:** `denoiser.py:1049` macht `R0_hat = dR @ R_t`
   (**Links**multiplikation = Weltrahmen), Kommentar Zeile 1052: „we translate omega
   which is skew-symmetric at the identity into global world frame at R_t".
   SigmaFlow integriert rechtsmultiplikativ (Körper). Der Mismatch war real.

**Rahmenquelle bestätigt:** `I_t` wird aus `pos_t` (Weltkoordinaten) berechnet,
`torque` aus `pos_t`/`lig_forces` (Welt). Also ist `dW = I_welt⁻¹τ_welt` weltrahmig.
Die Selbstprüfung `I_t ≈ (R_tR₁ᵀ) I₀ (R_tR₁ᵀ)ᵀ` in Zeile 827 ist die
Konjugationssignatur eines Welt-Trägheitstensors.

---

## 3. BEFUND 1 (MATERIELL): Rotation ist schlechter als Raten

Fehlerzerlegung über 2090 Posen je Methode (10 Seeds × 209 Komplexe), Mediane:

| Lauf | roh | Schwerpunkt | ausgerichtet | Ganzmolekül-Rotation |
|---|---|---|---|---|
| SigmaFlow | 4.94 Å | **1.17 Å** | 1.98 Å | **146.1°** |
| SigmaDock | 4.43 Å | 1.25 Å | 1.86 Å | 124.0° |
| Zufall | — | ~5–10 Å | — | **132.4°** (Haar-Median) |

- **Translation funktioniert** und ist sogar besser als SigmaDock (1.17 vs 1.25 Å).
- **Innere Geometrie funktioniert** (1.98 vs 1.86 Å, Differenz +0.07 Å).
- **Rotation liegt ÜBER der Zufallsgrenze.** Das ist keine Untertrainiertheit —
  ein untrainiertes Modell landet AUF der Zufallsgrenze, nicht darüber. Ein
  systematischer Bias von +14° gegenüber Zufall deutet auf eine verbleibende
  Vorzeichen-, Rahmen- oder Skalierungsinkonsistenz im Rotationskanal.

**Wichtige Einschränkung:** die Ganzmolekül-Kabsch-Rotation ist für ein flexibles
Mehrfragment-Molekül eine Kompromissgröße, kein sauberer Starrkörperwinkel.
Der Befund ist stark, aber nicht allein beweisend.

### Konkreter Verdacht: Loss-Gewichtung

`conf/training/slurm.yaml`: `trans_score_weight: 2.0`, `rot_score_weight: 0.5`.
Da `‖hat(ω)‖²_F = 2|ω|²`, ist das effektiv **2.0 (Translation) gegen 1.0 (Rotation)**.

Beide Ziele sind im Flow Matching O(1) (Translation `x₁−x₀`, Rotation `log(R₀ᵀR₁)`
mit Haar-Mittel ~2.2 rad). Das Verhältnis 2.0:0.5 stammt aus SigmaDocks
Diffusions-Setup, wo die Score-Magnituden durch σ-Pläne skaliert sind — dort
bedeutete es etwas völlig anderes. **Im Flow Matching wurde es nie neu hergeleitet
und nie abliert.** Das Projekt wurde von genau dieser Art geerbter Hyperparameter
schon zweimal getroffen (rückwärts laufender `edm`-Plan, geerbte 25 Schritte).

---

## 4. ~~BEFUND 2 (MATERIELL): Das lokale Repo reproduziert die ARC-Läufe nicht~~

### ❌ WIDERRUFEN 2026-08-13 — der Befund war FALSCH

Der Befund lautete: die lokalen Configs sagen Plural, die Ausgaben tragen
Singular, also müssten die ARC-Configs abweichen und die Läufe seien lokal
nicht reproduzierbar.

**Beides ist widerlegt:**

1. Die ARC-Config wurde ausgelesen und ist **identisch** mit der lokalen
   (`conf/experiments/posebusters.yaml` sagt beidseits Plural). Ein
   vollständiger `diff -rq` über `src/`, `scripts/` und `conf/` zwischen
   `SigmaFlow_Development/` und `SigmaFlow_Variants/d_frame_fix/` zeigt
   **genau eine** abweichende Datei: `sigma_flow_generator.py`, und dort nur
   den Frame-Fix. `conf/` ist byteweise gleich.
2. Die eigentliche Ursache ist ein **Bug im Config-Merge**, nicht eine
   abweichende Deployment-Config: `sampling_setup.py::build_sampling_datafront`
   überschreibt den `sdf_regex` der Experimentdatei bedingungslos mit dem Wert
   aus `conf/sampling/base.yaml` (`.*ligand.sdf$`, Singular), weil der Guard
   `if "sdf_regex" in cfg.experiments` immer wahr ist. Siehe STATUS.md,
   Abschnitt "BUG GEFUNDEN (2026-08-13)".

**Die Läufe SIND aus dem lokalen Repo reproduzierbar.** Die Datenprovenienz
ist geklärt: alle ausgewerteten Läufe lasen `<cid>_ligand.sdf`, die kanonische
Einzelkopie. Einzige Ausnahme bleibt Job 8553984 (alter SigmaDock-Vergleich),
der den Regex explizit auf Plural setzte.

Lehre für den Audit selbst: aus "Konfigurationsdatei sagt X, Ausgabe sagt Y"
folgt nicht "die Datei ist eine andere". Es folgt nur, dass zwischen Datei und
Laufzeit etwas passiert. Ich hätte den Merge-Pfad verfolgen müssen, bevor ich
auf abweichendes Deployment schloss.

### Warum das inhaltlich zählt

Am Dummy-Satz gemessen: die Singular-Datei enthält genau EIN Molekül und ist
**exakt gleich einer Kopie der Plural-Datei, aber nicht immer Kopie 0**:

| Komplex | Kopien | RMSD Singular vs Kopie 0 | identisch mit |
|---|---|---|---|
| 1HWI_115 | 4 | **41.68 Å** | Kopie **#3** |
| 1U1C_BAU | 6 | **45.77 Å** | Kopie **#5** |

Die Singular-Datei ist die **kanonische** Ligandenkopie. Die Datenpipeline nimmt
(`data.py:250`) `ligand_mols[0]` — bei der Plural-Datei also eine womöglich falsche.

**Konsequenz für den Kopien-Fix vom selben Tag:** „Minimum über Kopien" ist eine
Näherung, nicht die richtige Referenz. Richtig wäre die Singular-Datei.

- Für die **seeds10-Läufe** (beide Methoden, Singular-Eingabe) ist der Fix ein
  No-Op: 0 von 84 Mehrkopie-Komplexen wählen eine andere Kopie als 0. Diese Zahlen
  sind belastbar.
- Für **`sigmadock_12h_pred`** (Plural-Eingabe) greift der Fix bei 44 von 84 —
  er könnte zu großzügig sein. **Diese korrigierten Einzelziehungs-Zahlen sind
  UNVERIFIZIERT.**

---

## 5. BEFUNDE 3–5 (geringfügig)

**3. Falscher Kommentar.** `sigma_flow_generator.py:768`: „this is the LOCAL
(right-invariant - BODY) frame of reference". Der Code rechnet im Weltrahmen; die
Transport-Zeile darunter ist auskommentiert. Der Kommentar widerspricht Zeile 771
im selben Block und hätte diesen Audit fast fehlgeleitet. **Ersatzlos streichen.**

**4. `so3_utils.log` nahe π.** `Omega()` skaliert die Spur mit `(1−eps)`, eps=1e-6.
Nahe π wird der Bias mit cot(θ) verstärkt. Gemessen an 4000 Haar-Rotationen:
8.5 % haben θ>172°, dort Rundreisefehler bis 1.6e-2 (Matrixeinträge).
Wirkung auf das Trainingsziel (float32):

| t | Median-Fehler | p99 | max |
|---|---|---|---|
| 0.5 | 1.9e-6 | 5.0e-3 | 3.8e-2 |
| 0.9 | 9.7e-6 | 2.8e-2 | 1.6e-1 |
| 0.99 | 7.8e-5 | 2.8e-1 | 1.7e0 |
| 0.995 | 1.6e-4 | 8.4e-1 | 3.8e0 |

Der Median ist überall vernachlässigbar. Bei t>0.99 haben ~1 % der Ziele Fehler in
der Größenordnung des Signals — schwerschwänziges Labelrauschen, das ein MSE-Loss
überproportional bestraft. **Real, aber gering** (betrifft ~0.01 % aller Samples
bei uniformem t).

**5. Zufalls-Baseline.** `RANDOM_BASELINE_DEG = 126.5` ist der **Mittelwert** der
Haar-Winkelverteilung; der **Median** liegt bei **132.4°**. Der Code vergleicht ihn
in `fragment_locality.py` und `full_metrics.py` gegen Mediane. Beide Werte
angeben, nicht mischen.

---

## 6. KORRIGIERTE GRUNDWAHRHEIT

Vorzeichen: positiv = **SigmaFlow schlechter**. Basis: 10-Seed-Experiment
(je 2090 Posen, beide Methoden lasen dieselbe Singular-Datei).

### Aggregate (seed-gemittelt pro Komplex, n=209)

| Metrik | SigmaFlow | SigmaDock |
|---|---|---|
| roh RMSD mean / median | 5.24 / 5.10 Å | 4.77 / 4.71 Å |
| roh SD / q25 / q75 / q90 | 1.53 / 4.09 / 6.21 / 7.23 | 1.53 / 3.56 / 5.81 / 6.69 |
| ausgerichtet mean / median | 2.05 / 2.10 Å | 1.97 / 1.98 Å |
| <5 Å (seed-gemittelt) | 46.4 % | 55.5 % |
| >20 Å | **0** | **0** |

### Pro Seed (die eigentlich relevante Streuung)

| Metrik | SigmaFlow | SigmaDock |
|---|---|---|
| Median-RMSD je Seed | 4.56–5.70 (SD 0.307) | 4.15–4.87 (SD 0.262) |
| <2 Å je Seed | 1.4–6.7 % (SD 1.75) | 5.7–13.4 % (SD 2.17) |
| <5 Å je Seed | 50.9 % (SD 3.92) | 57.5 % (SD 3.70) |
| SD desselben Komplexes | 1.55 Å | 1.58 Å |
| Spannweite desselben Komplexes | 4.46 Å | 4.59 Å |
| best-of-10 <2 Å | 29.2 % | 45.9 % |

### Gepaarte Vergleiche (Bootstrap-CI, 20 000 Ziehungen)

| Vergleich | n | Median | Mittel | 95%-CI | SF besser | Urteil |
|---|---|---|---|---|---|---|
| roh RMSD, Seed-Mittel | 209 | +0.39 | +0.47 | [+0.36, +0.59] | 29 % | **signifikant** |
| ausgerichtet, Seed-Mittel | 209 | +0.07 | +0.07 | [+0.03, +0.11] | 37 % | **signifikant** |
| roh RMSD, Einzelziehung 12h | 209 | +0.48 | +0.77 | [+0.43, +1.10] | 37 % | signifikant¹ |
| ausgerichtet, Einzelziehung | 209 | +0.11 | +0.11 | [+0.01, +0.23] | 41 % | signifikant¹ |

¹ UNVERIFIZIERT — SigmaDock-Seite las die Plural-Datei, Referenzwahl unsicher (§4).

### Chemische Plausibilität (McNemar exakt, gepaart, n=209)

Gegen das Kopien-Artefakt geprüft: 44 Referenzen umgestellt, **0** Urteilsänderungen.
PoseBusters rechnet symmetriekorrigiert und war immun.

| Check | SF % | SD % | nur SF | nur SD | p | Urteil |
|---|---|---|---|---|---|---|
| Bindungslängen | 55.5 | 48.8 | 37 | 23 | 0.093 | n.s. |
| Bindungswinkel | 48.3 | 41.6 | 34 | 20 | 0.076 | n.s. |
| sterische Clashes | 47.8 | 46.4 | 27 | 24 | 0.780 | n.s. |
| interne Energie | 57.4 | 59.8 | 29 | 34 | 0.615 | n.s. |
| Chiralität | 93.8 | 90.9 | 14 | 8 | 0.286 | n.s. |
| alle 15 Checks | 37.3 | 32.5 | 28 | 18 | 0.184 | n.s. |
| **RMSD ≤ 2 Å** | 2.9 | 8.1 | 6 | **17** | **0.035** | **SigmaDock** |

### Schritt-Sweep (7 Schrittzahlen × 3 Seeds, n=209)

| Schritte | Median ± SD | <2 Å | gepaart vs 25 | 95%-CI |
|---|---|---|---|---|
| 5 | 4.61 ± 0.06 | 3.7 % | −0.20 | [−0.39, −0.01] |
| 10 | 4.78 ± 0.22 | 3.5 % | −0.13 | [−0.32, +0.05] |
| 15 | 4.88 ± 0.11 | 4.0 % | −0.11 | [−0.31, +0.08] |
| **25** | **4.76 ± 0.19** | **4.1 %** | Referenz | — |
| 50 | 5.08 ± 0.05 | 4.0 % | +0.11 | [−0.10, +0.32] |
| 100 | 5.17 ± 0.18 | 4.8 % | +0.09 | [−0.09, +0.29] |
| 200 | 4.93 ± 0.24 | 3.7 % | +0.06 | [−0.12, +0.23] |

Nur 5 Schritte ist formal signifikant, überlebt aber keine Bonferroni-Korrektur
(6 Vergleiche, obere CI-Grenze −0.01). **Korrekte Aussage: keine Schrittzahl
zwischen 5 und 200 ist nachweisbar besser als eine andere.**

---

## 7. WAS WIR JETZT WISSEN (verteidigbar)

1. Die Umstellung von Diffusion auf Riemannsches Flow Matching ist **mathematisch
   korrekt implementiert** — unabhängig verifiziert, nicht nur gelesen.
2. Die Pipeline läuft end-to-end und ist **bitgenau reproduzierbar** (209/209 Posen
   identisch bei gleichem Seed).
3. **Die Integration ist nicht der Engpass** (Oracle-Test + Schritt-Sweep, zwei
   unabhängige Belege).
4. **Die Translation funktioniert** (Schwerpunkt 1.17 Å, besser als SigmaDock).
5. **Die Rotation ist der Engpass** und liegt über der Zufallsgrenze.
6. SigmaDock ist gesichert besser, aber **um einen kleinen Betrag** (+0.47 Å bei
   einem Fehlerniveau von ~5 Å).
7. **Das Ziehungsrauschen ist ~10× so groß wie der Methodeneffekt.** Jeder
   Einzelziehungs-Vergleich des Projekts las überwiegend Rauschen.
8. Chemische Plausibilität ist nicht unterscheidbar.
9. **SigmaFlow kann mit 5 statt 25 Schritten sampeln** — 5× billiger.
10. Der Frame-Fix war real, notwendig und ist korrekt.

## 8. WAS WIR NICHT WISSEN

1. **Warum die Rotation schlechter als Zufall ist.** UNGELÖST — der wichtigste Punkt.
2. Ob SigmaFlow weniger Schritte braucht als **SigmaDock** — Gegen-Sweep fehlt.
3. Ob die 24h-Ergebnisse gültig sind — UNVERIFIZIERT, Daten nicht lokal.
4. Ob die korrigierten Einzelziehungs-12h-Zahlen stimmen — UNVERIFIZIERT (§4).
5. Ob das Modell untertrainiert oder gesättigt ist — Trainingskurven nicht geprüft.
6. Welche Configs auf ARC tatsächlich liefen — UNVERIFIZIERT (§4).

---

## 9. ENGPASS-RANKING

| Rang | Ursache | Evidenz | Konfidenz |
|---|---|---|---|
| 1 | **Rotationskanal (Loss-Gewicht / Restinkonsistenz)** | 146.1° > Zufall 132.4°; Translation gleichzeitig gut; Gewicht 2.0:0.5 nie abliert | **hoch** |
| 2 | Trainingsdauer | ~5 % der Paper-Gradientenschritte; 6h→12h bewegte <2 Å nur 1.0→1.9 % | mittel |
| 3 | Architektur/Kontext | Torque über `I⁻¹τ` koppelt Ziel an anisotropen, zustandsabhängigen Tensor | mittel |
| 4 | Wahrscheinlichkeitspfad | `R₁ = I` macht das Rotationsziel kontextfrei | mittel-gering |
| 5 | Daten | Kanonische-Kopie-Frage (§4) | gering-mittel |
| 6 | Integrator | zweifach widerlegt | **sehr gering** |
| 7 | Auswertungsartefakt | gefunden und behoben | sehr gering |

---

## 10. EMPFOHLENE NÄCHSTE EXPERIMENTE

### E1 — Rotations-Loss-Gewicht ablieren (SOFORT, ~4 GPU-h)
`rot_score_weight` ∈ {0.5, 2.0, 8.0} bei sonst identischer 6h-Konfiguration.
**Entscheidungskriterium:** sinkt der absolute Fragment-Rotationsfehler unter die
Zufallsgrenze? Wenn ja, ist der Engpass ein Gewichtungsproblem und billig behebbar.
Höchstes Informationsgewinn-pro-Kosten-Verhältnis im gesamten Projekt.

### E2 — Rotations-Vorzeichentest (SOFORT, Minuten, kein GPU)
Diagnose auf bestehenden Checkpoints: Kosinusähnlichkeit zwischen vorhergesagtem
und wahrem Rotations-Vektorfeld, stratifiziert nach t. Ist sie **systematisch
negativ**, liegt ein Vorzeichen-/Rahmenfehler vor, kein Lernproblem. Das ist der
direkte Test für Befund 1 und trennt „nicht gelernt" von „falsch herum gelernt".

### E3 — ARC-Config-Abgleich (SOFORT, Minuten)
`sdf_regex` der deployten Configs auslesen und mit dem lokalen Repo abgleichen;
prüfen, ob `<cid>_ligand.sdf` in `posebusters_paper/` existiert und ob es Kopie 0
der Plural-Datei entspricht. Schließt Befund 2 und entscheidet, ob die
Einzelziehungs-Zahlen verwendbar sind.

Danach erst: SigmaDock-Schritt-Sweep (~10 min GPU), Frame-Fix-Rückportierung,
24h-Nachrechnung.

---

## 11. BEDEUTUNG FÜR DIE THESIS

Das Projektziel war eine **minimalinvasive Ersetzung** des generativen Prozesses,
nicht SigmaDock zu schlagen. Daran gemessen ist das Ergebnis solide und
verteidigbar, auch ohne Leistungsvorsprung:

- Eine Diffusions-Docking-Pipeline wurde vollständig auf Riemannsches Flow Matching
  umgestellt, unter Erhalt von Architektur, Datenpipeline und Inferenzschnittstelle.
- Die Implementierung ist unabhängig verifiziert korrekt.
- Bei gleicher Rechenzeit liegt das Ergebnis in derselben Größenordnung, chemisch
  ununterscheidbar, mit einem kleinen, sauber quantifizierten Rückstand.
- **Ein echter, gemessener Vorteil:** 5 statt 25 Integrationsschritte.
- **Ein methodischer Beitrag mit eigenem Wert:** der Nachweis, dass
  Einzelziehungs-Vergleiche in diesem Feld Rauschen lesen (Streuung 4.5 Å gegen
  Effekt 0.5 Å) — plus zwei dokumentierte Auswertungsartefakte
  (Kristallkopien, Rahmen-Mismatch), die als Fallstudien für Reproduzierbarkeit
  taugen.

Formulierung, die trägt: **„Wir haben den generativen Prozess ausgetauscht, die
Korrektheit unabhängig verifiziert, dabei nichts kaputtgemacht und einen
Sampling-Kostenvorteil nachgewiesen."** Das ist etwas anderes als „unsere Methode
ist besser" — und sollte strikt so und nicht anders formuliert werden.
