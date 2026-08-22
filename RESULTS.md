# SigmaFlow — Ergebnis-Log

Konsolidierte Übersicht aller quantitativen Modell-Ergebnisse (RMSD,
PoseBusters, Trainings-Loss) über die Projektlaufzeit, chronologisch. Die
ausführliche Diskussion/Herleitung jedes Ergebnisses steht in `STATUS.md`
(Spalte "Quelle" verweist auf den jeweiligen PAUSE-PUNKT); diese Datei ist
die schnelle "wie gut hat welches Setup abgeschnitten"-Referenz, ohne den
Debugging-Weg dorthin nachlesen zu müssen.

Alle Zahlen unten sind auf denselben 10 Dummy-Komplexen (`notebooks/dummy_data/`,
Überanpassungstest, siehe `CLAUDE.md`) gemessen, nicht auf dem vollen
Datensatz — es gibt bisher noch keinen großen Trainingslauf (siehe
`STATUS.md` PAUSE-PUNKT #13, "was fehlt vor dem großen Lauf").

## RMSD gegen die wahre Pose (Å, alle 10 Komplexe)

"raw" = direkter Koordinatenvergleich, "aligned" = nach optimaler starrer
Kabsch-Ausrichtung (isoliert Form-/Platzierungsfehler von reiner
Translations-/Rotations-Verschiebung).

| Datum | Setup | raw (Mittel / Median) | aligned (Mittel / Median) | Quelle |
|---|---|---|---|---|
| 2026-07-22 | SigmaFlow, Ad-hoc-Gewichte, **`edm`-Sampling-Bug** (Zeitplan lief rückwärts) | 10.21 / 10.37 | 7.54 / 7.76 | #9 |
| 2026-07-24 | SigmaFlow, Ad-hoc-Gewichte, `edm`-Bug **gefixt** | 6.04 / 5.72 | 3.09 / 3.06 | #10 |
| 2026-07-25 | **SigmaFlow, `edm`-Fix + Produktions-Hyperparameter** (`trans_score_weight=2.0, rot_score_weight=0.5`), 3h-Bestätigungslauf | **4.31 / 3.67** | **2.86 / 2.81** | #11 |
| 2026-07-26 | SigmaFlow, identischer Checkpoint wie oben, aber auf **V100 statt L40S** gesampelt | 4.32 / 3.70 | 2.87 / 2.82 | #13 |
| 2026-07-22 | SigmaDock (Original-Diffusion), eigene Ad-hoc-Gewichte | 6.90 / 6.85 | 2.85 / 2.90 | #9 |
| 2026-07-25 | SigmaDock (Original-Diffusion), **Produktions-Gewichte** | 5.08 / 4.76 | 2.59 / 2.60 | #11 |

**Einordnung:** SigmaFlow (Produktions-Gewichte) liegt bei der rohen
Abweichung leicht **vor** SigmaDocks eigenem Produktionslauf (4.31 vs. 5.08 Å),
bei den ausgerichteten Werten praktisch gleichauf (2.86 vs. 2.59 Å) — in
Summe gleichwertig, keine Methode dominiert.

**Cross-GPU-Check (2026-07-26, #13):** derselbe Checkpoint auf V100 vs. L40S
gesampelt weicht im Schnitt nur **0.04 Å** ab (max. 0.10 Å) — bestätigt, dass
das Ergebnis nicht von der GPU-Architektur abhängt, nur Fließkomma-Rauschen.

## PoseBusters (chemische Plausibilität, 28 Checks, `redock`-Konfiguration)

Verglichen: SigmaFlow (`edm`-Fix + Produktions-Gewichte) vs. SigmaDock
(Produktions-Gewichte), beide vom 2026-07-25/26-Lauf (#11).

- **23 von 28 Checks exakt identisch** zwischen beiden Methoden.
- Bei den 5 verbleibenden Unterschieden gewinnt **durchgehend SigmaDock**:

| Check | SigmaFlow | SigmaDock | Differenz |
|---|---|---|---|
| `double_bond_stereochemistry` | 0.9 | 1.0 | 1/10 Komplexe |
| `tetrahedral_chirality` | 0.8 | 0.9 | 1/10 Komplexe |
| `volume_overlap_with_protein` | 0.1 | 0.2 | 1/10 Komplexe |
| `volume_overlap_with_organic_cofactors` | 0.9 | 1.0 | 1/10 Komplexe |
| `internal_energy` | 0.0 | 0.2 | 2/10 Komplexe |

Vier der fünf Unterschiede betreffen nur 1/10 Komplexe (bei der bekannten
Auswertungs-Stochastizität kein verlässliches Signal). `internal_energy`
(2/10) ist der einzige mit doppeltem Gewicht — passt mechanistisch zum
leichten Rest-Unterschied bei der ausgerichteten RMSD oben.

**Gemeinsame Schwäche beider Methoden** (0 % Pass-Rate bei **beiden**,
architektur-bedingt, kein Methodenunterschied): `bond_lengths`,
`bond_angles`, `minimum_distance_to_protein` — jedes Fragment wird starr
unabhängig platziert, nichts erzwingt chemisch sinnvolle Bindungsgeometrie
genau an der Torsionsbindungs-Schnittstelle zwischen zwei Fragmenten. Geerbt
aus der ursprünglichen SigmaDock-Architektur, nicht Teil des
Flow-Matching-Konversionsauftrags.

## Übergangsbindungslängen, volles PoseBusters-Set (209 Komplexe, 5576 Bindungen)

Gestufter 6h→12h→24h-Vergleich auf dem echten `pdbbind-general`-Trainingssatz
(nicht die 10 Dummy-Komplexe oben), jeweils frisch von Null trainiert bei
24h. "Übergangsbindung" = Bindung mit `|vorhergesagte Länge − wahre Länge| >
0.02 Å`, typischerweise an der starren Fragment-Torsions-Schnittstelle.

| Datum | Lauf | n Übergangsbindungen | % aller Bindungen | mean | median | max | Quelle |
|---|---|---|---|---|---|---|---|
| 2026-08-0x | SigmaFlow, 12h | 728 | 13.1% | 2.68 Å | 2.23 Å | 15.73 Å | STATUS.md, "Vollständiges PoseBusters-Set" |
| 2026-08-08 | **SigmaFlow, 24h** | 733 | 13.1% | 2.41 Å | 1.93 Å | 14.22 Å | STATUS.md, "24h-Ergebnis ausgewertet" |
| 2026-08-09 | **SigmaFlow, rotdata (Variante b), 12h** | 733 | 13.1% | 2.71 Å | 2.24 Å | 15.25 Å | STATUS.md, "Job 8490675 ... ausgewertet" |
| 2026-08-09 | **SigmaFlow, anchordist (Variante c), 12h** | 730 | 13.1% | 2.69 Å | 2.26 Å | 15.68 Å | STATUS.md, "Variante C ausgewertet" |
| 2026-08-13 | SigmaDock, 12h **(korrigiert)** | 715 | **12.8%** | 1.22 Å | 1.07 Å | 18.08 Å | `bond_length_check.py`, Minimum über Kopien |
| ~~2026-08-0x~~ | ~~SigmaDock, 12h~~ | ~~830~~ | ~~14.9%~~ | ~~1.25 Å~~ | ~~1.22 Å~~ | ~~18.08 Å~~ | artefaktbehaftet, ersetzt |
| 2026-08-08 | **SigmaDock, 24h** ⚠️ | 813 | 14.6% | 1.40 Å | 1.40 Å | 5.33 Å | ⚠️ NICHT nachgerechnet, s.u. |

**⚠️ Einordnung KORRIGIERT 2026-08-13 — die Trade-off-Lesart ist gefallen.**

Sie lautete: „SigmaFlow hat durchgehend weniger betroffene Bindungen,
SigmaDock hat durchgehend die kleinere typische Abweichung." Die erste Hälfte
stimmt nicht mehr. Korrigiert hat **SigmaDock weniger betroffene Bindungen
(12.8% gegen 13.1%) UND die kleinere Abweichung (1.22 gegen 2.68 Å Mittel)**.
Es gibt keinen Trade-off; SigmaDock ist auf beiden Achsen besser.

⚠️ Die **24h-Zeilen sind NICHT nachgerechnet** — die zugehörigen
Vorhersageordner liegen nicht lokal vor. Sie stammen aus derselben
artefaktbehafteten Methodik und sind für die SigmaDock-Seite entsprechend
unzuverlässig; insbesondere die Aussagen über den schrumpfenden
Einzelausreißer (18.08 → 5.33 Å) und über `7T1D_E7K` / `6TW5_9M2` als
„komplex-spezifisches strukturelles Problem" sind hinfällig, bis sie
nachgerechnet sind. `6TW5_9M2` ist einer der 42 Phantom-Ausreißer
(3 Kopien, alt 68.9 Å gegen Kopie 0, korrigiert 4.29 Å).

**Aber:** dies ist nur ein Ausschnitt der chemischen Qualität — s. nächster
Abschnitt für das vollständigere Bild, das diese "Trade-off"-Lesart
relativiert.

## Volle chemische Plausibilität (PoseBusters, 20 liganden-intrinsische Checks, 209 Komplexe)

Anders als die Bindungslängen-Tabelle oben (die nur misst, WIE SCHLIMM eine
bereits abweichende Bindung ist) misst dies den ANTEIL der Komplexe, die
den jeweiligen PoseBusters-Check überhaupt bestehen — das direktere Maß für
"ist das generierte Molekül insgesamt realistisch". Protein-Kontext-Checks
(Volume-Overlap/Abstand zu Protein/Kofaktoren/Wasser, 8 von 28) fehlen
noch (Rezeptor-PDBs nicht lokal vorhanden); die 20 übrigen sind
liganden-intrinsisch und laufen ohne Protein. 6h nicht vertreten (nie auf
dem vollen Set gesampelt).

**Pass-Raten, nur Checks mit Unterschied zwischen Methode/Stufe gezeigt
(restliche ~9 Checks: 100% bei allen vier Läufen):**

| Check | SigmaFlow 12h | SF rotdata (Var. b) 12h | SF anchordist (Var. c) 12h | SigmaFlow 24h | SigmaDock 12h | SigmaDock 24h |
|---|---|---|---|---|---|---|
| `double_bond_stereochemistry` | 0.962 | 0.957 | 0.957 | 0.981 | 0.976 | 0.990 |
| `tetrahedral_chirality` | 0.756 | 0.751 | 0.751 | 0.785 | 0.856 | **0.914** |
| `bond_lengths` (strenges DG-Kriterium) | 0.120 | 0.124 | 0.124 | 0.124 | 0.148 | **0.258** |
| `bond_angles` | 0.062 | 0.057 | 0.067 | 0.081 | 0.043 | **0.187** |
| `internal_steric_clash` | **0.129** | 0.129 | 0.134 | 0.144 | 0.062 | **0.268** |
| `internal_energy` | 0.176 | 0.167 | 0.163 | 0.197 | 0.081 | **0.372** |
| `rmsd_≤_2å` | 0.000 | 0.000 | 0.000 | 0.014 | 0.000 | 0.043 |

**Einordnung (2026-08-08, revidiert gegenüber der reinen
Bindungslängen-Sicht oben):** bei 12h noch gemischt, bei 24h dreht sich das
fast vollständig zugunsten SigmaDock — SigmaDock verbessert sich zwischen
12h und 24h auf JEDEM dieser Checks deutlich stärker als SigmaFlow und
liegt bei 24h überall vorne. SigmaFlow stagniert zwischen 12h und 24h auf
fast jedem Check. Absolute Werte bleiben niedrig bei beiden (schwerer
Blind-Generalisierungstest, kurze Trainingszeit).

**Ungetestete Hypothese, warum:** SigmaDocks Loss hat zwei zusätzliche
direkte Daten-Raum-Terme (`T0`,`R0` — implizite saubere Struktur direkt
gegen wahre Struktur), SigmaFlow nur Geschwindigkeits-Raum-Terme. Fast
jeder Check, bei dem SigmaDock stärker zulegt, ist eine Eigenschaft der
finalen Koordinaten — genau das, was `T0`/`R0` direkt bestrafen und
SigmaFlow nur indirekt erreicht. **DREI unabhängig GETESTETE, VERWORFENE
Alternativ-Hypothesen** schließen die Lücke nachweislich NICHT: (1)
analoge Zeitgewichtung wie SigmaDocks `λ(s)` nachrüsten
(`SigmaFlow_Variants/a_time_weighting/`, Test 3), (2) ein rotations-seitiger
Daten-Raum-Term analog zu SigmaDocks `R0`, aber mit SigmaFlows eigenen
Bausteinen konstruiert (`SigmaFlow_Variants/b_rotation_data_space_loss/`),
(3) ein direkter Anker-Atom-Distanz-Loss, der GENAU die gemessene
Übergangsbindungslänge bestraft (`SigmaFlow_Variants/
c_anchor_atom_distance_loss/`, 2026-08-09). Alle drei Varianten-Spalten
oben liegen praktisch exakt auf der SigmaFlow-12h-Baseline.

**Wichtig zur Belastbarkeit dieser Nullergebnisse:** die drei
SigmaFlow-12h-Läufe (Baseline, b, c) liefern 728/733/730
Übergangsbindungen bzw. mean 2.68/2.71/2.69 Å — die Lauf-zu-Lauf-Streuung
liegt also bei ±3 Bindungen / ±0.03 Å, während SigmaDocks Vorsprung
(830 / 1.25 Å) weit außerhalb liegt. Die Metrik ist empfindlich genug für
einen echten Effekt; sie sieht schlicht keinen.

**Diese Hypothese wurde am 2026-08-09 GEPRÜFT und WIDERLEGT** (Korrelation
zwischen Platzierungsfehler und Bindungsfehler nur r=0.05-0.18) — dabei
ergab sich aber ein weit wichtigerer Befund, siehe nächster Abschnitt.

## 🚨 Fragment-Platzierungsgenauigkeit (2026-08-09) — beide Methoden bei 12h faktisch untrainiert

Gemessen mit `SigmaFlow_Variants/posebusters_full_comparison/
placement_vs_bond_error.py` (Fragmentierung aus den Vorhersagen
rekonstruiert, dann Kabsch-Ausrichtung pro Fragment; Methodik
selbst-validiert: Residual-RMSD nach starrer Ausrichtung 0.019-0.066 Å).

**SigmaDock-Spalte korrigiert 2026-08-13** (Minimum über Kristallkopien).
SigmaFlow unverändert.

| Metrik (12h-Checkpoints, 209 Komplexe) | SigmaFlow | SigmaDock | Zufalls-Baseline |
|---|---|---|---|
| Pro-Fragment-Schwerpunktfehler (mean/median) | 5.19 / 4.57 Å | 6.26 / 5.29 Å | — |
| **Pro-Fragment-Rotationsfehler (mean)** | **122.2°** | **116.4°** | **126.5°** |
| Gesamt-Molekül-RMSD (mean/median) | 5.58 / 5.14 Å | 6.40 / 5.59 Å | — |
| Anteil RMSD < 2 Å | 0.000 | 0.000 | — |

Frühere, artefaktbehaftete SigmaDock-Werte: Schwerpunktfehler 12.92 / 6.39 Å,
Rotationsfehler 118.7°, Gesamt-RMSD 13.57 / 6.55 Å.

**Die Kernaussage des Abschnitts bleibt:** beide Methoden liegen beim
Pro-Fragment-Rotationsfehler nahe der Zufallsgrenze von 126.5° und treffen
KEINEN einzigen Komplex unter 2 Å. Die Korrektur verschiebt SigmaDock von
„weit daneben" auf „ähnlich daneben wie SigmaFlow" — sie ändert nichts daran,
dass beide bei 12h faktisch untrainiert sind.

## LR-Schedule-Korrektur (2026-08-09): 6h korrigiert schlägt 12h kaputt

Alle früheren pdbbind-Läufe steckten wegen `--max_epochs 1000` dauerhaft in
der LR-Warmup-Phase (5-20% der Peak-LR, Annealing nie erreicht). Mit
`--max_epochs 3` (auf das 6h-Budget kalibriert) läuft der Schedule erstmals
vollständig durch.

| Lauf | Rotationsfehler | RMSD | Schwerpunktfehler |
|---|---|---|---|
| SigmaFlow 12h, kaputter LR | 122.2° | 5.58 Å | 5.19 Å |
| **SigmaFlow 6h, LR korrigiert** | **120.1°** | **5.09 Å** | **4.75 Å** |
| SigmaDock 12h, kaputter LR | 118.7° | 13.57 Å | 12.92 Å |
| **SigmaDock 6h, LR korrigiert** | **117.3°** | **12.19 Å** | **11.21 Å** |
| *Zufall* | *126.5°* | — | — |

**Rauschgrenze:** drei unabhängige 12h-Läufe mit identischer Konfiguration
(Baseline, Var. b, Var. c) streuen um nur 0.1° (σ=0.04°) — die
−2.1°-Verbesserung ist also real (~50× die Streuung), konsistent über beide
Methoden und beide Metriken, bei halber Rechenzeit. **Aber klein:** 120.1°
gegen 126.5° Zufall bleibt praktisch Zufall.

**Eigentlicher Blocker: Gradientenschritte, nicht Epochen.** Paper: 155.544
Schritte bei Batch 32 (4,98 Mio Samples). Unser 6h-Lauf: ~7.290 Schritte bei
Batch 8 (0,06 Mio Samples) — **~5%**. Der Dummy-3h-Lauf absolvierte 614
Epochen (mehr als die 256 des Papers) und blieb dennoch bei 102° — es ist
also nicht die Epochenzahl, sondern die absolute Schrittzahl. Details:
STATUS.md, "Eigentliche Ursache identifiziert".

### Übergangsbindungen unter identischen Bedingungen (LR-korrigiert, 6h)

| Lauf | n | % | mean | median | max |
|---|---|---|---|---|---|
| SigmaFlow 12h (kaputter LR) | 728 | 13.1% | 2.68 Å | 2.23 Å | 15.73 Å |
| SigmaDock 12h (kaputter LR) | 830 | 14.9% | 1.25 Å | 1.22 Å | 18.08 Å |
| **SigmaFlow 6h (LR korrigiert)** | 720 | 12.9% | 2.41 Å | 1.97 Å | 11.72 Å |
| **SigmaDock 6h (LR korrigiert)** | 779 | 14.0% | 1.40 Å | 1.42 Å | 4.40 Å |

Der Trade-off bleibt bestehen (SigmaFlow weniger, aber größere Abweichungen),
der Faktor schrumpft leicht von 2,1 auf 1,7.

### 🔴 ROOT CAUSE (2026-08-09): Rahmen-Mismatch in der Rotationsvorhersage

**Bestätigter Implementierungsfehler:** SigmaFlows Rotations-Loss vergleicht
eine Netzwerkausgabe im **Welt-Rahmen** (`omega = hat(dW_world)`,
`newton_maruyama`) mit einem Ziel im **Körper-Rahmen**
(`log(R_tᵀ R_1)/(1-t)`, "right trivialised"). Der Adjoint-Transport
`u_body = R_tᵀ ω_world R_t` fehlt. `euler_step` multipliziert rechts
(Körper), SigmaDock links (Welt, konsistent). Fix: eine Zeile in
`_compute_vector_field`. Vollständige Belege und Ausschlussliste:
STATUS.md, "ROOT CAUSE GEFUNDEN".

**Messsignatur, die exakt dazu passt** (identische Fragmentierung, 111
Komplexe, 397 Bindungen):

| Größe | SigmaFlow | SigmaDock |
|---|---|---|
| Schwerpunkt-Paare, rel. Verschiebung | 4.44 Å | 4.47 Å (**identisch** → Translation intakt) |
| Anker-Paare, rel. Verschiebung | **2.91 Å** | **1.82 Å** (Rotation defekt) |
| rel. Rotationsfehler, **benachbarte** Fragmente | **121.8°** | **102.9°** |
| rel. Rotationsfehler, nicht benachbarte (Kontrolle) | 124.9° | 126.6° |
| *Zufall* | *126.5°* | *126.5°* |

SigmaDock zeigt einen klaren Lokalitätseffekt (benachbart 24° besser als
Zufall, nicht benachbart exakt zufällig); SigmaFlow praktisch keinen (3°).
**Konstant über 6h/12h/24h** → struktureller Fehler, kein Trainingsdefizit.

### ✅ ROOT CAUSE BESTÄTIGT (2026-08-10): Frame-Fix erzeugt Lokalität

Der oben beschriebene Einzeiler (`pred_u_t_R = R_tᵀ ω_world R_t` in
`_compute_vector_field`) wurde implementiert und unter compute-gematchten
Bedingungen getestet: 6h, `max_epochs=3`, identische Hyperparameter,
**identische Schrittzahl** (7050) wie die Kontrolle, volles 209er-Set.

| Lauf | gebunden | ungebunden | Lücke (gepaart/Komplex) | 95%-CI | Anteil <0 |
|---|---|---|---|---|---|
| SigmaFlow Kontrolle (8512798) | 122.8° | 124.9° | **+1.1°** | [−4.2, +6.6] | 53% |
| **SigmaFlow Frame-Fix (8530243)** | **113.5°** | 126.0° | **−12.8°** | [−18.3, −7.2] | 68% |
| SigmaDock (8512922) | 105.5° | 126.0° | −22.0° | [−27.3, −16.8] | 81% |
| *Zufall* | *126.5°* | *126.5°* | *0°* | | |

- Kontroll-CI enthält die Null, Frame-Fix-CI nicht, **kein Überlapp** der
  beiden → der Effekt ist gesichert. Rund **60%** des Abstands zur
  SigmaDock-Referenz geschlossen.
- Der **ungebundene** Wert bleibt auf der Zufallsgrenze, nur der gebundene
  fällt — die Signatur eines echten Lokalitätseffekts, nicht bloß besserer
  globaler Orientierung.
- **SigmaDock bleibt tendenziell besser, aber nicht gesichert:** CIs
  überlappen knapp (−18.3 vs. −16.8).
- **Weiterhin keine brauchbaren Posen** (~5% der Paper-Gradientenschritte,
  gebundene Paare mit 113.5° nahe am Zufall). Diagnostisch, nicht kompetitiv.

Reproduzierbar mit
`SigmaFlow_Variants/posebusters_full_comparison/fragment_locality.py`
(Bootstrap über Komplexe, nicht über Fragmentpaare — Paare innerhalb eines
Moleküls sind nicht unabhängig). **Achtung:** dieses Skript weicht leicht von
den historischen Inline-Zahlen der Tabelle darüber ab (Kontrolle 121.8° vs.
122.8°, SigmaDock 102.9° vs. 105.5°; ungebunden identisch), vermutlich durch
die Behandlung kleiner Fragmente. Historische und neue Einzelzahlen **nicht
mischen** — die Tabelle hier ist intern konsistent, alle drei Läufe gingen
durch dasselbe Skript.

### Loss-Varianten auf dem korrigierten Code (2026-08-11)

Alle drei Varianten wurden auf dem frame-korrigierten Stand neu gefahren
(6h, `max_epochs=3`, Hyperparameter identisch zur Kontrolle **8530243**).

| Lauf | gebunden | ungebunden | Lücke (gepaart) | 95%-CI |
|---|---|---|---|---|
| Frame-Fix allein (8530243, Kontrolle) | 113.5° | 126.0° | −12.8° | [−18.3, −7.2] |
| + Variante b, rot-data-space (8534746) | 113.6° | 126.5° | **−12.8°** | [−18.9, −6.8] |
| + Variante c, anchor-dist (8534747) | 113.5° | 126.0° | −12.8° | *No-Op, s.u.* |

**Variante b: echtes Nullergebnis.** Sie hat nachweislich anders trainiert
(alle 209 Vorhersagen unterscheiden sich byte-weise von der Kontrolle), die
Lokalitätslücke bleibt aber unverändert und die CIs überlappen fast
vollständig. Der Rotations-Datenraum-Term bringt nichts — diesmal belastbar,
weil auf korrekt gerahmter Basis gemessen.

### 🔴 Variante c ist ein NO-OP — und war es immer

**Befund:** die 209 Vorhersagen von 8534747 sind **byte-identisch** mit denen
der Kontrolle 8530243. Bei gleichem Seed, gleichen Daten und gleicher
Schrittzahl heißt das: der Zusatzterm hat den Gradienten in keinem einzigen
Schritt verändert.

**Ursache:** `_compute_anchor_distance_loss` baut seinen Fehler aus
`pos_1_hat`, das über `_apply_transformations` →
`get_transformations_from_rototranslations` entsteht — und diese Methode ist
**`@torch.no_grad()`** dekoriert (`sigma_flow_generator.py`, Zeile 275).
`pos_1_hat` ist damit von Autograd abgeschnitten. Der Loss wird berechnet, zu
`loss_trans` addiert und sogar geloggt, ist aber bezüglich der Parameter eine
Konstante. Ein stiller Fehler der unangenehmsten Sorte: kein Crash, kein NaN,
ein plausibel aussehender Loss-Wert im Log — und null Wirkung.

**Rückwirkende Konsequenz:** die `no_grad`-Dekoration steht unverändert im
Basiscode, der Mechanismus galt also auch im 12h-Lauf **8505487**. Das dort
notierte „kein Effekt" ist **keine Aussage über die Idee** und darf nicht
länger als drittes Nullergebnis gezählt werden. Damit stehen aktuell nur
**zwei** belastbare Nullergebnisse (a offen, b bestätigt).

**Kontrast, der die Diagnose stützt:** Variante b berechnet ihr `R_1_hat`
direkt in `compute_losses`, ohne den `no_grad`-Helfer — und unterscheidet
sich prompt in allen 209 Vorhersagen.

**Merksatz fürs Projekt:** ein neuer Loss-Term muss vor dem Absenden eines
6h-Laufs einmal daraufhin geprüft werden, ob er überhaupt einen
Gradientenpfad hat (`loss.requires_grad`, oder ein Schritt auf einem
Mini-Batch mit anschließendem `.grad`-Vergleich). Byte-Vergleich der
Vorhersagen gegen die Kontrolle ist der billigste nachträgliche Test.

## 📊 Vollständiger Metrikvergleich nach dem Frame-Fix (2026-08-11)

Reproduzierbar mit `SigmaFlow_Variants/posebusters_full_comparison/
full_metrics.py`. **Alle Zahlen dieses Abschnitts stammen aus EINEM Skript
mit EINER Methodik** — nicht mit historischen Einzelwerten weiter oben
mischen.

**Methodische Grundlage:** die Fragmentierung wird aus den Vorhersagen
rekonstruiert und fällt zwischen Läufen unterschiedlich aus. RMSD hängt nicht
davon ab (alle 209 Komplexe); alle Fragment- und Bindungsmetriken schon —
sie stehen deshalb nur auf den **94 Komplexen, bei denen alle drei Läufe
dieselbe Fragmentierung rekonstruieren**. Genau diese Einschränkung fehlte
den älteren Bindungslängenzahlen.

### A. RMSD gegen die Kristallpose (alle 209, index-basiert, nicht symmetriekorrigiert)

| | roh | nach Ausrichtung | <2 Å | <5 Å |
|---|---|---|---|---|
| SigmaFlow ohne Fix (8512798) | 5.09 / 4.70 | 2.47 / 2.44 | 1.0% | 54.1% |
| **SigmaFlow Frame-Fix (8530243)** | 5.43 / 5.10 | **2.17 / 2.07** | 1.0% | 48.8% |
| SigmaDock (8512922) | 12.19 / 5.32 | 2.09 / 1.96 | 3.8% | 45.9% |

Der Fix verbessert die **innere** Geometrie (ausgerichteter RMSD 2.47→2.17,
fast auf SigmaDock-Niveau) und lässt die **globale** Platzierung unverändert
bis leicht schlechter. Erwartete Signatur.

### ❌ WIDERRUFEN (2026-08-13): „Zwei verschiedene Fehlermodi — SigmaDocks Ausreißer"

Dieser Abschnitt behauptete, SigmaDock verlege ~20% der Liganden komplett aus
dem Bindungsbereich (42 Komplexe jenseits 20 Å, bis 80 Å), SigmaFlow nie.
**Das war ein Auswertungsartefakt. Die 42 Ausreißer haben nie existiert.**

Ursache: 84 der 209 `<complex>_ligands.sdf` enthalten MEHRERE
kristallographische Kopien desselben Liganden (bis zu 6, alle mit identischer
Topologie — verifiziert). Alle Auswertungsskripte nahmen stillschweigend die
erste Kopie. Der SigmaDock-Vergleichslauf las per
`experiments.sdf_regex=".*ligands.sdf$"` die Plural-Datei und lieferte für 44
dieser Komplexe eine Pose, die zu einer ANDEREN Kopie gehört.

Belege, jeder einzeln gemessen:
- Alle 42 Ausreißer stammen aus Mehrkopie-Dateien, keiner aus einer
  Einkopie-Datei.
- Bei 42 von 42 liegt die beste Übereinstimmung auf einer anderen Kopie als 0.
- Gegen die richtige Kopie gemessen: **Median 3.44 Å** statt 45–80 Å, 29 der
  42 unter 5 Å.
- Fehlerzerlegung: 99% des rohen Fehlers ist reine Verschiebung
  (Schwerpunktabstand Median 36.5 Å), RMSD nach Ausrichtung **1.71 Å** — also
  BESSER als die 1.91 Å der unauffälligen Komplexe. Gute Posen am falschen Ort.

Korrigierte Zahlen (`ligand_reference.py`, Minimum über Kopien):

| | >10 Å | >20 Å | >50 Å | max |
|---|---|---|---|---|
| SigmaFlow Frame-Fix 12h | 8 | 0 | 0 | 13.8 Å |
| SigmaDock 12h | **0** | **0** | **0** | **9.8 Å** |

**Beide Methoden haben null Ausreißer.** Der über 10 Seeds bestätigte Befund:
0 von 2090 Posen jenseits 20 Å, auf beiden Seiten. Das Argument „SigmaFlow ist
robuster" ist ersatzlos gestrichen.

Nicht überkorrigieren: der alte SigmaDock-Lauf war **nicht kaputt**. Er wurde
falsch gemessen. Das Modell lieferte durchgehend vernünftige Posen.

### C. Übergangsbindungen (94 Komplexe, 271 Schnittbindungen, wahre Länge 1.49 Å)

| | vorhergesagte Länge | absoluter Fehler |
|---|---|---|
| SigmaFlow ohne Fix | 2.49 Å | **1.20 Å** |
| **SigmaFlow Frame-Fix** | 1.62 Å | **0.36 Å** |
| SigmaDock | 1.36 Å | 0.37 Å |

**Die Erzählung „SigmaFlow baut systematisch zu lange Übergangsbindungen" ist
damit erledigt.** Sie war kein Merkmal von Flow Matching, sondern eine Folge
des Rahmenfehlers. Nach dem Fix ist der Fehler gleichauf mit SigmaDock
(0.36 vs. 0.37 Å); SigmaFlow liegt leicht zu lang, SigmaDock leicht zu kurz.

### B/D. Fragment-Orientierung (94 Komplexe)

| | absolut | rel. gebunden | rel. ungebunden | Lücke |
|---|---|---|---|---|
| SigmaFlow ohne Fix | 118.2° | 121.5° | 126.9° | −3.7° [−12.3, +5.1] |
| **SigmaFlow Frame-Fix** | 128.1° | 115.3° | 126.4° | **−14.2° [−22.4, −5.9]** |
| SigmaDock | 115.4° | 103.8° | 131.2° | −26.3° [−34.2, −18.4] |

**Nicht glätten:** der ABSOLUTE Rotationsfehler wurde nach dem Fix schlechter
(118.2°→128.1°, über der Zufallsgrenze 126.5°), während die RELATIVE
Orientierung gebundener Fragmente klar besser wurde. In sich stimmig — die
relative Orientierung ist innermolekular, die absolute erfordert Kenntnis der
Bindetasche. Aussage: das Modell hat innere Geometrie gelernt, Verankerung im
Protein nicht. Konsistent mit unveränderten 1.0% unter 2 Å.

## 📊 12h-Stufe: sauberer compute-gematchter Vergleich (2026-08-12)

Erstmals ein 12h-Paar unter korrigiertem LR-Schedule auf BEIDEN Seiten
(SigmaFlow 8541310, SigmaDock 8541439, je `max_epochs=6`, beide
`COMPLETED` vor dem Zeitlimit, 11:06 bzw. 10:52). SigmaFlow endete bei
global_step 13.750.

### Lokalitätslücke, volles Set (n≈133 Komplexe, fragment_locality.py)

| Lauf | Lücke | 95%-CI | Anteil <0 |
|---|---|---|---|
| SigmaFlow 6h ohne Fix (8512798) | +1.1° | [−4.2, +6.6] | 53% |
| SigmaFlow 6h Frame-Fix (8530243) | −12.8° | [−18.3, −7.2] | 68% |
| **SigmaFlow 12h Frame-Fix (8541310)** | **−15.0°** | [−21.6, −8.1] | 69% |
| SigmaDock 6h (8512922) | −22.0° | [−27.3, −16.8] | 81% |
| **SigmaDock 12h (8541439)** | **−20.4°** | [−26.0, −14.5] | 71% |
| + Variante a, time-weight (8540758) | −12.8° | [−19.0, −6.6] | 64% |
| + Variante b, rot-data-space (8534746) | −12.8° | [−18.9, −6.8] | 65% |
| + Variante c, anchor-dist (8534747) | −12.8° | *No-Op* | 68% |

### RMSD, alle 209 Komplexe (full_metrics.py) — KORRIGIERT 2026-08-13

Die SigmaDock-Zeilen sind gegenüber der ursprünglichen Fassung korrigiert
(Minimum über Kristallkopien, s.o.). SigmaFlow ist unverändert — es war nie
betroffen, weil es die Singular-Datei mit genau einer Kopie liest.

| | roh | ausgerichtet | <2 Å | <5 Å |
|---|---|---|---|---|
| SigmaFlow 6h | 5.43 / 5.10 | 2.17 / 2.07 | 1.0% | 48.8% |
| **SigmaFlow 12h** | 5.38 / 4.78 | **2.03 / 2.00** | **1.9%** | 52.2% |
| SigmaDock 6h | 4.80 / 4.51 | 2.08 / 1.94 | 4.8% | 57.9% |
| **SigmaDock 12h** | 4.62 / 4.23 | **1.92 / 1.80** | **6.7%** | 60.3% |

Frühere, artefaktbehaftete Werte zum Nachvollziehen: SigmaDock 6h
12.19 / 5.32, 3.8%, 45.9%; SigmaDock 12h 12.11 / 5.35, 4.8%, 45.5%.

### Befunde

1. **Doppelte Rechenzeit hilft beiden, aber wenig.** <2Å: SigmaFlow
   1.0→1.9%, SigmaDock 4.8→6.7%. Ausgerichteter RMSD je ~0.1 Å besser.
   Auf die 79.9% des Papers wäre das eine Extrapolation über zwei
   Größenordnungen — die Kurve trägt das nicht.
2. **SigmaFlow schließt auf, SigmaDock stagniert:** −12.8→−15.0 gegen
   −22.0→−18.6 (letzteres flach, Bewegung ist Rauschen). **Vorbehalt:** die
   CIs überlappen stark, das ist eine Tendenz, kein Nachweis.
3. ~~**SigmaDocks Ausreißerproblem ist strukturell.**~~ **GESTRICHEN
   2026-08-13.** Es gab kein Ausreißerproblem; der rohe Mittelwert von 12.11
   war das Kopien-Artefakt. Korrigiert liegt er bei 4.62 — SigmaDock hat
   **null** Komplexe jenseits 20 Å, ebenso wie SigmaFlow.
4. **Alle drei Loss-Varianten sind erledigt.** a und b liefern exakt die
   Kontrollzahl −12.8°; bei a ist das ein ECHTES Nullergebnis (Vorhersagen
   unterscheiden sich in allen 209 Dateien von der Kontrolle), bei c ein
   No-Op ohne Gradientenpfad. Keine weitere Loss-Variante testen, bevor der
   Engpass verstanden ist.

### ⚠️ Subset-Vorbehalt (gilt für ALLE Fragment- und Bindungsmetriken)

Der Matched-Subset schrumpft mit jedem verglichenen Lauf: 3 Läufe → 94
Komplexe, 4 Läufe → **57** (n=25 für die Lücke). Werte aus verschiedenen
Subsets sind NICHT vergleichbar. Konkrete Fälle, die sonst als Fortschritt
fehlgedeutet würden:
- Übergangsbindungsfehler SigmaFlow 6h: **0.36 Å** auf dem 94er-Subset,
  **0.40 Å** auf dem 57er-Subset — derselbe Lauf.
- Absoluter Rotationsfehler SigmaFlow 6h: **128.1°** (94er) vs. **126.1°**
  (57er) — derselbe Lauf.
Auf dem 57er-Subset: Bindungsfehler SigmaFlow 12h 0.28 Å vs. SigmaDock 12h
0.31 Å; absoluter Rotationsfehler SigmaFlow 126.1→123.6°, SigmaDock
115.8→118.5°. **Zu schwach für die Aussage "SigmaFlow lernt jetzt absolute
Orientierung"** — es bleibt an der Zufallsgrenze.
Die Lokalitätslücke ist von alldem NICHT betroffen: sie wird innerhalb jedes
Laufs berechnet und steht auf n≈133.

### Chemische Plausibilität + symmetriekorrigierter RMSD (2026-08-12)

PoseBusters 0.6.5, `redock_noprotein.yml`, alle 209 Komplexe, 12h-Läufe.
Reproduzierbar: `run_posebusters_ligandonly.py <pred_dir> <csv>`.

| Check (Anteil bestanden) | SigmaFlow 12h | SigmaDock 12h |
|---|---|---|
| Bindungslängen | **55.5%** | 48.8% |
| Bindungswinkel | **48.3%** | 41.6% |
| interne sterische Clashes | **47.8%** | 46.4% |
| tetraedrische Chiralität | **93.8%** | 90.9% |
| interne Energie | 57.4% | **59.8%** |
| **alle 15 ligandinternen Checks** | **37.3%** | 32.5% |

Zehn weitere Checks bestehen beide zu 100%. SigmaFlow liegt in vier von fünf
unterscheidenden Checks vorn — konsistent zum kleineren
Übergangsbindungsfehler und zur Wirkung des Frame-Fix auf die INNERE
Geometrie. **Einordnung:** 37% gegen 32% heißt, dass bei BEIDEN Methoden rund
zwei Drittel der Moleküle chemisch fehlerhaft sind.

**⚠️ Korrektur unserer RMSD-Zahlen:** PoseBusters rechnet
**symmetriekorrigiert**, `full_metrics.py` indexbasiert.

| <2 Å | indexbasiert (unser) | symmetriekorrigiert |
|---|---|---|
| SigmaFlow 12h | 1.9% | **2.9%** |
| SigmaDock 12h | 4.8% | **8.1%** |

Beide steigen, SigmaDocks Vorsprung wächst von Faktor 2.5 auf 2.8. Für
Vergleiche ZWISCHEN unseren Läufen bleiben beide Maße gültig (identisch
angewandt); für den Vergleich mit publizierten Zahlen nur das
symmetriekorrigierte.

### 🚨 Signifikanztest: KEINER der Plausibilitäts-Vorsprünge ist gesichert

Gepaarter exakter McNemar-Test über dieselben 209 Komplexe
(`validity_significance.py`). Aussagekräftig sind nur die **diskordanten**
Komplexe — die, bei denen sich die Methoden unterscheiden:

| Check | nur SigmaFlow | nur SigmaDock | p | Urteil |
|---|---|---|---|---|
| Bindungslängen | 37 | 23 | 0.093 | n.s. |
| Bindungswinkel | 34 | 20 | 0.076 | n.s. |
| sterische Clashes | 27 | 24 | 0.780 | n.s. |
| interne Energie | 29 | 34 | 0.615 | n.s. |
| Chiralität | 14 | 8 | 0.286 | n.s. |
| **alle 15 Checks** | 28 | 18 | 0.184 | n.s. |
| **RMSD ≤ 2 Å** | 6 | **17** | **0.035** | **SigmaDock besser** |

**Die chemische Plausibilität ist zwischen den Methoden NICHT
UNTERSCHEIDBAR.** Der 6.7-Punkte-Vorsprung bei Bindungslängen entspricht
37 gegen 23 diskordanten Komplexen — das produziert eine faire Münze in 60
Würfen ohne Weiteres. Die gleichgerichtete Tendenz bei vier von fünf Checks
(zwei davon bei p≈0.08) ist ein Hinweis auf einen kleinen echten Effekt,
kein Nachweis.

**Der einzige statistisch gesicherte Unterschied im gesamten Vergleich ist
die Erfolgsquote — und die geht an SigmaDock.**

Vorbehalt, den kein Test behebt: ein Trainingslauf, ein Sampling-Seed pro
Methode. Der Test deckt die Streuung zwischen Komplexen ab, nicht die
zwischen Läufen.

**✅ Gegen das Kopien-Artefakt geprüft (2026-08-13):** Diese Tabelle ist
NICHT betroffen. `run_posebusters_ligandonly.py` wurde auf die
nächstgelegene Kopie umgestellt und neu gerechnet — bei SigmaDock änderte
sich die Referenz für 44 Komplexe, bei SigmaFlow für keinen. Das `<2 Å`-
Urteil kippte bei **0 von 44**. PoseBusters rechnet symmetriekorrigiert und
behandelt die Kopien bereits richtig; betroffen war ausschließlich unsere
eigene indexbasierte Auswertung. Alle Zahlen oben bleiben unverändert
gültig, `p=0.0347` eingeschlossen.

## 🧩 Fragmentzahl als Schwierigkeitsachse (2026-08-19, CPU)

Verknüpfung von `arc/count_fragments.py` (208 Liganden mit Fragmentzahl) mit
den Per-Komplex-RMSDs der 10-Seed-Läufe. Erfolg = Oracle@10, RMSD < 2 Å.

| Fragmente | n | SigmaFlow | SigmaDock | SF med. bester RMSD | SD med. bester RMSD |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 60.0 % | **100.0 %** | 1.24 Å | 1.12 Å |
| 2 | 34 | 64.7 % | 79.4 % | 1.71 Å | 1.45 Å |
| 3 | 41 | 48.8 % | 68.3 % | 2.02 Å | 1.72 Å |
| 4 | 29 | 17.2 % | 44.8 % | 2.68 Å | 2.22 Å |
| 5 | 42 | 21.4 % | 40.5 % | 2.69 Å | 2.05 Å |
| 6 | 18 | 22.2 % | 33.3 % | 2.68 Å | 2.44 Å |
| 7 | 16 | **0.0 %** | 12.5 % | 3.40 Å | 3.38 Å |
| 8 | 15 | 0.0 % | 6.7 % | 4.01 Å | 3.29 Å |
| 9 | 5 | 0.0 % | 0.0 % | 5.06 Å | 3.89 Å |
| 10 | 1 | — | — | 2.97 Å | 3.71 Å |
| 11 | 2 | — | — | 4.08 Å | 3.70 Å |

### 1. Die Fragmentzahl ist die dominante Schwierigkeitsachse

Logistische Regression `Erfolg ~ Arm + F + Arm:F` über beide Arme:
**jedes zusätzliche Fragment multipliziert die Erfolgs-Odds mit 0.53**
(`coef = −0.636`, `p < 1e-4`). Spearman gegen den besten RMSD:
ρ = +0.570 (SigmaFlow, `p = 2.6e-19`) und +0.622 (SigmaDock, `p = 1.1e-23`).

Geschichtet: `F ≤ 3` gegen `F ≥ 4` bricht SigmaFlow von 56.2 % auf 14.1 %
(Fisher `p = 1.8e-10`), SigmaDock von 75.0 % auf 30.5 % (`p = 4.1e-10`).

Ab **7 Fragmenten** trifft SigmaFlow keinen einzigen Komplex mehr (0/39 für
F ≥ 7), SigmaDock noch 3/39.

### 2. SigmaFlow ist GLEICHMÄSSIG schlechter, nicht überproportional

Der Interaktionsterm `Arm:F` ist **nicht signifikant** (`coef = −0.030`,
**`p = 0.85`**). Das Odds-Verhältnis SigmaFlow/SigmaDock ist über die
Schichten stabil: 0.429 bei `F ≤ 3`, 0.373 bei `F ≥ 4`.

**Das ist die zulässige Formulierung:** SigmaFlow verliert einen konstanten
Faktor von rund 0.43 auf der Odds-Skala, unabhängig von der Fragmentzahl. Die
auf der Prozentskala scheinbar wachsende Lücke ist überwiegend ein
Bodeneffekt. Die Hypothese „die Rotationsbehandlung skaliert schlechter mit
der Zahl starrer Körper" wird von diesen Daten **nicht** gestützt.

### 3. 🚨 Korrektur: die Schnittmenge ist NICHT leer

Gepaart über 208 Liganden, Oracle@10:

| | Anzahl |
|---|---:|
| **beide lösen** | **48** |
| nur SigmaFlow | 15 |
| nur SigmaDock | 51 |
| keiner | 94 |

Die bisher dokumentierte Aussage „die Schnittmenge der gelösten Komplexe ist
leer — beide Modelle lösen disjunkte Fälle" stammt aus **Einzelziehungen**
(SigmaFlow 6/209, SigmaDock 17/209). Bei einem Ziehungsrauschen, das laut
Seed-Varianz-Analyse etwa zehnmal so groß ist wie der Methodeneffekt, war das
ein Kleinstichprobenartefakt. Mit zehn Seeds überlappen **48 Komplexe**.

McNemar über die diskordanten Paare (15 gegen 51): `p = 1.0e-05`. SigmaDock
ist signifikant besser — aber auf demselben Komplexsatz, nicht auf einem
anderen.

**Diese Korrektur muss in den Thesis-Entwurf.**

### 4. Der einfachste Fall wird nicht zuverlässig gelöst

Fünf Liganden haben **genau ein Fragment**, also keine Torsion. Für sie ist
die gesamte Generierung eine einzige globale Rototranslation, ohne jede
Fragmentkoordination. SigmaDock löst 5/5, **SigmaFlow nur 3/5**.

Bei n = 5 ist das statistisch nichts. Als Diagnosehinweis ist es trotzdem
wertvoll: wenn SigmaFlow schon einen einzelnen starren Körper nicht sicher
platziert, liegt der Rückstand nicht an der Koordination mehrerer Fragmente,
sondern an Rotation und Translation selbst. Diese fünf Komplexe sind die
saubersten Testfälle für die 72h-Snapshots.

Abbildungen: `A_fragment_distribution`, `B_fragments_vs_size`,
`C_fragments_vs_performance` (PNG und PDF), erzeugt mit
`visualization/plot_fragments.py`. Rohdaten: `fragments_vs_performance.csv`.


## 🎯 Oracle@K auf den 10-Seed-Läufen (2026-08-19, CPU)

`SigmaFlow_Evaluation.evaluate_run` über die bereits vorhandenen Sampling-
Ausgaben der Jobs 8554147 / 8554149. Kein neues Sampling, keine GPU.
**2090/2090 Posen gewertet, TFD-Abdeckung 100 %, keine einzige Pose verworfen.**

| | SigmaFlow 12h | SigmaDock 12h |
|---|---:|---:|
| RMSD Median, alle 2090 Posen | 4.901 Å | 4.381 Å |
| Einzelzug (Seed 0) < 2 Å | 2.4 % | 10.0 % |
| Einzelzug (Seed 0) < 5 Å | 53.1 % | 59.8 % |
| TFD Median | 0.2344 | 0.2257 |
| **Oracle@1** | **4.4 %** | **10.9 %** |
| **Oracle@5** | **19.1 %** | **34.0 %** |
| **Oracle@10** | **30.1 %** | **47.4 %** |
| Oracle@10 Median-RMSD | 2.55 Å | 2.05 Å |

**Konsistenzprüfung:** Oracle@1 mittelt über alle Einzelziehungen und trifft
damit exakt die frühere Seed-Varianz-Analyse vom 2026-08-13 (SigmaFlow 4.4 %,
SigmaDock 9.8 % gegen hier 10.9 %). Der Einzelzug ist Seed 0 und liegt in
beiden Armen innerhalb der dort dokumentierten Spannweiten (SF 1.4–6.7 %,
SD 5.7–13.4 %). Zwei unabhängige Codepfade, dasselbe Ergebnis.

### Was daran neu ist

**Ranking ist der größte einzelne Hebel, den wir bisher gemessen haben.**
Von Seed 0 zu Oracle@10 gewinnt SigmaFlow den Faktor **12.5**, SigmaDock den
Faktor **4.7**. Mit einem perfekten Ranker und nur zehn Seeds erreicht unsere
SigmaDock-Reproduktion **47.4 %**.

Das ordnet den Abstand zum Paper neu ein. Die 79.9 % dort stehen für Top-1 aus
**40** Seeds **mit** Ranking auf **308** Komplexen bei **384 GPU-h**. Unsere
10.0 % waren ein Einzelzug ohne Ranking bei 12 GPU-h. Der Vergleich
„10 % gegen 79.9 %" misst also überwiegend Seeds und Ranking, nicht
Generatorqualität.

**SigmaFlow ist stärker seedabhängig.** Der Faktor 12.5 gegen 4.7 heißt: die
typische SigmaFlow-Pose ist schlechter, aber unter zehn Ziehungen ist
überdurchschnittlich oft eine gute dabei. Das Verhältnis SF/SD steigt von 0.24
(Einzelzug) über 0.56 (Oracle@5) auf 0.64 (Oracle@10).

**Beide liegen dicht an der 2-Å-Schwelle.** Oracle@10-Median 2.55 Å und
2.05 Å. Eine Verbesserung des Generators um wenige Zehntel Ångström würde sich
deshalb überproportional in der Erfolgsquote niederschlagen — was die
72h-Läufe zu einem empfindlichen Test macht.

### Offene Anschlussfrage

Die Lücke zwischen Oracle@K und Top-1@K ist genau das, was ein
Confidence-Modell zurückholen könnte (EXP-105, nicht gebaut). Nachdem der
Quellverteilungsstrang am 2026-08-19 als Negativergebnis geschlossen wurde,
ist das die Richtung mit der bislang größten gemessenen Hebelwirkung.
**Nicht vor den 72h-Läufen anfangen.**

Ebenfalls offen: die frühere Aussage „Schnittmenge der gelösten Komplexe ist
leer" stammt aus **Einzelziehungen**. Mit 10 Seeds je Arm ist sie neu zu
prüfen — die Daten dafür liegen in den beiden `eval_*_10seeds.json`.


## 📊 Seed-Varianz: 10 unabhängige Sampling-Seeds (2026-08-13)

Jobs 8554147 (SigmaFlow) und 8554149 (SigmaDock), je 10 Seeds × 209
Komplexe = 2090 Posen pro Methode, 25 Schritte, `seed_variance.py`.
Zweck: erstmals messen, wie breit die Verteilung ist, aus der jeder bisherige
Einzelvergleich EINE Ziehung genommen hat.

### 🚨 Kernbefund: das Ziehungsrauschen ist ~10× so groß wie der Methodeneffekt

| | SigmaFlow | SigmaDock |
|---|---|---|
| Median-RMSD je Seed | 4.56–5.70 Å | 4.15–4.87 Å |
| Anteil <2 Å je Seed | **1.4–6.7%** | **5.7–13.4%** |
| Mittel <2 Å über 10 Seeds | 4.4% | 9.8% |
| SD desselben Komplexes über Seeds | 1.55 Å | 1.58 Å |
| Spannweite desselben Komplexes | **4.46 Å** | **4.59 Å** |
| Ausreißer >20 Å | 0 | 0 |

Derselbe Komplex bewegt sich zwischen zwei Seeds im Median um 4.5 Å; der
gepaarte Methodenunterschied beträgt 0.47 Å. **SigmaFlows bester Seed (6.7%)
liegt über SigmaDocks schlechtestem (5.7%)** — mit Seed-Glück hätte man in
beide Richtungen „gewinnen" können. Rückwirkend heißt das: jeder frühere
Einzelziehungs-Vergleich dieses Projekts las überwiegend Rauschen.

### Der Methodenunterschied selbst ist real, aber klein

Erst über 10 Seeds pro Komplex mitteln, dann gepaart vergleichen:

| Größe | Wert |
|---|---|
| Mittlere Differenz (SigmaFlow − SigmaDock) | **+0.47 Å** |
| Bootstrap-CI 95% | **[+0.36, +0.59]** — schließt 0 aus |
| Median der Differenz | +0.39 Å |
| SigmaFlow besser bei | 29% der Komplexe |

Das ist die erste methodisch saubere Aussage des Projekts: **SigmaDock ist
gesichert besser, um einen kleinen Betrag.**

Konsistenzcheck geht auf: die Seed-zu-Seed-Streuung der Aggregate entspricht
reiner Binomialstreuung (n=209; p=0.044 → erwartet 1.42, beobachtet 1.75
Prozentpunkte; p=0.098 → erwartet 2.05, beobachtet 2.17). Kein unerklärter
Streuungsanteil.

### Best-of-10: wie viel im fehlenden Ranking steckt

| | ein Seed | best-of-10 |
|---|---|---|
| SigmaFlow <2 Å | 1.9% | **29.2%** |
| SigmaDock <2 Å | 9.1% | **45.9%** |

Das Paper erreicht 79.9% als Top-1 aus N Ziehungen, **gerankt von einem
trainierten Confidence-Modell** (App. F.2). Wir haben keines und ranken gar
nicht. Ein erheblicher Teil des Abstands zum Paper liegt also im fehlenden
Ranking, nicht allein im generativen Modell. Best-of-10 ist die Obergrenze,
die ein perfektes Ranking erreichen könnte — kein erzielter Wert.

### Reproduzierbarkeit verifiziert

SigmaFlow-Einzellauf ↔ neuer seed_0: **209 von 209 Posen bitgleich**. Das
Sampling ist bei festem Seed deterministisch. Genau deshalb war klar, dass
die SigmaDock-Abweichung ein echter Konfigurationsunterschied sein musste
und kein Rauschen — was zur Aufdeckung des Kopien-Artefakts führte.

## 📊 Wie viele ODE-Integrationsschritte braucht SigmaFlow? (2026-08-13)

Job 8554148, 7 Schrittzahlen × 3 Seeds = 21 Läufe à 209 Komplexe,
`stepsweep_curve.py`. Anlass: die 25 Schritte sind von SigmaDock GEERBT, wo
das Paper sie für einen **Diffusions**-Rückwärtsprozess optimiert hat
(„diminishing returns with more than 20-30 steps", Prat et al. 2026, App. F).
Für eine ODE muss das nicht gelten.

### Die Kurve ist flach

| Schritte | Median roh (± SD über 3 Seeds) | <2 Å | <5 Å | ausgerichtet |
|---|---|---|---|---|
| 5 | 4.61 ± 0.06 | 3.7% | 54.9% | 2.11 |
| 10 | 4.78 ± 0.22 | 3.5% | 53.0% | 1.97 |
| 15 | 4.88 ± 0.11 | 4.0% | 51.8% | 1.95 |
| **25 (geerbt)** | **4.76 ± 0.19** | **4.1%** | **53.3%** | **2.01** |
| 50 | 5.08 ± 0.05 | 4.0% | 49.1% | 1.91 |
| 100 | 5.17 ± 0.18 | 4.8% | 48.8% | 1.99 |
| 200 | 4.93 ± 0.24 | 3.7% | 50.7% | 1.99 |

**Vierzigfache Integrationsarbeit ändert nichts.** Gepaart pro Komplex gegen
den Default von 25 Schritten:

| Schritte | Diff. Mittel | 95%-CI | besser bei |
|---|---|---|---|
| 5 | −0.199 Å | [−0.391, −0.007] | 67% |
| 10 | −0.130 Å | [−0.321, +0.052] | 59% |
| 15 | −0.115 Å | [−0.308, +0.078] | 55% |
| 50 | +0.111 Å | [−0.097, +0.323] | 39% |
| 100 | +0.094 Å | [−0.095, +0.286] | 44% |
| 200 | +0.057 Å | [−0.116, +0.233] | 40% |

Auch der ausgerichtete RMSD bleibt bei ~2 Å — **nicht einmal die innere
Geometrie profitiert.** Wäre die ODE unter-integriert, müsste vor allem die
Platzierung besser werden; sie wird es nicht.

**Der 5-Schritt-Punkt ist formal signifikant, darf aber nicht als Effekt
behauptet werden:** obere CI-Grenze −0.007, also praktisch auf Null, und bei
sechs Vergleichen gegen dieselbe Referenz überlebt das keine Korrektur für
multiples Testen. Korrekte Aussage: **keine Schrittzahl zwischen 5 und 200
ist nachweisbar besser als eine andere.**

Größenordnung: ein Komplex streut über 3 Seeds mit SD 1.36 Å, über alle 7
Schrittzahlen (bereits seed-gemittelt) nur mit Spanne 1.66 Å — bei reinem
Rauschen wäre die erwartete Spanne über 7 Ziehungen schon ~2.4 Å. Die
Schrittzahl trägt **nichts** bei.

### Was das für die Bewertung heißt

**Positiv, und das erste belastbare Argument für Flow Matching in diesem
Projekt:** SigmaFlow liefert bei 5 Schritten dasselbe wie bei 25 und kann
damit **5× billiger sampeln**. Das ist die theoretisch erwartete Signatur
einer ODE mit gerader bedingter Bahn: die Trajektorie ist nahezu linear, also
reichen wenige Euler-Schritte.

**Einschränkung, nicht überdehnen:** SigmaDock wurde NICHT mit 5 Schritten
getestet. Belegt ist „SigmaFlow braucht die geerbten 25 nicht", **nicht**
„SigmaFlow braucht weniger Schritte als SigmaDock". Für Letzteres fehlt der
Gegen-Sweep auf der SigmaDock-Seite — billig nachzuholen, ~10 min GPU.

**Negativ:** die Sampling-Auflösung ist damit als Ursache für SigmaFlows
Rückstand ausgeschlossen. Das Modell integriert sein Vektorfeld sauber; das
Vektorfeld selbst zeigt nur nicht auf die richtige Stelle. Der Engpass bleibt
die **absolute** Orientierung und Verankerung in der Bindetasche.

**⚠️ Die Wall-Clock-Zeiten dieses Job-Arrays sind KEINE Kostenmessung.** Mit
`--array=0-20%5` liefen die frühen Tasks zu fünft parallel und teilten sich
CPU und Dateisystem: Seed 0 ist über alle Schrittzahlen flach bei 12–15 min,
während Seed 2 sauber skaliert (~1.3 min Fixkosten + ~2.4 s je Schritt). Für
echte Kosten ein Extra-Lauf mit `%1`.

### ⚠️ Korrektur: die folgende Zerlegung war fehlerhaft

Die unten stehende „kohärent/inkohärent"-Zerlegung maß Fragment-SCHWERPUNKTE
und ist damit **irreführend** — die Schwerpunkte sind bei beiden Methoden
gleich weit auseinander (s. Tabelle oben). Die Aussage „SigmaFlow reißt das
Molekül auseinander, SigmaDock hält es zusammen" ist in dieser Form NICHT
haltbar. Ebenfalls zu beachten: die rekonstruierte Fragmentierung
unterscheidet sich bei **39% der Komplexe** zwischen den Läufen, weshalb
alle historischen Bindungslängen-Tabellen dieser Datei teilweise
verschiedene Bindungsmengen vergleichen. Richtung bestätigt, absolute
Zahlen nur eingeschränkt gültig.

### 🔍 (fehlerhaft, s.o.) Zerlegung des Platzierungsfehlers

Aufgeteilt in (a) gemeinsame Verschiebung aller Fragmente eines Komplexes
und (b) Streuung der Fragmente um diese gemeinsame Verschiebung:

| | kohärent (Molekül am falschen Ort, zusammenhängend) | inkohärent (Molekül auseinandergerissen) | Verhältnis |
|---|---|---|---|
| **SigmaFlow** | **1.68 Å** | 4.04 Å | 2.41 |
| **SigmaDock** | 9.39 Å | **3.66 Å** | 0.39 |

- **SigmaFlow findet die Bindetasche** (1.68 Å daneben), **reißt das Molekül
  aber auseinander** — Fehler überwiegend inkohärent.
- **SigmaDock platziert weit daneben** (9.39 Å), **hält das Molekül aber
  zusammen** — Fehler überwiegend eine gemeinsame Verschiebung.

Das löst beide scheinbaren Widersprüche auf: SigmaFlows besserer RMSD kommt
fast vollständig aus der guten globalen Platzierung, SigmaDocks bessere
Übergangsbindungen aus der besseren inneren Kohärenz.

**Damit ist die alte Erzählung "SigmaFlow baut systematisch längere
Bindungen" zu korrigieren:** präziser ist "SigmaFlow trifft den Ort besser,
SigmaDock den Zusammenhalt". Keine Methode ist schlicht besser.
**Einschränkung:** die inkohärente Streuung unterscheidet sich nur um ~10%
(4.04 vs. 3.66 Å), die Bindungslängen aber um 70% — die Zerlegung erklärt
die Richtung, nicht die volle Größe. Und beide Modelle liegen weiterhin bei
praktisch zufälligen Fragment-Orientierungen; verglichen werden zwei
Fehlermodi, keine zwei funktionierenden Modelle.

**Die Fragment-Orientierungen sind bei BEIDEN Methoden praktisch
zufällig** (Zufallsgrenze 126.5°, analytisch `(π²/2+2)/π` und empirisch
über 20.000 Stichproben bestätigt). Alle bisherigen
Übergangsbindungs-Vergleiche oben sind damit Vergleiche zwischen zwei
faktisch untrainierten Modellen — das erklärt, warum alle drei
Loss-Varianten wirkungslos blieben.

**Compute-Einordnung (Paper, Appendix E.3):** die publizierten
SigmaDock-Zahlen (79.9% Top-1 RMSD<2Å) stammen aus 4 Tagen auf 4 A100
(DDP), Batch-Size 32 ≈ 384 GPU-Stunden. Unsere Läufe: 12h auf einer L40S,
Batch-Size 8 ≈ 12 GPU-Stunden — **~1/32 der Rechenzeit bei 1/4 der
Batch-Size**. 0% RMSD<2Å ist damit erwartbar, kein SigmaFlow-Defekt.
Details: STATUS.md, "Hypothese GEPRÜFT".

## Trainings-Verlust-Entwicklung (`best_model_score` / `loss_val/total`)

| Datum | Setup | Ergebnis | Quelle |
|---|---|---|---|
| 2026-07-17 | 300 Epochen, `rot_score_weight=2.0` (früher Test, vor jedem Bugfix) | `loss_R=6.76`, `loss_trans=3.5` (aggregiert ~18.8) | #4 |
| 2026-07-22 | 3h-Lauf bis Epoche 638, Ad-hoc-Gewichte | Plateau bei **12.69–12.94** (Epoche 620–3105, fällt kaum noch) | #9 |
| 2026-07-25 | 3h-Lauf bis Epoche 614, **Produktions-Gewichte** | **5.23** — grob halbiert gegenüber dem Ad-hoc-Lauf | #11 |

**Einordnung:** der große Sprung von ~12.7 auf 5.23 kam nicht von mehr
Trainingszeit, sondern vom Hyperparameter-Wechsel (Ad-hoc → SigmaDocks
eigene Produktionswerte, siehe #11 Sweep-Ergebnis: höheres
`rot_score_weight` verschlechtert `loss_trans`, ohne `loss_R` zu verbessern).

---

**Wie diese Datei aktuell halten:** nach jedem neuen Sampling-/Trainings-Lauf
mit echten Vergleichszahlen (nicht jeder Zwischenschritt/Bugfix — dafür ist
`STATUS.md` da) eine Zeile in der passenden Tabelle ergänzen, mit Datum und
Verweis auf den `STATUS.md`-PAUSE-PUNKT, der die Herleitung dokumentiert.

---

# EXP-110 Zwei-Kopf: erstes vollständiges Ergebnis (2026-08-22)

Trainingslauf `8625634`, Sampling `8629345` (10 Seeds, CPU, nfe 25, ohne
Ranking). Auswertung mit `SigmaFlow_Evaluation/evaluate_run.py` gegen
`true_ligands/`, **dieselbe Kette für alle drei Arme**. Die Werte für Minimal
und SigmaDock reproduzieren die dokumentierten 30.3 % und 47.6 % auf 0.2
Punkte genau, die Kette ist also konsistent.

## Kopfzahlen, 209 Komplexe, je 10 Seeds

| | Top-1 (Seed 0) | Oracle@5 | Oracle@10 | med. bester RMSD | med. Seed-0-RMSD |
|---|---:|---:|---:|---:|---:|
| SigmaFlow-Minimal | 2.4 % | 19.1 % | 30.1 % | 2.55 Å | 4.72 Å |
| SigmaDock | 10.0 % | 34.0 % | **47.4 %** | 2.05 Å | 4.39 Å |
| **EXP-110 Zwei-Kopf** | **7.2 %** | 21.9 % | **34.4 %** | 2.39 Å | 4.88 Å |

## Der Befund: die beiden Ebenen trennen sich

Gepaarter exakter McNemar über dieselben 209 Komplexe:

| Vergleich | Top-1 | Oracle@10 |
|---|---|---|
| Minimal gegen SigmaDock | p = 0.0015 signifikant | p < 0.0001 signifikant |
| **Minimal gegen EXP-110** | **p = 0.021 signifikant** | **p = 0.27 nicht signifikant** |
| SigmaDock gegen EXP-110 | p = 0.38 nicht signifikant | p = 0.0013 signifikant |

**Bei Top-1 schlägt EXP-110 Minimal signifikant, und der Abstand zu SigmaDock
verschwindet. Bei Oracle@10 kippt es: der Vorsprung gegenüber Minimal ist
nicht mehr signifikant, SigmaDock bleibt klar vorn.**

Das ist kein Widerspruch, sondern eine Aussage über die Art des Gewinns.
EXP-110 macht nicht mehr Komplexe grundsätzlich lösbar, es löst sie
**zuverlässiger je Ziehung**.

## Die vorregistrierte Kenngröße zeigt es am schärfsten

| | Oracle@10 | Oracle@1 | Verhältnis |
|---|---:|---:|---:|
| Minimal | 63 | 5 | **12.60** |
| SigmaDock | 99 | 21 | 4.71 |
| **EXP-110** | 72 | 15 | **4.80** |

Minimal braucht rund dreizehn Ziehungen für das, was ein einzelner Wurf
erreichen könnte. EXP-110 braucht knapp fünf, also **exakt SigmaDocks Wert**.
Der getrennte Rotationskopf hat die enorme Ziehungsvarianz beseitigt, ohne die
Obergrenze anzuheben.

**Das war so nicht vorhergesagt.** Die Vorregistrierung nannte diese Kenngröße,
erwartete aber, dass sie sich mit der Genauigkeit mitbewegt. Stattdessen
trennen sich die beiden Größen sauber. Lesart, mit Vorbehalt: der zweite Kopf
reduziert Rauschen in der Rotationsvorhersage, statt die grundsätzliche
Fähigkeit zu erweitern. Das passt zur Konstruktionsidee, ein Kopf der nur
Rotation lernen muss streut weniger als einer der sich das Feld teilt.

**Nicht festschreiben vor dem 72h-Lauf.** Bei 12 h Training kann sich das
verschieben.

## Chemische Plausibilität, Seed 0

| | RMSD < 2 Å | PB-valid | beides |
|---|---:|---:|---:|
| Minimal | 2.9 % | 37.3 % | 2.4 % |
| SigmaDock | 10.0 % | 31.6 % | 5.7 % |
| EXP-110 | 7.2 % | 36.4 % | 4.3 % |

EXP-110 behält Minimals Plausibilität und liegt über SigmaDock. Die
Zehn-Seed-Fassung läuft noch.

Einzelchecks mit Unterschied über einen Punkt: `bond_lengths` 55.5 / 46.4 /
49.8, `bond_angles` 48.3 / 39.7 / 46.4, `internal_steric_clash` 47.8 / 45.5 /
50.2, `internal_energy` 57.4 / 61.2 / 58.9, `tetrahedral_chirality` 93.8 /
92.8 / 90.0 (Reihenfolge Minimal / SigmaDock / EXP-110).

## Offene Einschränkungen

1. **Geräteunterschied.** EXP-110 wurde auf CPU gesampelt, Minimal und
   SigmaDock auf GPU (8554147 / 8554149). Die Integration ist deterministisch
   und die Abweichung liegt weit unter 2 Å, sauber ist es trotzdem erst, wenn
   alle Arme auf demselben Gerät gesampelt sind. Nachziehen kostet je Arm eine
   halbe Stunde mit `arc/sample_pb_seeds_cpu.slurm`.
2. **12h-Budget.** Absolute Zahlen liegen weit unter dem Paper (79.9 %). Was
   verglichen wird, sind die Arme untereinander.
3. **Ohne Ranking.** Alle Zahlen sind `RANKING=raw`. Die Lücke zwischen
   Oracle@10 und Top-1 ist genau das, was ein Ranker zurückholen könnte.

## gnina ist verfügbar (2026-08-22)

Modul `gnina/1.3.2`. **Läuft auf CPU-Knoten** (getestet auf `htc-c052`), aber
**nicht auf Login-Knoten** — dort scheitert das Mappen von `libcublasLt.so.12`
an den Speicherlimits. `CUDA/12.6.0` muss mitgeladen werden, auch für
`--no_gpu`.

Damit ist SigmaDocks Rankingheuristik reproduzierbar. Gemessen: 3.7 s je
Einzelaufruf, davon der Großteil Startzeit. gnina bewertet mehrere Posen aus
einer Multi-Model-SDF in einem Aufruf; 209 Aufrufe statt 2090 bringen einen
Arm auf rund 15 Minuten.

Erster Testwert für `5S8I_2LY` Seed 0: `Affinity = +2.88 kcal/mol`, also
**positiv**, mit `repulsion` 9.79. Plausible Bindungen liegen bei −6 bis −10.
Erwartbar bei diesem Trainingsbudget, aber ein Hinweis darauf, dass die
Energie bei durchweg schlechten Posen schlecht trennt. Das wäre selbst ein
berichtenswerter Befund.

**Nicht reproduzierbar bleibt die gemischte Heuristik**: `score_bias` und
`pb_exponent` stehen nirgends im SigmaDock-Repository. Die reinen Modi
`vinardo` und `pb` sind exakt nachbaubar, die Mischung nur als
Sensitivitätsanalyse über ein Gitter beider Konstanten.
