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

> **ÜBERHOLT am 2026-08-22.** Die Top-1-Spalte unten beruht auf **Seed 0
> allein**. Über alle zehn Seeds hält sie nicht: EXP-110 liegt nur in 5 von 10
> Seeds vor Minimal, und das Aggregat ist nicht signifikant (p = 0.13). Siehe
> den Abschnitt "Korrektur des Top-1-Befunds" weiter unten. Die Oracle@10-Spalte
> und die Aussagen zu SigmaDock bleiben gültig.

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

## PoseBusters-Validität, alle drei Arme über 10 Seeds (2026-08-22)

> **TEILWEISE ÜBERHOLT am 2026-08-22, später am Tag.** Alle Zahlen dieses
> Abschnitts sind Modus `regen`, also OHNE Protein. Mit Proteinprüfung
> (`redock`) verschwindet der Validitätsvorsprung der Flow-Matching-Arme
> vollständig: 7.0 / 6.7 / 6.9 %, p >= 0.72. Die Aussage "beide
> Flow-Matching-Arme sind klar plausibler als SigmaDock" gilt NUR für die
> innere Ligandengeometrie. Siehe "PB-Validität MIT Protein" weiter unten.
> Die Einzelcheck-Tabelle und die Aussage zum Konformer-Problem bleiben gültig.

Nachgerechnet lokal für alle drei Arme, 209 Komplexe × 10 Seeds = 2090 Posen
je Arm. Bis dahin gab es Zehn-Seed-PB nur für EXP-110, die anderen beiden
hatten nur Seed 0.

**Gegenprobe bestanden:** die neu erzeugten Seed-0-Tabellen stimmen mit den
alten *komplexweise* überein, 209/209 für Minimal und für SigmaDock. Die
Kette ist also dieselbe.

### Wichtige Einschränkung zum Modus

Alle Tabellen laufen im PoseBusters-Modus **`regen`**, nicht `redock`. Belegt
in den Daten: jede Zeile trägt `mol_cond_loaded = False`, und die Spalten
`minimum_distance_to_protein` und `volume_overlap_with_protein` fehlen.
PoseBusters wählt `regen` automatisch, wenn `mol_pred` und `mol_true`
vorliegen, aber kein Protein (`posebusters/cli.py:_select_mode`).

**Die Protein-Ligand-Prüfungen fehlen damit.** Geprüft wird nur die innere
Geometrie des Liganden. Die 79.9 % im SigmaDock-Paper sind `redock`-Zahlen.
Unsere Werte sind eine **Obergrenze**: Proteinprüfungen können nur weitere
Posen aussortieren. Nicht mit der Literatur vergleichbar.

### Kopfzahlen

| | PB-valid | Streuung über Seeds | RMSD < 2 Å | beides |
|---|---:|---:|---:|---:|
| SigmaFlow-Minimal | 34.4 % | 27.3–37.3 | 5.3 % | 4.0 % |
| SigmaDock | 29.3 % | 25.4–31.6 | 11.3 % | 6.2 % |
| **EXP-110 Zwei-Kopf** | **35.8 %** | 30.6–40.2 | 6.2 % | 4.4 % |

| | ≥1/10 valid | ≥1/10 genau | ≥1/10 beides |
|---|---:|---:|---:|
| SigmaFlow-Minimal | 70.8 % | 31.6 % | 23.4 % |
| SigmaDock | 60.3 % | 49.3 % | 30.1 % |
| **EXP-110** | **71.3 %** | 35.4 % | 25.8 % |

### Der Befund: Plausibilität und Genauigkeit sind entkoppelt

**SigmaDock ist der genaueste und zugleich der am wenigsten plausible Arm.**
Gepaarter exakter McNemar auf PB-Validität:

| Vergleich | je Seed signifikant | Richtung | gepoolt |
|---|---|---|---|
| Minimal gegen SigmaDock | 3/10 | Minimal besser in 9/10 | p = 8.5e-07 |
| Minimal gegen EXP-110 | 1/10 | EXP-110 besser in 5/10 | p = 0.19 |
| SigmaDock gegen EXP-110 | 5/10 | **EXP-110 besser in 10/10** | p = 5.1e-10 |

Die beiden Flow-Matching-Varianten sind untereinander gleich plausibel und
beide klar plausibler als SigmaDock. Der Vorsprung von EXP-110 gegenüber
SigmaDock gilt in **jedem einzelnen der zehn Seeds**, was ihn belastbar macht;
der gepoolte p-Wert allein wäre optimistisch, weil Seeds innerhalb eines
Komplexes nicht unabhängig sind.

Die Streuung der Validität über Seeds ist mit 30.6–40.2 (EXP-110) deutlich
enger als die der Genauigkeit (3.3–9.1, Faktor 2.8). Plausibilität ist
offenbar eine Eigenschaft des Modells, Platzierungsgenauigkeit stark eine
Eigenschaft der Ziehung.

### Woran die Validität scheitert

Alle topologischen Prüfungen bestehen zu 100 % in allen drei Armen
(Sanitisierung, Formel, Bindungen, Ringplanarität, Radikale). Es liegt
ausschließlich an vier geometrischen Prüfungen (Mittel über 2090 Posen,
Reihenfolge Minimal / SigmaDock / EXP-110):

| Check | Minimal | SigmaDock | EXP-110 | Spanne |
|---|---:|---:|---:|---:|
| `bond_lengths` | 54.4 % | 45.1 % | 51.2 % | 9.2 |
| `bond_angles` | 45.6 % | 39.5 % | 46.5 % | 7.0 |
| `internal_steric_clash` | 45.5 % | 44.1 % | 47.5 % | 3.4 |
| `internal_energy` | 59.4 % | 56.7 % | 59.9 % | 3.3 |
| `tetrahedral_chirality` | 90.9 % | 92.5 % | 91.9 % | 1.6 |

**Das ist ein Konformer-Problem, kein Platzierungsproblem.** Die starren
Fragmente werden bewegt, aber Bindungslängen und Winkel innerhalb der
Fragmente beziehungsweise an den Verbindungsstellen stimmen nicht. Bei 12 h
Trainingsbudget erwartbar, und es trifft alle drei Arme. Als Vergleichsgröße
zwischen den Armen daher schwach, als Aussage über das Budget aussagekräftig.

### Konsequenz für das Ranking

`≥1/10 beides` ist die Obergrenze für jeden Ranker, der Plausibilität und
Genauigkeit zugleich fordert: 25.8 % für EXP-110 gegenüber 35.4 % bei reiner
Genauigkeit. Zusammen mit dem Befund, dass das PB-Mittel als Ranker (6.2 %)
schlechter abschneidet als ein beliebiger Seed (7.2 %), heißt das: chemische
Plausibilität trägt hier **kein** Rankingsignal.

### Offen

Die echte, literaturvergleichbare PB-Validität braucht `redock` mit Protein
und damit einen ARC-Job; lokal liegen nur 2 der 209 Protein-PDBs.

## Rotationsqualität der drei Arme (2026-08-22)

Gemessen mit `SigmaFlow_Variants/posebusters_full_comparison/rotation_translation_compare.py`
über alle 2090 Posen je Arm (209 Komplexe × 10 Seeds), rund 7100 Fragmente je
Arm. Fragmentierung aus den Posen selbst zurückgewonnen: Bindungen, deren
Länge zwischen wahrer und vorhergesagter Pose erhalten bleibt, liegen
innerhalb eines starren Fragments. Kopienwahl über `best_copy`, weil 84 der
209 Dateien mehrere kristallographische Kopien enthalten.

**Nullhypothese**, numerisch nachgerechnet statt erinnert: bei Haar-verteilter
Rotation hat der Winkel die Dichte (1 − cos t)/π auf [0, π], Mittel **126.5°**,
Median 132.3°, Anteil unter 30° = 6.7 %.

### (1) Absoluter Fragment-Rotationsfehler

| Arm | Mittel | Median | 25 % | 75 % | < 30° |
|---|---:|---:|---:|---:|---:|
| SigmaFlow-Minimal | 126.7° | 136.7° | 100.4° | 161.2° | 2.6 % |
| SigmaDock | **113.6°** | 121.7° | 77.0° | 154.2° | **5.7 %** |
| EXP-110 Zwei-Kopf | 126.7° | 136.8° | 100.4° | 161.0° | 2.8 % |
| Haar-Zufall | 126.5° | 132.3° | 101.4° | 157.3° | 6.7 % |

**Minimal und EXP-110 sind von Zufall nicht zu unterscheiden.** Nur SigmaDock
liegt meßbar darunter. Der absolute Fehler enthält allerdings den globalen
Orientierungsfehler des ganzen Moleküls, deshalb die zweite Größe.

### (2) Relativer Fehler zwischen Fragmentpaaren

R_rel = R_f^T R_g; ein gemeinsamer Fehler beider Fragmente kürzt sich exakt
heraus.

| Arm | gebunden | ungebunden | Abstand |
|---|---:|---:|---:|
| SigmaFlow-Minimal | 109.7° | 124.4° | −14.7° |
| SigmaDock | 105.1° | 123.5° | **−18.4°** |
| EXP-110 | 108.6° | 124.8° | −16.3° |
| Haar-Zufall | 126.5° | 126.5° | 0.0° |

Alle drei haben lokale Geometrie gelernt; EXP-110 liegt zwischen Minimal und
SigmaDock. Die *innere Konsistenz* ist also erlernt, die *globale
Orientierung* bei den Flow-Matching-Armen nicht.

### Gepaarter Test, Bootstrap über 209 Komplexe

Unabhängige Einheit ist der Komplex, nicht das Fragment: Fragmente eines
Moleküls teilen sich dieses Molekül.

| Vergleich | Rotation | 95 %-KI | p |
|---|---:|---|---:|
| EXP-110 minus Minimal | **−0.06°** | [−1.58, +1.49] | **0.93** |
| EXP-110 minus SigmaDock | +13.66° | [+11.65, +15.68] | 0.0001 |
| Minimal minus SigmaDock | +13.72° | [+11.66, +15.79] | 0.0001 |

| Vergleich | Translation | 95 %-KI | p |
|---|---:|---|---:|
| EXP-110 minus Minimal | **−0.02 Å** | [−0.13, +0.08] | **0.65** |
| EXP-110 minus SigmaDock | +0.33 Å | [+0.20, +0.45] | 0.0001 |
| Minimal minus SigmaDock | +0.35 Å | [+0.23, +0.47] | 0.0001 |

**Der zweite Rotationskopf hat die Rotation nicht verbessert.** −0.06° bei
einem Konfidenzintervall von ±1.5° ist nicht "klein", sondern nicht vorhanden.
Dasselbe für die Translation. Das ist ein Nullergebnis gegen die
Konstruktionsidee von EXP-110, und es ist belastbar: 2090 gepaarte Posen.

### Korrektur des Top-1-Befunds

Der Rotationsbefund zwang zur Gegenprobe: wenn weder Rotation noch
Translation besser sind, woher käme dann ein Genauigkeitsgewinn? Antwort: er
existiert über zehn Seeds nicht. Der frühere Befund beruhte auf **Seed 0
allein**.

Erfolgsrate je einzelner Ziehung (RMSD < 2 Å), je Seed:

| Seed | Minimal | SigmaDock | EXP-110 |
|---:|---:|---:|---:|
| 0 | 2.9 % | 10.0 % | 7.2 % |
| 1 | 5.3 % | 12.0 % | 6.2 % |
| 2 | 6.2 % | 8.6 % | 3.3 % |
| 3 | 9.1 % | 12.0 % | 8.1 % |
| 4 | 7.2 % | 11.5 % | 7.2 % |
| 5 | 2.4 % | 14.4 % | 4.8 % |
| 6 | 6.2 % | 8.1 % | 7.2 % |
| 7 | 5.7 % | 12.9 % | 4.3 % |
| 8 | 4.3 % | 11.5 % | 4.3 % |
| 9 | 3.3 % | 12.0 % | 9.1 % |
| **Mittel** | **5.3 %** | **11.3 %** | **6.2 %** |

Seed 0 war für Minimal der zweitschlechteste von zehn (2.9 % gegen 5.3 %
Mittel) und für EXP-110 überdurchschnittlich. Der Vergleich auf Seed 0 war
damit Glück, nicht Signal.

Gepaarter exakter McNemar je Seed, Minimal gegen EXP-110: EXP-110 liegt in
**5 von 10** Seeds vorn, zwei Seeds sind signifikant (0 und 9), beide
zugunsten von EXP-110, acht zeigen nichts.

Aggregat, Erfolge je Komplex aus 10 Ziehungen, Bootstrap über 209 Komplexe:

| Arm | Erfolge je Komplex |
|---|---:|
| SigmaFlow-Minimal | 0.53 |
| SigmaDock | 1.13 |
| EXP-110 | 0.62 |

| Vergleich | Diff | 95 %-KI | p |
|---|---:|---|---:|
| EXP-110 minus Minimal | +0.09 | [−0.02, +0.21] | **0.13** |
| EXP-110 minus SigmaDock | −0.51 | [−0.70, −0.33] | 0.0001 |
| Minimal minus SigmaDock | −0.60 | [−0.81, −0.41] | 0.0001 |

**EXP-110 schlägt Minimal bei der Einzelziehung nicht signifikant.** Die
Richtung ist durchgehend leicht positiv (+0.09 Erfolge je Komplex), das
Intervall enthält die Null. SigmaDock schlägt beide klar, in 10 von 10 Seeds.

### Was davon bestehen bleibt

Bestätigt und unverändert: EXP-110 ist der **plausibelste** Arm (PB-Validität,
10/10 Seeds vor SigmaDock, p = 5.1e-10) und liegt bei der Ziehungsvarianz auf
SigmaDocks Niveau (Verhältnis Oracle@10/Oracle@1: 4.80 gegen 4.71).

Aufgegeben: die Aussage, EXP-110 verbessere Genauigkeit oder Rotation
gegenüber Minimal. Beides ist über zehn Seeds nicht nachweisbar.

### Vorbehalt Symmetrie

Für ein symmetrisches Fragment (Phenyl, tert-Butyl) ist die Rotation nur bis
auf die Symmetriegruppe bestimmt. Kabsch auf der gegebenen Atomnummerierung
liefert einen bestimmten Vertreter, nicht den nächstliegenden. Das **bläht den
gemessenen Fehler auf**, für alle drei Arme gleichermaßen. Der Vergleich
zwischen den Armen bleibt gültig, die absolute Höhe ist eine Obergrenze. Wie
groß der Effekt ist, ist ungeprüft.

## PB-Validität MIT Protein: `redock` über alle 10 Seeds (2026-08-22)

Der abschließende und maßgebliche Stand. Alle 2090 Posen je Arm, 209 Komplexe
× 10 Seeds, `mol_cond_loaded = True` durchgehend geprüft.

**Es wurde nichts neu gesampelt.** `regen` und `redock` sind zwei
PoseBusters-Presets auf denselben Posen. Der Unterschied ist allein, dass
`redock` das Protein mitbekommt und die Kollisions- und Volumenprüfungen
ausführt. Lokal machbar: die 209 Protein-PDBs von ARC geholt (20 MB, md5
38cf9027b29e46a6284b0f62b4bba339, 209/209 Deckung mit `true_ligands/`), dann
rund eine Stunde Rechenzeit für alle drei Arme parallel.

### Kopfzahlen

| Arm | `regen` (ohne Protein) | **`redock` (mit Protein)** | Verlust |
|---|---:|---:|---:|
| SigmaFlow-Minimal | 34.4 % | **7.0 %** | −27.5 pp |
| SigmaFlow-Separate | 35.8 % | **6.7 %** | −29.1 pp |
| SigmaDock | 29.3 % | **6.9 %** | −22.3 pp |

Gepaart über alle 2090 Posen (Komplex × Seed):

| Größe | Vergleich | p |
|---|---|---:|
| redock-Validität | Minimal gegen SigmaDock | **1.00** |
| redock-Validität | Separate gegen SigmaDock | **0.78** |
| redock-Validität | Separate gegen Minimal | **0.72** |
| RMSD < 2 Å | Minimal gegen SigmaDock | 8.7e-14 |
| RMSD < 2 Å | Separate gegen SigmaDock | 5.1e-10 |
| RMSD < 2 Å | Separate gegen Minimal | 0.16 |

**Der Validitätsvorsprung der Flow-Matching-Arme verschwindet restlos.** Er
galt ausschließlich für die innere Ligandengeometrie. Nur die Genauigkeit
trennt die Verfahren noch, und dort gewinnt SigmaDock klar.

### Warum: eine einzige Prüfung dominiert

| Check | Minimal | Separate | SigmaDock |
|---|---:|---:|---:|
| **`minimum_distance_to_protein`** | **18 %** | **16 %** | **20 %** |
| `volume_overlap_with_protein` | 66 % | 68 % | 73 % |
| `minimum_distance_to_organic_cofactors` | 86 % | 87 % | 86 % |
| übrige sechs | 94–100 % | 94–100 % | 94–100 % |

Rund vier von fünf Posen ragen ins Protein hinein, bei allen drei Armen fast
gleich stark. Dieser Einbruch von etwa 25 Punkten überschreibt die 5 Punkte
Unterschied, die unter `regen` das ganze Signal waren.

Nebenbei widerlegt: die Vermutung, SigmaDock stoße wegen der näheren
Platzierung häufiger an. Beim Volumenüberlapp ist SigmaDock der beste der drei.

### Die Metrik des Papers: RMSD < 2 Å UND PB-valid mit Protein

| Seed | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 0 | 1.0 % (2) | 1.4 % (3) | 2.9 % (6) |
| 1 | 1.4 % (3) | 1.9 % (4) | 1.9 % (4) |
| 2 | 1.9 % (4) | 1.4 % (3) | 0.0 % (0) |
| 3 | 3.3 % (7) | 2.9 % (6) | 1.9 % (4) |
| 4 | 1.9 % (4) | 1.9 % (4) | 1.9 % (4) |
| 5 | 0.5 % (1) | 1.0 % (2) | 2.9 % (6) |
| 6 | 0.5 % (1) | 1.0 % (2) | 2.4 % (5) |
| 7 | 1.9 % (4) | 1.0 % (2) | 2.9 % (6) |
| 8 | 1.0 % (2) | 1.4 % (3) | 1.9 % (4) |
| 9 | 1.0 % (2) | 1.4 % (3) | 1.9 % (4) |
| **Mittel** | **1.4 %** | **1.5 %** | **2.1 %** |
| **≥ 1/10** | 9.6 % | 11.0 % | 12.9 % |

SigmaDock führt, aber das sind 0 bis 7 Komplexe von 209 je Seed. Bei dieser
Größenordnung ist die Rangfolge nicht belastbar. **Warnung aus eigener
Erfahrung heute:** die Zwischenmeldung nach 4 Seeds ergab 1.9 / 1.9 / 1.7 %
und zeigte die Rangfolge falsch herum.

Vergleichswert des SigmaDock-Papers: 79.9 %. Wir liegen bei 1–2 %. Das ist
eine Aussage über 12 h Trainingsbudget ohne Ranking, nicht über die Verfahren.

### Die Zeile, die interessant bleibt

P(PB-valid mit Protein | RMSD < 2 Å), Bootstrap über 209 Komplexe:

| Arm | Schätzer | 95 %-KI | n Treffer |
|---|---:|---|---:|
| SigmaFlow-Minimal | 27.3 % | [17.1, 37.9] | 110 |
| SigmaFlow-Separate | 24.8 % | [16.4, 33.3] | 129 |
| SigmaDock | 18.2 % | [11.7, 25.4] | 236 |

| Vergleich | Diff | p |
|---|---:|---:|
| Minimal minus SigmaDock | +9.0 pp | **0.071** |
| Separate minus SigmaDock | +6.5 pp | 0.116 |
| Separate minus Minimal | −2.5 pp | 0.668 |

Die Richtung ist konsistent: SigmaDocks Treffer sind seltener physikalisch
sauber. **Signifikant ist es nicht** (p = 0.071). Als Hinweis formulieren,
nicht als Befund.

### Bilanz zu EXP-110 / SigmaFlow-Separate

Über 10 Seeds und mit Proteinprüfung bleibt kein positiver Befund:

| Hypothese | Ergebnis |
|---|---|
| bessere Genauigkeit als Minimal | nein, p = 0.16 |
| bessere Rotation als Minimal | nein, −0.06° bei ±1.5° |
| bessere Validität als Minimal | nein, p = 0.72 |
| bessere Validität als SigmaDock | nein, p = 0.78 |

**Der Zwei-Kopf-Ansatz ist bei diesem Trainingsbudget ein sauberes
Nullergebnis.** Das ist berichtbar: eine vorregistrierte Hypothese, sauber
gemessen, widerlegt.

Was bleibt, ist der Verfahrensvergleich: Diffusion trifft bei gleichem Budget
rund doppelt so oft wie Flow Matching, und beide erzeugen weit überwiegend
Posen, die ins Protein ragen.

### Methodische Lehre dieses Tages

Drei Befunde sind an einem Tag gekippt, alle in dieselbe Richtung, alle durch
dieselbe Ursache: **eine Auswertung, die weniger prüft als die Zielgröße.**

1. Top-1-Vorsprung von EXP-110: galt nur für Seed 0 (5/10 Seeds, p = 0.13).
2. Validitätsvorsprung der Flow-Arme: galt nur ohne Protein (p = 0.78 mit).
3. Rangfolge der gemeinsamen Metrik: nach 4 Seeds falsch herum.

Regel für den Rest der Arbeit: keine Zahl in die Thesis, die nicht über alle
Seeds und mit der vollständigen Prüfung gerechnet ist.

## Lokale Vertiefung auf 10 Seeds (2026-08-22, spät)

Alle Zahlen aus `threshold_and_overlap.py`, `error_budget.py` und
`pose_geometry.py` (alle in `SigmaFlow_Variants/posebusters_full_comparison/`).
Basis: 2090 gepaarte Posen je Arm, 209 Komplexe, RMSD durchgehend
symmetriekorrigiert (spyrmsd) — damit stimmen die Zahlen exakt mit den
dokumentierten Oracle@10-Werten überein.

**Hinweis zu zwei RMSD-Quellen:** PoseBusters rechnet einen eigenen RMSD. Er
stimmt zu 99.4–99.7 % mit spyrmsd überein; alle Abweichungen liegen zwischen
2.00 und 2.16 Å, also direkt an der Schwelle. Wo Komplexzahlen berichtet
werden, gilt **spyrmsd**, sonst verschieben sich Zählungen um bis zu vier
Komplexe.

### 1. Die Arme lösen unterschiedliche Komplexe

Mindestens ein Treffer unter 2 Å aus 10 Ziehungen:

| Arm | gelöst |
|---|---:|
| SigmaFlow-Minimal | 63 (30.1 %) |
| SigmaFlow-Separate | 72 (34.4 %) |
| SigmaDock | 99 (47.4 %) |
| **Vereinigung aller drei** | **126 (60.3 %)** |
| Schnitt aller drei | 33 |

**Ein perfekter Ensemble-Ranker käme auf 60.3 %, der beste Einzelarm auf
47.4 %.** Die Verfahren sind komplementär, nicht nur unterschiedlich stark.
Das ist der bislang stärkste Punkt zugunsten von Flow Matching in dieser
Arbeit.

### 2. Fehlerhaushalt: exakte Zerlegung statt Korrelation

**Korrektur einer früheren Aussage.** Ich hatte Spearman-Korrelationen
verglichen: Translation in Ångström gegen Rotation in Grad, jeweils mit dem
RMSD. Das ist keine Zerlegung, sondern ein Vergleich zweier Einheiten, und er
verzerrt zugunsten der Translation: der Versatz eines Fragments ist fast schon
dessen RMSD-Beitrag, ein Winkel wirkt erst über den Trägheitsradius.

Die exakte Identität, mit t = mean(P) − mean(Q) je Fragment:

    (1/n) Summe |P_i − Q_i|^2  =  |t|^2  +  (1/n) Summe |P_i^c − Q_i^c|^2

Der Kreuzterm verschwindet, weil die zentrierten Abweichungen im Mittel null
sind. Beide Terme in Å^2, direkt vergleichbar. Die Identität wird in
`error_budget.py` bei **jeder einzelnen Pose** per `assert` geprüft und hielt
in allen 6270 Fällen.

| Arm | MSD gesamt | Translation | Rotation | Anteil Translation |
|---|---:|---:|---:|---:|
| SigmaFlow-Minimal | 32.66 | 25.38 | 7.28 | 77.7 % |
| SigmaFlow-Separate | 32.28 | 25.30 | 6.98 | 78.4 % |
| SigmaDock | 28.10 | 22.17 | 5.93 | 78.9 % |

Aufgeschlüsselt nach Fragmentzahl der Pose (Anteil Translation):

| Fragmente | n | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|---:|
| **1** | 171 | **17.4 %** | **44.1 %** | **53.3 %** |
| 2 | 374 | 56.0 % | 54.4 % | 59.5 % |
| 3 | 401 | 71.0 % | 71.8 % | 69.3 % |
| 4–5 | 664 | 82.0 % | 81.8 % | 82.0 % |
| >= 6 | 480 | 88.1 % | 87.4 % | 87.7 % |

**Bei einfragmentigen Liganden kehrt es sich um**: Minimals Fehler liegt dort
zu 82.6 % in der Rotation. Die 78 % im Mittel stammen aus den
mehrfragmentigen Molekülen, und dort ist der Translationsterm nicht nur
globale Verschiebung, sondern auch die relative Anordnung der Fragmente.

Richtige Formulierung: **78 % des Fehlers liegen darin, WO die Fragmente
sind, 22 % darin, WIE sie gedreht sind.** Für EXP-110 heißt das: der zweite
Kopf adressiert rund ein Fünftel des Fehlers, und ausgerechnet bei
einfragmentigen Molekülen, wo Rotation dominiert, hat er nicht geholfen
(11.1 % gegen 11.7 % Treffer).

**Randbedingung, die die Zerlegung gültig macht:** mit
`graph.sample_conformer=false` stammt die innere Fragmentgeometrie aus der
gebundenen Pose. Innerhalb eines Fragments unterscheiden sich Vorhersage und
Wahrheit daher nur durch eine starre Bewegung, und der Rotationsterm enthält
keine Konformerabweichung.

### 3. Die Fragmentierung ist randomisiert

`fragmentation_strategy: Literal[...] = "random"` in
`SigmaFlow_Minimal/src/sigmadock/data.py:75`. Die Zerlegung wird **je Lauf neu
gezogen**. Belegt in den Daten: nur 35 % der Fragmentzahlen stimmen zwischen
den Armen überein, 193 von 209 Komplexen schwanken schon innerhalb eines Arms
über die Seeds.

Die Fragmentzahl ist damit eine Eigenschaft der einzelnen Pose, nicht des
Moleküls. Binning je Pose ist richtig; eine Zahl "die Fragmentzahl von Komplex
X" gibt es nicht.

### 4. Alles nach Fragmentzahl

**Median-RMSD in Å**

| Fragmente | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 1 | 3.32 | 3.18 | 2.70 |
| 2 | 3.67 | 3.47 | 3.23 |
| 3 | 4.33 | 4.59 | 3.91 |
| 4–5 | 5.38 | 5.33 | 4.84 |
| >= 6 | 6.77 | 6.75 | 6.29 |

**RMSD < 2 Å**

| Fragmente | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 1 | 11.7 % | 11.1 % | 24.6 % |
| 2 | 10.4 % | 14.2 % | 20.1 % |
| 3 | 5.7 % | 6.7 % | 12.5 % |
| 4–5 | 2.9 % | 2.9 % | 7.4 % |
| >= 6 | 0.2 % | 0.8 % | 1.7 % |

**PB-valid OHNE Protein**

| Fragmente | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 1 | 98.2 % | 94.2 % | 87.1 % |
| 2 | 67.4 % | 70.9 % | 56.4 % |
| 3 | 42.4 % | 46.1 % | 38.4 % |
| 4–5 | 17.3 % | 19.1 % | 12.3 % |
| >= 6 | 3.1 % | 2.3 % | 3.3 % |

**PB-valid MIT Protein**

| Fragmente | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 1 | 33.3 % | 24.6 % | 29.8 % |
| 2 | 11.8 % | 13.4 % | 13.6 % |
| 3 | 5.7 % | 8.2 % | 8.2 % |
| 4–5 | 2.9 % | 2.3 % | 0.9 % |
| >= 6 | 0.6 % | 0.0 % | 0.8 % |

Die innere Validität ist fast reine Fragmentarithmetik: 98 % bei einem
Fragment, 3 % bei sechs. Jede Schnittstelle ist eine Gelegenheit, eine
Bindungslänge oder einen Winkel zu verfehlen, und die Wahrscheinlichkeiten
multiplizieren sich. **Der Validitätsvorsprung der Flow-Arme kommt aus den
mittleren Klassen (2 bis 5 Fragmente)**, nicht aus den Randfällen. Mit Protein
verschwindet das Muster: dann liegen alle drei in jeder Klasse dicht
beieinander.

### 5. Wirkung der RMSD-Schwelle

**Die PB-Validität selbst hängt nicht von der Schwelle ab** — sie prüft
Geometrie und kennt den RMSD nicht. Fest: 34.4 / 35.8 / 29.3 % ohne Protein,
7.0 / 6.7 / 6.9 % mit.

**Trefferquote je Ziehung**

| Schwelle | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 1.0 Å | 0.2 % | 0.5 % | 0.9 % |
| 2.0 Å | 4.9 % | 5.8 % | 10.7 % |
| 2.5 Å | 10.0 % | 11.8 % | 18.5 % |
| 3.0 Å | 18.1 % | 21.5 % | 28.2 % |

**RMSD < X UND valid MIT Protein**

| Schwelle | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 1.0 Å | 0.2 % | 0.3 % | 0.4 % |
| 2.0 Å | 1.4 % | 1.5 % | 2.1 % |
| 2.5 Å | 2.0 % | 2.1 % | 2.7 % |
| 3.0 Å | 2.3 % | 2.8 % | 3.8 % |

**Die gemeinsame Metrik sättigt.** Von 2.0 auf 3.0 Å steigt SigmaDocks reine
Trefferquote von 10.7 auf 28.2 %, die gemeinsame Metrik aber nur von 2.1 auf
3.8 %. Die zusätzlichen Treffer zwischen 2 und 3 Å sind fast alle physikalisch
unbrauchbar. Die Schwelle zu lockern kauft nichts, wenn Validität mitverlangt
wird.

**P(valid MIT Protein | RMSD < X)**

| Schwelle | Minimal | Separate | SigmaDock | n (Min/Sep/SD) |
|---:|---:|---:|---:|---|
| 1.0 Å | 80.0 % | 60.0 % | 47.4 % | 5 / 10 / 19 |
| 2.0 Å | 28.4 % | 25.4 % | 19.2 % | 102 / 122 / 224 |
| 2.5 Å | 19.6 % | 17.8 % | 14.5 % | 209 / 247 / 386 |
| 3.0 Å | 13.0 % | 13.1 % | 13.6 % | 378 / 449 / 590 |
| alle | 7.0 % | 6.7 % | 6.9 % | 2090 |

Ohne Protein bei 1.0 Å: Minimal und Separate zu 100 % valide (5 bzw. 10
Posen).

**Genauigkeit und Plausibilität sind gekoppelt**, monoton und stark. Bei jeder
Schwelle bis 2.5 Å sind die Flow-Arme bedingt sauberer als SigmaDock; bei
3.0 Å verschwindet der Abstand.

Zwei Vorbehalte. **Die 1-Å-Zeile trägt nicht**: 5, 10 und 19 Posen, die 80 %
sind vier von fünf. **Die Kopplung ist teilweise eingebaut**: kleine RMSDs
treten überproportional bei einfragmentigen Liganden auf, und die sind fast
immer valide. Ein Teil des Anstiegs von 7 auf 28 % ist Fragmentzahl, nicht
Genauigkeit.

**Empfehlung für die Thesis:** falls eine zweite Schwelle berichtet wird, dann
**2.5 Å** — sie verdoppelt die Fallzahlen, erhält die Rangfolge und den
bedingten Validitätsvorsprung der Flow-Arme. 3.0 Å nicht, dort verschwindet
genau dieser Unterschied.

### 6. Oracle@k sättigt nicht

Erwartungstreu über zufällige Auswahl von k der 10 Seeds:

| k | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 1 | 5.3 % | 6.3 % | 11.4 % |
| 2 | 9.7 % | 11.1 % | 19.7 % |
| 3 | 13.6 % | 15.5 % | 25.9 % |
| 5 | 20.1 % | 22.7 % | 35.3 % |
| 10 | 31.6 % | 35.4 % | 49.3 % |

Nahezu linear bis k = 10, kein Abflachen. Die 40 Seeds werden weiter Ertrag
bringen.

### 7. Größenabhängigkeit (pose-unabhängige Achse)

Nach Schweratomen der wahren Struktur, RMSD < 2 Å je Ziehung:

| Schweratome | Komplexe | Minimal | Separate | SigmaDock |
|---|---:|---:|---:|---:|
| <= 20 | 82 | 9.3 % | 12.0 % | 20.0 % |
| 21–28 | 53 | 3.4 % | 3.0 % | 8.7 % |
| 29–36 | 45 | 1.8 % | 1.8 % | 2.9 % |
| >= 37 | 29 | **0.0 %** | **0.0 %** | 0.3 % |

Bei großen Liganden trifft kein Flow-Matching-Arm auch nur einmal in 290
Ziehungen. Die Gesamtzahlen sind fast vollständig ein Kleinmolekül-Ergebnis.

## 80 Seeds, alle drei Arme, mit und ohne Protein (2026-08-23)

**Maßgeblicher Stand.** Ersetzt alle 10- und 40-Seed-Zahlen der Abschnitte
darüber. 209 Komplexe × 80 Ziehungen × 3 Arme = **50.160 Posen**,
TFD-Abdeckung 100 %, keine Pose verworfen. RMSD durchgehend
symmetriekorrigiert (spyrmsd), Kristallkopien über `best_copy`. PoseBusters
im Modus `redock`: 15 ligandenintrinsische Checks, 9 mit Protein/Kofaktor/
Wasser.

Herkunft belegt: Sampling `8629867`/`8629909`/`8629910` (Seeds 0–39) und
`8633967`–`8633969` (Seeds 40–79), Redock `8632472`/`8632473`/`8632877` und
`8634241`–`8634243`. Vor dem Nachsampeln hat
`arc/preflight_more_seeds.sh` geprüft, dass alle vorhandenen Seeds je Arm aus
**genau einem** Checkpoint stammen und die neuen denselben benutzen — sonst
wären sie nicht additiv.

### Zwei Verifikationen, die vor den Zahlen kommen

**Maschinenunabhängigkeit.** Für Minimal und Separate wurden die Seeds 0–39
zweimal unabhängig durch PoseBusters geschickt: lokal (RDKit 2026.03.5) und
auf ARC. Ergebnis der aggregierten Raten:

| | ohne Protein | mit Protein |
|---|---:|---:|
| Minimal lokal / ARC | 34,14 % / **34,14 %** | 6,63 % / **6,63 %** |
| Separate lokal / ARC | 35,22 % / **35,22 %** | 6,77 % / **6,77 %** |

Identisch. Bemerkenswert ist, dass **52 % der einzelnen
`internal_energy`-Urteile abweichen** — der Check erzeugt Konformer-Ensembles
(ETKDG) und optimiert mit UFF, ist also umgebungsabhängig. Er ist aber nie
ausschlaggebend: wo er kippt, ist ohnehin ein anderer Check gefallen. Lässt
man ihn ganz weg, ändert sich die Rate um 0,09 Punkte. Alle übrigen 23 Checks
stimmen bis auf 3 von 8360 Posen überein.

**Abgrenzung der Checkgruppen.** `redock_noprotein.yml` (redock minus die 8
protein-abhängigen Module) liefert exakt die 15 intrinsischen Checks, keinen
Protein-Check, und reproduziert die aus der Redock-Tabelle abgeleiteten
Urteile 209/209. Die Aufteilung „ohne Protein" / „mit Protein" ist damit
belegt, nicht behauptet.

### Anteil je Ziehung (16.720 Posen je Arm, 95 %-Wilson)

| Größe | Minimal | Separate | SigmaDock |
|---|---:|---:|---:|
| RMSD < 1 Å | 0,38 % [0,30–0,49] | 0,42 % [0,34–0,54] | **0,83 %** [0,70–0,98] |
| RMSD < 2 Å | 5,43 % [5,10–5,78] | 5,86 % [5,51–6,22] | **10,34 %** [9,89–10,81] |
| RMSD < 2,5 Å | 11,54 % [11,06–12,03] | 12,08 % [11,59–12,58] | **18,34 %** [17,76–18,93] |
| RMSD < 3 Å | 19,72 % [19,12–20,33] | 20,51 % [19,91–21,13] | **27,39 %** [26,72–28,07] |
| PB-valid OHNE Protein | 33,89 % [33,18–34,61] | **35,02 %** [34,30–35,74] | 28,43 % [27,75–29,12] |
| PB-valid MIT Protein | 6,89 % [6,52–7,28] | **6,97 %** [6,59–7,36] | 6,19 % [5,83–6,57] |
| < 2 Å UND valid OHNE Protein | 4,01 % [3,72–4,32] | 4,31 % [4,01–4,63] | **5,71 %** [5,37–6,07] |
| < 2 Å UND valid MIT Protein | 1,39 % [1,22–1,58] | 1,37 % [1,20–1,56] | **1,64 %** [1,46–1,85] |

### Gepaarte Vergleiche

Bootstrap über die **209 Komplexe**, nicht über die 16.720 Posen: die 80
Ziehungen eines Komplexes sind nicht unabhängig, ein Test über Posen würde
die Freiheitsgrade um den Faktor 80 überschätzen.

| Größe | Sep − Min | SD − Min | SD − Sep |
|---|---|---|---|
| Treffer < 2 Å je Komplex | +0,43 pp, p = 0,044 | +4,91 pp, p < 0,001 | +4,48 pp, p < 0,001 |
| Median-RMSD je Komplex | −0,010 Å, p = 0,819 | −0,630 Å, p < 0,001 | −0,620 Å, p < 0,001 |
| PB-valid OHNE Protein | **+1,12 pp, p = 0,007** | −5,46 pp, p < 0,001 | −6,58 pp, p < 0,001 |
| PB-valid MIT Protein | +0,08 pp, p = 0,79 | −0,70 pp, p = 0,082 | −0,78 pp, p = 0,088 |
| < 2 Å und valid OHNE | +0,31 pp, p = 0,105 | +1,70 pp, p < 0,001 | +1,40 pp, p = 0,003 |
| < 2 Å und valid MIT | −0,02 pp, p = 0,91 | +0,26 pp, p = 0,22 | +0,28 pp, p = 0,22 |

### Oracle@k, erwartungstreu über zufällige Seed-Teilmengen

Gemittelt über 400 zufällige k-Teilmengen der 80 Seeds, nicht „die ersten k" —
sonst hängt das Ergebnis an der Seed-Reihenfolge.

**RMSD < 2 Å** — sättigt bei 80 noch nicht

| k | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 1 | 5,45 % | 5,91 % | 10,35 % |
| 5 | 20,38 % | 21,87 % | 34,01 % |
| 10 | 31,83 % | 33,75 % | 47,27 % |
| 20 | 44,64 % | 47,31 % | 59,22 % |
| 40 | 56,61 % | 60,19 % | 68,62 % |
| 80 | **65,55 %** | **70,81 %** | **76,08 %** |

**PB-valid OHNE Protein** — sättigt vollständig

| k | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 1 | 34,13 % | 34,95 % | 28,47 % |
| 10 | 69,37 % | 68,90 % | 57,51 % |
| 40 | 83,96 % | 84,03 % | 72,07 % |
| 80 | **89,00 %** | **89,00 %** | 77,99 % |

**PB-valid MIT Protein**

| k | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 1 | 6,90 % | 6,93 % | 6,14 % |
| 10 | 29,82 % | 29,76 % | 25,96 % |
| 40 | 47,92 % | 48,05 % | 42,22 % |
| 80 | 55,98 % | **56,46 %** | 48,33 % |

**< 2 Å UND valid OHNE Protein** — hier kreuzen sich die Kurven

| k | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 1 | 3,99 % | 4,35 % | **5,75 %** |
| 10 | 23,82 % | 24,68 % | **29,11 %** |
| 40 | 44,81 % | 45,97 % | **47,05 %** |
| 60 | 50,64 % | **52,27 %** | 50,82 % |
| 80 | 54,55 % | **56,46 %** | 53,11 % |

**< 2 Å UND valid MIT Protein** — die harte Größe

| k | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 1 | 1,37 % | 1,37 % | **1,68 %** |
| 10 | 9,69 % | 9,43 % | **11,69 %** |
| 40 | 22,58 % | 22,20 % | **26,84 %** |
| 80 | 32,06 % | 31,58 % | **35,89 %** |

### Sechs Befunde

1. **Flow Matching gewinnt die Ligandenchemie deutlich und stabil.** +5,5
   bzw. +6,6 Prozentpunkte gegenüber SigmaDock, p < 0,001. Der Befund hat 10,
   40 und 80 Seeds überstanden und ist der robusteste Vorteil der Arbeit.
2. **SigmaDock gewinnt die Genauigkeit ebenso deutlich.** Doppelte
   Trefferquote unter 2 Å, Median-RMSD −0,63 Å, alle p < 0,001.
3. **Mit Protein liegt SigmaDock leicht hinten** (6,19 % gegen 6,89/6,97 %,
   p ≈ 0,08) — nicht signifikant, aber das Vorzeichen hat sich gegenüber 40
   Seeds gefestigt.
4. **Auf der gemeinsamen Metrik mit Protein ist kein Paar signifikant.**
   1,39 / 1,37 / 1,64 %, alle p > 0,21. Genauigkeitsvorsprung und
   Validitätsvorsprung heben sich auf.
5. **Ein Kreuzungspunkt bei „genau und sauber ohne Protein":** SigmaDock
   führt bis k ≈ 40 und wird dann von beiden Flow-Armen überholt.
6. **Die Sättigung trennt die Größen.** Validität ohne Protein läuft bei
   89,00 % aus, bei Minimal und Separate auf zwei Nachkommastellen gleich.
   `RMSD < 2 Å` steigt bei 80 noch. Was die Arme unterscheidet, ist
   Genauigkeit; was sie teilen, ist die Menge der überhaupt lösbaren Komplexe.

### EXP-110 gegen Minimal: ein einziger robuster Unterschied

**Validität ohne Protein, +1,12 pp, p = 0,007** — überlebt die
Bonferroni-Korrektur über drei Vergleiche (Schwelle 0,017). Alle anderen
Größen sind null (p = 0,11 bis 0,91).

Der zweite Kopf verbessert die **chemische Plausibilität**, nicht die
Platzierung. Das ist nicht, wofür die Variante gebaut wurde — sie sollte die
Rotation behandeln, und die ist bei beiden Armen von Zufall nicht zu
unterscheiden (126,9° gegen 126,48° Haar-Erwartung, gemessen bei 40 Seeds).

### Fehlerhaushalt, 80 Seeds

Exakte Zerlegung des Lagefehlers, Identität bei **jeder** der 50.160 Posen per
`assert` geprüft:

| Arm | MSD gesamt | Translation | Rotation | Anteil Translation |
|---|---:|---:|---:|---:|
| Minimal | 31,67 | 24,54 | 7,14 | 77,5 % |
| Separate | 31,87 | 24,87 | 6,99 | 78,0 % |
| SigmaDock | 28,36 | 22,46 | 5,90 | 79,2 % |

Anteil Translation nach Fragmentzahl der Pose:

| Fragmente | n (Min/Sep/SD) | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|---:|
| 1 | 1333/1322/1167 | 14,7 % | 14,3 % | 23,6 % |
| 2 | 3115/3113/3180 | 56,7 % | 56,8 % | 57,5 % |
| 3 | 3187/3035/3085 | 70,7 % | 70,5 % | 72,0 % |
| 4–5 | 5319/5341/5599 | 82,3 % | 82,6 % | 83,2 % |
| ≥ 6 | 3766/3909/3689 | 87,7 % | 88,0 % | 88,3 % |

**78 % des Fehlers liegen darin, WO die Fragmente sind, 22 % darin, WIE sie
gedreht sind.** Bei einfragmentigen Posen kehrt es sich um: dort liegen ~85 %
in der Rotation.

### Rotations- und Translationsgeometrie, 80 Seeds

Nullhypothese numerisch nachgerechnet: bei Haar-verteilter Rotation hat der
Winkel die Dichte (1 − cos t)/π auf [0, π], Mittel **126,48°** (Monte Carlo
über 400.000 Quaternionen: 126,49°).

| Größe | Minimal | Separate | SigmaDock |
|---|---:|---:|---:|
| Rotationsfehler je Fragment | **126,49°** | **126,90°** | 113,73° |
| größter Fragmentwinkel | 154,21° | 154,59° | 145,83° |
| Fragmentversatz | 4,06 Å | 4,08 Å | 3,82 Å |
| Schwerpunktversatz Ligand | 1,34 Å | **1,31 Å** | 1,58 Å |

**Minimals Rotationsfehler beträgt 126,49° gegen eine Zufallserwartung von
126,48°.** Nach 12 h Training ist die globale Fragmentorientierung bei beiden
Flow-Matching-Armen von Zufall nicht zu unterscheiden. SigmaDock liegt mit
113,73° messbar darunter, aber ebenfalls weit von „gelernt" entfernt.

Gepaart, Bootstrap über 209 Komplexe:

| | Rotation | Schwerpunktversatz |
|---|---|---|
| Separate − Minimal | +0,404°, p = 0,15 | **−0,034 Å, p < 0,001** |
| SigmaDock − Minimal | **−12,76°, p < 0,001** | +0,241 Å, p < 0,001 |
| SigmaDock − Separate | **−13,16°, p < 0,001** | +0,274 Å, p < 0,001 |

### Der mechanistische Gegensatz

SigmaFlow setzt den **Schwerpunkt des Moleküls besser** (1,31–1,34 gegen
1,58 Å), SigmaDock die **innere Anordnung** (Rotation 13° besser,
Fragmentversatz kleiner). Flow Matching findet die Tasche, ordnet aber innen
falsch an; Diffusion ordnet innen besser, sitzt als Ganzes aber daneben.
Beides hochsignifikant und gegenläufig.

### EXP-110 bei 80 Seeds: zwei robuste Effekte, keiner davon die Rotation

| Behauptung | Befund |
|---|---|
| bessere Rotation als Minimal | **nein** — +0,40°, p = 0,15, Vorzeichen sogar falsch |
| bessere Genauigkeit (Median-RMSD) | **nein** — −0,010 Å, p = 0,82 |
| bessere Trefferquote < 2 Å | grenzwertig — +0,43 pp, p = 0,044 |
| **bessere Validität ohne Protein** | **ja** — +1,12 pp, p = 0,007 |
| **bessere Schwerpunktplatzierung** | **ja** — −0,034 Å, p < 0,001 |
| bessere Validität mit Protein | nein — +0,08 pp, p = 0,79 |
| bessere gemeinsame Metrik mit Protein | nein — −0,02 pp, p = 0,91 |

Der zweite Kopf wurde gebaut, um die **Rotation** zu behandeln. Bei 80 Seeds
wirkt er messbar auf **Translation und Ligandenchemie** und nachweislich
nicht auf die Rotation. Das ist berichtbar, aber es ist nicht die Wirkung,
die die Konstruktion begründet hat.

### Was durch die 80 Seeds gefallen ist

- **Der vorregistrierte Median-RMSD-Effekt.** Bei 40 Seeds −0,137 Å
  (p = 0,013), bei 80 Seeds −0,010 Å (p = 0,82). Ein Permutationstest über
  2000 zufällige Halbierungen zeigt, warum: der je-Komplex-Median schwankt
  zwischen zwei Hälften desselben Laufs mit SD 0,069 Å. Der gemessene Effekt
  war nur das Doppelte dieses Rauschens und hat die Verdoppelung der Daten
  nicht überstanden.
- **Die 44,1-%-Zeile im Fehlerhaushalt** (Separate, ein Fragment, 10 Seeds).
  Bei 80 Seeds sind es 14,3 % über 1322 Posen. Die 10-Seed-Zahl beruhte auf
  171 Posen; MSD ist varianzartig und bricht bei kleinen Stichproben zuerst.
- **Die Größe des Komplementaritätsvorteils.** Bei 10 Ziehungen holte ein
  Ensemble aller drei Arme 12,9 Punkte über den besten Einzelarm, bei 40 noch
  5,2. Die Arme lösen nicht dauerhaft verschiedene Komplexe — sie streuen
  unterschiedlich.

**Regel, die sich daraus ergibt:** keine Zahl in die Thesis, die nicht über
alle Seeds, mit vollständiger Prüfung und mit einem gepaarten Test über
Komplexe gerechnet ist. In diesem Projekt sind inzwischen sechs Befunde an
genau dieser Stelle gekippt.

## Ranking und Auswahl mit gnina/Vinardo, 80 Seeds (2026-08-23)

Alle Zahlen aus `SigmaFlow_Variants/posebusters_full_comparison/build_thesis_datasets.py`,
Datensätze in `Thesis Visualisierungen/data/`. Scoring auf ARC, Jobs `8634498`
bis `8634500`, gnina 1.3.2, `--scoring vinardo --cnn_scoring none --no_gpu`,
je 209 Aufrufe mit allen 80 Posen eines Komplexes in einer Multi-Model-SDF.
16.720 Werte je Arm, kein übersprungener Komplex.

### Warum die Auswahl legitim ist

Der Affinitätsscore braucht nur die **vorhergesagte Pose** und die
**Proteinstruktur**. Keine Referenzpose, kein RMSD. Beides liegt zur
Inferenzzeit vor, und `gnina --score_only` ist bei gegebener Pose
deterministisch. Es ist damit eine echte Auswahlregel und keine nachträgliche
Beschönigung.

Ein Vorbehalt: bei der PB-Validität *mit* Protein misst man etwas, das mit der
Affinität konzeptuell verwandt ist — beide bestrafen Kollision. Zirkulär ist
es nicht (der Score ist keiner der Checks), aber die Verwandtschaft gehört
beim Berichten genannt.

### SigmaDock kennt vier Rankingmodi

`SigmaDock/src_sigmadock/src_sigmadock_chem/statistics#Evaluationspipeline.py`,
Funktion `compute_ordering`, Zeilen 650–760:

| Modus | Sortierschlüssel |
|---|---|
| `None` | zufällige Permutation, fester Seed — die Nullreferenz |
| `"vinardo"` / `"cnn"` | eine benannte gnina-Kennzahl; `Affinity` und `Intramolecular energy` aufsteigend, `CNNscore` absteigend |
| `"pb"` | Mittelwert über gewählte PoseBusters-Checks, absteigend |
| `"heuristic"` | Produkt aus richtungsnormalisiertem Score und `(score_bias + avg_pb ** pb_exponent)` |

Verfügbare Kennzahlen (`postprocessor.py:11`): `Affinity`, `CNNscore`,
`CNNaffinity`, `CNNvariance`, `Intramolecular energy`.
`conf/sampling/base.yaml:84` setzt zur Samplingzeit `scoring: vinardo`.

**Ausgewertet ist Modus 2 mit score_name = Affinity.** Der `cnn`-Modus fehlt —
er braucht `--cnn_scoring all` und damit gninas neuronalen Teil, der für
Posenauswahl trainiert ist und vermutlich besser abschneidet. Offen.

### Der Score ist informativ, aber schwach

Spearman zwischen Affinität und RMSD, je Komplex über 80 Ziehungen. Positiv
ist richtig: bessere (negativere) Affinität soll kleinerem RMSD entsprechen.

| Arm | Median rho | rho > 0 | p < 0,05 und rho > 0 |
|---|---:|---:|---:|
| Minimal | +0,113 | 72,2 % | 28,7 % |
| Separate | +0,108 | 73,2 % | 32,1 % |
| SigmaDock | **+0,172** | **82,8 %** | **39,7 %** |

Bemerkenswert ist die Größenordnung der Affinitäten selbst: Median **+8,05**
(Minimal, Separate) und **+5,67** (SigmaDock) kcal/mol, nur 18–25 % negativ,
Maximum bis +299. Vinardo-Werte sind normalerweise negativ. Die Mehrheit der
Posen sitzt also energetisch ungünstig — sie kollidiert mit dem Protein. Das
deckt sich damit, dass nur rund 7 % die Proteinprüfung bestehen.

### Auswahl nach Affinität: Anteil der gewählten Posen

**Minimal**

| Kenngröße | k=1 | k=5 | k=10 | k=20 | k=40 | k=80 |
|---|---:|---:|---:|---:|---:|---:|
| RMSD < 1,0 Å | 0,38 % | 1,32 % | 1,91 % | 2,76 % | 3,57 % | 3,83 % |
| RMSD < 2,0 Å | 5,44 % | 11,15 % | 14,19 % | 17,84 % | 21,70 % | 25,36 % |
| RMSD < 2,5 Å | 11,62 % | 19,26 % | 22,76 % | 26,73 % | 30,42 % | 33,01 % |
| RMSD < 3,0 Å | 19,90 % | 28,41 % | 32,52 % | 37,13 % | 41,07 % | 44,02 % |
| PB-valid ohne Protein | 34,06 % | 35,01 % | 35,43 % | 35,96 % | 36,78 % | 38,76 % |
| PB-valid mit Protein | 6,98 % | 16,30 % | 20,26 % | 23,67 % | 26,28 % | 29,67 % |
| < 2 Å und valid ohne | 4,00 % | 7,75 % | 9,87 % | 12,57 % | 15,72 % | 19,14 % |
| **< 2 Å und valid mit** | **1,44 %** | 4,13 % | 6,01 % | 8,35 % | 10,95 % | **13,88 %** |

**Separate**

| Kenngröße | k=1 | k=5 | k=10 | k=20 | k=40 | k=80 |
|---|---:|---:|---:|---:|---:|---:|
| RMSD < 1,0 Å | 0,42 % | 1,35 % | 2,12 % | 3,12 % | 4,35 % | 5,26 % |
| RMSD < 2,0 Å | 5,82 % | 11,44 % | 14,40 % | 17,83 % | 21,37 % | 24,88 % |
| RMSD < 2,5 Å | 12,03 % | 19,50 % | 22,94 % | 26,70 % | 30,20 % | 33,01 % |
| RMSD < 3,0 Å | 20,56 % | 28,74 % | 32,06 % | 35,69 % | 39,32 % | 43,06 % |
| PB-valid ohne Protein | 35,03 % | 36,94 % | 36,97 % | 37,19 % | 37,24 % | 37,80 % |
| PB-valid mit Protein | 7,00 % | 16,32 % | 20,15 % | 23,65 % | 26,82 % | 29,19 % |
| < 2 Å und valid ohne | 4,27 % | 7,98 % | 9,65 % | 11,48 % | 13,15 % | 15,31 % |
| **< 2 Å und valid mit** | **1,39 %** | 4,01 % | 5,42 % | 7,28 % | 8,98 % | **11,00 %** |

**SigmaDock**

| Kenngröße | k=1 | k=5 | k=10 | k=20 | k=40 | k=80 |
|---|---:|---:|---:|---:|---:|---:|
| RMSD < 1,0 Å | 0,82 % | 2,89 % | 4,49 % | 6,38 % | 8,50 % | **11,48 %** |
| RMSD < 2,0 Å | 10,30 % | 19,98 % | 24,55 % | 28,62 % | 32,15 % | **36,36 %** |
| RMSD < 2,5 Å | 18,26 % | 30,46 % | 35,79 % | 40,50 % | 44,38 % | **48,80 %** |
| RMSD < 3,0 Å | 27,25 % | 39,35 % | 44,34 % | 48,70 % | 51,70 % | **54,07 %** |
| PB-valid ohne Protein | 28,30 % | 30,38 % | 31,06 % | 31,36 % | 31,39 % | 31,10 % |
| PB-valid mit Protein | 6,18 % | 13,85 % | 17,43 % | 20,78 % | 23,77 % | 26,32 % |
| < 2 Å und valid ohne | 5,68 % | 10,10 % | 12,14 % | 14,12 % | 15,70 % | 17,22 % |
| **< 2 Å und valid mit** | **1,63 %** | 4,78 % | 6,74 % | 9,06 % | 11,33 % | **14,35 %** |

### Der Score wirkt sehr ungleich

Zuwachs von k = 1 auf k = 80, Beispiel Minimal:

| Kenngröße | Faktor |
|---|---:|
| PB-valid ohne Protein | **1,1x** |
| RMSD < 3 Å | 2,2x |
| PB-valid mit Protein | 4,3x |
| RMSD < 2 Å | 4,7x |
| < 2 Å und valid mit Protein | **9,6x** |
| RMSD < 1 Å | **10,1x** |

Vinardo bewertet die Protein-Ligand-Wechselwirkung. Über die innere Geometrie
des Liganden weiß es nichts — deshalb bewegt sich die ligandenintrinsische
Validität kaum. Über Kollisionen weiß es alles — daher der Faktor 4 bei der
Proteinvalidität. **Der Score ist ein Platzierungsfilter, kein Chemiefilter.**

Und je schärfer die Schwelle, desto mehr bringt die Auswahl: Faktor 10 bei
1 Å, Faktor 2 bei 3 Å. Grobe Treffer hätte man auch zufällig erwischt.

### Rückholquote: welcher Anteil der Lücke geschlossen wird

Definiert als (Top-1 minus Zufall) geteilt durch (Oracle minus Zufall), k = 80:

| Kenngröße | Minimal | Separate | SigmaDock |
|---|---:|---:|---:|
| PB-valid ohne Protein | 8,8 % | 5,2 % | 5,4 % |
| < 2 Å und valid ohne Protein | 29,9 % | 21,1 % | 24,3 % |
| RMSD < 2 Å | 33,1 % | 29,3 % | 39,6 % |
| < 2 Å und valid mit Protein | 40,7 % | 31,9 % | 37,1 % |
| PB-valid mit Protein | **46,4 %** | **44,9 %** | **47,8 %** |

Bei k = 2 liegt die Rückholquote der Proteinvalidität sogar bei 72–77 % — die
billigsten Gewinne kommen zuerst.

### Von den vier Modi ist der einfachste der beste

Zielgröße RMSD < 2 Å, Top-1@80. Nicht zirkulär, weil der RMSD in keinen Ranker
eingeht. Gleichstände zufällig aufgelöst — die PB-Mittelwerte haben nur 25
Stufen, sonst entschiede die Seed-Reihenfolge.

| Ranker | Minimal | Separate | SigmaDock |
|---|---:|---:|---:|
| Zufall | 4,31 % | 3,83 % | 10,05 % |
| **Affinität (vinardo)** | **25,36 %** | **24,88 %** | **36,36 %** |
| pb: alle 24 Checks | 13,25 % | 14,64 % | 24,25 % |
| pb: 15 intrinsische | 8,10 % | 8,80 % | 14,54 % |
| pb: 9 mit Protein | 10,91 % | 11,42 % | 20,60 % |
| Oracle | 65,55 % | 70,81 % | 76,08 % |

Die Heuristik über ein Gitter aus 24 Parameterkombinationen erreicht
28,23 / 26,79 / 38,28 % im besten Feld, also **ein bis drei Punkte über der
reinen Affinität** — bei einer Gitterstreuung derselben Größenordnung und
einem Standardfehler von rund drei Punkten. Kein Feld ist von der reinen
Affinität zu unterscheiden, und ein nachträglich ausgewähltes bestes Feld wäre
wertlos.

**Ein struktureller Defekt der Heuristik:** sie multipliziert einen
vorzeichenbehafteten Wert mit einem positiven Faktor. Ist der normalisierte
Score positiv, hebt ein besserer PB-Faktor den Wert; ist er negativ, senkt er
ihn — bessere Chemie wird dann bestraft. Anteil der Posen mit negativem
Score: **81,7 / 81,1 / 75,4 %**. Für rund vier Fünftel aller Posen wirkt die
Formel in die falsche Richtung. Die additive Gegenprobe erreicht
28,71 / 28,71 / 39,71 %, also ebenfalls nur ein bis drei Punkte — die schwache
Wirkung liegt nicht am Vorzeichen, sondern daran, dass die PB-Checks über den
RMSD wenig aussagen.

### Zwei Beobachtungen ohne statistische Absicherung

**Separate verliert unter Ranking.** Bei k = 1 gleichauf (1,39 gegen 1,44 %),
bei k = 80 deutlich hinten (11,00 gegen 13,88 %). Bei den reinen
RMSD-Schwellen passiert das nicht; bei < 1 Å ist Separate sogar besser (5,26
gegen 3,83 %). Der gepaarte Test über 209 Komplexe gibt p = 0,34 — **das ist
eine Beobachtung, kein Befund.**

**Unter Ranking sind alle drei Arme auf der harten Kenngröße
ununterscheidbar:** 13,88 / 11,00 / 14,35 %, kein Paar signifikant (p = 0,29
bis 0,93). Nur SigmaDocks RMSD-Vorsprung überlebt (+11,00 pp gegen Minimal,
p = 0,006).

### Was das für den Ausblick bedeutet

Der Ausblick auf ein Confidence-Modell hat jetzt eine Untergrenze: **ein
klassischer Score ohne jedes Lernen holt rund 40 % der Lücke** und hebt die
harte Kenngröße von 1,4 % auf 13,9 %. Was ein gelerntes Modell darüber hinaus
schafft, ist offen — aber der Boden steht.

## Der Rotationsprior erklärt den gesamten Methodenunterschied (2026-08-23)

**Dies ist der wichtigste Befund des Projekts und er widerruft die
Kernaussage des bisherigen Verfahrensvergleichs.**

Alle Zahlen aus `SigmaFlow_Variants/posebusters_full_comparison/confsampled_compare.py`,
40 Seeds, 8360 Posen je Arm und Modus. Referenz durchgehend der wahre Ligand
aus `true_ligands/<cid>_ligands.sdf`, nächstgelegene kristallographische Kopie
über `best_copy`.

### Die Beobachtung, die alles ausgelöst hat

Die Trajektorien aus `predictions.pt` zeigen den Rotationsfehler bei
**Schritt 0**, also vor jeder gelernten Bewegung:

| Arm | Rotation bei Schritt 0 |
|---|---:|
| SigmaFlow-Minimal | 126,2° |
| SigmaFlow-Separate | 126,4° |
| **SigmaDock** | **115,5°** |

126,48° ist der Erwartungswert unter dem Haar-Maß. **SigmaDock startet elf
Grad näher an der Wahrheit als der Zufall** — vor dem ersten Schritt.

### Die Ursache, aus dem Code belegt

`SigmaDock/src_sigmadock/src_sigmadock_diff/so3_diffuser - UMBAUEN.py`:

```python
def sample_ref(self, n, device):
    # NOTE: replace with sample uniform?
    return self.sample(torch.ones([n], device=device))   # IGSO(3) bei t = 1
```

SigmaDock zieht die Startrotation aus **IGSO(3) beim maximalen Rauschpegel**,
und `max_sigma = 2.25**0.5 = 1.5`. IGSO(3) ist der Wärmeleitungskern auf SO(3)
und konvergiert erst für σ → ∞ gegen das Haar-Maß. Mit SigmaDocks eigener
Dichteformel nachgerechnet:

| σ | mittlerer Winkel |
|---:|---:|
| 1,00 | 87,16° |
| **1,50** | **114,97°** |
| 2,00 | 124,47° |
| 3,00 | 126,46° |
| Haar | **126,48°** |

Gemessen 115,5°, theoretisch 114,97° — Übereinstimmung auf 0,6°.

SigmaFlow zieht dagegen exakt gleichverteilt
(`so3_flow_matcher.py:13`, `so3_utils.sample_uniform`). Das war eine
dokumentierte Designentscheidung: Flow Matching kann die exakte
Gleichverteilung als Quelle nehmen, Diffusion muss sie über hinreichend großes
Terminalrauschen annähern. Der Audit hatte den Unterschied bereits als
„Confounder 1" mit Totalvariationsabstand 0,130 festgehalten — neu ist, dass
er das Hauptergebnis erklärt.

### Warum die Identität die Antwort ist

Mit `graph.sample_conformer=false` stammt die Fragmentgeometrie aus der
**gebundenen Pose**. Die Identitätsrotation ist dann die **wahre**
Orientierung, und IGSO(3) hat um sie herum mehr Masse als das Haar-Maß. Der
Prior ist damit informativ.

`alignment_tries: 0` schließt aus, dass nachträglich ausgerichtet wird.

### Standardisierung: der Vorsprung verschwindet

Posen nach Startrotation gebinnt, Trefferquote unter 2 Å je Bin:

| Startrotation | Minimal | Separate | SigmaDock |
|---|---:|---:|---:|
| 60–90° | 35,1 % | 31,9 % | 35,8 % |
| 90–110° | 9,9 % | 9,0 % | 9,4 % |
| 110–130° | 2,6 % | 2,7 % | 3,7 % |
| 130–150° | 2,2 % | 1,1 % | 1,5 % |

**Innerhalb jedes Bins identisch.** Was sich unterscheidet, ist die
Besetzung: SigmaDock hat 285 Posen unter 90° Startfehler, Minimal 117.

Standardisiert auf Minimals Startverteilung fällt SigmaDock von 9,24 % auf
**5,21 %** [4,41 – 6,09] gegen Minimals 5,17 %. Umgekehrt steigt Minimal auf
SigmaDocks Startverteilung auf **8,95 %** gegen SigmaDocks 9,24 %.

### Der direkte Test: `sample_conformer=true`

Mit generiertem Konformer ist die Identität nicht mehr die Antwort. Der
Prior-Vorteil sollte verschwinden. 40 Seeds, alle drei Arme:

| Arm | Modus | Rotation Start | Rotation Ende | RMSD Ende | < 2 Å |
|---|---|---:|---:|---:|---:|
| Minimal | bound | 127,16° | 126,87° | 5,179 | 4,40 % |
| Minimal | sampled | 126,74° | 126,89° | 5,167 | 4,50 % |
| Separate | bound | 126,67° | 126,78° | 5,149 | 4,65 % |
| Separate | sampled | 126,65° | 126,34° | 5,130 | 4,72 % |
| **SigmaDock** | **bound** | **115,20°** | 113,79° | 4,790 | **9,37 %** |
| **SigmaDock** | **sampled** | **126,47°** | 126,64° | 5,245 | **4,77 %** |

**SigmaDocks Startrotation springt auf 126,47°. Der Haar-Erwartungswert ist
126,48°.** Übereinstimmung auf eine Hundertstel Grad.

**Die Trefferquote halbiert sich: 9,37 % → 4,77 %.**

Die beiden SigmaFlow-Arme bewegen sich nicht (4,40 → 4,50 % und 4,65 →
4,72 %). Nur der Arm ändert sich, dessen Prior nicht uniform ist — das
schließt aus, dass der Effekt von der geänderten Aufgabenschwierigkeit kommt.

### Bei gleicher Startbedingung kein Unterschied

Gepaart über 209 Komplexe, Modus `sampled`:

| Vergleich | Differenz | 95 %-KI | p |
|---|---:|---|---:|
| Separate − Minimal | +0,23 pp | [−0,37, +0,83] | 0,45 |
| SigmaDock − Minimal | +0,28 pp | [−0,38, +0,93] | 0,42 |
| SigmaDock − Separate | +0,05 pp | [−0,54, +0,65] | 0,88 |

**Kein Paar signifikant. 4,50 / 4,72 / 4,77 Prozent.**

### Was das für die berichteten Zahlen heißt

**ÜBERHOLT:** „SigmaDock trifft rund doppelt so oft unter 2 Å (11,3 % gegen
5,3 / 6,2 %, p = 8,7e-14)" und alle davon abgeleiteten Aussagen zum
Genauigkeitsvorsprung. Die Messung ist korrekt, die Deutung als
Methodenunterschied ist es nicht.

**ÜBERHOLT:** „SigmaDock − Minimal: −12,76° Rotation, p < 0,001". Der
Absolutwert misst den Startpunkt. Vergleichbar ist nur die **Änderung** über
die Trajektorie, und die ist bei allen drei Armen ununterscheidbar von null
(−0,03° / +0,35° / −0,82° bei rund 59° tatsächlicher Drehung).

> **Nachgerechnet 2026-08-24.** `traj_agg_Minimal.csv` war am 23.08. um 17:21
> versehentlich von einem kleineren Lauf (2 Seeds, 418 Posen) überschrieben
> worden, `traj_pose_Minimal.csv` enthielt nur die Kopfzeile. Die oben
> berichtete Zahl stammte aber aus dem ursprünglichen 10-Seed-Lauf und ist
> korrekt: die Wiederholung mit `--max_seeds 10 --detail_seeds 2` ergibt
> **−0,031°** bei 59,6° kumulierter Drehung. Die 2-Seed-Fassung hätte −0,245°
> ergeben. Alle drei Arme stehen jetzt wieder auf je 2090 Posen:
>
> | Arm | Start | Ende | Änderung | kumulierte Drehung |
> |---|---:|---:|---:|---:|
> | Minimal | 126,19° | 126,16° | −0,031° | 59,6° |
> | Separate | 126,41° | 126,76° | +0,349° | 58,8° |
> | SigmaDock | 115,53° | 114,71° | −0,821° | 58,3° |
>
> Die Lehre: eine Ausgabedatei, die bei jedem Lauf denselben Namen trägt, sagt
> nichts über den Umfang des Laufs, der sie erzeugt hat. `n_poses` steht
> deshalb in der Aggregatdatei — ohne diese Spalte wäre die Verkürzung nicht
> aufgefallen.

**STEHT:** bei gleicher Startbedingung sind Flow Matching und Diffusion bei
diesem Trainingsbudget nicht unterscheidbar.

### Einordnung, die dazugehört

SigmaDocks Prior ist kein Implementierungsfehler. Sein Score-Modell ist auf
diesen Rauschplan trainiert; ein uniformer Prior zur Samplingzeit wäre
inkonsistent mit dem Training. Die Designentscheidung ist `max_sigma = 1.5`,
und sie stammt aus dem Paper.

Entscheidend ist: in der Konfiguration, die das Paper selbst vorsieht
(`sample_conformer: true`), tritt der Effekt **gar nicht auf**. Er entsteht
erst durch unsere Entscheidung, die gebundene Pose als Konformerquelle zu
nehmen. Diese Entscheidung war gut begründet — sie isoliert die Platzierung
und macht die exakte Fehlerzerlegung gültig — aber sie hat den Vergleich
verzerrt.

Die faire Formulierung: *In der Auswertungskonfiguration mit gebundenen
Konformeren startet SigmaDock systematisch näher an der Lösung, weil sein
Rotationsprior das Haar-Maß nicht erreicht. Unter der Konfiguration des
Papers verschwindet der Unterschied vollständig.*

### Ein Fehler in der Auswertung, gefunden und behoben

Die erste Messung im `sampled`-Modus ergab RMSD 49,7 Å und 0 % Treffer.
Ursache: `x0` aus `predictions.pt` wurde als Wahrheit benutzt. Das stimmt nur
bei `sample_conformer=false`; mit `true` trägt `x0` den generierten Konformer
in seinem eigenen Bezugssystem, im Mittel **50,4 Å** von der wahren Pose
entfernt.

Die Gegenprobe lokalisierte den Bruch eindeutig: `x0_hat` stimmt in **beiden**
Modi auf 0,0000 Å mit der geschriebenen SDF überein, und
`trajectory[-1] × 2,7 + com` ebenfalls. Nur die Referenz war falsch.
`trajectory_geometry.py` und `confsampled_compare.py` benutzen jetzt den
Referenzliganden aus der SDF-Datei.

## MASSGEBLICH: Der volle Metriksatz im Paper-Setup (2026-08-23, spät)

**Dieser Abschnitt ersetzt die Tabellen der `bound`-Auswertung als
maßgeblichen Stand.** Die dortigen Zahlen bleiben gültig als Messung, aber sie
gelten für eine Auswertungskonfiguration, in der die Identitätsrotation die
Antwort ist und SigmaDocks Prior deshalb informativ. Siehe den Abschnitt
„Der Rotationsprior erklärt den gesamten Methodenunterschied".

40 Seeds, 8360 Posen je Arm und Modus, RMSD durchgehend symmetriekorrigiert
(spyrmsd), Referenz über `best_copy`. PoseBusters `redock`: 15
ligandenintrinsische Checks, 9 mit Protein.

`bound` = `graph.sample_conformer=false`, Fragmente aus der gebundenen Pose.
`sampled` = `true`, generierter Konformer — **die Vorgabe des Papers**
(`conf/sampling/base.yaml:49`).

### Anteil je Ziehung

| Kenngröße | Min bound | **Min sampled** | Sep bound | **Sep sampled** | SD bound | **SD sampled** |
|---|---:|---:|---:|---:|---:|---:|
| RMSD < 2 Å | 5,32 % | **5,33 %** | 5,79 % | **5,85 %** | 10,33 % | **5,80 %** |
| PB-valid ohne Protein | 34,14 % | **33,85 %** | 35,22 % | **34,96 %** | 28,64 % | **28,19 %** |
| PB-valid mit Protein | 6,63 % | **6,61 %** | 6,77 % | **6,94 %** | 6,51 % | **5,60 %** |
| < 2 Å und valid ohne Prot. | 3,86 % | **3,84 %** | 4,29 % | **4,43 %** | 5,80 % | **3,22 %** |
| < 2 Å und valid mit Prot. | 1,27 % | **1,15 %** | 1,30 % | **1,48 %** | 1,72 % | **0,94 %** |

**Nur SigmaDock bewegt sich.** Die beiden Flow-Arme ändern sich beim Wechsel
der Konformerquelle um Hundertstel bis Zehntel Prozentpunkte; SigmaDocks
Trefferquote halbiert sich von 10,33 auf 5,80 %. Das ist der Prior-Effekt,
und er trifft ausschließlich den Arm, dessen Startverteilung nicht das
Haar-Maß ist.

### Gepaarte Vergleiche im Paper-Setup

Bootstrap über die 209 Komplexe, nicht über die 8360 Posen.

| Kenngröße | Separate − Minimal | SigmaDock − Minimal | SigmaDock − Separate |
|---|---|---|---|
| RMSD < 2 Å | +0,51 pp, p = 0,12 | +0,47 pp, p = 0,20 | −0,05 pp, p = 0,89 |
| PB-valid ohne Protein | **+1,11 pp, p = 0,030** | **−5,66 pp, p < 0,0001** | **−6,77 pp, p < 0,0001** |
| PB-valid mit Protein | +0,32 pp, p = 0,30 | −1,02 pp, p = 0,054 | **−1,34 pp, p = 0,003** |
| < 2 Å und valid ohne Prot. | **+0,59 pp, p = 0,023** | **−0,62 pp, p = 0,039** | **−1,21 pp, p = 0,0003** |
| < 2 Å und valid mit Prot. | +0,33 pp, p = 0,052 | −0,20 pp, p = 0,21 | **−0,54 pp, p = 0,003** |

### Die Umkehrung auf der gemeinsamen Metrik

Bei `bound` führte SigmaDock deutlich: **5,80 gegen 3,86 und 4,29 %**. Im
Paper-Setup ist es umgekehrt: **3,22 gegen 3,84 und 4,43 %** — SigmaDock ist
der schlechteste der drei, gegen Separate mit p = 0,0003.

Mit Protein dasselbe: aus 1,72 gegen 1,27/1,30 wird 0,94 gegen 1,15/1,48.

Der Mechanismus ist durchgehend derselbe. SigmaDock verliert den Startvorteil,
seine Genauigkeit fällt auf das Niveau der anderen — und weil seine
Ligandenchemie ohnehin rund sechs Punkte schlechter ist, fällt die
**Kombination** aus Genauigkeit und Validität unter die der Flow-Arme.

### Stand der Methodenfrage

| Größe | Ergebnis im Paper-Setup |
|---|---|
| Genauigkeit allein | **kein Unterschied**, alle p > 0,11 |
| Ligandenchemie | **Flow Matching**, +5,7 bis +6,8 pp, p < 0,0001 |
| Validität mit Protein | **Flow Matching**, bis +1,3 pp, p = 0,003 |
| genau UND valide | **Flow Matching**, bis +1,2 pp, p = 0,0003 |
| Robustheit bei nfe 5 | **Flow Matching**, Faktor 7 |
| Rotation | von keinem Verfahren gelernt |

**EXP-110 ist auf beiden gemeinsamen Metriken der beste Arm** (+0,59 pp gegen
Minimal ohne Protein, p = 0,023; +0,33 pp mit Protein, p = 0,052). Der
Zwei-Kopf-Ansatz, der als Nullergebnis begann, hat damit zwei robuste
Effekte: bessere Ligandenchemie, und in der Folge die beste kombinierte
Kenngröße.

### Multiples Testen

Fünfzehn Vergleiche. Die beiden stärksten — Ligandenchemie mit p < 0,0001 —
überleben jede Korrektur. Die Werte zwischen p = 0,02 und 0,05 tun das
einzeln nicht, zeigen aber **alle in dieselbe Richtung** und sind über zwei
unabhängige Konformerquellen hinweg konsistent. Sie gehören als Muster
berichtet, nicht als Einzelbefunde.

### Die Robustheit gegen grobe Integration

Aus dem Schrittzahl-Sweep, 40 Seeds je Arm, `bound`:

| Arm | nfe 5 | nfe 25 |
|---|---:|---:|
| Minimal | 4,13 % | 4,40 % |
| Separate | 4,61 % | 4,65 % |
| **SigmaDock** | **0,63 %** | 9,37 % |

SigmaDock bricht bei fünf Schritten vollständig zusammen: Median-RMSD
23,45 Å, 146 von 209 Komplexen über 20 Å, kein einziger unter 5 Å.
Konfiguration geprüft, `diffusion.num_steps: 5` war gesetzt, die Trajektorie
hat fünf Einträge.

Gepaart innerhalb der Arme, 5 gegen 25 Schritte:

| Arm | Trefferquote | Median-RMSD |
|---|---|---|
| Minimal | −0,28 pp, p = 0,18 | **−0,230 Å, p < 0,001** |
| Separate | −0,05 pp, p = 0,86 | −0,055 Å, p = 0,32 |
| SigmaDock | **−8,73 pp, p < 0,0001** | **+18,42 Å, p < 0,0001** |

Bei fünf Schritten direkt verglichen: SigmaDock **−3,49 pp** gegen Minimal und
**−3,97 pp** gegen Separate, beide p < 0,0001.

Der Mechanismus ist strukturell. Der Flow-Matching-Pfad ist die Geodäte
zwischen Quelle und Ziel, also nahezu gerade — Euler trifft sie mit wenigen
Schritten. SigmaDocks Rückwärtsprozess integriert über einen EDM-Rauschplan
von σ = 1,5 bis t_min = 0,005 mit ρ = 3; bei fünf Schritten sind die Sprünge
in σ zu groß, und ein Euler-Schritt bei hohem Rauschpegel multipliziert den
Score mit einem großen Faktor.

**Praktisch: SigmaFlow liefert mit einem Fünftel der Netzwerkauswertungen
dieselbe Qualität, SigmaDock nicht.** Das ist der einzige Methodenvorteil in
dieser Arbeit, der weder vom Prior noch von der Auswertungskonfiguration
abhängt.

### Nebenbefund: die kumulierte Drehung hängt nicht an der Schrittzahl

| | nfe 5 | nfe 25 |
|---|---:|---:|
| Drehung je Fragment, kumuliert | 51,9° | 59,4° |
| Rotationsfehler am Ende | 126,92° | 126,87° |

Fünfmal so viele Schritte erzeugen 14 % mehr Gesamtdrehung — die Weglänge ist
also nahezu erhalten, wie es für die Integration derselben ODE sein muss. Der
Rotationsfehler bleibt in beiden Fällen unverändert. **Die Rotation wird nicht
gelernt, und das hängt nicht an der Integrationsauflösung.**

### Oracle@k im Paper-Setup

209 Komplexe, 40 Seeds, erwartungstreu über zufällige k-Teilmengen (400
Wiederholungen). Datensatz: `Thesis Visualisierungen/data/selection_curves_papersetup40.csv`.

**RMSD < 2 Å — die Kurven liegen aufeinander**

| k | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 1 | 5,43 % | 5,81 % | 5,84 % |
| 5 | 20,74 % | 21,41 % | 21,80 % |
| 10 | 32,81 % | 33,12 % | 33,45 % |
| 20 | 46,96 % | 46,48 % | 46,47 % |
| 40 | **59,81 %** | 59,33 % | 57,42 % |

Bei kleinem k führt SigmaDock minimal, ab k = 20 dreht es sich. Zum Vergleich
bei `bound`: dort stand SigmaDock bei k = 40 auf **68,62 %** gegen 56,61 und
60,19. Der gesamte Abstand ist verschwunden.

**Übrige Schwellen bei k = 40**

| Schwelle | Minimal | Separate | SigmaDock |
|---|---:|---:|---:|
| < 1 Å | 7,66 % | **8,61 %** | **8,61 %** |
| < 2,5 Å | 75,60 % | **77,03 %** | 73,68 % |
| < 3 Å | 85,17 % | **87,56 %** | 86,12 % |

**PB-valid ohne Protein — hier öffnet sich die Schere mit k**

| k | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 1 | 33,83 % | 34,97 % | 28,47 % |
| 3 | 52,34 % | 52,69 % | 41,80 % |
| 10 | 69,82 % | 70,68 % | 55,86 % |
| 20 | 77,20 % | 79,26 % | 63,23 % |
| 40 | 82,30 % | **85,17 %** | **70,33 %** |

Der Abstand wächst von **6,5 Punkten** bei einer Ziehung auf **14,8 Punkte**
bei vierzig. SigmaDock erreicht selbst mit 40 Versuchen für rund 30 % der
Komplexe keine einzige chemisch saubere Pose, die Flow-Arme nur für 15–18 %.

Bemerkenswert im Vergleich zu `bound`: dort liefen Minimal und Separate bei
k = 40 auf **exakt denselben** Wert (85,65 %). Im Paper-Setup trennen sie sich,
Separate liegt vorn.

**PB-valid mit Protein**

| k | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 1 | 6,67 % | 6,99 % | 5,52 % |
| 10 | 28,22 % | 28,51 % | 22,37 % |
| 40 | **44,98 %** | 44,50 % | 35,89 % |

**< 2 Å UND valid ohne Protein**

| k | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 1 | 3,83 % | 4,46 % | 3,25 % |
| 10 | 24,10 % | 24,95 % | 19,74 % |
| 40 | **44,98 %** | 44,02 % | 35,89 % |

**< 2 Å UND valid mit Protein — die harte Größe**

| k | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 1 | 1,15 % | **1,48 %** | 0,94 % |
| 5 | 4,85 % | **6,31 %** | 4,09 % |
| 10 | 8,16 % | **10,45 %** | 7,15 % |
| 20 | 13,17 % | **16,13 %** | 11,42 % |
| 40 | 20,57 % | **22,49 %** | 16,75 % |

**Separate führt hier über den gesamten k-Bereich.** Gegen SigmaDock sind es
bei k = 40 fast sechs Punkte.

### Was die Kurven zeigen und die Einzelzahlen nicht

**Die Trennung der Verfahren liegt nicht in der Genauigkeit, sondern in der
Chemie — und sie wächst mit k.** Bei `RMSD < 2 Å` laufen alle drei Kurven
aufeinander. Bei `PB-valid ohne Protein` öffnet sich die Schere von 6,5 auf
14,8 Punkte. SigmaDock erzeugt nicht nur seltener saubere Posen, es erreicht
auch mit vielen Versuchen weniger Komplexe überhaupt.

**EXP-110 gewinnt genau dort, wo beide Kriterien zusammenkommen.** Auf
`RMSD < 2 Å` allein liegt es gleichauf mit Minimal, auf `PB-valid` knapp vorn
— aber auf der Kombination führt es über jeden Wert von k. Das ist konsistent
mit den gepaarten Tests und betrifft die Kenngröße, die praktisch zählt.

## Der Chemievorsprung auf 80 Seeds im Paper-Setup (2026-08-24)

Die Validitaetstabellen fuer `graph.sample_conformer=true` liegen jetzt fuer
alle 80 Seeds vor: 209 Komplexe x 80 Seeds x 3 Arme = **50160 Posen**. Die
Tabellen der Seeds 0-39 sind byte-identisch mit den zuvor ausgewerteten, die
Datenbasis ist also additiv erweitert, nicht neu erzeugt.

Rechnung: `SigmaFlow_Variants/posebusters_full_comparison/papersetup80.py`.
Datensaetze: `Thesis Visualisierungen/data/validity_papersetup80.csv` und
`selection_validity_papersetup80.csv`.

### Anteil je Ziehung

| Kenngroesse | Minimal | Separate | SigmaDock |
|---|---:|---:|---:|
| PB-valid ohne Protein | 33,71 % [33,00; 34,43] | **34,89 %** [34,17; 35,61] | 28,28 % [27,60; 28,97] |
| PB-valid mit Protein | 6,62 % [6,25; 7,01] | **7,17 %** [6,78; 7,57] | 5,84 % [5,49; 6,20] |

Gegen die 40-Seed-Fassung (33,85 / 34,96 / 28,19 bzw. 6,61 / 6,94 / 5,60)
bewegt sich nichts ausserhalb der Intervalle. Die Verdopplung bestaetigt, sie
korrigiert nicht.

### Gepaart, Bootstrap ueber die 209 Komplexe

4000 Ziehungen, Aufloesungsgrenze p = 0,00025.

| Kenngroesse | Separate - Minimal | SigmaDock - Minimal | SigmaDock - Separate |
|---|---|---|---|
| PB-valid ohne Protein | **+1,17 pp, p = 0,0045** | **-5,44 pp, p < 0,00025** | **-6,61 pp, p < 0,00025** |
| PB-valid mit Protein | **+0,54 pp, p = 0,041** | -0,78 pp, p = 0,069 | **-1,33 pp, p = 0,0025** |

**Der Vorsprung von Separate gegenueber Minimal ist mit der doppelten
Datenbasis von p = 0,030 auf p = 0,0045 gefallen.** Er ueberlebt damit auch
eine Bonferroni-Korrektur ueber die sechs Vergleiche dieser Tabelle
(0,0045 x 6 = 0,027). Bei 40 Seeds tat er das nicht. Das ist der Unterschied
zwischen einem Befund, der nur als Muster berichtet werden darf, und einem,
der allein stehen kann.

### Oracle@k, PB-valid ohne Protein

| k | Minimal | Separate | SigmaDock | Sep - SD |
|---:|---:|---:|---:|---:|
| 1 | 33,75 % | 34,91 % | 28,27 % | +6,64 |
| 5 | 59,81 % | 60,28 % | 48,04 % | +12,24 |
| 10 | 69,78 % | 70,28 % | 56,23 % | +14,05 |
| 20 | 78,20 % | 78,59 % | 64,01 % | +14,58 |
| 40 | 84,39 % | 84,11 % | 71,26 % | +12,85 |
| 80 | **89,47 %** | 87,56 % | 78,47 % | +9,09 |

Die Schere zu SigmaDock oeffnet sich bis k = 20 auf 14,6 Punkte und schliesst
sich danach wieder -- mit genuegend Versuchen holt auch ein schlechteres
Verfahren auf. Das ist Saettigung, kein Aufholen im Sinne der Qualitaet.

**Bei grossem k ueberholt Minimal die Separate-Variante** (89,47 gegen
87,56 % bei k = 80), obwohl Separate je Ziehung besser ist. Das ist kein
Widerspruch, sondern eine Aussage ueber Vielfalt: Separate erzeugt im Mittel
sauberere Posen, Minimal erzeugt *verschiedenere*. Wer eine Pose zieht, nimmt
Separate; wer achtzig zieht und die beste behalten darf, nimmt Minimal.

### Oracle@k, PB-valid mit Protein

| k | Minimal | Separate | SigmaDock |
|---:|---:|---:|---:|
| 1 | 6,59 % | 7,20 % | 5,86 % |
| 10 | 28,37 % | 29,83 % | 22,50 % |
| 40 | 45,13 % | 47,04 % | 36,29 % |
| 80 | 52,15 % | **55,98 %** | 42,58 % |

Hier bleibt Separate ueber den ganzen Bereich vorn und der Abstand waechst.
Die Umkehrung bei k = 80 gilt also nur fuer die ligandenintrinsischen Checks.

### Komplexe, fuer die auch 80 Versuche nicht reichen

| Arm | ohne Protein | mit Protein |
|---|---:|---:|
| Minimal | 10,53 % | 47,85 % |
| Separate | 12,44 % | 44,02 % |
| SigmaDock | **21,53 %** | **57,42 %** |

Fuer jeden fuenften Komplex erzeugt SigmaDock in achtzig Versuchen keine
einzige chemisch saubere Pose, die Flow-Arme fuer jeden achten bis zehnten.

## Auswahl nach gnina/Vinardo im Paper-Setup, 80 Seeds

Die Affinitaet kennt die PoseBusters-Checks nicht; "nach Affinitaet ranken und
dann Validitaet messen" ist deshalb nicht zirkulaer. Die Auswertung gegen den
RMSD folgt, sobald die SDF-Dateien der Seeds 40-79 lokal sind.

**Vorzeichen:** bei Vinardo ist negativ gut. Median hier +8, Maximum +293
(Clashes), nur rund 19 % der Posen sind negativ.

### Zielgroesse PB-valid ohne Protein: der Scorer hilft kaum

| k | Min Top-1 | Sep Top-1 | SD Top-1 |
|---:|---:|---:|---:|
| Zufall | 33,71 | 34,89 | 28,28 |
| 5 | 34,76 | 36,76 | 28,75 |
| 20 | 35,50 | 37,72 | 27,66 |
| 80 | 37,32 | 37,80 | **27,27** |

Trefferquote (Top1 - Random)/(Oracle - Random) bei k = 80: **6,5 %** fuer
Minimal, **5,5 %** fuer Separate, **-2,0 %** fuer SigmaDock.

**Bei SigmaDock schadet die Auswahl.** Ab k = 20 liegt Top-1 unter dem
Zufallsniveau: die Pose mit der besten Vinardo-Affinitaet ist dort im Mittel
chemisch schlechter als eine zufaellig gezogene. Das ist plausibel, weil
Vinardo enge Kontakte belohnt und ein Teil der Ligandenchemie-Checks
(`internal_steric_clash`, `bond_lengths`, `internal_energy`) genau bei
gequetschten Konformationen bricht.

### Zielgroesse PB-valid mit Protein: der Scorer hilft stark

| k | Min Top-1 | Sep Top-1 | SD Top-1 |
|---:|---:|---:|---:|
| Zufall | 6,62 | 7,17 | 5,84 |
| 5 | 15,32 | 16,76 | 12,26 |
| 20 | 22,41 | 24,13 | 17,03 |
| 80 | 28,71 | **29,19** | 20,57 |

Trefferquote: **48,5 / 45,1 / 40,1 %**.

**Dieser Befund ist mit Vorsicht zu lesen.** Vinardo ist eine Funktion der
Protein-Ligand-Abstaende, und die neun Protein-Checks von PoseBusters
(`minimum_distance_to_protein`, `volume_overlap_with_protein`, ...) sind es
auch. Die beiden Groessen sind nicht identisch, aber sie schauen auf dasselbe
Merkmal. Der Sprung von 6,62 auf 28,71 % misst daher zu einem Teil, dass zwei
Clash-Detektoren uebereinstimmen -- nicht, dass der Scorer die *richtige* Pose
findet. Die saubere Frage bleibt die Auswertung gegen den RMSD.

### Was daraus fuer die Empfehlung folgt

Separate fuehrt je Ziehung (34,89 %), unter Auswahl (37,80 %) und auf der
Oracle-Kurve mit Protein. Der Vorsprung ueberlebt also die Auswahl -- anders
als im `bound`-Fall, wo Separate zwar die beste Oracle-Kurve hatte, unter
gnina-Ranking aber hinter Minimal zurueckfiel. **Im Paper-Setup ist der
Vorteil von EXP-110 kein reiner Oracle-Effekt.**

Die einzige Ausnahme ist Oracle@80 ohne Protein, wo Minimal fuehrt. Das ist
ein Vielfaltseffekt und betrifft ein Szenario, in dem achtzig Posen erzeugt
und alle geprueft werden.

## Schrittzahl: die Korrektur zum 5-gegen-25-Vergleich (2026-08-24)

**"Fuenf Schritte genuegen" ist so nicht haltbar.** Der Schluss vom 23.08. kam
aus den RMSD-Zahlen, und diese Zahlen stimmen -- aber der RMSD war die einzige
Kenngroesse, bei der es zutrifft. Auf jeder anderen verlieren **alle drei
Arme** signifikant, auch die Flow-Arme.

40 Seeds je Zelle, 8360 Posen, `bound`, RMSD symmetriekorrigiert.

| Kenngroesse | Min 5 | Min 25 | Sep 5 | Sep 25 | SD 5 | SD 25 |
|---|---:|---:|---:|---:|---:|---:|
| RMSD < 2 A | 4,84 | 5,32 | 5,49 | 5,79 | **0,73** | 10,33 |
| PB-valid ohne Protein | 16,45 | 34,14 | 16,78 | 35,22 | **5,44** | 28,64 |
| < 2 A und valid mit Prot. | 0,84 | 1,27 | 0,85 | 1,30 | **0,12** | 1,72 |

Gepaart, 5 minus 25 Schritte: alles signifikant schlechter, mit einer
Ausnahme -- der RMSD von Separate (-0,30 pp, p = 0,27).

**Was faellt und was bleibt.** Es faellt: die Behauptung, fuenf Schritte
kosteten nichts. Die Validitaet halbiert sich bei den Flow-Armen
(-17,69 und -18,43 pp) und faellt bei SigmaDock um den Faktor fuenf.

Es bleibt: der *relative* Vorteil von Flow Matching. SigmaDock verliert auf
der Trefferquote 93 % seines Wertes, die Flow-Arme 5 bis 9 %. Die Aussage
lautet also nicht "fuenf Schritte genuegen", sondern **"wer die Schrittzahl
senken muss, verliert mit Flow Matching sehr viel weniger"**.

### Rechenaufwand

Netzwerkauswertungen: exakt ein Fuenftel (25 statt 5 Euler-Schritte, je eine
Auswertung).

Wall-Clock: **nicht** ein Fuenftel. Aus den gemessenen Laufzeiten
(nfe 5: 398 s, nfe 200: 8591 s je Seed, CPU) ergibt sich
Steigung = (8591 - 398)/195 = 42,0 s je Schritt und Grundlast = 188 s.
Damit 25 Schritte = 1238 s, 5 Schritte = 398 s -- eine Ersparnis von **68 %**,
nicht 80 %. Die Grundlast (Datenaufbau, Modell laden, Nachbearbeitung)
skaliert nicht mit der Schrittzahl.

## Voller Metriksatz und Auswahl: Paper-Setup gegen fuenf Schritte (2026-08-24)

Zwei Auswertungskonfigurationen, je 209 Komplexe x 40 Seeds = 8360 Posen je
Arm. `build_thesis_datasets.py sampled` und `build_thesis_datasets.py nfe5`.

`sampled` = `graph.sample_conformer=true`, 25 Schritte -- die Vorgabe des
Papers. `nfe5` = `bound`, 5 Schritte. Die Kreuzung (sampled x 5 Schritte)
existiert nicht.

### Anteil je Ziehung, mit 95%-Wilson-Intervall

| Kenngroesse | Min sampled | Sep sampled | SD sampled | Min nfe5 | Sep nfe5 | SD nfe5 |
|---|---:|---:|---:|---:|---:|---:|
| PB-valid ohne Protein | 33,85 | **34,96** | 28,19 | 16,45 | **16,78** | 5,44 |
| PB-valid mit Protein | 6,61 | **6,94** | 5,60 | 4,51 | **4,57** | 0,62 |
| RMSD < 1 A | 0,32 | 0,35 | 0,36 | 0,29 | **0,33** | 0,07 |
| RMSD < 2 A | 5,33 | **5,85** | 5,80 | 4,84 | **5,49** | 0,73 |
| RMSD < 2,5 A | 11,20 | **11,96** | 11,85 | 10,92 | **11,60** | 1,63 |
| RMSD < 3 A | 19,74 | **20,38** | 19,83 | 18,80 | **20,85** | 3,09 |
| < 2 A UND valid ohne Prot. | 3,84 | **4,43** | 3,22 | 2,21 | **2,34** | 0,36 |
| < 2 A UND valid mit Prot. | 1,15 | **1,48** | 0,94 | 0,84 | **0,85** | 0,12 |

**Separate fuehrt im Paper-Setup auf sieben von acht Kenngroessen.** Die
Ausnahme ist RMSD < 1 A, wo SigmaDock mit 0,36 gegen 0,35 % vorn liegt --
innerhalb der Intervalle, also kein Unterschied.

**Bei fuenf Schritten fuehrt Separate auf allen acht.** SigmaDock ist dort
nicht knapp schlechter, sondern um Faktoren: Faktor 7,5 auf RMSD < 2 A,
Faktor 3 auf die Ligandenchemie, Faktor 7 auf die gemeinsame Kenngroesse.

### Oracle@k, Paper-Setup

| Kenngroesse | k | Minimal | Separate | SigmaDock |
|---|---:|---:|---:|---:|
| PB-valid ohne Protein | 5 | 60,33 | 60,58 | 47,97 |
| | 40 | 82,30 | **85,17** | 70,33 |
| PB-valid mit Protein | 5 | 20,07 | 20,46 | 16,19 |
| | 40 | **44,98** | 44,50 | 35,89 |
| RMSD < 2 A | 5 | 20,76 | 21,37 | 21,55 |
| | 40 | **59,81** | 59,33 | 57,42 |
| < 2 A UND valid mit Prot. | 5 | 4,88 | **6,30** | 4,06 |
| | 40 | 20,57 | **22,49** | 16,75 |

Auf dem RMSD allein laufen die drei Kurven praktisch aufeinander -- der
Unterschied der Verfahren liegt in der Chemie, nicht in der Platzierung. Auf
der gemeinsamen Kenngroesse fuehrt Separate ueber den ganzen k-Bereich.

### Oracle@k bei fuenf Schritten (ohne gnina, diese Posen wurden nie gescort)

| Kenngroesse | k | Minimal | Separate | SigmaDock |
|---|---:|---:|---:|---:|
| PB-valid ohne Protein | 40 | **58,85** | 56,94 | 27,27 |
| PB-valid mit Protein | 40 | 32,54 | **34,93** | 11,00 |
| RMSD < 2 A | 40 | **57,89** | 57,42 | 16,27 |
| < 2 A UND valid mit Prot. | 40 | 14,83 | **17,70** | 2,87 |

Auf `RMSD < 2 A` erreichen die Flow-Arme mit 40 Versuchen 57-58 %, praktisch
denselben Wert wie bei 25 Schritten (59,3-59,8 %). **SigmaDock kommt auf
16,27 % statt 57,4 %.** Der Zusammenbruch bei grober Integration ist also
nicht durch mehr Versuche zu heilen.

### Auswahl nach gnina/Vinardo, Paper-Setup, Zielgroesse RMSD < 2 A

| Ranker | Min k=40 | Sep k=40 | SD k=40 |
|---|---:|---:|---:|
| Zufall | 5,34 | 5,86 | 5,75 |
| **Affinitaet (Vinardo)** | 20,10 | **25,36** | 18,66 |
| PB alle Checks | 13,22 | 13,23 | 11,44 |
| PB nur intrinsisch | 7,77 | 8,48 | 7,36 |
| PB nur Protein | 10,83 | 11,20 | 11,19 |
| Oracle | 59,81 | 59,33 | 57,42 |

Trefferquote (Top1 - Random)/(Oracle - Random) bei k = 40, Affinitaet:
**27,1 % / 36,5 % / 25,0 %**.

**Das ist die Umkehrung gegenueber `bound`.** Dort hatte Separate die beste
Oracle-Kurve, fiel unter gnina-Ranking aber hinter Minimal zurueck -- die
guten Posen waren da, der Scorer fand sie nicht. Im Paper-Setup ist es
umgekehrt: bei praktisch identischer Oracle-Kurve (59,33 gegen 59,81 %) holt
der Scorer aus Separate **25,36 %** heraus und aus Minimal nur 20,10 %.

Separates Posen sind also nicht nur haeufiger richtig, sie sind auch
**besser rankbar**. Das ist die praktisch entscheidende Eigenschaft, weil ein
Anwender keine Orakelauswahl hat.

Die PB-basierten Ranker liegen durchgehend hinter der Affinitaet. Bemerkenswert
ist `pb_intrinsic` mit 4,5-4,9 % Trefferquote: die Ligandenchemie sagt fast
nichts darueber aus, ob die Pose am richtigen Ort sitzt. Das ist die saubere
Bestaetigung, dass Validitaet und Genauigkeit zwei verschiedene Dinge messen.

### Ein Fehler in der Zufallsgrundlinie, behoben

Die `random`-Zeile im Rankervergleich schwankte zwischen 4,31 und 6,52 %,
obwohl das Zufallsniveau nicht von k abhaengen darf. Ursache: die
Zufallsmatrix wurde **einmal** vor der Wiederholungsschleife gezogen. Bei
k = NS waehlt `argmax` darauf in allen 400 Wiederholungen dieselbe Pose --
es wird nichts gemittelt, man sieht eine einzelne Ziehung.

Behoben durch `np.zeros_like(A[arm])` als Score: `top1()` addiert ohnehin je
Wiederholung frisches Rauschen zur Gleichstandsaufloesung, und genau das ist
die gesuchte Zufallsauswahl. Die Grundlinie ist jetzt flach (5,33-5,40 fuer
Minimal ueber alle k) und trifft den Ziehungsanteil.

Betroffen waren ausschliesslich die `random`-Zeile und die daraus berechneten
Trefferquoten, nicht die Top-1-Werte selbst.

### Was fuer die fuenf Schritte fehlt

Kein Ranking: diese Posen wurden nie mit gnina gescort. Und die Kreuzung
`sampled` x 5 Schritte wurde nie gesampelt -- offen bleibt damit, ob
SigmaDocks Zusammenbruch bei grober Integration teilweise am Prior haengt.
Beides ist nachholbar, beides ist CPU-Arbeit.

## Die praktische Kenngroesse unter realistischer Auswahl (2026-08-24)

Was ein Anwender tatsaechlich bekommt: 40 Posen erzeugen, die mit der besten
Vinardo-Affinitaet nehmen, und fragen, ob sie unter 2 A liegt **und** alle 24
PoseBusters-Checks besteht. Paper-Setup, 25 Schritte, 209 Komplexe x 40 Seeds.

| Arm | Top-1 nach Vinardo, k = 40 |
|---|---:|
| **Separate (EXP-110)** | **10,53 %** |
| Minimal | 9,09 % |
| SigmaDock | 5,26 % |

Gepaart ueber die 209 Komplexe, 2000 Bootstrap-Ziehungen:

| Vergleich | Differenz | 95-%-Intervall | p |
|---|---|---|---|
| Separate - SigmaDock | **+5,26 pp** | [+0,96; +9,57] | **0,024** |
| Minimal - SigmaDock | +3,83 pp | [-0,96; +8,61] | 0,146 |
| Separate - Minimal | +1,44 pp | [-3,35; +6,70] | 0,598 |

**Belegt ist Flow Matching gegen Diffusion, nicht Separate gegen Minimal.**
Der Abstand zwischen den beiden Flow-Armen verschwindet auf dieser Kenngroesse
im Rauschen. Und das Intervall ist knapp: die Untergrenze liegt bei +0,96 pp.
Mit 80 statt 40 Seeds wuerde der Effekt deutlich fester.

### Zum Trainingsbudget: walltime-gematcht, nicht schrittgematcht

| Arm | Job | Laufzeit | Epochen |
|---|---|---|---|
| Minimal | 8541310 | 11:06 | 6 |
| Separate | 8625634 | 11:10:59 | 6 |
| SigmaDock | 8541439 | 10:52 | 6 |

Rund 11 h in einem 12-h-Budget. "12h" bezeichnet das Budget, nicht die
Laufzeit. SigmaFlow endete bei `global_step` 13.750, SigmaDock bei 13.200 --
rund **4 % weniger Optimierungsschritte fuer SigmaDock**. Das schwaecht den
Befund nicht, gehoert aber als Fussnote dazu statt als "gleich" verbucht.

### Wie SigmaDock im Original rankt -- nachgelesen, nicht erinnert

`compute_ordering` (`statistics#Evaluationspipeline.py`, ~660-760) kennt vier
Modi: `None` (zufaellig), `vinardo`/`cnn`, `pb`, `heuristic`. Im Modus
`vinardo`:

```python
if score_name in ["Affinity", "Intramolecular energy"]:
    reverse = False     # aufsteigend -- negativ ist besser
```

Das ist zeichengenau der hier verwendete Ranker, Sortierrichtung
eingeschlossen. `conf/sampling/base.yaml:84` setzt `scoring: vinardo`, das
README nennt "GNINA/Vinardo and PoseBusters" die **Vorgaben**.

**Die Einschraenkung, die in die Arbeit gehoert:** `score_name` ist in keiner
Konfigurationsdatei des Repositories festgelegt, und `run_permutation_topk` --
die Funktion, die das Ranking anstoesst -- hat dort **keinen Aufrufer**. Die
Zahlen des Papers stammen aus Code, der nicht mitgeliefert ist. Dass
"Affinity" gemeint ist, folgt daraus, dass `CNNscore` zum Modus `cnn` gehoert
und `Affinity` der einzige Vinardo-Wert in `GNINA_METRICS` ist. Das ist eine
begruendete Rekonstruktion, kein Beleg.

Formulierbar ist: "nach der Vorgabe des Originals, GNINA/Vinardo-Affinitaet,
aufsteigend sortiert". Nicht formulierbar ist: "genau wie im Paper".

Dasselbe gilt fuer `score_bias` und `pb_exponent` im Modus `heuristic` -- auch
sie haben im Repository keine Vorgabewerte. Deshalb das Parametergitter statt
eines Wertes; ein nachtraeglich ausgewaehltes bestes Feld waere wertlos.

### Zwei Korrekturen an der Gesamttabelle

**k = 80 fehlte bei `sampled` nicht, es war nur nicht eingezogen.** Die
gnina-Scores decken 0..79 ab; blockiert ist allein der RMSD. Fuer die reinen
Validitaetsgroessen steht k = 80 aus `papersetup80.py` zur Verfuegung und ist
jetzt eingebunden, mit Kennzeichnung "80 Seeds" in der Zeilenbeschriftung.

| Kenngroesse | Arm | Zufall | Top-1 | Oracle@80 |
|---|---|---:|---:|---:|
| PB-valid ohne Protein | Minimal | 33,71 | 37,32 | **89,47** |
| | Separate | 34,89 | **37,80** | 87,56 |
| | SigmaDock | 28,28 | 27,27 | 78,47 |
| PB-valid mit Protein | Minimal | 6,62 | 28,71 | 52,15 |
| | Separate | 7,17 | **29,19** | **55,98** |
| | SigmaDock | 5,84 | 20,57 | 42,58 |

**Die Zelle `sampled` x 5 Schritte existiert doch.** Sie liegt auf ARC:
40 Seeds, 8360 SDF und 40 `predictions.pt` je Arm. Ich hatte sie faelschlich
als leer gemeldet, weil meine Pruefung `seed_*` direkt unter dem
Modellverzeichnis suchte statt unter `results/posebusters/last/`. Dieselbe
Pruefung meldete auch fuer die bekannt vollen Baeume null Seeds -- das haette
auffallen muessen.

## Die vierte Zelle: Schrittzahl und Prior sind unabhaengig (2026-08-24)

`sampled` x 5 Schritte ist gerechnet. Damit ist der Versuchsplan
vollstaendig, jedenfalls fuer die Validitaet (die RMSD-Haelfte wartet noch
auf die SDF-Dateien).

Rechnung: `vierzellen_validitaet.py`. Alle vier Zellen auf 40 Seeds
beschnitten, damit der Vergleich nicht daran haengt, dass eine Zelle die
doppelte Datenbasis hat. 209 Komplexe x 40 Seeds x 3 Arme je Zelle.

### PB-valid ohne Protein, alle vier Zellen

| Zelle | Minimal | Separate | SigmaDock |
|---|---:|---:|---:|
| 25 Schritte, bound | 34,14 % | **35,22 %** | 28,64 % |
| 25 Schritte, sampled | 33,85 % | **34,96 %** | 28,19 % |
| 5 Schritte, bound | 16,45 % | **16,78 %** | 5,44 % |
| 5 Schritte, sampled | 16,28 % | **17,00 %** | 5,16 % |

### Die entscheidende Frage: ist der Zusammenbruch ein Prior-Effekt?

Bei `bound` ist SigmaDocks Rotationsprior informativ, weil die
Identitaetsrotation die richtige Antwort ist. Denkbar waere gewesen, dass ein
Teil seines Absturzes bei fuenf Schritten daher ruehrt, dass dieser Vorteil
bei grober Integration nicht mehr zur Geltung kommt. Der Test ist eine
Wechselwirkung: faellt der Abfall in beiden Konformerquellen gleich aus?

| Konformer | Arm | 25 Schritte | 5 Schritte | Abfall | Faktor |
|---|---|---:|---:|---:|---:|
| bound | Minimal | 34,14 % | 16,45 % | -17,69 pp | 2,08x |
| bound | Separate | 35,22 % | 16,78 % | -18,43 pp | 2,10x |
| bound | **SigmaDock** | 28,64 % | 5,44 % | **-23,19 pp** | **5,26x** |
| sampled | Minimal | 33,85 % | 16,28 % | -17,57 pp | 2,08x |
| sampled | Separate | 34,96 % | 17,00 % | -17,97 pp | 2,06x |
| sampled | **SigmaDock** | 28,19 % | 5,16 % | **-23,04 pp** | **5,47x** |

Alle sechs Abfaelle p < 0,00025, gepaart ueber die 209 Komplexe.

**Die Wechselwirkung ist null.**

| Arm | Abfall bound | Abfall sampled | Unterschied |
|---|---:|---:|---:|
| Minimal | -17,69 pp | -17,57 pp | +0,12 pp |
| Separate | -18,43 pp | -17,97 pp | +0,47 pp |
| SigmaDock | -23,19 pp | -23,04 pp | +0,16 pp |

Bei allen drei Armen unterscheiden sich die Abfaelle um weniger als einen
halben Prozentpunkt. **Der Zusammenbruch bei grober Integration ist ein
reiner Integrationseffekt und hat mit dem Rotationsprior nichts zu tun.**

Damit ist die Robustheitsaussage **unkonfundiert** -- als einzige der drei
Kernaussagen dieser Arbeit, ohne Vorbehalt zur Auswertungskonfiguration.

### Mit Protein: dasselbe Bild, noch schaerfer

| Konformer | Arm | 25 Schritte | 5 Schritte | Faktor |
|---|---|---:|---:|---:|
| bound | Minimal | 6,63 % | 4,51 % | 1,47x |
| bound | Separate | 6,77 % | 4,57 % | 1,48x |
| bound | **SigmaDock** | 6,51 % | 0,62 % | **10,46x** |
| sampled | Minimal | 6,61 % | 4,02 % | 1,65x |
| sampled | Separate | 6,94 % | 4,96 % | 1,40x |
| sampled | **SigmaDock** | 5,60 % | 0,61 % | **9,18x** |

Auf der Kenngroesse, die praktisch zaehlt, verliert SigmaDock bei fuenf
Schritten rund neun Zehntel seines Werts, die Flow-Arme etwa ein Drittel.
Die Wechselwirkung bleibt klein (hoechstens 0,90 pp bei SigmaDock).

### Was das fuer die Formulierung bedeutet

Die Aussage lautet: **bei einem Fuenftel der Netzwerkauswertungen behaelt Flow
Matching rund die Haelfte seiner Ligandenchemie, Diffusion ein Fuenftel.**
Sie gilt unabhaengig von der Konformerquelle, ist also nicht durch den
Rotationsprior erklaerbar.

Nicht formulierbar bleibt: "fuenf Schritte kosten nichts". Alle drei Arme
verlieren signifikant, und zwar auf jeder Kenngroesse ausser dem RMSD.

Der Mechanismus ist strukturell: der Flow-Matching-Pfad ist die Geodaete
zwischen Quelle und Ziel, also nahezu gerade -- Euler trifft sie mit wenigen
Schritten. SigmaDocks Rueckwaertsprozess integriert ueber einen EDM-Rauschplan
von sigma = 1,5 bis t_min = 0,005 mit rho = 3; bei fuenf Schritten sind die
Spruenge in sigma zu gross, und ein Euler-Schritt bei hohem Rauschpegel
multipliziert den Score mit einem grossen Faktor.

## Der volle Metriksatz der vierten Zelle (2026-08-24)

`sampled` x 5 Schritte ist jetzt vollstaendig, RMSD eingeschlossen. Damit ist
der Versuchsplan geschlossen. 209 Komplexe x 40 Seeds = 8360 Posen je Arm und
Zelle, RMSD symmetriekorrigiert (spyrmsd, `best_copy`).

Datensaetze: `per_draw_sampled5_40seeds.csv`,
`selection_curves_sampled5_40seeds.csv`.

### Anteil je Ziehung, Paper-Setup, 25 gegen 5 Schritte

| Kenngroesse | Arm | 25 Schritte | 5 Schritte | Differenz | Faktor | p |
|---|---|---:|---:|---:|---:|---|
| **RMSD < 2 A** | Minimal | 5,33 % | **5,42 %** | +0,08 pp | 0,98x | 0,73 |
| | Separate | 5,85 % | **5,51 %** | -0,33 pp | 1,06x | 0,20 |
| | SigmaDock | 5,80 % | **0,54 %** | -5,26 pp | **10,78x** | < 0,00025 |
| **PB-valid ohne Prot.** | Minimal | 33,85 % | 16,28 % | -17,57 pp | 2,08x | < 0,00025 |
| | Separate | 34,96 % | 17,00 % | -17,97 pp | 2,06x | < 0,00025 |
| | SigmaDock | 28,19 % | 5,16 % | -23,04 pp | **5,47x** | < 0,00025 |
| **PB-valid mit Prot.** | Minimal | 6,61 % | 4,02 % | -2,60 pp | 1,65x | < 0,00025 |
| | Separate | 6,94 % | 4,96 % | -1,97 pp | 1,40x | < 0,00025 |
| | SigmaDock | 5,60 % | 0,61 % | -4,99 pp | **9,18x** | < 0,00025 |
| **< 2 A UND valid o.P.** | Minimal | 3,84 % | 2,57 % | -1,27 pp | 1,49x | < 0,00025 |
| | Separate | 4,43 % | 2,63 % | -1,79 pp | 1,68x | < 0,00025 |
| | SigmaDock | 3,22 % | 0,30 % | -2,92 pp | **10,76x** | < 0,00025 |
| **< 2 A UND valid m.P.** | Minimal | 1,15 % | 0,80 % | -0,35 pp | 1,43x | 0,0075 |
| | Separate | 1,48 % | 1,00 % | -0,48 pp | 1,48x | 0,0005 |
| | SigmaDock | 0,94 % | 0,06 % | -0,89 pp | **15,80x** | < 0,00025 |

### Der schaerfste Einzelbefund der Arbeit

**Auf der Platzierungsgenauigkeit verlieren die Flow-Arme bei fuenf Schritten
NICHTS.** Minimal 5,33 -> 5,42 % (p = 0,73), Separate 5,85 -> 5,51 %
(p = 0,20). Beide Aenderungen sind ununterscheidbar von null.

**SigmaDock verliert den Faktor 10,8.** 5,80 -> 0,54 %, p < 0,00025.

Das ist die klarste Trennung der beiden Verfahren in dieser Arbeit, und sie
steht im Paper-Setup, also ohne den Prior-Vorbehalt. Bei `bound` war derselbe
Befund vom Prior mitverdeckt: dort fiel SigmaDock von 10,33 auf 0,73 %, aber
die 10,33 waren selbst ein Artefakt. Hier fallen 5,80 auf 0,54 -- Ausgangs-
und Endwert sind beide unverfaelscht.

Der Mechanismus ist strukturell. Der Flow-Matching-Pfad ist die Geodaete
zwischen Quelle und Ziel, also nahezu gerade -- Euler trifft sie mit fuenf
Schritten so gut wie mit fuenfundzwanzig. SigmaDocks Rueckwaertsprozess
integriert ueber einen EDM-Rauschplan von sigma = 1,5 bis t_min = 0,005 mit
rho = 3; bei fuenf Schritten sind die Spruenge in sigma zu gross, und ein
Euler-Schritt bei hohem Rauschpegel multipliziert den Score mit einem grossen
Faktor.

**Die Ligandenchemie faellt dagegen bei allen dreien.** Sie haengt nicht an
der Genauigkeit der Platzierung, sondern an der Zahl der Korrekturschritte,
die das Netz auf die innere Geometrie anwenden kann. Fuenf Schritte genuegen
fuer den Ort, nicht fuer die Chemie.

### Zwischen den Armen bei fuenf Schritten

| Kenngroesse | Sep - Min | Min - SD | Sep - SD |
|---|---|---|---|
| RMSD < 2 A | +0,10 pp, p = 0,76 | **+4,88 pp**, p < 0,00025 | **+4,98 pp**, p < 0,00025 |
| PB-valid ohne Prot. | +0,72 pp, p = 0,17 | **+11,12 pp**, p < 0,00025 | **+11,84 pp**, p < 0,00025 |
| PB-valid mit Prot. | **+0,94 pp, p = 0,0015** | **+3,41 pp**, p < 0,00025 | **+4,35 pp**, p < 0,00025 |
| < 2 A UND valid o.P. | +0,06 pp, p = 0,84 | **+2,27 pp**, p < 0,00025 | **+2,33 pp**, p < 0,00025 |
| < 2 A UND valid m.P. | +0,20 pp, p = 0,23 | **+0,74 pp**, p < 0,00025 | **+0,94 pp**, p < 0,00025 |

**Separate schlaegt Minimal bei fuenf Schritten auf der Proteinvaliditaet**
(+0,94 pp, p = 0,0015). Bei 25 Schritten war derselbe Vergleich nicht
signifikant. Der Zwei-Kopf-Ansatz vertraegt grobe Integration also besser --
das ist der dritte eigenstaendige Effekt von EXP-110, nach der Ligandenchemie
und der besseren Rankbarkeit.

Auf allen uebrigen Kenngroessen bleiben die beiden Flow-Arme auch bei fuenf
Schritten ununterscheidbar.

**Gegen SigmaDock ist bei fuenf Schritten jede einzelne Kenngroesse
signifikant, alle p < 0,00025.** Bei 25 Schritten war das nur fuer die
Ligandenchemie so.

### Praktische Lesart

Wer die Rechenzeit drittelt (fuenf statt fuenfundzwanzig Schritte, 32 %
Wall-Clock-Ersparnis, exakt ein Fuenftel der Netzwerkauswertungen), bekommt
mit Flow Matching dieselbe Trefferquote und rund die Haelfte der
Ligandenchemie. Mit Diffusion bekommt er nichts Brauchbares mehr:
0,54 % Trefferquote und 0,06 % auf der gemeinsamen Kenngroesse.

## Ein Fuenftel des Aufwands gegen den vollen Fahrplan (2026-08-24)

SigmaFlow mit fuenf Integrationsschritten gegen SigmaDock mit
fuenfundzwanzig, beides im Paper-Setup. 209 Komplexe x 40 Seeds, gepaart
ueber die Komplexe.

**Der Vergleich ist bewusst NICHT compute-gematcht.** Er ist unfair zulasten
von SigmaFlow: ein Fuenftel der Netzwerkauswertungen gegen den vollen
Fahrplan des Originals. Genau so gehoert er auch berichtet -- sonst liest er
sich, als sei SigmaFlow schlicht besser.

### Anteil je Ziehung

| Kenngroesse | Minimal 5 | Separate 5 | SigmaDock 25 |
|---|---:|---:|---:|
| RMSD < 2 A | 5,42 % | 5,51 % | 5,80 % |
| PB-valid ohne Protein | 16,28 % | 17,00 % | **28,19 %** |
| PB-valid mit Protein | 4,02 % | 4,96 % | 5,60 % |
| < 2 A und valid ohne Prot. | 2,57 % | 2,63 % | **3,22 %** |
| < 2 A und valid mit Prot. | 0,80 % | **1,00 %** | 0,94 % |

### Gepaart gegen SigmaDock 25

| Kenngroesse | Minimal 5 | Separate 5 |
|---|---|---|
| RMSD < 2 A | -0,38 pp, p = 0,30 | -0,29 pp, p = 0,46 |
| PB-valid ohne Protein | **-11,91 pp**, p < 0,00025 | **-11,20 pp**, p < 0,00025 |
| PB-valid mit Protein | **-1,58 pp**, p = 0,0010 | -0,63 pp, p = 0,20 |
| < 2 A und valid ohne Prot. | **-0,65 pp**, p = 0,0055 | **-0,59 pp**, p = 0,039 |
| < 2 A und valid mit Prot. | -0,14 pp, p = 0,34 | **+0,06 pp, p = 0,79** |

### Gleichstand auf Genauigkeit und auf der gemeinsamen Kenngroesse

Beide Flow-Arme mit fuenf Schritten sind auf `RMSD < 2 A` von SigmaDock mit
fuenfundzwanzig nicht zu unterscheiden. Bei Oracle@40 treffen sich Separate 5
und SigmaDock 25 exakt: **57,42 % gegen 57,42 %.**

Auf `< 2 A UND PB-valid mit Protein` ebenfalls Gleichstand, mit Separate
nominell vorn (1,00 gegen 0,94 %, p = 0,79). Oracle@40: 16,27 gegen 16,75 %.

**EXP-110 erreicht mit einem Fuenftel der Netzwerkauswertungen dieselbe
Trefferquote und dieselbe kombinierte Kenngroesse wie SigmaDock mit dem
vollen Fahrplan.**

### Wo SigmaFlow bei fuenf Schritten verliert

Die Ligandenchemie, rund elf Punkte, p < 0,00025. Konsistent mit dem Befund
der vierten Zelle: fuenf Schritte genuegen fuer den Ort, nicht fuer die
Chemie. Deshalb liegt SigmaDock auch auf `< 2 A und valid ohne Protein` vorn
(3,22 gegen 2,63 %, p = 0,039) -- dort schlaegt die Ligandenchemie durch,
waehrend sie mit Protein von den strengeren Kollisionspruefungen ueberdeckt
wird.

### Oracle@k, die wichtigsten Zeilen

| Kenngroesse | k | Minimal 5 | Separate 5 | SigmaDock 25 |
|---|---:|---:|---:|---:|
| RMSD < 2 A | 10 | 31,59 % | 31,48 % | 33,54 % |
| | 40 | 54,07 % | **57,42 %** | **57,42 %** |
| PB-valid mit Protein | 40 | 33,97 % | 35,41 % | 35,89 % |
| < 2 A und valid mit Prot. | 10 | 6,34 % | **7,15 %** | 6,97 % |
| | 40 | 14,35 % | 16,27 % | 16,75 % |

### Einordnung

Der Fuenf-Schritte-Vergleich ist nicht der staerkste Befund der Arbeit, aber
der eindruecklichste. Bei gleicher Schrittzahl fuehrt SigmaFlow auf derselben
Kenngroesse deutlicher: Separate 1,48 gegen SigmaDock 0,94 % bei je
fuenfundzwanzig Schritten.

### Datenbasis verdoppelt

Die RMSD-Werte fuer das Paper-Setup bei 25 Schritten liegen jetzt fuer alle
80 Seeds vor (je 16720 Posen). Die ueberlappenden 8360 Posen stimmen mit den
frueheren Tabellen exakt ueberein -- **null Abweichungen**, die Erweiterung
ist additiv.

## Rankbarkeit bei fuenf Schritten (2026-08-24, spaet)

Die gnina-Scores fuer `sampled` x 5 liegen vor (Jobs 8638560-62, je 8360
Posen, `sampling_root` gegen das Slurm-Log geprueft). Damit ist die letzte
offene Spalte des Versuchsplans gefuellt.

### Der Rankervergleich, Zielgroesse RMSD < 2 A, k = 40

| Ranker | Min 25 | Sep 25 | SD 25 | Min 5 | Sep 5 | SD 5 |
|---|---:|---:|---:|---:|---:|---:|
| Zufall | 5,40 | 5,82 | 6,19 | 5,46 | 5,54 | 0,54 |
| **Affinitaet (Vinardo)** | 20,79 | **23,45** | 20,65 | 20,10 | **22,97** | 2,87 |
| PB alle Checks | 12,93 | 14,40 | 13,73 | 13,06 | 14,72 | 1,51 |
| PB nur intrinsisch | 7,79 | 8,86 | 8,42 | 9,15 | 9,68 | 1,87 |
| PB nur Protein | 10,39 | 11,87 | 12,15 | 9,65 | 9,55 | 0,80 |
| Oracle | 57,36 | 61,05 | 59,07 | 54,07 | 57,42 | 12,44 |

Trefferquote `(Top1 - Random)/(Oracle - Random)` bei Affinitaet:

| | Minimal | Separate | SigmaDock |
|---|---:|---:|---:|
| 25 Schritte | 29,6 % | **31,9 %** | 27,4 % |
| 5 Schritte | 30,1 % | **33,6 %** | 19,6 % |

**Die Rankbarkeit ueberlebt die grobe Integration.** Separate liefert mit
fuenf Schritten unter gnina-Auswahl **22,97 %** gegen 23,45 % mit
fuenfundzwanzig -- praktisch unveraendert. Die Trefferquote steigt sogar
leicht (33,6 gegen 31,9 %), weil die Oracle-Obergrenze faellt, der Scorer
aber gleich viel davon hebt.

**Bei SigmaDock bricht sie ein**, von 27,4 auf 19,6 %. Der Scorer findet in
den grob integrierten Posen weniger.

### Der Vergleich, auf den es hinauslaeuft

Beide Zellen auf 40 Seeds beschnitten, damit `k = 40` in beiden dasselbe
bedeutet -- naemlich alle Seeds. Im 80-Seed-Datensatz waere `k = 40` ein
Mittel ueber zufaellige Haelften und damit eine andere Groesse.

**Zielgroesse RMSD < 2 A, Top-1 nach Vinardo**

| Konfiguration | Top-1@40 |
|---|---:|
| Separate, 25 Schritte | 25,36 % |
| Minimal, 25 Schritte | 20,10 % |
| **Separate, 5 Schritte** | **22,97 %** |
| SigmaDock, 25 Schritte | 18,66 % |
| SigmaDock, 5 Schritte | 2,87 % |

Gepaart ueber die 209 Komplexe:

| Vergleich | Differenz | 95-%-Intervall | p |
|---|---|---|---|
| Separate 5 - SigmaDock 25 | +4,31 pp | [-2,87; +11,48] | 0,24 |
| Minimal 5 - SigmaDock 25 | +1,44 pp | [-5,26; +8,13] | 0,73 |
| **Separate 5 - SigmaDock 5** | **+20,10 pp** | [+14,35; +25,84] | < 0,00025 |
| Separate 25 - SigmaDock 25 | +6,70 pp | [0,00; +13,40] | 0,066 |

**Zielgroesse `< 2 A UND valid mit Protein`**

| Konfiguration | Top-1@40 |
|---|---:|
| Separate, 25 Schritte | 10,53 % |
| Minimal, 25 Schritte | 9,09 % |
| **Separate, 5 Schritte** | **7,18 %** |
| SigmaDock, 25 Schritte | 5,26 % |
| SigmaDock, 5 Schritte | 0,96 % |

| Vergleich | Differenz | p |
|---|---|---|
| Separate 5 - SigmaDock 25 | +1,91 pp | 0,44 |
| **Separate 5 - SigmaDock 5** | **+6,22 pp** | < 0,00025 |
| **Separate 25 - SigmaDock 25** | **+5,26 pp** | 0,018 |

### Was daraus formulierbar ist

**Unter realistischer Auswahl liegt Separate mit fuenf Schritten nicht hinter
SigmaDock mit fuenfundzwanzig.** Nominell vorn (22,97 gegen 18,66 % bzw. 7,18
gegen 5,26 %), statistisch ununterscheidbar (p = 0,24 und 0,44). Das
Intervall ist breit, die Aussage lautet also "gleichauf", nicht "besser".

**Bei gleicher Schrittzahl ist der Abstand riesig:** +20,10 pp auf der
Trefferquote und +6,22 pp auf der gemeinsamen Kenngroesse, beide
p < 0,00025.

Die praktische Lesart: wer den Rechenaufwand auf ein Fuenftel der
Netzwerkauswertungen senkt, bekommt mit Flow Matching **dasselbe Ergebnis wie
mit dem vollen Fahrplan der Diffusionsvariante** -- und die Auswahl per
Affinitaet funktioniert dabei unvermindert. Mit Diffusion bei fuenf Schritten
bekommt er 2,87 statt 18,66 %.

### Einschraenkung

`Separate 25 - SigmaDock 25` auf `RMSD < 2 A` ist mit p = 0,066 nicht
signifikant, das Intervall beruehrt die Null. Auf der gemeinsamen Kenngroesse
mit Protein ist derselbe Vergleich signifikant (p = 0,018). Der
Rankingvorteil von EXP-110 steht also auf der kombinierten Groesse, nicht auf
der Genauigkeit allein.
