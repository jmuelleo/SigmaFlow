# SigmaFlow — Forschungsroadmap

> Zentrales Steuerungsdokument für den Rest der Thesis.
> Stand 2026-08-13. Alle Zahlen in diesem Dokument sind **in dieser Sitzung
> lokal gemessen** oder mit Quelle belegt. Nicht belegte Vermutungen sind als
> solche markiert.

---

## 1. Current State

| Komponente | Stand |
|---|---|
| `SigmaFlow_Minimal` | eingefroren, Tag `sigflow-minimal-baseline-v1`, Commit `267fb69` |
| Frame-Fix | verifiziert notwendig und korrekt (dreifach) |
| ODE-Integration | freigesprochen: Oracle-Feld rekonstruiert exakt; NFE 5→200 wirkungslos |
| Zeitgewichtung (Variante a) | getestet, **kein Effekt** |
| EXP-100 State Reparameterization | `LOCALLY VALIDATED — READY FOR ARC` |
| Bester Vergleich | 12h, volles PoseBusters (209 Komplexe), 10 Seeds je Methode |

### Leistungsstand bei gleichem Budget (12h, Seed 0 bzw. 10 Seeds)

| | SigmaFlow | SigmaDock |
|---|---|---|
| RMSD < 2 Å, Einzelseed (Mittel über 10) | 4.4 % | **9.8 %** |
| Median-RMSD | 4.78 Å | **4.48 Å** |
| **Best-of-10 (Oracle@10)** | 29.2 % | **45.9 %** |
| Median-RMSD Oracle@10 | 2.62 Å | **2.10 Å** |

SigmaFlow liegt bei diesem Budget **hinter** SigmaDock. Das ist der
Ausgangspunkt, nicht ein Detail.

---

## 2. Diagnosed Weaknesses

### 2.1 Die Rotation trägt derzeit **keine** Information — gemessen

Rotationsfehler je starrem Fragment (711 Fragmente, 209 Komplexe, Seed 0),
gegen die Haar-Referenz einer reinen Zufallsrotation:

| | SigmaFlow | SigmaDock | Haar (Zufall) |
|---|---|---|---|
| Median-Rotationsfehler | **138.2°** | 123.8° | 132.3° |
| Anteil < 30° | **2.1 %** | 5.8 % | ~2 % |
| Anteil < 60° | **8.4 %** | 15.8 % | ~10 % |

> **SigmaFlows Rotationskanal ist statistisch nicht von Raten zu unterscheiden.**
> SigmaDock erreicht bei < 30° eine ~3-fache Anreicherung über Zufall, SigmaFlow
> keine. Bei identischer Architektur, identischen Daten und identischem Budget.

Das ist der wichtigste Einzelbefund dieses Dokuments. Er verschiebt die
Priorität aller nachfolgenden Ideen.

### 2.2 Die Translation trägt sehr wohl Information

| | SigmaFlow | SigmaDock |
|---|---|---|
| Fragmente mit \|Δt\| < 2 Å | 22.6 % | 29.3 % |
| Fragmente mit \|Δt\| < 1 Å | 8.2 % | 9.2 % |

Die Behauptung „Translation funktioniert besser als Rotation" ist damit
**bestätigt und quantifiziert** — und zwar als *qualitativer* Unterschied
(Signal vs. kein Signal), nicht als Gradunterschied.

### 2.3 Mechanismus: ein Kraftfeld, zwei sehr ungleiche Kanäle

Der Code hat **genau einen** Output-Block (`model.py:407 force_block`,
ein 3-Vektor je Ligandenatom). Beide Freiheitsgrade werden daraus abgeleitet:

```
f_i  (ein l=1-Vektor je Atom, EIN Kopf)
 ├─► F = Σ_i f_i                    ─► ΔT = F / m          Translation
 └─► τ = Σ_i (r_i − c) × f_i        ─► ω  = I⁻¹ τ          Rotation
```

Das sind zwei **orthogonale Projektionen** desselben Feldes: ein räumlich
konstanter Anteil von `f` erzeugt exakt null Drehmoment (weil `Σ(r_i − c) = 0`),
ein rein rotatorisches Muster exakt null Kraft. Es gibt also *keinen*
Kapazitätskonflikt im naiven Sinn — aber die **Konditionierung** unterscheidet
sich drastisch. Gemessen auf den 711 realen Fragmentgeometrien:

| | Wert |
|---|---|
| Rauschverstärkung Rotation / Translation | **8.8×** |
| Rauschbudget Rotation relativ zu Translation (bei gleichem Relativfehler) | **0.25×** |
| … bei 3–4-Atom-Fragmenten (ρ = 1.09 Å) | **0.14×** |
| Median-Fragmentradius ρ | 1.52 Å |

Die Translation liest den **Mittelwert** von `f` — eine Mittelung, die Rauschen
mit 1/√n dämpft. Die Rotation liest das **erste Moment der Abweichung vom
Mittelwert**, also genau den Anteil, den die Mittelung verwirft, skaliert mit
dem kurzen Hebelarm ρ ≈ 1.5 Å.

> **Ehrliche Einschränkung.** Dieser Mechanismus sagt voraus, dass *kleine*
> Fragmente die schlechteste Rotation haben sollten. Das ist in den Daten
> **nicht** sichtbar (SigmaFlow 135.8° klein vs. 141.7° groß; SigmaDock 123.5°
> vs. 125.6°). Bei einem Modell, dessen Rotation ohnehin auf Zufallsniveau
> liegt, fehlt allerdings der Dynamikbereich, um so einen Trend nachzuweisen.
> Die Vorhersage ist damit **nicht bestätigt** — sie bleibt offen.
> `PENDING`: Wiederholung an einem Modell mit tatsächlichem Rotationssignal.

### 2.4 Der FM-Rotationszielwert ist konstant in t — es gibt kein leichtes Regime

Aus der Pfaddefinition (`so3_flow_matcher.py`), analytisch:

```
R_t = R_0 exp(t·log(R_0ᵀR_1))
  ⇒  log(R_tᵀR_1) = (1−t)·log(R_0ᵀR_1)
  ⇒  u_t = log(R_tᵀR_1)/(1−t) = log(R_0ᵀR_1)     ← unabhängig von t
```

Das Ziel hat bei **jedem** t dieselbe Größe (im Mittel 126.5°, Haar). Bei
Diffusion ist das anders: der Score bei hohem Rauschen ist klein, das Modell
muss dort nur eine kleine Korrektur nennen und lernt das Schwere erst spät.
Unter Flow Matching wird bei t ≈ 0 sofort die **vollständige** Antwort verlangt.

**Daraus folgt eine konkrete, testbare Hypothese (H-Shrink):**

> Bei kleinem t ist `R_1` aus dem Zustand nicht bestimmbar. Der L2-optimale
> Ausgang ist dann der bedingte Mittelwert, und Mittelung von
> Rotations**vektoren** schrumpft gegen null. Ein geschrumpftes Feld erzeugt
> beim Integrieren fast keine Drehung — der Endzustand bleibt nahe `R_0 ~ Haar`.
> Genau das wird beobachtet (§2.1).

Die Translation hat dieses Problem **nicht**, weil sie einen informativen Prior
hat: `pocket_com` liegt im Ursprung, „zur Taschenmitte" ist auch bei t = 0 eine
brauchbare Antwort. Für die Rotation existiert kein Analogon — Haar hat keinen
Mittelwert.

**Diese Asymmetrie ist der eigentliche mechanistische Kern des Problems** und
sie verbindet §2.1, §2.3 und die Motivation für die konditionierte Quelle.

### 2.5 Symmetrie erklärt die Rotationsfehler **nicht** — gemessen

Die naheliegende Erklärung („53 % Graphautomorphismen ⇒ Kopf-Schwanz-Flips sind
Symmetrieartefakte") ist **falsch**. Gemessen auf Fragmentebene:

| | Wert |
|---|---|
| Fragmente mit echter, räumlich realisierbarer Drehsymmetrie | **4.8 %** (34 von 711) |
| Flip-Kandidaten (θ > 120°) | 467 (65.7 %) |
| davon durch Symmetrie erklärt (θ_sym < 30°) | **0 von 467 (0.0 %)** |
| Senkung des medianen Rotationsfehlers durch Symmetriekorrektur | **1.3°** |

Zwei Fehlschlüsse werden hier ausgeräumt:
1. Ein Automorphismus des *Moleküls* muss das *Fragment* nicht auf sich abbilden.
2. Selbst dann muss die Atompermutation nicht durch eine **räumliche Drehung**
   realisierbar sein — sonst ist sie bloß eine Umbenennung.

Die vorhandenen Symmetrieoperationen sind zudem 3-zählig (Median 120°,
81 % zwischen 90° und 150°), nur 19 % sind 180°-Flips. Sie betreffen fast nur
kleine Fragmente (3–4 Atome: 13 %, ab 7 Atomen: 0 %).

> Skript: `SigmaFlow_Variants/posebusters_full_comparison/symmetry_flip_analysis.py`

### 2.6 Oracle@10 ≫ Top-1 — aber die Lesart ist heikel

SigmaFlow 1.9 % → **29.2 %**, SigmaDock 9.1 % → **45.9 %**.

Die naheliegende Lesart „Ranking ist der Engpass" ist **nur zur Hälfte richtig**.
Wenn der Rotationskanal auf Zufallsniveau arbeitet, ist Best-of-10 im Kern
„zehnmal würfeln und das beste Ergebnis behalten". Ein hoher Oracle@K-Wert ist
dann **kein Beleg für einen guten Generator**, sondern teilweise ein Symptom
eines zufälligen. Beides gilt gleichzeitig:

- Ranking ist ein echter, großer, billiger Hebel (Faktor ~7 Kopfraum).
- Der Generator ist trotzdem der tiefere Engpass — SigmaDocks Oracle@10 liegt
  16.7 Prozentpunkte höher.

### 2.7 Was bereits ausgeschlossen ist

| Hypothese | Status |
|---|---|
| ODE-Integrationsmechanik | ausgeschlossen (Oracle exakt) |
| Zu wenige Integrationsschritte | ausgeschlossen (5→200 flach) |
| Globale Zeitgewichtung des Loss | getestet, **kein Effekt** |
| Loss-Gewichte begünstigen Translation | widerlegt (Gradientenmessung 1.84 zugunsten Rotation) |
| Frame-Konvention | gefixt und verifiziert |
| SigmaDock-Ausreißer / Bimodalität | Auswertungsartefakt, widerrufen |

---

## 3. FM-Specific Opportunities

### A1 — Conditional / Informative Source Distribution `[Kategorie A]`

**Motivation, jetzt mechanistisch statt vage.** Nach §2.4 schrumpft das
Rotationsfeld, weil `R_1` bei kleinem t aus einem Haar-verteilten Zustand nicht
bestimmbar ist. Eine konditionierte Quelle verkleinert die Streuung von `R_0`
und macht das Ziel damit **vorhersagbarer** — der Effekt liegt also nicht nur
darin, näher am Ziel zu starten, sondern darin, dass das Modell überhaupt etwas
zu lernen bekommt.

> Das ist der entscheidende Unterschied zur schwachen Version des Arguments
> („wir schenken dem Modell die Antwort"). Die starke Version lautet: die
> konditionierte Quelle **repariert die Lernbarkeit des Rotationskanals**.

**Was konditionieren?** Nach §2.1/§2.2 eindeutig **Rotation zuerst**. Die
Translation hat bereits einen informativen Prior (`pocket_com` im Ursprung) und
messbares Signal; dort ist wenig zu holen. Rotation hat keinen Prior und kein
Signal.

**Warum sollte SigmaDock davon weniger profitieren?** Weil bei Diffusion der
Prior durch den Vorwärtsprozess festgelegt ist. Man kann ihn nicht frei wählen,
ohne den Prozess selbst umzubauen. Genau das ist die FM-Freiheit.

**Was würde die Hypothese widerlegen?**
- EXP-101 zeigt, dass keine inferenzverfügbare Heuristik die mediane
  Rotationsdistanz spürbar unter 132.3° drückt.
- Oder: `q_φ` reduziert die Distanz, aber das trainierte Modell bleibt bei
  Zufallsniveau — dann ist nicht die Quelle das Problem, sondern der Kanal (§2.3).
- Oder: der Gewinn ist vollständig durch `q_φ` allein erklärbar (Kontrolle:
  `q_φ` ohne Flow auswerten).

**Realistische Effektgröße:** ehrlich unsicher. Wenn H-Shrink stimmt, potenziell
groß (Rotation von Zufall auf Signal). Wenn §2.3 dominiert, nahe null.
**Confidence: mittel.**

**Varianten**
| | Bewertung |
|---|---|
| **A** Hand-crafted geometrisch (Trägheitsachsen ↔ Taschenanisotropie) | erste Wahl. Keine Parameter, äquivariant, leakage-frei, interpretierbar. Ein positiver Effekt ist eindeutig der Idee zuzuschreiben, nicht der Kapazität. |
| **B** Kleines gelerntes Proposal | zweite Stufe. Nur wenn A trägt. |
| **C** Multimodal `Σ π_k q_k(R)` | dritte Stufe. Interessant, aber §2.5 nimmt der Symmetriemotivation die Grundlage; die Multimodalität müsste aus Taschengeometrie stammen, nicht aus Molekülsymmetrie. |
| **D** Klassische Docking-Pose (Vina/Smina) als Quelle | **wissenschaftlich problematisch.** Es entstünde „Vina → Verfeinerung", und der Beitrag von SigmaFlow wäre nur noch die Differenz zu Vina. Als *Kontrollexperiment* dagegen sehr wertvoll: es liefert die Obergrenze dessen, was eine gute Quelle bringen kann. **Empfehlung: als Oberschranken-Kontrolle fahren, nicht als Beitrag verkaufen.** |

### A2 — Docking-spezifischer Probability Path `[Kategorie B, nicht A]`

**Herabgestuft von A auf B, mit Begründung:** DiffDock benutzt bereits getrennte
Rauschzeitpläne für Translation, Rotation und Torsion. Getrennte Zeitpläne sind
also *nicht* FM-exklusiv. FM-natürlicher ist nur, dass man eine Komponente
buchstäblich **einfrieren** kann.

**Der einzige Mechanismus, der hier trägt** (und er ist gut): unter einem
gestaffelten Pfad wäre bei t = 0.6 die Translation bereits gelöst, und die
Orientierung müsste in einem **korrekt platzierten** Taschenkontext bestimmt
werden statt gleichzeitig mit einer falschen Platzierung. Das ist eine echte
Konditionierungsverbesserung.

**Was dagegen spricht:**
1. Eine Umparametrisierung `s_R(t)` ändert nur die **Magnitude** des Ziels, nicht
   seine **Richtung** — die Richtung ist bei jedem t die volle Antwort. Der
   Schwierigkeitskern bleibt.
2. Die globale Zeitgewichtung (Variante a) war bereits **wirkungslos**. Ein
   gestaffelter Pfad ist nicht dasselbe, aber es ist ein warnender Präzedenzfall.
3. Das Modell sieht `t` und könnte eine Staffelung selbst lernen.

**Urteil:** nicht „nur theoretisch hübsch", aber auch nicht der erste Hebel.
Mittlere Priorität. **Confidence: mittel-niedrig.**

### A3 — Source + Path Co-Design `[Kategorie A]`

Gemeinsame Optimierung von `q_φ` und `v_θ` mit einem Trajektorienterm
(Transportdistanz, Krümmung, NFE).

**Urteil: für diese Thesis zu groß.** Zwei gekoppelte, jeweils instabile
Lernprobleme, und die Zuschreibung eines Effekts wäre kaum sauber möglich. Als
Ausblickskapitel wertvoll. **Kategorie A, aber Priorität niedrig.**

### A4 — Few-Step / 1-Step SigmaFlow `[Kategorie B]`

Die NFE-Kurve ist bereits **flach von 5 bis 200 Schritten**. Das heißt: SigmaFlow
braucht schon jetzt kaum Schritte — aber es heißt auch, dass hier **kein
Genauigkeitsgewinn** zu holen ist, nur Rechenzeit. Und die Rechenzeit ist
aktuell nicht der Engpass.

**Urteil:** korrekt eingeordnet als Effizienz-, nicht Genauigkeitsthema.
Niedrige Priorität, solange die Genauigkeit das Problem ist. Als Nebenresultat
in der Thesis aber billig mitzunehmen (die Messung existiert schon).

---

## 4. General Improvements

### G1 — Direkter äquivarianter Rotationskopf `[Kategorie C]` ⭐ **eigene Idee**

**Motivation:** §2.3. Die Rotation entsteht als inertianormiertes erstes Moment
eines Kraftfeldes und verstärkt Rauschen 8.8× stärker als die Translation.

**Mathematische Änderung:** ein **zweiter** `l=1`-Output-Block, dessen
Knotenvektoren über die Fragmentatome gemittelt werden:

```
ω_F = mean_{i ∈ F} g_i          statt      ω_F = I⁻¹ Σ_i (r_i − c) × f_i
```

**Äquivarianz bleibt erhalten:** unter globalem `Q ∈ SO(3)` gilt `g_i → Q g_i`,
also `mean → Q·mean` — dieselbe Konstruktion, die die Translation schon benutzt.
Da nur eigentliche Rotationen auftreten (det = +1), ist der Unterschied
axialer/polarer Vektor hier ohne Belang.

**Codebereiche:** `net/model.py` (zweiter `force_block` + Rückgabe),
`sigma_flow_generator.py::_compute_forces`, `linear_mechanics`,
`newton_maruyama`, `_predict_fragment_updates`.

**Risiko:** mittel. Bricht die physikalische Kraft-/Drehmoment-Analogie, die
SigmaDock geerbt hat — die aber ohnehin nur eine Analogie ist, kein Constraint.
Bricht Checkpoint-Kompatibilität (neuer Parameterblock).

**Erwartete Wirkung:** potenziell die größte Einzelwirkung im ganzen Dokument,
weil sie den nachweislich toten Kanal direkt adressiert.
**Aufwand: mittel. Confidence: mittel-hoch, dass es hilft; niedrig, wie viel.**

> Wichtig: das ist **Kategorie C** — SigmaDock würde genauso profitieren. Es darf
> nicht als FM-Vorteil dargestellt werden. Es ist aber möglicherweise die
> Voraussetzung dafür, dass die FM-Ideen überhaupt messbar werden.

### G2 — Endpunkt-Parametrisierung der Rotation (`R_1`-Vorhersage) `[Kategorie B]` ⭐ **eigene Idee**

**Motivation:** direkt aus H-Shrink (§2.4). Mittelung von Rotations**vektoren**
schrumpft gegen null. Mittelung von Rotations**matrizen** mit anschließender
Projektion auf SO(3) tut das **nicht** — sie liefert eine gültige Rotation voller
Magnitude. Wenn die beobachtete Nullwirkung Schrumpfung ist, ist das der
strukturell passende Gegenentwurf.

**Mathematische Änderung:** das Netz sagt `R̂_1` vorher (etwa über 6D-Darstellung
mit Gram-Schmidt), der Loss ist geodätischer Abstand auf SO(3), das Vektorfeld
wird abgeleitet als `u_t = log(R_tᵀ R̂_1)/(1−t)`.

Das ist exakt die Analogie zu `x_0`- statt `ε`-Vorhersage bei Diffusion —
deshalb **Kategorie B, nicht A**.

**Risiko:** der `1/(1−t)`-Faktor verstärkt Fehler in `R̂_1` nahe t = 1. Muss mit
`t_max < 1` bzw. Clipping abgefangen werden.
**Aufwand: mittel. Confidence: mittel.**

### G3 — Confidence Head + Multi-Sample Ranking `[Kategorie C]`

**Hebel:** groß. 4.4 % → bis zu 29.2 % Kopfraum (Faktor ~7).
**Aber** siehe §2.6: ein Teil davon ist Symptom eines zufälligen Generators.

**Zielwahl.** `P(RMSD < 2 Å)` ist wahrscheinlich **nicht** das beste Ziel:
bei 4.4 % Positivrate ist es stark unbalanciert und liefert kaum Gradient.
Bessere Kandidaten, in dieser Reihenfolge:
1. **Regression auf RMSD** (oder `log RMSD`) — nutzt jeden Datenpunkt, dichtes Signal.
2. **Paarweises Ranking** innerhalb der K Kandidaten desselben Komplexes — das
   ist die tatsächliche Aufgabe, und es eliminiert komplexspezifische Offsets.
3. `P(RMSD < 2 Å)` als Kalibrierungskopf **obendrauf**, nicht als Hauptziel.

PoseBusters-Validität als Ziel wäre falsch: sie korreliert nur schwach mit RMSD
und beide Methoden versagen bei denselben Checks.

**Aufwand: hoch** (eigenes Netz, eigene Trainingsschleife, Kandidatengenerierung).
**Confidence: hoch, dass es hilft. Hoch, dass es nicht FM-spezifisch ist.**

### G4 — Symmetry-aware Rotation `[Kategorie D]` ❌

**Abgelehnt, auf Basis eigener Messung.** 4.8 % betroffene Fragmente,
0 von 467 Flips erklärt, 1.3° medianer Gewinn (§2.5).

Das ist Korrektheitspflege, kein Leistungshebel. Es sollte in der Thesis als
**geprüfte und verworfene Hypothese** erscheinen — das ist ein wertvolles
Negativergebnis, weil die 53 %-Automorphismus-Zahl den gegenteiligen Schluss
nahelegt und andere denselben Fehlschluss ziehen würden.

**Dies ist mein stärkster Widerspruch zur bisherigen Priorisierung.**

### G5 — Rotation Direction/Magnitude Objective `[Kategorie C]`

Getrennte Terme für Richtung (`1 − cos`) und Magnitude.

**Kritik:** Die MSE `‖v_R − u_R‖²` zerlegt sich bereits implizit in beide
Anteile. Ein expliziter Richtungsterm ändert vor allem die **Gewichtung** — er
nimmt großen Zielvektoren Dominanz. Da `|u_R|` nach §2.4 **konstant in t** und
Haar-verteilt ist (also ohne große Ausreißer), ist der Gewichtungseffekt
vermutlich klein.

Zusätzlich instabil: `cos` ist bei kleinen `|u_R|` schlecht konditioniert.

**Urteil: Kategorie C, niedrige Priorität.** Ein billiger Zusatztest, wenn G1/G2
laufen — aber kein eigenständiger Kandidat. **Confidence: mittel-hoch, dass der
Effekt klein ist.**

### G6 — Getrennte Translation-/Rotations-/Torsions-Köpfe `[Kategorie C]`

Teilweise **schon durch G1 abgedeckt**. Die wichtige Erkenntnis: Translation und
Rotation lesen bereits **orthogonale** Projektionen desselben Feldes — es gibt
also *keinen* Kapazitätskonflikt in dem Sinne, dass sie sich gegenseitig
überschreiben. Das Problem ist Konditionierung, nicht Konkurrenz.

Torsion ist ein separater Fall: sie entsteht in SigmaFlow gar nicht aus dem
Vektorfeld, sondern implizit aus der Starrfragment-Zerlegung. Ein Torsionskopf
wäre eine echte Architekturerweiterung, keine Aufteilung.

**Urteil:** als eigenständige Idee **redundant zu G1**. Nicht separat ansetzen.

### G7 — Exact Likelihood `[Kategorie B]`

Mathematisch implementierbar (Hutchinson-Spur-Schätzer entlang der ODE).

**Aber:** Likelihood misst, wie wahrscheinlich das Modell eine Pose erzeugt —
nicht, ob sie korrekt ist. Bei einem Modell, dessen Rotationskanal auf
Zufallsniveau arbeitet, wäre die Likelihood bezüglich der Rotation nahezu
uniform und damit als Ranking-Signal wertlos.

**Ihre Einschätzung wird bestätigt:** hinter dem Confidence Head.
Zusätzliches Argument: die Likelihood ist **nicht** FM-exklusiv (kontinuierliche
Normalizing Flows und Diffusions-ODEs können das ebenfalls). **Kategorie B.**

### G8 — Rotationskanal-Diagnose vor allen Interventionen `[Diagnose]` ⭐ **eigene Idee**

**Die billigste und entscheidendste Messung im ganzen Dokument.** Sie
diskriminiert zwischen den beiden konkurrierenden Erklärungen:

| Beobachtung | Schlussfolgerung |
|---|---|
| `‖v_R^pred‖ ≪ ‖u_R‖`, besonders bei kleinem t | **H-Shrink** (§2.4): Ziel unvorhersagbar, Modell schrumpft → konditionierte Quelle (A1) und Endpunkt-Parametrisierung (G2) sind die richtigen Hebel |
| `‖v_R^pred‖ ≈ ‖u_R‖`, aber `cos(v_R, u_R) ≈ 0` | **H-Noise** (§2.3): Magnitude da, Richtung Rauschen → direkter Rotationskopf (G1) ist der richtige Hebel |
| beide klein und t-unabhängig | Kanal komplett tot, Architekturproblem |

Zu erheben: `‖v_R^pred‖`, `‖u_R‖`, `cos(v_R^pred, u_R)`, jeweils gebinnt über t,
getrennt für Translation und Rotation.

Das Skript existiert bereits: `SigmaFlow_Minimal/diagnostics/rotation_completion/cosine_by_t.py`.
Es ist auf ARC **eingereicht, aber wegen der Wartung nicht gelaufen**.

> **Ohne dieses Ergebnis ist jede Priorisierung zwischen A1, G1 und G2 geraten.**

---

## 5. Dependency Graph

```
                        SigmaFlow_Minimal (eingefroren)
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
              ┌─────▼─────┐               ┌───────▼────────┐
              │  EXP-100  │               │  G8 Diagnose   │  ← ZUERST, billig
              │  fertig   │               │  cosine_by_t   │     entscheidet den Rest
              └─────┬─────┘               └───────┬────────┘
                    │                             │
                    │            ┌────────────────┼────────────────┐
                    │            │                │                │
                    │      falls H-Shrink   falls H-Noise      unabhängig
                    │            │                │                │
                    │      ┌─────▼─────┐    ┌─────▼─────┐    ┌─────▼─────┐
                    └─────►│ A1 Cond.  │    │ G1 Rot-   │    │ G3 Conf.  │
                           │   Source  │    │    Kopf   │    │  Ranking  │
                           └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
                                 │                │                │
                           ┌─────▼─────┐          │          ┌─────▼─────┐
                           │  Multi-   │◄─────────┘          │  Early    │
                           │  Sampling │                     │  Pruning  │
                           └─────┬─────┘                     └───────────┘
                                 └──────────────►  G3 (Nutzen steigt)

     Unabhängige Äste (keine Voraussetzung, jederzeit):
       G2 Endpunkt-Parametrisierung     A2 Probability Path
       G5 Rotation Direction/Magnitude  A4 Few-Step
       G4 Symmetry-aware  ❌ verworfen
```

**Korrekturen an der vermuteten Abhängigkeitsstruktur:**

- **G8 fehlte** und ist die eigentliche Wurzel. A1 vs. G1 ist ohne sie geraten.
- **Multi-Sampling braucht die konditionierte Quelle nicht.** Varianten A und C
  des Ablationsrasters laufen sofort mit dem bestehenden Modell. Das ist die
  billigste Messung überhaupt — und sie ist bereits gemacht (Oracle@10, §2.6).
- **„Symmetry-aware macht die Quelle stabiler"** — die Voraussetzung entfällt,
  da G4 verworfen ist (§2.5).
- **EXP-100 ist Voraussetzung für A1, aber nicht für G1, G2, G3, A2.** Diese
  Äste können parallel und unabhängig laufen.

---

## 6. Experiment Matrix

Bewertung 1–5. „FM" = FM-Spezifität (5 = exklusiv FM). Confidence = wie sicher
ich mir bei der Einschätzung bin, gegeben die vorhandenen Messungen.

| # | Idee | Kat. | FM | Accuracy | Effizienz | Aufwand | Risiko | Wiss. Wert | Confidence |
|---|---|---|---|---|---|---|---|---|---|
| G8 | Rotationskanal-Diagnose | Diag | 1 | – | – | **1** | 1 | **5** | **hoch** |
| G1 | Direkter Rotationskopf | C | 1 | **4** | 2 | 3 | 3 | 4 | mittel-hoch |
| A1-A | Hand-crafted cond. Source | A | **5** | 3 | 3 | 2 | 2 | **5** | mittel |
| G2 | Endpunkt-Param. Rotation | B | 2 | **4** | 2 | 3 | 3 | 4 | mittel |
| G3 | Confidence Ranking | C | 1 | **5** | 1 | **5** | 2 | 3 | hoch |
| A1-B | Gelernte cond. Source | A | **5** | 3 | 3 | 4 | **4** | 4 | niedrig |
| A1-D | Vina-Pose als Kontrolle | A/D | 3 | – | – | 2 | 1 | 4 | hoch |
| Multi | Multi-Sampling (A/C) | C | 1 | 3 | 1 | **1** | 1 | 3 | **hoch** |
| A2 | Staged Probability Path | B | 3 | 2 | 3 | 3 | 2 | 3 | mittel-niedrig |
| G5 | Rot. Direction/Magnitude | C | 1 | 2 | 1 | **2** | 2 | 2 | mittel-hoch |
| A4 | Few-Step / Destillation | B | 3 | 1 | **5** | 3 | 2 | 3 | hoch |
| G7 | Exact Likelihood | B | 2 | 2 | 1 | 4 | 3 | 3 | mittel |
| A3 | Source + Path Co-Design | A | **5** | 2 | 3 | **5** | **5** | 4 | niedrig |
| G4 | Symmetry-aware Rotation | **D** | 1 | **1** | 1 | 3 | 1 | 2 | **hoch** |
| G6 | Separate Köpfe (allgemein) | C | 1 | – | – | – | – | – | redundant zu G1 |

---

## 7. Prioritization

### A. Beste FM-spezifische Ideen

1. **A1-A Hand-crafted conditional source** — die einzige Idee, die zugleich
   FM-exklusiv ist *und* einen gemessenen Mechanismus adressiert (§2.4).
2. **A1-B gelernte conditional source** — nur nach A1-A, mit strenger
   Kapazitätsbegrenzung und `q_φ`-allein-Kontrolle.
3. **A2 staged probability path** — ein realer Mechanismus (Orientierung im
   korrekt platzierten Kontext), aber warnender Präzedenzfall.

### B. Beste allgemeine Verbesserungen

1. **G1 direkter Rotationskopf** — adressiert den nachweislich toten Kanal.
2. **G3 Confidence Ranking** — größter reiner Zahleneffekt, teuerste Umsetzung.
3. **G2 Endpunkt-Parametrisierung** — strukturell passender Gegenentwurf zu H-Shrink.

### C. Beste Ideen pro Aufwand

1. **G8 Diagnose** — Aufwand 1, entscheidet die gesamte Priorisierung.
2. **Multi-Sampling A/C** — Aufwand 1, im Wesentlichen **schon gemessen**.
3. **A1-A hand-crafted source** — Aufwand 2, FM-exklusiv, interpretierbar.
4. **G5 Direction/Magnitude** — Aufwand 2, aber vermutlich kleiner Effekt.

---

## 8. Rejected / Low-Priority Ideas

| Idee | Grund |
|---|---|
| **G4 Symmetry-aware Rotation** | **Gemessen widerlegt**: 4.8 % betroffene Fragmente, 0/467 Flips erklärt, 1.3° Gewinn. Als Negativergebnis dokumentieren. |
| **G6 Separate Köpfe als eigene Idee** | Redundant zu G1. Translation und Rotation lesen bereits orthogonale Projektionen — kein Kapazitätskonflikt. |
| **A1-D Vina als produktive Quelle** | Macht SigmaFlow zum Nachbearbeiter. Als *Oberschranken-Kontrolle* dagegen wertvoll. |
| **A3 Source + Path Co-Design** | Zwei gekoppelte instabile Lernprobleme; Zuschreibung unmöglich. Ausblickskapitel. |
| **Globale Zeitgewichtung** | Bereits getestet, kein Effekt. Nicht wiederholen. |
| **Mehr Integrationsschritte** | Bereits ausgeschlossen (5→200 flach). |
| **PoseBusters-Validität als Confidence-Ziel** | Korreliert schwach mit RMSD; beide Methoden versagen bei denselben Checks. |

---

## 9. ARC Validation Plan

Alles Folgende ist **auf ARC**, nicht lokal entscheidbar.

| Priorität | Job | Zweck | Kosten |
|---|---|---|---|
| **1** | `diag_rotation_completion.sh` (G8) | H-Shrink vs. H-Noise entscheiden | Minuten |
| **2** | `train_exp100_vs_minimal_12h.sh` | EXP-100 gegen Minimal, identischer Seed | 12 h |
| 3 | EXP-101 Source Distance Audit | Reduziert eine Heuristik `d_SO(3)` unter 132.3°? | Minuten |
| 4 | Der aus G8 folgende Ast (G1 **oder** A1-A) | die eigentliche Intervention | 12 h |
| 5 | Durchsatzmessung EXP-100 vs. Minimal | Kostet die Konformergenerierung Schritte/s? | im Lauf 2 enthalten |

**Offene Punkte mit `PENDING ARC VALIDATION`:**
- `cosine_by_t` und globale SE(3)-Äquivarianz (eingereicht, nicht gelaufen)
- EXP-100-Trainingsvergleich und Durchsatz
- Die Größenabhängigkeit aus §2.3 an einem Modell mit Rotationssignal

---

## 10. Recommended Next Steps

Fünf Experimente, in dieser Reihenfolge:

1. **G8 — Rotationskanal-Diagnose.** Minuten Rechenzeit, Skript existiert.
   Entscheidet, ob G1 oder A1 der richtige Ast ist. Ohne sie ist die
   Priorisierung geraten.
2. **EXP-100 gegen Minimal.** Fertig, geprüft, wartet nur auf ARC. Klärt, ob die
   inferenzsaubere Parametrisierung trainierbar ist — Voraussetzung für A1.
3. **Der aus G8 folgende Ast.** Bei H-Shrink → A1-A; bei H-Noise → G1.
4. **Multi-Sampling-Ablation A vs. C** mit dem dann besten Modell. Billig, und
   der Oracle@K-Wert sagt sofort, ob G3 sich lohnt.
5. **G3 Confidence Ranking** — nur wenn Schritt 4 weiterhin eine große
   Oracle-Top1-Lücke zeigt *und* der Generator repariert ist.

**Leitprinzip:** Nicht viele Experimente, sondern die wenigen, für die eine
Messung einen konkreten Mechanismus nahelegt. Nach diesem Dokument sind das
G8 (Diagnose), dann genau ein Ast daraus, und EXP-100 parallel.

---

## Anhang: in dieser Sitzung erzeugte Messskripte

| Skript | Was es misst |
|---|---|
| `SigmaFlow_Variants/posebusters_full_comparison/symmetry_flip_analysis.py` | Symmetrie vs. echter Rotationsfehler (§2.5) |
| `SigmaFlow_Variants/posebusters_full_comparison/seed_variance.py` | Oracle@10 vs. Einzelseed (§2.6), bestand bereits |

Die Messungen zu §2.1–§2.3 (Translation/Rotation-Aufspaltung, Rauschverstärkung)
wurden ad hoc erhoben; die Skripte sind in der Sitzungshistorie und können bei
Bedarf als dauerhafte Diagnostik abgelegt werden.
