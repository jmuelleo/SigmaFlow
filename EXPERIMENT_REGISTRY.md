# SigmaFlow — Experiment Registry

**Regel:** Jede Variante entsteht aus einer frischen Kopie von
`SigmaFlow_Minimal/` (Tag `sigflow-minimal-baseline-v1`, Commit `267fb69`) und
ändert **genau eine** Sache. Keine Variante erbt Änderungen einer anderen.

**Status-Werte:** `SPECIFIED` (Hypothese + Mathematik dokumentiert, kein Code) ·
`IMPLEMENTING` · `LOCAL TESTS PASSED` · `PENDING ARC VALIDATION` · `VALIDATED` ·
`REJECTED`

---

## EXP-000 — SigmaFlow_Minimal (Baseline)

| | |
|---|---|
| Ordner | `SigmaFlow_Minimal/` |
| Tag | `sigflow-minimal-baseline-v1` (Commit `267fb69`) |
| Änderung ggü. SigmaDock | nur die generative Engine |
| Status | **`PENDING ARC VALIDATION`** |

Offen: Äquivarianz-Follow-up, `num_bad_batches`, 3-Tage-Vergleich.
Details in `SigmaFlow_Minimal/README.md`.

---

## Flow-Matching-spezifisch

### EXP-101 — Informative Quellverteilung (Translation)
| | |
|---|---|
| Ordner | `SigmaFlow_FM_Specific/01_informative_source/` |
| Einzige Änderung | `R3_FlowMatcher.sample_init`: `N(0,I)` → `N(0, σ²I)` mit σ aus der Taschengeometrie |
| FM-spezifisch | **vollständig** — bei Diffusion legt der Vorwärtsprozess die Quelle fest |
| Hypothese | Weniger Transportweg → weniger Inferenzlast; sollte den Schwerpunktfehler nicht verschlechtern |
| Primärmetrik | Schwerpunktfehler, RMSD ≤ 2 Å |
| Leakage-Risiko | **hoch** — `μ` darf NIE aus der Ground-Truth-Pose stammen. Nur Taschen-COM aus dem Rezeptor. |
| Status | `SPECIFIED` |

**Wichtige Einschränkung:** Die Translation ist bereits SigmaFlows Stärke
(Schwerpunkt 1.16 Å, signifikant besser als SigmaDock). Der erwartete Gewinn ist
gering. Wissenschaftlich ist die Variante trotzdem wertvoll, weil sie die
FM-Exklusivität demonstriert.

### EXP-102 — Informative Quellverteilung (Rotation)
| | |
|---|---|
| Ordner | `SigmaFlow_FM_Specific/01_informative_source/` (zweite Ablation) |
| Einzige Änderung | `SO3_FlowMatcher.sample_init`: Haar-uniform → konzentriert |
| FM-spezifisch | **vollständig** |
| Hypothese | Trifft die gemessene Schwachstelle direkt: Kosinus 0.025 bei t<0.1, **weil** R₀ Haar-uniform ist |
| Status | **`NOT IMPLEMENTED — MATHEMATICAL VALIDATION REQUIRED`** |

**Warum nicht implementiert:** Eine konzentrierte Quelle auf SO(3) braucht eine
wohldefinierte Verteilung (z.B. Matrix-Fisher oder IGSO3 mit kleinem σ) **und**
ein Zentrum. Woher das Zentrum kommt, ohne Leakage, ist ungelöst: die
Taschengeometrie liefert Hauptachsen, aber es ist eine offene Frage, ob diese
überhaupt Information über die Ligandenorientierung tragen. Das muss zuerst
empirisch geprüft werden (siehe „Vorabtest" unten), nicht implementiert werden.

**Vorabtest, lokal möglich, noch nicht durchgeführt:** Korreliert die
Hauptachse der Tasche mit der Hauptachse des Kristallliganden? Wenn die
Verteilung des Winkels zwischen beiden nahe uniform ist, trägt die Tasche keine
Orientierungsinformation und die ganze Idee ist wertlos.

### EXP-103 — Klassische Docking-Pose als Quelle
| | |
|---|---|
| FM-spezifisch | **vollständig** — Diffusion kann das nicht ohne Prozesswechsel |
| Status | `SPECIFIED`, zurückgestellt |

Braucht externe Software (Vina/Smina) und einen Präprozessierungslauf über den
gesamten Trainingssatz. Erst nach EXP-101/102 sinnvoll.

### EXP-111 — Endpunkt-Supervision
| | |
|---|---|
| Ordner | `SigmaFlow_FM_Specific/02_endpoint_supervision/` |
| Einzige Änderung | Zusatzterm `λ·L_endpoint` im Loss |
| FM-spezifisch | **teilweise** — `x̂₁ = x_t + (1−t)v` ist im FM exakt, bei Diffusion nur im Erwartungswert |
| Hypothese | Trifft die gemessene Längenunterschätzung (Normverhältnis 0.47) und die frühe Blindheit |
| Mathematik | Translation: `x̂₁ = x_t + (1−t)v`, Verlust `\|x̂₁ − x₁\|²`. Rotation: `R̂₁ = R_t·exp((1−t)v)`, Verlust **geodätisch** `‖log(R̂₁ᵀR₁)‖²` — NICHT euklidisch auf Matrizen |
| Risiko | λ-Tuning; bei t→1 wird `(1−t)v` klein und der Term degeneriert |
| Status | `SPECIFIED` |

### EXP-112 — Endpunkt-Selbstkonsistenz
| | |
|---|---|
| Ordner | `SigmaFlow_FM_Specific/03_endpoint_consistency/` |
| Einzige Änderung | Regularisierer `d(x̂₁(t_a), x̂₁(t_b))` |
| FM-spezifisch | **teilweise** |
| Status | **`NOT IMPLEMENTED — MATHEMATICAL VALIDATION REQUIRED`** |

**Offene Fragen, bewusst nicht improvisiert:** Braucht zwei Vorwärtspässe pro
Sample (Kosten ×2). Beide Punkte müssen auf **derselben** Trajektorie liegen —
bei der konditionalen FM-Konstruktion ist die Trajektorie durch `(x₀, x₁)`
festgelegt, zwei t-Werte sind also billig zu bekommen; aber ob der Gradient
durch beide Pfade stabil ist, ist ungeprüft. Kein Leakage, weil nur
Modellvorhersagen verglichen werden.

### EXP-121 — Pfad-/Kopplungsdesign
| | |
|---|---|
| Ordner | `SigmaFlow_FM_Specific/04_path_coupling/` |
| FM-spezifisch | **vollständig** |
| Status | **`REJECTED — mathematisch nicht überzeugend`** |

**Begründung für die Ablehnung ohne Implementierung:** Optimal-Transport-Kopplung
paart x₀ und x₁ transportkostenminimal über eine Minibatch. Im bedingten Docking
ist x₁ pro Komplex **festgelegt** — es gibt nichts umzupaaren. Die einzige
anwendbare Variante ist Kopplung über mehrere x₀-Ziehungen desselben Komplexes,
und das entspricht schlicht „wähle das nächstgelegene x₀", was den Effekt einer
informativen Quelle (EXP-101/102) reproduziert, nur teurer. Dokumentiert, nicht
implementiert.

### EXP-131 — Getrennte Köpfe für Translation / Rotation / Torsion
| | |
|---|---|
| Ordner | `SigmaFlow_FM_Specific/05_structured_heads/` |
| Einzige Änderung | Zweiter äquivarianter Kopf gibt `ω` direkt aus, statt es über `Σ r×F` aus dem Kraftfeld abzuleiten |
| FM-spezifisch | **gering** — auf R³×SO(3) faktorisiert auch der Diffusions-Score |
| Hypothese | Translation (`ΣF`) und Rotation (`Σ r×F`) sind zwei Funktionale **desselben** Per-Atom-Ausgangs und konkurrieren um dieselbe Kapazität; eines funktioniert, das andere nicht |
| Erwartete Wirkung | **hoch** — die stärkste Evidenzgrundlage aller Varianten |
| Status | `SPECIFIED` |

**Ehrliche Einordnung:** geringste FM-Spezifität, höchste erwartete Wirkung.
Beides muss in der Thesis so stehen.

---

## Allgemein (nicht als FM-Vorteil zu verkaufen)

### EXP-201 — Confidence-Kopf (nur beobachtend)
| | |
|---|---|
| Ordner | `SigmaFlow_General/01_confidence_head/` |
| Einzige Änderung | Zusatzkopf sagt `P(RMSD_final ≤ 2 Å \| h_t)` voraus; **kein Feedback** ins Flussnetz |
| FM-spezifisch | **nein** — in SigmaDock genauso baubar |
| Empfohlenes Target | **Option A**, siehe Begründung im Bericht |
| Evaluation | AUROC, AUPRC, Brier, ECE, Reliability, Spearman vs. RMSD, Top-1-von-10 |
| Status | `SPECIFIED` |

**Warum das die wertvollste Erweiterung überhaupt ist:** best-of-10 liegt bei
29.2 % gegen 4.4 % einer Einzelziehung. Ein funktionierendes Ranking ist
**25 Prozentpunkte** wert — mehr als jeder gemessene Architekturunterschied.

### EXP-202 — Confidence-konditionierter Fluss
| | |
|---|---|
| Status | `SPECIFIED`, blockiert durch EXP-201 |

Zirkularität `c_t = f(x_t, v_θ)` und `v_θ = g(x_t, t, c_t)` muss aufgelöst
werden. Empfohlen: `c_{t−Δt}` aus dem vorherigen ODE-Schritt (kausal, keine
Zirkularität, kein zusätzlicher Vorwärtspass).

### EXP-203 — Exakte ODE-Likelihood
| | |
|---|---|
| Ordner | `SigmaFlow_General/03_exact_likelihood/` |
| Status | **`NOT IMPLEMENTED — MATHEMATICAL VALIDATION REQUIRED`** |

**Verifiziert (lokal, aus Code und Daten):** Zustandsdimension
`D = 6 × Fragmente`, gemessen über 209 Komplexe: **Median 24, q90 42, Maximum
66**. Exakte Jacobi-Spur kostet D JVPs je Schritt, also 600 (Median) bis 1650
(worst case) je Pose bei 25 Schritten. **Hutchinson ist nicht nötig.**

**Nicht verifiziert:** dass die Divergenz im invarianten Rahmen ohne
Christoffel-Terme auskommt. SO(3) ist kompakt und unimodular, das Haar-Maß
bi-invariant — die Aussage ist plausibel, aber ich habe sie **nicht** gegen ein
Testfeld mit analytisch bekannter Divergenz validiert. Genau die Art Annahme,
die in diesem Projekt schon zweimal danebenlag. Vor jeder Implementierung
nachrechnen.

### EXP-204 — Likelihood-basiertes Posen-Ranking
| | |
|---|---|
| Status | `SPECIFIED`, blockiert durch EXP-203 |

Zentrale Frage: `Spearman(log p(x), −RMSD)`. **Likelihood ≠ Confidence** —
Likelihood misst Typizität unter dem Modell, nicht Korrektheit. Unsere
Winkelverteilung ist bimodal mit einer Flip-Mode bei 36.5 %; eine geflippte Pose
könnte hohe Likelihood haben. Muss gemessen werden, bevor irgendetwas gebaut wird.

---

## Weitere Kandidaten (kritisch geprüft)

| ID | Idee | Urteil |
|---|---|---|
| EXP-141 | Zeitgewichteter Flow-Loss `w(t)·\|v−u\|²` | **REJECTED** — Variante a hat Zeitgewichtung bereits als Nullergebnis abgeschlossen (Job 8540758) |
| EXP-142 | Separate Zeit-Embeddings je Freiheitsgrad | `SPECIFIED`, niedrige Priorität |
| EXP-143 | Auxiliary Rotations-**Richtungs**-Loss (Kosinus) | `SPECIFIED` — passt zur Diagnose (Kosinus 0.025 früh); Vorsicht bei `\|u\|→0` |
| EXP-144 | Magnituden-Kalibrierung `(\|v\|−\|u\|)²` | `SPECIFIED` — passt direkt zum Normverhältnis 0.47 |
| EXP-145 | Endpunkt-abgeleitete Confidence `U_t = d(x̂₁(t), x̂₁(t−Δt))` | `SPECIFIED` — attraktiv, weil kein zusätzlicher Kopf nötig |

**EXP-143 und EXP-144 sind die billigsten Kandidaten mit direktem Bezug zur
Diagnose** — beide sind Ein-Zeilen-Zusätze im Loss und je eine saubere Ablation.

---

## Evaluation

### EVAL-01 — TFD (RDKit Torsional Fingerprint Deviation)
| | |
|---|---|
| Ordner | `SigmaFlow_Evaluation/metrics/` |
| Status | **`LOCAL TESTS PASSED`** — empirisch geprüft, Urteil im Bericht |

Verifiziert: exakt invariant gegen globale Starrkörperbewegung (40/40, max
5.4e-13). Median 8 Torsionen je Molekül (6 Nicht-Ring, 2 Ring). Spearman gegen
roh-RMSD **+0.03** — praktisch orthogonal. Gepaart SigmaFlow vs SigmaDock:
**nicht unterscheidbar** (CI [−0.025, +0.038]).
**Praktische Einschränkung:** schlägt bei ~40 % unserer Moleküle fehl
(Sanitisierung) — vor Einsatz beheben oder n explizit ausweisen.
