# EXP-100 — Zustandsparametrisierung relativ zu einem inferenzverfügbaren Referenzframe

## Research Question

Does replacing the crystal-oriented local fragment frame with an
inference-available conformer-relative frame affect Flow Matching docking
performance?

## Baseline

`SigmaFlow_Minimal/` — Tag `sigflow-minimal-baseline-v1`, Commit `267fb69`.
`src/`, `scripts/`, `conf/` wurden byteweise kopiert; Ausgangspunkt verifiziert
identisch.

## Single Intervention

Allgemeine Zielrotationen `R_1 ≠ I` relativ zu einem inferenzverfügbaren
Referenzkonformer, statt `R_1 ≡ I` relativ zur Kristallorientierung.

**Unverändert:** Source-Verteilungen (`N(0,I)` bzw. Haar), Architektur, Loss,
Loss-Gewichte, Optimizer, LR, Integrator, NFE, Sampling-Schedule.

## Motivation

Voraussetzung für leakage-freie informative Source Distributions (EXP-102).
In der Minimal-Parametrisierung bedeutet `R_1 = I` „die Orientierung, die das
Fragment in der Kristallpose bereits hat" — eine Source-Konzentration um diese
Identität ließe sich nicht als inferenzseitig gewonnene Information deuten.

**EXP-100 erwartet keinen Leistungsgewinn.** Es kann schlechter abschneiden.
Die Frage ist, ob SigmaFlow in einer inferenzsauberen Parametrisierung mit
echter unbekannter Zielrotation trainierbar ist.

---

## A. `R_1 = I`-Audit

| Datei | Funktion | setzt `R_1 = I` voraus? | Änderung nötig |
|---|---|---|---|
| `sigma_flow_generator.py` | `get_fragment_com_and_rot` | **JA** — gibt `I3` unbedingt zurück | **ja**, Ursprung |
| `sigma_flow_generator.py` | `_get_initial_states` | **JA** — reicht `I3` durch | **ja** |
| `sigma_flow_generator.py` | `_apply_transformations` | **implizit** — `delta_R = R_t @ R_1ᵀ`. Das Argument heißt `R_1`, meint aber die **Referenzrotation von `pos_0`**. In der alten Parametrisierung fielen Referenz- und Zielrotation zusammen, in der neuen nicht | **ja**, `R_ref = I` übergeben |
| `sigma_flow_generator.py` | `_compute_fragment_dynamics` | **implizit** — verbose-Trägheitscheck `delta_R_t = R_t @ R_1ᵀ` | ja, `R_ref = I` |
| `sigma_flow_generator.py` | `_sample_flow` | nein — reicht `R_1` generisch durch | nein |
| `sigma_flow_generator.py` | `_compute_vector_field` | nein — nur `R_t`, kein `R_1` | nein |
| `sigma_flow_generator.py` | `compute_losses` | nein | nein |
| `so3_flow_matcher.py` | `conditional_probability_path` | nein — `Δ = R_0ᵀ R_1` vollständig allgemein | nein |
| `so3_flow_matcher.py` | `calc_rot_vector_field` | nein — `log(R_tᵀ R_1)/(1−t)` allgemein | nein |
| `so3_flow_matcher.py` | `euler_step` | nein | nein |
| `se3_flow_matcher.py` | alle | nein | nein |
| `sampling.py` | `sampler`, `sample_notebook` | **implizit** — `_apply_transformations(..., R_1=R_1, ...)` beim Anfangszustand | **ja**, `R_ref = I` |
| `data.py` | `__getitem__` | Referenzkonformer fehlt | **ja**, neues Feld |

**Kernbefund:** Die Flow-Matching-Engine selbst (`so3_flow_matcher`,
`se3_flow_matcher`, `_compute_vector_field`, Loss) ist **bereits vollständig
allgemein** in `R_1`. Nur die Zustandskonstruktion und drei Aufrufstellen
setzen die Identität voraus.

## B. Neue Zustandskonvention

Ein Zustand ist `(R, p)` und bedeutet die Pose

```
pose_F(R, p) = R · C_F + p
```

mit `C_F` = zentriertes Referenzfragment aus dem generierten Konformer.
Zielzustand `(R_1F, p_1F)`, Quellzustand `(R_0, x_0)`.

**Die Falle:** `_apply_transformations` erwartet als `R_1`-Argument die
Referenzrotation der übergebenen Geometrie, nicht die Zielrotation. Da `pos_0`
nun bereits im `C_F`-Frame liegt, ist dort **`I` zu übergeben**, während `R_1F`
ausschließlich als Flussziel dient.

## C. Implementierungsstand

| Komponente | Status |
|---|---|
| `src/sigmadock/diff/state_reparam.py` | ✅ **implementiert und validiert** |
| `tests/test_exp100.py` | ✅ **7 Kategorien, 0 Fehlschläge** |
| Patch `sigma_flow_generator.py` (3 Stellen) | ⬜ offen |
| Patch `sampling.py` (2 Stellen) | ⬜ offen |
| `data.py`: Feld `ref_conf_pos` | ⬜ offen — **Komplikation:** der Ligandengraph enthält virtuelle Knoten (`num_lig_virtual`), Atomindizes in `pos` bilden also nicht 1:1 auf die RDKit-Molekülatome ab. Die Indexabbildung muss sauber gelöst werden, bevor der Konformer eingespeist wird |
| SLURM-Skript | ⬜ offen |

## D. Testergebnisse (lokal, alle bestanden)

| Kategorie | Ergebnis |
|---|---|
| Kabsch = Identität | 0 (exakt) |
| **Kabsch = bekannte Rotation `Q`, nicht `Qᵀ`** | max 3.19e-06 Grad über 200 Fälle |
| Kontrollprobe gegen `Qᵀ` | 20.2 Grad Abstand → Test ist sensitiv |
| Orthogonalität, `det = +1`, Reflexionsausschluss | bestanden |
| Endpunkte `R(0)=R_0`, `R(1)=R_1`, `\|Δ\|≤172°` | max 4.25e-03 Grad |
| **Oracle `R_0 → R_1 ≠ I`** bei 1°/30°/90°/150°/179°/zufällig, 5 und 25 Schritte | median 0 bis 1.7e-05 Grad |
| R3-Oracle | max 1.1e-16 |
| **Gauge-Invarianz** `R_1' = R_1 Sᵀ` | max 3.62e-06 Grad; rekonstruierte Pose invariant zu 3.3e-15 Å |
| **SE(3)-Äquivarianz** `R_1 → Q R_1`, `p_1 → Q p_1 + a` | max 2.96e-06 Grad / 4.4e-15 Å |
| Autograd-Hygiene | Ziel trägt keinen Graphen (`@torch.no_grad`) |
| Degenerierte Fälle (1, 2, 3 kollineare Atome) | endlich, `det = +1` |

**Numerische Einordnung:** `arccos` hat in float64 eine Auflösungsgrenze von
~6e-7 Grad. Werte um 3e-06 Grad liegen an dieser Grenze, nicht darüber.

## E. Rekonstruktionsfehler (Analysephase, 674 starre Fragmente / 209 Komplexe)

| | Wert |
|---|---|
| Median | **0.043 Å** |
| p90 / p95 | 0.329 / 0.572 Å |
| relativ ε/ρ | 0.032 |
| ε > 0.5 Å | 6.4 % |

Eine Größenordnung unter dem Übergangsbindungsfehler des Modells (0.36 Å).

## F. Known limitations

**Numerische Grenze nahe π.** `so3_utils.log` hat eine Präzisionsgrenze bei
Rotationen über ~172°. Gemessen: bei `R_1 ~ Haar` median 4.37e-05 / max 1.12
Grad, bei `R_1 = I` (Minimal) median 3.77e-05 / max 1.11 Grad. **EXP-100
verschlechtert das nicht** — es ist eine geerbte Eigenschaft.

**Symmetrie.** 53 % der Moleküle haben >1 Graphautomorphismus (Median 2).
`R_1` ist damit streng genommen eine Äquivalenzklasse `[R_1]`.
> EXP-100 does not yet perform symmetry-aware rotational supervision.

Relevant später für EXP-102, Kopf-Schwanz-Flips und die Rotationsbewertung.

**Flexible Ringe.** ~6 % der Fragmente haben ε > 0.5 Å, bei 10+-Atom-Ringen
p95 = 1.08 Å.
> Rigid-fragment SE(3) representation has a small irreducible reconstruction
> error for conformationally flexible ring systems.

**`pocket_com`.** Beim Re-Docking wird die Tasche aus dem Referenzliganden
geschnitten. Für den Methodenvergleich fair (beide Seiten identisch), aber für
absolute Blind-Docking-Aussagen relevant.

## Status

`IMPLEMENTING` — mathematischer Kern validiert, Pipeline-Anbindung offen.
