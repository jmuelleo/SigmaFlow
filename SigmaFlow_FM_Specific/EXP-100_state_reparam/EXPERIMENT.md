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
| `src/sigmadock/diff/state_reparam.py` | ✅ implementiert und validiert |
| `tests/test_exp100.py` (Mathematik) | ✅ 7 Kategorien, 0 Fehlschläge |
| `tests/validate_mapping.py` (Indexabbildung) | ✅ chemisch + geometrisch bewiesen |
| `data.py`: `ref_conf_pos` | ✅ implementiert, rein additiv |
| Patch `sigma_flow_generator.py` | ✅ 6 Stellen |
| Patch `sampling.py` | ✅ 6 Stellen |
| `tests/test_exp100_pipeline.py` (End-to-End) | ✅ T1–T6, 0 Fehlschläge |
| `tests/test_exp100_forward.py` (echter Trainingsschritt) | ✅ inkl. beider Sampler |
| SLURM `train_exp100_vs_minimal_12h.sh` | ✅ vorbereitet, **nicht gestartet** |

### C.1 Die Indexabbildung — empirisch bewiesen, nicht angenommen

```
Graph          = [ Protein 0..P-1 | Ligandenblock ]
Ligandenblock  = [ frag_mol-Atome 0..M-1 | virtuelle Knoten 0..V-1 ]
frag_mol       = [ Originalatome 0..n_orig-1 | Dummies n_orig..M-1 ]
```

Nachgewiesen auf allen lokalen Komplexen:

| Prüfung | Ergebnis |
|---|---|
| Atommerkmalsvektor Graph ↔ RDKit (Ordnungszahl, Grad, Formalladung, Hybridisierung, Aromatizität, Chiralität, Valenzen) | **0 Abweichungen** |
| Bindungsnachbarschaft Graph ↔ RDKit | identisch |
| Koordinaten | max 9.5e-06 Å |
| Dummy-Isotop benennt das Partneratom | gilt für alle Dummies |
| Dummy liegt exakt auf seinem Partner | < 1e-3 Å |

**Verworfener Ansatz, mit Grund.** Naheliegend wäre, die Fragmentierung ein
zweites Mal auf dem Konformer laufen zu lassen. Gemessen: `fragment_and_annotate`
ist **nicht rein topologisch**. Bei gleicher Konnektivität, aber anderer
Geometrie ergaben sich abweichende Bindungsmengen, `frag_atom_idx` und
`dummy_frag_sizes`. Ein daraus gebautes Kabsch-Ziel wäre still falsch gewesen.
Stattdessen wird der Graph **festgehalten** und nur die Geometrie ersetzt.

### C.2 EXP-100 ist an der Inferenz ein exakter No-Op

`conf/sampling/base.yaml` setzt `sample_conformer: true`. An der Inferenz **ist**
`frag_mol` bereits der generierte Konformer, also `ref_conf_pos == ref_pos` und
`R_1 = I` — gemessen exakt 0.00 Grad. Die Intervention wirkt ausschließlich auf
der Trainingsseite. Genau das macht sie zu einer sauberen Einzelintervention.

Der eigentliche Gehalt von EXP-100 lässt sich damit schärfer fassen als bisher:

> Im Training benutzt SigmaFlow-Minimal als Referenzgeometrie die
> **kristallnahe** Pose und supervidiert `R_1 = I`. An der Inferenz ist die
> Referenzgeometrie ein **generierter Konformer** in beliebiger Orientierung.
> EXP-100 schließt diese Lücke: Training benutzt dieselbe Konformerquelle wie
> die Inferenz und supervidiert die dazu passende echte Zielrotation.

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

## D.1 Pipeline-Tests (lokal, alle bestanden)

| Test | Ergebnis |
|---|---|
| T1 Toy End-to-End, bekanntes `R*`, `p*` | `R_1 = R*` auf 0.00e+00 Grad; Kontrollprobe gegen `R*ᵀ` 126 Grad entfernt |
| T1 Gegenprobe `R_ref = R_1` (der falsche Griff) | bricht mit 1.69 Å — der Test würde ihn bemerken |
| T2 Real End-to-End, 48 Fragmente / 9 Komplexe | median **0.057 Å**, p90 0.102, p95 0.327, max 0.499 |
| T3 Batch, 10 Graphen mit verschiedenen Größen | Batch == Einzelverarbeitung auf 0.00e+00 |
| T3 Protein unverändert | 0.00e+00 |
| T4a Leakage bei festem Graphen | **bitgleich** (0.00e+00) trotz 31.9 Å verschobener Kristallpose |
| T5 `trans_1` identisch zu Minimal | 0.00e+00 — Translationsziele unverändert |
| T5 `R_1` nicht-trivial | median 152.8 Grad, max 177.2 Grad |
| T6 Sampler-Invariante über 10 Schritte | max 4.8e-07 |
| Echter Trainingsschritt (EquiformerV2 + Loss + Backward) | Gradienten endlich, Ziele graphfrei |
| `sampler` und `sample_notebook` mit echtem Modell | laufen durch, Posen bewegen sich |

## D.2 Numerischer Befund: Kabsch muss in float64 rechnen

Gemessen an 2000 Zufallsfragmenten mit `C == Y` (exakte Antwort: Identität):

| Genauigkeit | größte Abweichung |
|---|---|
| float32 | **6.1e-02 Grad** |
| float64 | 3.0e-06 Grad |

Ursache sind fast entartete Singulärwerte: `U` und `V` sind einzeln schlecht
bestimmt. Da `R_1` ein **Trainingslabel** ist, hätte sich das direkt als Rauschen
auf das Ziel gelegt. `kabsch_rotation` promoviert deshalb intern immer auf
float64 und castet zurück. Kosten: vernachlässigbar, es sind 3×3-Matrizen.

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

**Die Fragmentierung selbst bleibt kristallabgeleitet (geerbt).** Im Training
wird der Graph — Fragmentzerlegung, Anker, Dummies, virtuelle Knoten — weiterhin
aus der gebundenen Pose gebaut. Test T4b zeigt, dass diese Zerlegung bei stark
veränderter Geometrie in einem von fünf Fällen abweicht. Das ist
SigmaFlow-Minimal-Verhalten und von EXP-100 **unverändert**; EXP-100 ersetzt nur
die *Geometrie*, nicht die *Topologie*. Eine Behebung wäre eine eigene,
separate Intervention und gehört nicht in dieses Experiment.
> EXP-100 removes the crystal dependence of the reference *geometry*, not of the
> fragment *decomposition*.

**Kosten.** Pro Trainingsbeispiel kommt eine ETKDGv3+MMFF-Konformergenerierung
hinzu. Bei `num_workers=6` läuft das im Dataloader parallel zur GPU; ob es zum
Engpass wird, ist auf ARC zu messen — lokal nicht entscheidbar.
> `PENDING ARC VALIDATION`: Durchsatz Schritte/s gegen SigmaFlow-Minimal.

## Status

`LOCALLY VALIDATED — READY FOR ARC`

Vier Testskripte, 0 Fehlschläge. Diff gegen `SigmaFlow_Minimal`: drei Dateien,
alle Änderungen Kategorie A (für EXP-100 notwendig), `data.py` rein additiv,
keine Datei außerhalb der Interventionsfläche berührt. `SigmaFlow_Minimal`
selbst ist unverändert.

Offen und nur auf ARC entscheidbar: der eigentliche Trainingsvergleich sowie
der Durchsatz. Das SLURM-Skript liegt bereit und wurde **nicht** gestartet.
