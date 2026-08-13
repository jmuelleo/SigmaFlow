# SigmaFlow-Minimal

**Die kontrollierte Baseline für „Diffusions-Engine → Flow-Matching-Engine".**

Diese Version existiert, um genau eine wissenschaftliche Frage beantwortbar zu
machen:

> Was ändert sich, wenn man in einem sonst identischen Docking-Modell den
> diffusions-/score-basierten generativen Prozess durch Riemannsches Flow
> Matching ersetzt?

Jede spätere Erweiterung wird als eigener Variantenordner **von hier** abgeleitet.
Dieser Ordner wird dabei **nicht verändert**.

---

## 1. Was gegenüber SigmaDock geändert wurde

Systematischer Diff über alle Quelldateien (2026-08-13):
**45 von 48 Dateien byteweise identisch.**

| Bereich | Änderung |
|---|---|
| `net/` (18 Dateien) | **keine** — Architektur vollständig unverändert |
| `geo/`, `core/`, `torch_utils/` | **keine** |
| `chem/` | nur `parsing.py`: toter Code nach unbedingtem `return None` entfernt, semantisch identisch |
| `diff/denoiser.py` → `sigma_flow_generator.py` | Score → Vektorfeld |
| `diff/{r3,so3,se3}_diffuser.py` → `_flow_matcher.py` | Diffusion → Flow Matching |
| `diff/sampling.py` | Reverse-SDE → ODE |
| `diff/so3_utils.py` | `Omega`-Clamp `[-0.99, 0.99]` → `[-1+1e-7, 1-1e-7]` |
| `trainer.py` | Loss-Kombination; `T0`/`R0`-Terme entfallen (Default-Gewicht war 0.0) |
| `utils.py`, `oracle.py` | Umbenennung / Kommentar |

**Die Architektur ist nachweislich unverändert:** Embeddings, Equiformer-Blöcke,
Hidden Dimensions, Layer-Zahl, Attention, Normalisierung, Aktivierungen,
Output-Heads, Zeit-Embedding, Graph- und Kantenkonstruktion, Cutoffs,
Radialfeatures — alle Dateien byteweise gleich.

### Warum der `so3_utils`-Clamp geändert werden MUSSTE

`arccos` wurde auf `[-0.99, 0.99]` geklemmt, was Winkel auf maximal 171.9°
begrenzt. Diffusion braucht nie eine exakte `exp(log(R)) == R`-Rundreise; die
geodätische Interpolation `R_t = R_0 exp(t·log(R_0ᵀR_1))` schon. Ohne den Fix
wären rund 8.5 % aller Haar-verteilten Rotationen falsch interpoliert worden.

**Ehrliche Asymmetrie:** der Fix verbessert die Genauigkeit dieser Primitive
auch für Diffusion. SigmaFlow hat hier einen kleinen numerischen Vorteil, den
SigmaDock nicht bekommt, weil `SigmaDock/` unverändert bleiben muss.

---

## 2. Was bewusst NICHT geändert wurde

| Element | Wert | Anmerkung |
|---|---|---|
| `trans_score_weight` | 2.0 | aus SigmaDocks Diffusions-Setup geerbt |
| `rot_score_weight` | 0.5 | dito |
| Batch-Größe, LR, Optimizer, EMA, Scheduler | identisch | |
| Netzarchitektur | identisch | |
| Datenpipeline, Splits, Featurisierung | identisch | |
| `num_steps` | 25 | von SigmaDock geerbt |

**Zu den Loss-Gewichten:** sie wurden für Diffusions-Score-Magnituden gewählt,
die durch σ-Pläne skaliert sind. Im Flow Matching sind beide Ziele O(1). Gemessen
ergibt sich am geteilten Netzausgang ein effektives Gradientenverhältnis von
**1.84 zugunsten der Rotation** (nicht, wie zunächst vermutet, zugunsten der
Translation). Bewusst unverändert gelassen — Kandidat für eine Ziel-2-Variante.

---

## 3. Mathematische Definition

### Zustand

Pro Ligandenfragment ein SE(3)-Element: Translation `x ∈ R³` (Fragment-COM,
normiert mit `dimensional_scale = 2.7 Å`) und Rotation `R ∈ SO(3)` relativ zur
Referenzkonformation `pos_0`.

**Wichtig:** `R_1 = I` für jedes Fragment (`get_fragment_com_and_rot` gibt `I3`
zurück) — der Rotationszustand ist relativ zur Eingabekonformation definiert.
Diese Konvention ist von SigmaDock geerbt (dort steht die PCA-Alternative in
einem `if False:`-Block).

### Zeitkonvention

`t = 0` → Quelle (Rauschen), `t = 1` → Daten. Durchgängig, in Training und
Sampling.

### Translation

```
x_0 ~ N(0, I)                          sample_init
x_t = (1-t)·x_0 + t·x_1                conditional_probability_path
u_t = x_1 - x_0                        Zielvektorfeld, KONSTANT in t
x_{t+dt} = x_t + v·dt                  euler_step
```

Identität, die im Code ausgenutzt wird:
`(x_1 - x_t)/(1-t) = x_1 - x_0` — die beiden Formen sind exakt gleich.

### Rotation

```
R_0 ~ Uniform(SO(3))                   Haar, so3_utils.sample_uniform
Δ   = R_0ᵀ R_1
R_t = R_0 · exp(t · log Δ)             geodätische Interpolation
u_t = log Δ = log(R_0ᵀ R_1)            Zielfeld, KÖRPERRAHMEN, konstant in t
R_{t+dt} = R_t · exp(v·dt)             euler_step, RECHTSmultiplikation
```

`dR/dt = R_t · u` bedeutet: `u` ist die Winkelgeschwindigkeit im **Körperrahmen**.

### Frame-Fix

Der Vorhersagekopf liefert `omega = hat(I_t⁻¹ τ)` im **Weltrahmen** (`I_t` und
`τ` werden beide aus `pos_t`, also Weltkoordinaten, berechnet). SigmaDock
konsumiert das konsistent per **Links**multiplikation (`dR @ R_t`). SigmaFlow
integriert rechtsmultiplikativ, braucht also die Adjungierte:

```
R·exp(u_body·dt) = exp(w_world·dt)·R   ⟺   u_body = Rᵀ · w_world · R
```

Implementiert in `_compute_vector_field`. Ohne diesen Fix ist der Rotationskanal
funktionslos (verifiziert: 104–120° mediane Abweichung, Zufall liegt bei 132.3°).

### Loss

```
L = 2.0 · Σ|v_trans - u_trans|²  +  0.5 · Σ‖v_rot - u_rot‖²_F
```

`‖hat(a) - hat(b)‖²_F = 2|a-b|²`, das effektive Rotationsgewicht ist also 1.0.

### Sampling-ODE

`timesteps = linspace(t_min, t_max, num_steps)`, `N-1` Euler-Schritte von
`t_min = 0.01` bis `t_max = 1.0`. Der letzte Schritt landet exakt auf `t_max`.

---

## 4. Verifikationsstand

### Lokal verifiziert ✅

- `hat`/`vee`-Rundreise, Schiefsymmetrie
- `exp(log(R)) == R` bei 0°, 1e-4°, 0.001°, 1°, 45°, 90°, 120°
- `R Rᵀ = I`, `det R = +1`, `log(Aᵀ) = -log(A)`, `Rz(30)Rz(40) = Rz(70)`
- Adjungierten-Identität, max. Abweichung 6.7e-16
- Zielfeld-Konsistenz: `calc_rot_vector_field == u_t` aus dem Pfad
- **Vorzeichen: 9/9 synthetische Fälle bewegen sich auf `R_1` zu**, Kontrollprobe
  mit `-v` entfernt sich in 8/8 (Test ist sensitiv)
- **Oracle-Integration exakt**: R³ 0.00 Å, SO(3) 0.0000° ab 2 Schritten,
  Körper- und Weltrahmen-Pfad
- Zeitkonvention t=0→Rauschen, t=1→Daten durchgängig
- Train und Sampling rufen dieselbe `_compute_vector_field`
- Kosinus über t: **kein negativer Bereich** → kein versteckter Vorzeichenfehler
- Architektur-Diff gegen SigmaDock: 45/48 Dateien identisch

### `PENDING ARC VALIDATION` ⏳

| Punkt | Was fehlt | Aufwand |
|---|---|---|
| Äquivarianz-Follow-up | MAIN 6e-3/1.4e-2 gegen vorab erklärte Toleranz 1e-3; Kontrollen 100–190× schlechter. Skript liegt in `diagnostics/rotation_completion/equivariance_followup.py` | ~12 min GPU |
| `num_bad_batches` | Der `try/except AssertionError` im `training_step` überspringt degenerierte Batches, die SigmaDock zum Absturz brächten. Ist der Zähler über den Lauf 0, ist die Abweichung beweisbar wirkungslos | `grep` im Trainingslog |
| `pocket_mol`-Check | `data.py` wirft jetzt früher; vermutlich identische Sample-Menge, nicht bewiesen | Sample-Zahl beider Pipelines vergleichen |
| Finaler Vergleich | 3 Tage SigmaDock vs. 3 Tage SigmaFlow-Minimal | 6 GPU-Tage |

### Bekannter Bug, bewusst NACH dem Freeze zu beheben

`sampling_setup.py::build_sampling_datafront` überschreibt den `sdf_regex` der
Experimentdatei bedingungslos mit dem Wert aus `conf/sampling/base.yaml`. Wirkung
war zufällig korrekt (alle Läufe lasen die kanonische Einzelkopie
`<cid>_ligand.sdf`). Details in `STATUS.md`. **Fix nur als Paar anwenden** — sonst
greift wieder die Mehrkopie-Datei.

---

## 5. Provenienz

| | |
|---|---|
| Checkpoint SigmaFlow 12h | `0-08-11_18-00-41/checkpoints/last.ckpt`, epoch 5, **global_step 13750** |
| Ergebnisordner | `SigmaFlow_Variants/d_frame_fix/experiments/` |
| Auswertung | `VERGLEICH_SigmaFlow_vs_SigmaDock.txt` (Repo-Root) |
| Audit | `AUDIT_2026-08-13_Vollaudit.md` |

Codeseitig ist dieser Ordner identisch mit `SigmaFlow_Variants/d_frame_fix/`
(verifiziert: `src/`, `scripts/`, `conf/` byteweise gleich). `d_frame_fix` ist
der Ergebnis-, `SigmaFlow_Minimal` der Code-Ordner.
