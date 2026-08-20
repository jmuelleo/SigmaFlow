# Rotations-Abschlussdiagnostik der SigmaFlow-Minimalkonversion

Zwei Tests, die entscheiden, ob die Ersetzung des Diffusionsprozesses durch
Riemannsches Flow Matching **implementierungsseitig abgeschlossen** ist.

Reine Diagnostik. Kein Training, keine Architekturänderung, keine Schreibzugriffe
ausserhalb von `diagnostics/rotation_completion/out/`. `SigmaDock/` bleibt unberührt.

---

## Ausführung auf ARC

```bash
cd /data/stat-cadd/shug8458/SigmaFlow_Variants_JulianMueller/d_frame_fix
mkdir -p slurm_logs

# 1) Den richtigen Checkpoint FINDEN, nicht raten:
ls -dlt experiments/sigmadock/*/checkpoints/last.ckpt | head

# 2) Kandidaten identifizieren (liest den Checkpoint selbst aus):
python diagnostics/rotation_completion/verify_checkpoint.py \
    experiments/sigmadock/*/checkpoints/last.ckpt

#    Der 12h-Frame-Fix-Lauf sollte global_step ~13.750 zeigen
#    und KEINE Diffusions-Marker (rot_score_method o.ae.).

# 3) Beide Tests einreichen:
CKPT=experiments/sigmadock/<TIMESTAMP>/checkpoints/last.ckpt \
  sbatch slurm/diag_rotation_completion.sh
```

Optionale Umgebungsvariablen: `DATA_DIR`, `OUT_DIR`, `N_COMPLEXES` (Standard 200),
`N_T` (20), `N_EQ_COMPLEXES` (30), `N_ROT` (5).

Laufzeit: wenige Minuten. Nur Vorwärtspässe, eine L40S genügt reichlich.

---

## Was jeder Test misst

### Test 1 — Kosinus über t (`cosine_by_t.py`)

Spiegelt `SigmaFlowGenerator.forward()` Aufruf für Aufruf, damit jede
Konvention die des Trainings ist. Zwei bewusste Abweichungen:

1. `t` ist auf ein Gitter `linspace(0.01, 0.99, 20)` fixiert statt gezogen.
2. Der Basiszug `(x_0, R_0)` wird **pro Komplex gepinnt**, sodass alle 20
   t-Werte auf *einem* kohärenten Pfad liegen. Ohne das mischt die Kurve
   unabhängige Pfade und der Trend verwässert.

**Ziel-Definition:** Der Trainingsloss konsumiert `out["u_t_R"]` aus
`conditional_probability_path`, also die **konstante Form** `log(R_0^T R_1)` —
nicht die durch `(1-t)` geteilte Variante aus `_compute_true_vector_field`
(die dient nur dem Logging im Sampler). Der Test benutzt die erste.

Rotation wird über `vee()` in R³ verglichen, was die Frobenius-Norm des
Losses treu abbildet (`‖hat(a)−hat(b)‖²_F = 2|a−b|²`).

### Test 2 — Globale SE(3)-Äquivarianz (`equivariance.py`)

Getestet auf **Vektorfeld-Ebene**, weil dort ein Rahmenfehler lebt und weil nur
dort der Vergleich exakt sein kann: ein Volltrajektorien-Test müsste den
transformierten Basiszug `(Q x_0, Q R_0)` in den Sampler injizieren, aber der
zieht ihn intern — gleicher Seed liefert `x_0`, nicht `Q x_0`, also eine andere
physikalische Anfangsbedingung.

Zwei **unterschiedliche** Bedingungen unter `x → Qx + a`:

| Grösse | Verhalten | Begründung |
|---|---|---|
| `v_trans` | **äquivariant**: `Q⁻¹ v(Qx+a) = v(x)` | Weltvektor, `+a` kürzt sich |
| `v_rot` (Körperrahmen) | **invariant**: `v(Qx+a) = v(x)` | `(QR)ᵀ(QωQᵀ)(QR) = RᵀωR` |

Die zweite ist die scharfe: sie gilt **nur**, wenn der Frame-Fix angewandt wird.

**Drei Kontrollen, die fehlschlagen MÜSSEN** — sonst ist der Test blind:

- `C1_state_not_rotated` — Positionen gedreht, `R_t`/`trans_t` nicht
- `C2_no_frame_fix` — `omega` direkt als Körperrahmen konsumiert
- `C3_wrong_inverse` — `Qᵀ` statt `Q` beim Zurückdrehen

**Toleranz, vorab festgelegt:** relativer Fehler < `1e-3` im Median.

---

## Ausgaben

| Datei | Inhalt |
|---|---|
| `cosine_by_t_raw.csv` | alle Einzelbeobachtungen (Komplex, Fragment, t, Grössen, Kosinus, Normen) |
| `cosine_by_t_summary.txt` | Tabelle je t-Bin + Stratifikation nach Fragment-/Ligandgrösse/Zielnorm |
| `cosine_by_t_meta.json` | Checkpoint, Git-Hash, Seed, Versionen, Gerät, Zeitstempel |
| `equivariance_results.json` | maschinenlesbar, inkl. aller Kontrollen |
| `equivariance_summary.txt` | Tabelle mit Urteil je Variante |
| `checkpoint_identity.txt` | wofür der Checkpoint sich ausweist |
| `FINAL_SUMMARY.txt` | Gesamturteil nach vorab festgelegter Entscheidungslogik |

---

## Entscheidungslogik (vorab festgelegt, in `summarize.py` kodiert)

| Outcome | Bedingung | Bedeutung |
|---|---|---|
| **C** | `cos_rot` systematisch negativ (>55 % negativ oder Mittel < −0.15) | **STOP.** Versteckter Rahmen-/Vorzeichenfehler. Nicht als vollständig einstufen. |
| **A** | `cos_trans` > 0.35, `cos_rot` < 0.15 bei kleinem t, Anstieg > 0.15 zu grossem t | Frühzeitige Rotationsblindheit → Hebel ist die **Zeitverteilung**, nicht die Architektur |
| **B** | `cos_rot` durchgehend 0.15–0.35 | Rotation schwach gelernt → rechtfertigt dedizierten Rotationskopf |
| **D** | `cos_rot` > 0.35 | Vorhersage gut → Fehler liegt **downstream** (Konversion, Indizierung, Integration, Auswertung) |

Gesamturteil **COMPLETE** nur, wenn Äquivarianz besteht **und** alle drei
Kontrollen fehlschlagen **und** Outcome ≠ C.
