# EXP-110 — Zwei-Kopf-Vektorfeld

Kopiert aus `SigmaFlow_Minimal` bei Commit `16a069a` (siehe `COPIED_FROM.txt`).
`SigmaFlow_Minimal` und `SigmaDock` sind unverändert.

## Die eine wissenschaftliche Änderung

```
Minimal:  ein l=1-Ausgang -> Pseudo-Kraft -> Newton-Euler -> (dT, omega)
EXP-110:  zwei l=1-Ausgaenge -> Mittel je Fragment -> (v, omega)
```

$$\hat u^{T}_{f} = \frac{1}{N_f}\sum_{i\in f} f^{T}_i \qquad
\hat w^{R}_{f} = \frac{1}{N_f}\sum_{i\in f} f^{R}_i \qquad
\hat u^{R}_{f} = R_t^{\top}\,\widehat{\hat w^{R}_{f}}\,R_t$$

Ziele, Losse, Pfade, Integrator, Daten und Rumpf bleiben unverändert.

## Warum Mittel und nicht Summe

Fragmente haben 1 bis ~30 Atome; eine Summe skaliert damit, das Ziel nicht.
Nebenbei ist das Mittel **exakt** das, was Minimal für die Translation ohnehin
rechnet: `dT = sum(f_i) / m_f` mit `m_f = Atomzahl`. Die Translation ändert
also nur ihre Quelle, nicht ihre Form.

## Warum kein Drehmoment mehr

Die Drehmomentkonstruktion `omega = I^-1 * sum (x_i - c) x f_i` hat zwei
Defekte, die für ein *Geschwindigkeits*ziel nicht nötig sind:

- **1-Atom-Fragmente:** `x_i = c`, also `tau ≡ 0`. Die Rotation ist dort
  **prinzipiell unerreichbar**, egal was das Netz ausgibt. Im Datensatz
  betrifft das 5 von 208 Liganden vollständig und viele weitere teilweise.
- **Lineare Fragmente:** `cond(I_reg) ≈ 1.5e8`. Die Inversion verstärkt entlang
  der entarteten Achse um acht Größenordnungen. Die Eigenwert-Klemmung und der
  `omega`-Clamp bei ±1e3 in Minimal sind genau dafür da.

Dazu kommt ein Inductive-Bias-Argument: `u^R = log(R_t^T R_1)/(1-t)` ist eine
reine Winkelrate **ohne** Formabhängigkeit. Die Division durch `I` zwingt das
Netz, eine fragmentgrößenabhängige Vorkompensation zu lernen.

## Der Rahmenwechsel bleibt — und zwar zwingend

Das Ziel ist unter globaler Rotation **invariant**:
`(QR_t)^T (QR_1) = R_t^T R_1`. Jede äquivariante Netzausgabe ist es **nicht**.
Die Konjugation `R_t^T (.) R_t` schließt diese Lücke, und zwar für **jede**
Konstruktion — auch für das Drehmoment. Der Frame-Fix war kein Artefakt des
Newton-Euler-Wegs. Numerisch belegt in `audits/test_two_head_vector_field.py`.

## Parameterzuwachs

Produktionsmaße (6 Layer, 128 Kanäle, lmax=3):

| | Parameter |
|---|---:|
| Rumpf + 1 Kopf (= Minimal) | 21 654 513 |
| zweiter Kopf | 2 811 905 |
| gesamt | 24 466 418 |
| **Zuwachs** | **+12.99 %** |

## Geänderte Dateien

| Datei | Änderung |
|---|---|
| `src/sigmadock/net/model.py` | `force_block` → `trans_block`, neuer `rot_block`, `forward` gibt beide Felder |
| `src/sigmadock/diff/sigma_flow_generator.py` | `_compute_forces` liefert zwei Felder; neues `pool_fragment_fields`; `_compute_vector_field` auf gepoolte Größen |
| `src/sigmadock/diff/sampling.py` | beide Sampler-Blöcke auf dieselbe Kette (Training und Inferenz dürfen nicht auseinanderlaufen) |

Alles andere ist byte-identisch zu Minimal — Daten, Fragmentierung, Config,
`train.py`, Flow-Matcher.

`linear_mechanics`, `newton_maruyama` und `_compute_fragment_dynamics` bleiben
im Code, werden aber nicht mehr aufgerufen. Bewusst: sie dokumentieren die
Vorgängerkonstruktion und erlauben einen direkten Vergleich im Audit.

## Audits

```bash
python audits/test_two_head_vector_field.py   # 22 Checks: Geometrie, Rahmen, Entartung
python audits/test_two_head_gradients.py      # 16 Checks: Gradientenzuordnung
```

## Noch nicht getan

Kein Produktionslauf. Kein Lauf auf echten Daten — die Audits arbeiten mit
Stellvertreterrümpfen, weil ein echter EquiformerV2-Vorwärtslauf einen
vollständigen Graphen braucht und die geprüften Fragen nicht schärfer
beantwortet. Der erste echte Test ist ein kurzer GPU-Lauf auf ARC.
