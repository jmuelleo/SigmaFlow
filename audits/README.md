# Architektur-Audits

Skripte, die Behauptungen über die Architektur gegen den tatsächlich
ausgeführten Code prüfen. Kein Kommentar und keine Dokumentzeile wird
geglaubt; jede Frame- und Symmetrieaussage wird numerisch bestätigt.

Ergebnisse sind in `Texte/theory.tex`, Kapitel *The Output Heads, Audited*
dokumentiert.

## Aufruf

Beide Skripte laufen lokal, ohne GPU, ohne Checkpoint.

```bash
python audits/frame_audit.py
python audits/s2_gauge_audit.py
```

Sie importieren `SigmaFlow_Minimal/src` und den Dummy-Datensatz unter
`SigmaFlow_Minimal/notebooks/dummy_data`. Keine Datei außerhalb von
`audits/` wird verändert.

## `frame_audit.py`

Beantwortet: in welchem Frame lebt jeder Output und wie transformiert er
unter einer globalen Rotation `Q`.

| Abschnitt | Prüft |
|---|---|
| A | Aktivierung isolieren: S²- gegen Gate-Aktivierung |
| B | Gruppenwirkung auf `R_t`, aus `_apply_transformations` abgeleitet |
| C | volle Kette Kraft → Drehmoment → `dW` → `omega` → `pred_u_t_R` |
| D | Ziel `u_t_R` gegen Vorhersage — ist der Loss rotationsinvariant |

Kernergebnis: alle Weltgrößen bei 1e-15; `pred_u_t_R` transformiert **nicht**
invariant (1.20), sondern durch Konjugation (2.3e-15), weil die Gruppe auf
`R_t` durch `Q R_t Qᵀ` wirkt und nicht durch `Q R_t`.

## `s2_gauge_audit.py`

Misst den Äquivarianzbruch der S²-Aktivierung bei Produktionsgröße
(128 Kanäle, 6 Blöcke, 15 M Parameter).

Kernergebnis: ~10 % relativ, sowohl als Nicht-Determinismus des Forward-Passes
als auch als Äquivarianzfehler. Mit Gate-Aktivierung 1e-15.

**Gemessen bei zufälliger Initialisierung.** Ob Training den Effekt
unterdrückt, ist offen und braucht einen echten Checkpoint —
`PENDING ARC VALIDATION`.

## Zwei Anmerkungen zur Methodik

`zero_init_last` muss auf `False` stehen. Der Default `True` nullt die letzte
Schicht, das Netz gäbe exakt 0 aus, und jeder Äquivarianztest wäre leer
erfüllt.

Der Zeitpfad läuft in float32, weil `timestep_embedder.py` hart
`timesteps.float()` aufruft. Er speist ausschließlich ℓ=0-Kanäle und kann die
Äquivarianz nicht beeinflussen; alles Geometrische läuft in float64.
