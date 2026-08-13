# Repository-Struktur (Stand 2026-08-13)

Aufgeräumt, damit in einigen Wochen sofort erkennbar ist, was was ist.

```
SigmaFlow/
├── SigmaDock/                       REFERENZ - NIEMALS AENDERN
│                                    Originale Diffusions-Implementierung.
│                                    Baseline fuer jeden Vergleich.
│
├── SigmaFlow_Minimal/               ⭐ DIE FINALE MINIMAL-CHANGES-BASELINE
│                                    = SigmaDock, Diffusion durch Riemannsches
│                                    Flow Matching ersetzt, sonst nichts.
│                                    Nur Code (9.3 MB): src/ scripts/ conf/
│                                    slurm/ diagnostics/ notebooks/
│                                    Erzeugt Checkpoint 0-08-11_18-00-41
│                                    (global_step 13750).
│
├── SigmaFlow_Variants/              ZIEL-2-EXPERIMENTE + Auswertungswerkzeuge
│   ├── d_frame_fix/                 Quelle von SigmaFlow_Minimal, plus die
│   │                                Ergebnisordner (experiments/ 924 MB).
│   │                                Code identisch mit SigmaFlow_Minimal.
│   ├── a_time_weighting/            Ziel 2: Zeitgewichtung -> Nullergebnis
│   ├── b_rotation_data_space_loss/  Ziel 2: Rotation im Datenraum -> Nullergebnis
│   ├── c_anchor_atom_distance_loss/ Ziel 2: NO-OP, nie wirksam getestet
│   ├── posebusters_full_comparison/ AUSWERTUNGSWERKZEUGE (aktiv genutzt)
│   ├── sigmadock_control_scripts/   SigmaDock-Gegenstuecke der SLURM-Skripte
│   └── ...                          weitere Vergleichsordner
│
├── archive/                         ABGELEGT, NICHT GELOESCHT
│   ├── SigmaFlow_Development_pre_framefix/
│   │                                Der versionierte Entwicklungsstand.
│   │                                OHNE Frame-Fix -> abgeloest.
│   │                                Enthaelt die Juli-Dummy-Sampling-Laeufe,
│   │                                auf die add_dummy_cases.py noch zeigt.
│   ├── SigmaFlow_Adapted/           frueher Zwischenstand, ohne Frame-Fix
│   └── SigmaFlow_MinimalChange/     frueher Zwischenstand, ohne Frame-Fix
│
├── papers/                          Theoriereferenzen
├── Texte/                           Schreibmaterial
└── *.md / *.txt                     STATUS.md, RESULTS.md, Berichte
```

## Was verschoben wurde und warum

| Von | Nach | Grund |
|---|---|---|
| `SigmaFlow_Development/` | `archive/SigmaFlow_Development_pre_framefix/` | Trug den Rahmenfehler; abgeloest durch `SigmaFlow_Minimal/`. Nicht geloescht, weil dort die Juli-Dummy-Laeufe liegen. |
| `SigmaFlow_Adapted/` | `archive/` | Frueher Zwischenstand, ohne Frame-Fix, kein Ergebnis haengt daran. |
| `SigmaFlow_MinimalChange/` | `archive/` | Ebenso. Der Name war zusaetzlich irrefuehrend - er ist NICHT die Minimal-Changes-Version. |

## Was neu ist

`SigmaFlow_Minimal/` — Kopie des verifizierten Codes aus `d_frame_fix`.
Verifiziert: `src/`, `scripts/`, `conf/` sind byteweise identisch mit der
Quelle, der Frame-Fix ist aktiv.

## Was angepasst wurde

`SigmaFlow_Variants/posebusters_full_comparison/add_dummy_cases.py` — drei
Pfadkonstanten auf die neuen Orte gezeigt. Getestet: laeuft weiterhin durch.

## Warum d_frame_fix bleibt

`SigmaFlow_Minimal/` enthaelt nur Code. Die 924 MB Checkpoints und
Sampling-Ausgaben liegen weiter in `d_frame_fix/experiments/`, weil alle
bisherigen Ergebnisse und die laufenden ARC-Auswertungen darauf zeigen.
Codeseitig sind beide identisch — `d_frame_fix` ist ab jetzt der
Ergebnis-, `SigmaFlow_Minimal` der Code-Ordner.

## Wo spaetere Erweiterungen hingehoeren

Ziel-2-Varianten kommen als neue Ordner unter `SigmaFlow_Variants/`, jeweils
als kontrollierte Ein-Aenderung-Ablation gegen `SigmaFlow_Minimal/`.
`SigmaFlow_Minimal/` selbst wird dabei NICHT veraendert.
