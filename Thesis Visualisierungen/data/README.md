
### Dritte Konfiguration: Paper-Setup auf 80 Seeds (2026-08-24)

```bash
cd SigmaFlow_Variants/posebusters_full_comparison
python papersetup80.py
```

| Datei | Inhalt |
|---|---|
| `validity_papersetup80.csv` | Anteil je Ziehung mit Wilson-Intervall, plus Anteil der Komplexe ohne einen einzigen Treffer in 80 Ziehungen |
| `selection_validity_papersetup80.csv` | Random / Top-1 nach Vinardo / Oracle je k, für beide Validitätsdefinitionen |

Diese Dateien enthalten **keine** RMSD-Größen. Die SDF-Dateien der Seeds 40–79
liegen noch auf ARC; Validität und affinitätsbasierte Auswahl brauchen sie
nicht. Sobald sie da sind, liefert `build_thesis_datasets.py sampled` die
gemeinsame Kenngröße „< 2 Å UND valide" auf derselben Basis.

Zwei Punkte für die Beschriftung von Abbildungen aus
`selection_validity_papersetup80.csv`:

- Bei **Vinardo ist negativ gut**. Die Kurve `top1_affinity_pct` wählt das
  Minimum, nicht das Maximum.
- Die Kurve zur Zielgröße `valid_mit_protein` ist teilweise selbstbezüglich:
  Vinardo und die neun PoseBusters-Protein-Checks bewerten beide
  Protein-Ligand-Abstände. Die Zielgröße `valid_ohne_protein` ist der saubere
  Test, und dort hebt der Scorer fast nichts.

### Vierte und fünfte Konfiguration (2026-08-24)

```bash
cd SigmaFlow_Variants/posebusters_full_comparison
python build_thesis_datasets.py sampled   # Paper-Setup, 40 Seeds, JETZT mit gnina
python build_thesis_datasets.py nfe5      # bound, 5 Schritte, 40 Seeds, ohne gnina
```

Neu bzw. geändert:

| Datei | Hinweis |
|---|---|
| `ranker_comparison_papersetup40.csv` | **neu** — fünf Ranker gegen RMSD < 2 Å |
| `heuristic_grid_papersetup40.csv` | **neu** — SigmaDocks Heuristik über ein Parametergitter |
| `selection_curves_papersetup40.csv` | hat jetzt die Spalte `top1_affinity_pct` |
| `per_draw_nfe5_40seeds.csv` | **neu** — 5 Integrationsschritte |
| `selection_curves_nfe5_40seeds.csv` | **neu** — ohne `top1_affinity_pct`, nie gescort |
| `ranker_comparison_80seeds.csv` | `random`-Zeile korrigiert, siehe unten |

**Korrektur an der Zufallsgrundlinie.** Die `random`-Zeile im Rankervergleich
schwankte mit k, obwohl sie das nicht darf: die Zufallsmatrix wurde einmal vor
der Wiederholungsschleife gezogen, sodass bei k = NS in allen Wiederholungen
dieselbe Pose gewählt wurde. Betroffen waren nur diese Zeile und daraus
berechnete Trefferquoten, nicht die Top-1-Werte. Beide
`ranker_comparison_*.csv` sind neu erzeugt.
