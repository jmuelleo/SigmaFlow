
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
