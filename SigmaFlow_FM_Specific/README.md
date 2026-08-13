# SigmaFlow_FM_Specific

Jede Unterordner-Variante entsteht aus einer **frischen Kopie von
`SigmaFlow_Minimal/`** (Tag `sigflow-minimal-baseline-v1`) und aendert
**genau eine** Sache. Keine Variante uebernimmt Aenderungen einer anderen.

Kopierbefehl fuer eine neue Variante:

```bash
cp -r SigmaFlow_Minimal/{src,scripts,conf,slurm} SigmaFlow_FM_Specific/<variante>/
```

Kombinationen erst, nachdem die Einzelaenderungen getestet sind
(-> `SigmaFlow_Combinations/`).
