# Thesis-Visualisierungen — Fragmentierung und Dockingerfolg

Stand 2026-08-20. Alles hier ist aus `data/fragments_vs_performance.csv`
reproduzierbar; die Abbildungen sind keine Handarbeit.

```bash
python -m visualization.plot_fragments \
    --joined "Thesis Visualisierungen/data/fragments_vs_performance.csv" \
    --lang en --out-dir "Thesis Visualisierungen/figures_en"
```

`--lang de` erzeugt dieselben Abbildungen mit deutschen Beschriftungen.

---

## Inhalt

```
figures_en/   Abbildungen mit englischen Beschriftungen (für die Thesis)
figures_de/   dieselben Abbildungen auf Deutsch (für Arbeitslogs)
data/         die Zahlen dahinter
```

| Abbildung | Zeigt |
|---|---|
| **A** `A_fragment_distribution` | Histogramm der Fragmentzahl, mit kumulativer Kurve und `D = 6F` als zweiter Achse |
| **B** `B_fragments_vs_size` | Fragmentzahl gegen Schweratome und gegen Torsionsbindungen, mit Pearson r |
| **C** `C_fragments_vs_performance` | Erfolg und bester RMSD gegen Fragmentzahl, beide Verfahren |

Jeweils als **PDF** (Vektor, für den Satz) und **PNG** (300 dpi).

| Datei | Inhalt |
|---|---|
| `data/fragments_vs_performance.csv` | eine Zeile je Ligand — die Rohdaten |
| `data/fragment_distribution.csv` | eine Zeile je Fragmentklasse — Histogramm plus Erfolgsquoten |
| `data/summary_statistics.csv` | die Kennzahlen der Verteilung |

---

## Was die Daten sind

**Fragmente `F`.** Jeder Ligand wird an seinen Torsionsbindungen in `F` starre
Fragmente zerlegt; die Generierung arbeitet damit in `D = 6F` Dimensionen —
drei Rotations- und drei Translationsfreiheitsgrade je Fragment.

Gezählt wird über die **minimalen Cut-Sets**, also genau so, wie das Training
mit `fragmentation_strategy="random"` fragmentiert. `F = min_cuts + 1`.

> ⚠️ Nicht zu verwechseln mit der Zahl **4.56** aus dem IS-1-Lauf. Die entsteht
> unter `fragmentation_strategy="canonical"` und ist eine andere Grösse. Für
> alles Trainingsbezogene gilt **4.51**.

**Erfolg** = Oracle@10: mindestens eine der zehn Sampling-Posen unter 2 Å,
symmetriekorrigiert und gegen die nächstgelegene kristallographische Kopie
gerechnet. Das ist eine **Obergrenze ohne Ranking**.

**Seed 0** ist eine einzelne Ziehung. Das Ziehungsrauschen ist rund zehnmal so
gross wie der Methodenunterschied — Einzelwerte deshalb nicht interpretieren.

**Checkpoints:** die 12h-Referenzläufe, SigmaFlow Job 8541310, SigmaDock Job
8541439. Sampling: Jobs 8554147 und 8554149, je 10 Seeds × 209 Komplexe.
Fragmentzählung: Job 8607523.

---

## Die Verteilung

| | Fragmente `F` | Zustandsdimension `D = 6F` |
|---|---|---|
| Mittel | **4.51** (SD 2.11) | 27.1 |
| Median | **4** | 24 |
| Modus | **5** | 30 |
| IQR | 3–6 | 18–36 |
| q90 | 8 | 48 |
| Spanne | **1–11** | 6–66 |

Rechtsschief (Pearson-Schiefe +0.73). 38.5 % haben ≤ 3 Fragmente, 18.8 %
haben ≥ 7, und **5 Liganden (2.4 %) haben genau eines** — keine Torsion, also
ein starrer Körper.

**`F` ist je Ligand deterministisch.** Obwohl das Training zufällig aus den
minimalen Cut-Sets zieht, schwankt die Fragmentzahl bei **keinem** der 208
Liganden zwischen den Alternativen — variiert wird nur, *welche* Bindungen
geschnitten werden. `D` ist damit über Epochen konstant.

---

## Die Befunde

**1. Die Fragmentzahl ist die dominante Schwierigkeitsachse.**
Jedes zusätzliche Fragment multipliziert die Erfolgs-Odds mit **0.53**
(`p < 1e-4`), für beide Verfahren. Spearman gegen den besten RMSD: ρ = +0.57
(SigmaFlow) und +0.62 (SigmaDock). Von `F ≤ 3` auf `F ≥ 4` fällt SigmaFlow von
56.2 % auf 14.1 %, SigmaDock von 75.0 % auf 30.5 % (beide Fisher `p ≈ 1e-10`).
Ab `F = 7` trifft SigmaFlow keinen einzigen Komplex mehr (0/39).

**2. SigmaFlow ist gleichmässig schlechter, nicht überproportional.**
Der Interaktionsterm `Arm × F` ist mit **`p = 0.85`** nicht signifikant; das
Odds-Verhältnis liegt stabil bei 0.43 (`F ≤ 3`) bzw. 0.37 (`F ≥ 4`). Die auf
der Prozentskala wachsende Lücke ist überwiegend ein Bodeneffekt. Die
Hypothese „die Rotationsbehandlung skaliert schlechter mit der Zahl starrer
Körper" wird von diesen Daten **nicht** gestützt.

**3. `F` misst Flexibilität, nicht Grösse.**
Pearson gegen die Torsionszahl **r = 0.93**, gegen die Atomzahl nur 0.81 (siehe
Abbildung B). Das stützt die Lesart, dass die Schwierigkeit aus der
Konditionierungslast wächst, die mit der Zahl unabhängig zu platzierender
starrer Körper zunimmt — beweisen lässt es sich mit diesen Daten nicht, weil
`D` eine deterministische Funktion von `F` ist.

**4. Der einfachste Fall wird nicht zuverlässig gelöst.**
Von den fünf Liganden mit genau einem Fragment löst SigmaDock 5/5,
**SigmaFlow 3/5**. Bei n = 5 statistisch nichts, als Diagnosehinweis aber
wertvoll: dort gibt es keine Fragmentkoordination, nur eine globale
Rototranslation.

---

## Beim Zitieren mitnennen

Die Modelle sahen **12 GPU-Stunden**, das entspricht **5.8 Epochen** oder
**2.3 %** des Originaltrainings (Paper: 256 Epochen, 384 GPU-h auf 4×A100).
Der Benchmark umfasst **209** Komplexe, das Paper wertet auf 308 aus.
Erfolg hier ist Oracle@10 **ohne** Ranking, die Paper-Zahl 79.9 % ist Top-1 aus
40 Seeds **mit** Ranking und zusätzlich an PoseBusters-Validität gekoppelt.

---

## Bekannte Lücke

Ein Ligand von 209 fehlt: bei ihm lief die kombinatorische Aufzählung der
Cut-Sets in das 120-Sekunden-Zeitlimit. Es wurde **nichts** geschätzt. Die
Abbildungen nennen deshalb `n = 208`.

Die zusammengeführte CSV enthält nur die 208 gemessenen Liganden — die
Abbildungen können daraus die fehlende Messung nicht mehr ausweisen. Die
vollständige Liste inklusive der leeren Zeile liegt auf ARC unter
`arc_runs/FRAGCOUNT-posebusters_8607523/fragment_counts_posebusters.csv`.
