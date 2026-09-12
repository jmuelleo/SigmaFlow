# Referenz: welcher PoseBusters-Satz ist welcher

Stand 2026-08-20. Angelegt, weil sich beim Nachgehen der Frage „warum 209 und
nicht 308" herausgestellt hat, dass unsere Datenkopie aus der **überholten**
Fassung des Benchmarks stammt.

## Dateien

| Datei | Inhalt |
|---|---|
| `posebusters_v2_308_ids.csv` | die offiziellen 308 IDs der Zeitschriftenfassung, Spalten `pdb_id,ccd_id` |
| `our_set_vs_posebusters_v2.csv` | unsere 208 gemessenen Komplexe mit Flag `in_posebusters_v2` |

## Die drei Zahlen

| Satz | n | was er ist |
|---|---:|---|
| PoseBusters v1 | 428 | der ursprünglich kuratierte Benchmark |
| **PoseBusters v2** | **308** | v1 minus 120 Strukturen mit Kristallkontakten. **Das benutzt SigmaDock.** |
| unsere ARC-Kopie | 209 | Teilsatz von v1, **nicht** von v2 |

Die Reduktion von 428 auf 308 ist von den Autoren dokumentiert
([Zenodo](https://zenodo.org/records/8278563)):

> During peer review we learned that some of the 428 structures contain crystal
> contacts (e.g. 5S8I_2LY). For the journal paper the results are reported on a
> subset containing 308 structures.

Das SigmaDock-Paper wertet auf diesen 308 aus, dazu Astex mit 85.

## Der Befund

Unsere 208 gemessenen Komplexe gegen die offizielle Liste:

| | |
|---|---:|
| Schnittmenge mit v2 | **151** |
| bei uns, **nicht** in v2 | **57** |
| in v2, nicht bei uns | 157 |

**Unsere Menge ist keine Teilmenge der 308.** Die 57 sind Strukturen, die die
Benchmark-Autoren wegen Kristallkontakten entfernt haben, und die alphabetisch
erste davon ist `5S8I_2LY` — genau ihr eigenes Beispiel. Damit ist belegt, dass
unsere Kopie aus v1 stammt.

## Warum 209: abgebrochene Entpackung (geklärt 2026-08-21)

Das Zenodo-Archiv liegt auf ARC unter
`data/posebusters_paper/posebusters_paper_data.zip` (52 MB) und **enthält alle
428 Komplexe**. Entpackt wurden davon nur **209**.

Drei Indizien belegen den Abbruch:

1. `astex_diverse_set` aus demselben Archiv hat exakt **85** Ordner, ist also
   vollständig. Beide wurden am 2026-07-05 um 19:31 geschrieben, 13 Minuten
   nach dem Download des Archivs.
2. Das Archiv enthält laut README auch `posebusters_benchmark_set_ids.txt`.
   Im entpackten Verzeichnis **fehlt** diese Datei. Alphabetisch steht sie
   hinter `posebusters_benchmark_set/`, der Entpackvorgang ist also mitten im
   grossen Ordner stehengeblieben.
3. `unzip -l` zaehlt 428 Komplexordner im Archiv, `ls` zaehlt 209 auf Platte.

**Es war also kein Filter und keine Auswahlentscheidung, sondern ein
unvollstaendiger `unzip`.** Damit erklaert sich auch, warum die 209 keine
Teilmenge von v2 sind: sie sind ein willkuerlicher Anfangsabschnitt von v1.

Weder SigmaDocks Repo noch das PoseBusters-Repo erzeugen 209; SigmaDocks README
kennt nur eine optionale Whitelist `posebusters_correct_ids.txt`, die eher in
Richtung v2 filtern wuerde.

## Wirkung auf die Ergebnisse (12h-Läufe, Oracle@10)

| Menge | n | SigmaFlow | SigmaDock | Median RMSD SF / SD |
|---|---:|---:|---:|---:|
| volle Kopie | 208 | 30.3 % | 47.6 % | 2.55 / 2.05 |
| **nur v2** | **151** | **33.8 %** | **53.0 %** | 2.37 / 1.91 |
| nur die aussortierten | 57 | 21.1 % | 33.3 % | 3.03 / 2.34 |

Die 57 sind für **beide** Arme deutlich schwerer, und das erklärt sich nicht
über die Fragmentzahl (4.72 gegen 4.44). Sie drücken die absoluten Zahlen um
3.5 Punkte (SigmaFlow) und 5.4 Punkte (SigmaDock).

**Der Methodenunterschied bleibt stabil**, rund 17 Punkte auf der vollen Menge
gegen rund 19 auf v2. Der interne Vergleich ist also nicht gefährdet, betroffen
sind nur die absoluten Zahlen.

## Betrifft es das Training?

**Nein.** `--train_exps pdbbind-general`, PoseBusters steht nur bei
`--val_exps` und `--test_exps`. Gradienten sehen den Satz nie.

Zwei Einschränkungen, damit die Aussage genau bleibt. PoseBusters ist der
**Validierungssatz**, und `ModelCheckpoint(monitor="loss_val/total",
save_top_k=3)` wählt danach Checkpoints aus. Wir sampeln aber aus
`last.ckpt`, nicht aus dem besten, und Early Stopping ist mit
`--early_stopping_patience 0` abgeschaltet, also greift die Auswahl faktisch
nirgends durch. Das ist ein glücklicher Umstand, keine Konstruktion.

Getestet werden sollte noch, ob einer unserer 209 Komplexe im PDBBind-general
Trainingssatz liegt. Das Risiko ist gering, weil der PoseBusters-Benchmark aus
Strukturen ab 2021 besteht und PDBBind v2020 davor endet, aber geprüft ist es
nicht:

```bash
# auf ARC
python - <<'EOF'
import csv, pathlib
ours = {r["complex"].split("_")[0].lower()
        for r in csv.DictReader(open("SigmaFlow_Evaluation/reference/our_set_vs_posebusters_v2.csv"))}
train = {p.name.lower() for p in pathlib.Path("/data/stat-cadd/shug8458/data/pdbbind/general-set").iterdir() if p.is_dir()}
print("Ueberlappung Training/Evaluation:", sorted(ours & train))
EOF
```

## Reproduzieren

```bash
python - <<'EOF'
import csv
off = {f"{r['pdb_id'].upper()}_{r['ccd_id'].upper()}"
       for r in csv.DictReader(open("SigmaFlow_Evaluation/reference/posebusters_v2_308_ids.csv"))}
ours = {r["complex"].upper()
        for r in csv.DictReader(open("SigmaFlow_Evaluation/reference/our_set_vs_posebusters_v2.csv"))}
print(len(ours & off), len(ours - off), len(off - ours))   # 151 57 157
EOF
```

## Herkunft der Liste

Die offizielle 308er-Liste liegt nicht im PoseBusters-Repo selbst. Übernommen
aus [inductive-bio/strong-docking-baseline](https://github.com/inductive-bio/strong-docking-baseline)
(`posebusters_308_ids.csv`) und **gegen eine zweite unabhängige Quelle geprüft**,
[degrado-lab/PoseBusters-Benchmark](https://github.com/degrado-lab/PoseBusters-Benchmark)
(`posebuster_benchmark.txt`). Beide stimmen auf 307 von 308 überein; die
einzige Abweichung ist dokumentiert, `7D6O` wurde dort als obsoletes Duplikat
durch `8J79` ersetzt.
