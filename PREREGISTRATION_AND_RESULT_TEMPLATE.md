# Vorregistrierung und Ergebnisvorlage

> **Vor den Ergebnissen geschrieben.** Zweck: festzulegen, welche Fragen zählen
> und welche Zahlen sie beantworten, bevor irgendeine davon bekannt ist. Das
> schützt gegen nachträgliche Metrikauswahl — die häufigste stille Fehlerquelle
> in Methodenvergleichen.
>
> Wird nach den Läufen **nur ausgefüllt**, nicht umgeschrieben. Wenn sich eine
> Frage im Nachhinein als falsch gestellt erweist, wird das als *zusätzlicher*
> Abschnitt notiert, nicht durch Überschreiben kaschiert.

---

## 1. Primärfragen

### Q1 — Gleiches Trainingsbudget

> Wie schneidet SigmaFlow Minimal gegen SigmaDock ab, bei 24 h auf derselben
> GPU-Klasse, identischem Seed und identischem `max_epochs`?

**Primärmetrik:** Anteil Komplexe mit RMSD < 2 Å, Einzelsample, gemittelt über
10 Seeds.
**Test:** McNemar exakt auf gepaarten Komplexen, Bootstrap-CI, Bonferroni über
die Zahl der berichteten Vergleiche.

### Q2 — Gleiche NFE

> Welcher Generator erzeugt bessere Posen bei gleicher Anzahl
> Funktionsauswertungen?

`COMPARISON=controlled`, `NUM_STEPS=25` auf beiden Seiten.
**Vorbehalt, vorab notiert:** NFE ist zwischen einem Diffusions-Sampler und
einem ODE-Integrator nur bedingt vergleichbar. Q2 ist deshalb sekundär zu Q3.

### Q3 — Methodeneigene Einstellungen

> Welches Gesamtsystem ist wirksamer, wenn jede Methode ihre Vorgabe benutzt?

`COMPARISON=default`. Das ist die praxisnähere Frage.

### Q4 — Diversität

> Wie groß ist `Oracle@K − Single` je Modell?

Trennt Generierungsqualität von Diversität. K ∈ {1, 5, 10, 20, 40}.

### Q5 — Ranking

> Wie viel des Abstands zwischen Oracle@K und Single holt ein Ranking zurück?

`Top-1@K` gegen `Oracle@K`, plus Regret `RMSD_ranked − RMSD_oracle`.
Zunächst nur mit SigmaDocks methodeneigenem Score; eine gelernte Confidence
existiert nicht.

### Q6 — EXP-100 (sekundär)

> Ist SigmaFlow in einer inferenzsauberen Parametrisierung trainierbar?

Gegen **SigmaFlow Minimal**, nicht gegen SigmaDock. Ein Gleichstand ist bereits
das Ergebnis, das den Quellen-Strang freischaltet. Eine Verschlechterung ist
möglich und wäre ebenfalls ein Ergebnis.

---

## 2. Was *vorab* als Erfolg gilt

| Frage | Erfolg | Neutral | Misserfolg |
|---|---|---|---|
| Q1 | SigmaFlow ≥ SigmaDock bei < 2 Å | Differenz im CI von 0 | SigmaFlow klar darunter |
| Q4 | Oracle@10 ≥ 3× Single | 1.5–3× | < 1.5× ⇒ Multi-Sampling lohnt nicht |
| Q5 | Ranking holt ≥ 30 % des Abstands | 10–30 % | < 10 % ⇒ Ranking ist nicht der Engpass |
| Q6 | EXP-100 innerhalb des CI von Minimal | — | EXP-100 deutlich schlechter ⇒ Quellen-Strang gefährdet |

Diese Schwellen sind **vor** den Ergebnissen gesetzt und werden nicht
nachträglich verschoben.

---

## 3. Ergebnisvorlage

Alle Zellen mit `—` sind vor dem ARC-Lauf leer. Nach den Läufen ausfüllen,
nichts löschen.

### 3.1 Haupttabelle

| Modell | Variante | Single <2 Å | Single Median | Oracle@5 | Oracle@10 | Oracle@40 | Top-1@10 | Regret |
|---|---|---|---|---|---|---|---|---|
| SigmaDock 24h | raw | — | — | — | — | — | n/a | n/a |
| SigmaDock 24h | ranked | — | — | — | — | — | — | — |
| SigmaFlow Minimal 24h | raw | — | — | — | — | — | n/a | n/a |
| EXP-100 24h | raw | — | — | — | — | — | n/a | n/a |
| *SigmaFlow 12h (Referenz, vorhanden)* | raw | **3.9 %** | **4.98 Å** | **18.1 %** | **29.2 %** | n/a | n/a | n/a |
| *SigmaDock 12h (Referenz, vorhanden)* | raw | **9.1 %** | **4.48 Å** | — | **45.9 %** | n/a | n/a | n/a |

### 3.2 Zerlegung nach Freiheitsgrad

| Modell | Centroid-Fehler Median | Rotation Median | Rotation < 30 % | Aligned RMSD | TFD Median | TFD-Abdeckung |
|---|---|---|---|---|---|---|
| SigmaDock 24h | — | — | — | — | — | — |
| SigmaFlow Minimal 24h | — | — | — | — | — | — |
| EXP-100 24h | — | — | — | — | — | — |
| *SigmaFlow 12h (Referenz)* | — | **138.2°** | **2.1 %** | — | **0.253** | **100 %** |
| *SigmaDock 12h (Referenz)* | — | **123.8°** | **5.8 %** | — | — | — |
| *Haar-Referenz (Zufall)* | n/a | **132.3°** | **~2 %** | n/a | n/a | n/a |

> Die Zeile „Haar-Referenz" gehört in jede Rotationstabelle. Ohne sie ist
> „138°" eine Zahl; mit ihr ist es die Aussage „nicht von Raten unterscheidbar".

### 3.3 PoseBusters

| Modell | PB-valid | PB-valid ∧ RMSD < 2 Å | schwächster Check |
|---|---|---|---|
| SigmaDock 24h | — | — | — |
| SigmaFlow Minimal 24h | — | — | — |
| EXP-100 24h | — | — | — |

**PB-valid ∧ RMSD ≤ 2 Å ist die Schlagzeilenmetrik des Felds.** Sie wird
berichtet, auch wenn eine andere Zahl günstiger aussieht.

### 3.4 Rechenaufwand

| Modell | Schritte in 24 h | s/Schritt | NFE je Pose | Sampling-Zeit je Komplex |
|---|---|---|---|---|
| SigmaDock 24h | — | — | — | — |
| SigmaFlow Minimal 24h | — | — | — | — |
| EXP-100 24h | — | — | — | — |

> EXP-100 erzeugt zusätzlich einen ETKDGv3+MMFF-Konformer je Trainingsbeispiel.
> Erreicht es dadurch weniger Schritte in 24 h, ist der Vergleich **zeit**-fair,
> aber nicht **schritt**-fair. Beide Zahlen werden berichtet.

---

## 4. Regeln für die Auswertung

1. **Nächstgelegene Kristallkopie** wird gewertet. 84 der 209 PoseBusters-
   Dateien enthalten mehrere; naiv die erste zu nehmen erzeugte früher 42
   Phantom-Ausreißer.
2. **Der Seed steckt im Verzeichnis, nicht im Dateinamen.** Alle Ausgaben heißen
   `_seed0.sdf`. Wer auf den Dateinamen globt, bekommt zehnmal Seed 0.
3. **Oracle@K:** erst minimieren, dann schwellen. Nie RMSDs mitteln und den
   Mittelwert schwellen.
4. **Raw und ranked** werden nie in derselben Zeile berichtet.
5. **Fehlende Posen** werden gezählt und ausgewiesen, nie still weggelassen.
6. **TFD** wird nur mit Abdeckungsangabe berichtet (aktuell 100 % nach der
   Koordinatentransplantation).
7. Jede Zahl bekommt ihr `n`.

---

## 5. Was *nicht* berichtet wird

- Metriken, die erst nach Sicht der Ergebnisse gesucht wurden — es sei denn,
  sie werden ausdrücklich als *post hoc* gekennzeichnet.
- Einzelseed-Zahlen ohne die Streuung über Seeds. Die Streuung desselben
  Komplexes zwischen Seeds beträgt median 1.47 Å; ein Einzelseed-Vergleich
  unterhalb dieser Größenordnung ist bedeutungslos.
- Teilmengen von Komplexen ohne vorab festgelegtes Kriterium.
