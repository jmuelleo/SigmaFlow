# FM Source Roadmap — die konditionierte Quellverteilung als Forschungsstrang

> Status: **Planungsdokument.** Kein Code außerhalb von `EXP-100_state_reparam/`.
> EXP-101 bis EXP-106 sind Absichtserklärungen mit definierten Metriken und
> Abbruchkriterien, keine Implementierungen.

---

## 0. Die Leitfrage

Diffusion zwingt die Quellverteilung fest: das Vorwärtsrauschen endet in einem
vorgegebenen Prior (isotropes Gauß, Haar auf SO(3)). Flow Matching tut das
nicht. Der bedingte Pfad wird zwischen **frei wählbaren** Endpunkten
konstruiert. Damit wird `p_0` zu einem Entwurfsobjekt statt zu einer Konstante.

> **Kann SigmaFlow davon profitieren, dass Flow Matching eine frei wählbare,
> konditionierte Quellverteilung erlaubt?**
>
> ```
> p_0(z | P, L)  ──►  p_1(z | P, L)
> ```

Das ist der zentrale Flow-Matching-spezifische Strang dieser Arbeit. Alles
Weitere in diesem Dokument hängt daran.

---

## 1. Warum EXP-100 zwingend vorausgeht

In SigmaFlow-Minimal gibt `get_fragment_com_and_rot` unbedingt `R_1 = I`
zurück. Die Fragmentkoordinaten stammen dort aus der gebundenen Pose, also
bedeutet die Identitätsrotation „genau die Orientierung, die das Fragment in
der Kristallpose bereits hat".

Die präzise Formulierung des Problems — die grobe Fassung („jede Reduktion der
Quellentropie ist Leakage") ist zu stark und wurde verworfen:

> In der Minimal-Parametrisierung ist die Rotationsidentität bereits **relativ
> zu der ground-truth-orientierten Fragmentrepräsentation** definiert. Eine
> Quellkonzentration um `I` ließe sich deshalb nicht als inferenzseitig
> gewonnene Information interpretieren.

Erst wenn `R_1` eine echte, aus inferenzverfügbaren Größen bestimmte
Zielrotation ist, wird die Aussage „die Quelle liegt näher am Ziel" überhaupt
messbar und nicht tautologisch. **EXP-100 ist deshalb keine Verbesserung,
sondern eine Voraussetzung.**

---

## 2. Wissenschaftliche Einordnung — was ist wirklich FM-spezifisch?

Diese Tabelle ist der Grund, warum das Dokument existiert. In der Thesis darf
nichts als FM-exklusiv erscheinen, was es nicht ist.

| Baustein | Einordnung | Begründung |
|---|---|---|
| Frei wählbare Quellverteilung | **FM-specific** | Diffusion bindet `p_0` an den Endzustand des Vorwärtsprozesses. FM konstruiert den Pfad direkt zwischen den Endpunkten. |
| Informative / konditionierte Quelle `q(z_0\|P,L)` | **FM-specific** | Folgt unmittelbar aus der freien Wahl von `p_0`. |
| Design der Quellverteilung als eigenes Forschungsobjekt | **FM-specific** | In Diffusion gibt es dieses Objekt nicht. |
| Gelerntes Proposal → deterministische Flow-Verfeinerung | **FM-enabled** | Prinzipiell auch mit Diffusion konstruierbar (z. B. über Bridges), unter FM aber die natürliche und billigste Konstruktion. |
| Quellkonditioniertes multimodales Sampling | **FM-enabled** | Multi-Start-Sampling gibt es überall; dass die Modi aus der *Quelle* stammen, ist FM-natürlich. |
| Mehrere Kandidaten erzeugen | **general** | Jeder generative Docker kann das. |
| Confidence-/Ranking-Kopf | **general** | Steht orthogonal zum generativen Verfahren. |
| Oracle@K als Metrik | **general** | Reine Auswertungskonvention. |
| Early Pruning über Zwischen-Confidence | **general** | Compute-Heuristik, kein FM-Argument. |
| Confidence-gesteuerte Schrittweiten | **general** | Adaptive Integration ist Standard-ODE-Technik. |

Merksatz für die Thesis:
> Die *Wahl* der Quelle ist FM-spezifisch. Was man mit mehreren Kandidaten
> anschließend tut, ist es nicht.

---

## 3. Rollentrennung — nicht vermischen

Drei Komponenten, drei verschiedene Fragen. Die Trennung ist bewusst und muss
in Code, Auswertung und Text durchgehalten werden.

| Komponente | Objekt | Frage |
|---|---|---|
| Conditional Source | `q_φ(z_0 \| P, L)` | **Wo sollten wir anfangen zu suchen?** |
| SigmaFlow | `v_θ(x_t, t)` | **Wie transportieren wir den Kandidaten zur finalen Pose?** |
| Confidence | `C_ψ(ẑ_1, P, L)` | **Welcher fertige Kandidat ist wahrscheinlich korrekt?** |

Ein häufiger Fehler in der Literatur ist, Confidence-Gewinne als Gewinne des
generativen Modells auszuweisen. Die Ablationen in §8 sind genau so gebaut,
dass das hier nicht passieren kann.

---

## 4. Zielarchitektur (Roadmap, **nicht** EXP-100)

```
                    (P, L)
                      │
          ┌───────────▼───────────┐
          │   q_φ(z_0 | P, L)     │   Conditional Source     [FM-specific]
          └───────────┬───────────┘
                      │
        z_0^(1), z_0^(2), ..., z_0^(K)
                      │
          ┌───────────▼───────────┐
          │      SigmaFlow        │   v_θ, ODE-Integration
          └───────────┬───────────┘
                      │
        ẑ_1^(1), ẑ_1^(2), ..., ẑ_1^(K)
                      │
          ┌───────────▼───────────┐
          │        C_ψ            │   Confidence / Ranking   [general]
          └───────────┬───────────┘
                      │
                 ranked final pose
```

EXP-100 ist **keine** Stufe dieser Pipeline. Es ist die Zustandsdarstellung,
auf der sie überhaupt erst definierbar ist.

---

## 5. Stufenplan

### EXP-101 — Source Distance Audit *(Messung, kein Training)*

Bevor irgendetwas gelernt wird: **wie weit ist die uninformierte Quelle vom
Ziel entfernt, und wie weit käme eine einfache Heuristik?** Rein
diagnostisch, auf vorhandenen Daten, ohne einen einzigen Trainingsschritt.

Erhoben wird, je Fragment:

- `‖p_0 − p_1‖` — anfängliche Translationsdistanz
- `d_SO(3)(R_0, R_1) = ‖log(R_0ᵀ R_1)‖` — anfängliche Rotationsdistanz
- die Referenzverteilung: Haar auf SO(3) hat Winkel-**Median 132.3°**,
  Mittelwert 126.5°. Jede Kandidatenquelle muss sich daran messen.

Abbruchkriterium: reduziert keine inferenzverfügbare Heuristik die mediane
Rotationsdistanz spürbar unter 132.3°, ist der ganze Strang schwach motiviert
und EXP-102 wird nicht gebaut.

### EXP-102 — Heuristische konditionierte Quelle *(einfach, interpretierbar)*

Bewusst **kein** gelerntes Netz. Eine simple, inferenzverfügbare,
interpretierbare Quelle, damit ein positiver Effekt eindeutig der *Idee* und
nicht der *Kapazität* zugeschrieben werden kann.

```
R_0 ~ q(R | P, L)      statt   R_0 ~ Haar
p_0 ~ q(p | P, L)      statt   p_0 ~ N(0, I)
```

Kandidaten für die Heuristik (Auswahl offen, in EXP-102 zu entscheiden):
Trägheitshauptachsen des Fragments gegen Hauptachsen der Taschenöffnung,
Taschenschwerpunkt plus Formanisotropie, klassische Scoring-Startpunkte.

### EXP-103 — Gelernte konditionierte Quelle

```
q_φ(z_0 | P, L)
```

als kleines Proposal-Netz — zunächst nur für die Rotation, `q_φ(R_0 | P, L)`,
später ggf. gemeinsam `q_φ(p_0, R_0 | P, L)`.

**Bewusst nicht deterministisch.** Docking ist multimodal; eine Abbildung
`R_0 = f_φ(P, L)` kann Multimodalität nicht darstellen. Angestrebt ist

```
R_0^(k) ~ q_φ(R | P, L)
```

sodass die Quelle mehrere plausible Orientierungen trägt.

**Risiko, das ausdrücklich festgehalten wird:** löst `q_φ` bereits fast das
vollständige Docking, ist das System faktisch *Docking-Modell → Docking-Modell*
und die wissenschaftliche Interpretation bricht zusammen. Gegenmaßnahmen von
Anfang an: kleine Kapazität, grobe Verteilung, Proposal statt Posenprädiktor.
Als Kontrolle wird `q_φ` **allein** ausgewertet — ist es schon nahe am Ziel,
ist der FM-Beitrag nicht mehr isolierbar.

### EXP-104 — Multi-Sampling aus der konditionierten Quelle

```
z_0^(1..K) ~ q_φ(z_0 | P, L)     →     ẑ_1^(1..K)
```

Der Gewinn gegenüber einem einzelnen Zug: **Exploration verschiedener
plausibler Docking-Modi.** Enthält die Quelle mehrere Modi, kann ein einzelner
Zug den falschen erwischen; mehrere Züge erhöhen die Chance auf mindestens
einen guten Start.

**Die Kernhypothese des ganzen Strangs:**

> Eine konditionierte Quelle deckt mit deutlich weniger Samples relevante Modi
> ab als eine uninformierte. Beispielsweise 10 konditionierte Starts gegen 40
> Haar-Starts.

Interessante Ausgänge, in absteigender Stärke:
1. gleiche Genauigkeit bei weniger Compute (Samples oder NFE),
2. bessere Genauigkeit bei gleichem Compute,
3. gleiche Genauigkeit bei gleichem Compute → Idee trägt nicht.

Ausgang 1 wäre ein echter, FM-spezifischer Effizienzbeitrag.

### EXP-105 — Confidence / Ranking *(general, klar so gekennzeichnet)*

Aus EXP-104 folgt zwangsläufig: **welche der K Posen wählen wir?**

```
C_ψ(ẑ_1, P, L)  ≈  P(RMSD < 2 Å)          k* = argmax_k C_ψ(ẑ_1^(k), P, L)
```

Wird als **allgemeiner**, nicht FM-exklusiver Baustein geführt. Der Gewinn
durch Ranking darf niemals dem Flow-Matching zugeschrieben werden.

### EXP-106 — Early Pruning *(Idee, general)*

```
K konditionierte Starts → je 2–3 ODE-Schritte → C_t → beste 3 behalten
                        → nur diese bis t = 1 integrieren
```

Setzt voraus, dass Confidence auch zu Zwischenzeiten `C_t` funktioniert. Reine
Compute-Ersparnis, kein FM-Argument.

**Noch später, nur notiert:** confidence-konditionierte Integration
`v_θ(x_t, t, c_t)` oder confidence-gesteuerte adaptive Schrittweiten. Ebenfalls
**general extension, not intrinsically Flow-Matching-specific.**

---

## 6. Die drei Leistungszahlen — immer getrennt ausweisen

| Zahl | Definition | Was sie misst |
|---|---|---|
| **Single Sample** | Qualität eines einzelnen Laufs | Generierungsqualität |
| **Oracle@K** | `min_k RMSD(ẑ_1^(k), z*)` | Obergrenze bei perfektem Ranking → **Diversität** |
| **Ranked Top-1@K** | Qualität der tatsächlich gewählten Pose | **Rankingqualität** |

Die Lücke zwischen Oracle@K und Ranked Top-1@K **ist** die Ranking-Schwäche.
Die Lücke zwischen Single Sample und Oracle@K **ist** der Diversitätsgewinn.
Nur so lässt sich sagen, welcher Bestandteil welchen Effekt hat.

Headline-Metrik bleibt wie bisher **PB-valid ∧ RMSD ≤ 2 Å**, mit McNemar-Test,
Bootstrap-CIs und Bonferroni-Korrektur über die Varianten.

---

## 7. Metriken für die Quelle selbst

Unabhängig von der Endleistung — eine Quelle kann gut sein, ohne dass das
Gesamtsystem sofort besser wird.

| Metrik | Definition |
|---|---|
| Initial translation distance | `‖p_0 − p_1‖` |
| Initial rotation distance | `d_SO(3)(R_0, R_1)`, gegen Haar-Median 132.3° |
| **Coverage@K** | `min_k d(z_0^(k), z_1)` — wie nah kommt der beste von K Zügen? |
| **Source diversity** | paarweise Distanzen der K Züge — kollabiert die Verteilung? |
| **Mode coverage** | decken die Züge verschiedene plausible Orientierungen ab? |

Coverage@K und Source diversity gehören zusammen gelesen: eine Quelle, die
kollabiert, hat exzellente Diversitätswerte von null und ist wertlos für
Multi-Sampling.

---

## 8. Ablationsraster

Damit am Ende exakt sagbar ist, welcher Bestandteil welchen Effekt hat:

| | Quelle | Samples | Ranking |
|---|---|---|---|
| **A** | Haar / N(0,I) | 1 | — |
| **B** | konditioniert | 1 | — |
| **C** | Haar / N(0,I) | K | Oracle@K |
| **D** | konditioniert | K | Oracle@K |
| **E** | konditioniert | K | `C_ψ` |
| **F** | konditioniert | K + Early Pruning | `C_ψ` |

Die Vergleiche, die zählen:

- **B − A** → Effekt der Quelle allein
- **C − A** → Effekt von Multi-Sampling allein
- **D − C** → Effekt der Quelle *unter* Multi-Sampling ← die eigentliche These
- **E − D** → Effekt des Rankings *(general, nicht FM zuschreiben)*
- **F − E** → Compute-Ersparnis durch Pruning, bei gleicher Genauigkeit

Compute muss dabei mitgeführt werden (Samples × NFE **und** Wall-Clock),
sonst ist „besser" nicht interpretierbar.

---

## 9. Denkbare Ausgänge der gelernten Quelle *(konzeptionell, keine Entscheidung)*

- **Rotation:** Parameter einer Verteilung auf SO(3)
- **Translation:** `μ(P,L)`, `Σ(P,L)`
- **Multimodal:** `q_φ(z) = Σ_k π_k q_k(z)`

Bewusst offen bis EXP-103.

---

## 10. Übertrag aus EXP-100

Zwei bereits gemessene Punkte, die für diesen Strang unmittelbar relevant sind:

- **Symmetrie.** 53 % der Moleküle haben mehr als einen Graphautomorphismus
  (Median 2). `R_1` ist streng genommen eine Äquivalenzklasse `[R_1]`. Für
  Coverage@K und Rotationsbewertung heißt das: Distanzen müssen perspektivisch
  über die Symmetriegruppe minimiert werden, sonst zählen korrekte Posen als
  falsch. Bisher nicht implementiert.
- **Kopf-Schwanz-Flips.** Die gemessene Rotationsverteilung von SigmaFlow ist
  **bimodal**, nicht antikorreliert — ein zweiter Modus nahe der Umkehrung.
  Genau so ein Fehlerbild ist der plausibelste Kandidat dafür, dass eine
  konditionierte Quelle hilft: sie könnte den falschen Modus von vornherein
  unwahrscheinlicher machen.

---

## 11. Reihenfolge und Priorität

| Priorität | Experiment | Aufwand | Voraussetzung |
|---|---|---|---|
| 1 | **EXP-100** ARC-Lauf | mittel | fertig, wartet auf ARC |
| 2 | **EXP-101** Distance Audit | gering | EXP-100 trainiert |
| 3 | **EXP-102** heuristische Quelle | mittel | EXP-101 positiv |
| 4 | **EXP-104** Multi-Sampling (auf A/C zuerst) | gering | keine — A/C brauchen keine gelernte Quelle |
| 5 | **EXP-103** gelernte Quelle | hoch | EXP-102 positiv |
| 6 | **EXP-105** Confidence | hoch | EXP-104 zeigt Oracle-Ranking-Lücke |
| 7 | **EXP-106** Early Pruning | mittel | EXP-105 funktioniert |

Hinweis zur Reihenfolge: **EXP-104 in den Varianten A und C ist billig und
sofort machbar**, sobald EXP-100 ein Modell hat — es braucht keine neue
Quelle, nur K Züge aus der bestehenden. Die Oracle@K-Lücke, die dabei sichtbar
wird, sagt bereits, ob sich der ganze Strang lohnt: ist Oracle@K kaum besser
als Single Sample, ist das Problem nicht die Startverteilung.

---

## 12. Abgrenzung

Ausdrücklich **nicht** Teil dieses Strangs, obwohl technisch benachbart:
exakte Likelihood, Affinitätsvorhersage, Architekturwechsel, Änderungen an
Datenpipeline oder Backbone. Diese laufen, falls überhaupt, als eigene
Zielgruppe 3 und dürfen nicht in den FM-Vorteil hineinargumentiert werden.
