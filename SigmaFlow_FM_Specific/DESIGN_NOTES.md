# Entwurfsnotizen — bis an die Implementierungsgrenze, ohne Implementierung

> **Status: ENTWURF. Kein Code.** Diese Datei existiert, damit nach dem
> Rotationsdiagnose-Ergebnis auf ARC sofort implementiert werden kann, ohne
> vorher noch einmal zu entwerfen. Jede Notiz ist so weit getrieben, dass nur
> noch Tippen fehlt.
>
> Was hier **nicht** passieren darf: eine dieser Varianten zu bauen, bevor die
> Diagnose sagt, welche gebraucht wird. Siehe „DO NOT IMPLEMENT YET" am Ende.

---

## Inhalt

| Notiz | Zweig | Voraussetzung |
|---|---|---|
| [D1 Heuristische konditionierte Quelle](#d1) | Quelle | EXP-101 positiv |
| [D2 Gelernte konditionierte Quelle](#d2) | Quelle | D1 positiv |
| [D3 Direkter Rotationskopf](#d3) | Repräsentation | Diagnose = DIRECTION-NOISE |
| [D4 Endpunkt-Parametrisierung](#d4) | Repräsentation | Diagnose = SHRINKAGE |
| [D5 Staged Probability Path](#d5) | Pfad | — |
| [D6 Confidence: Datensatz](#d6) | Ranking | Multi-Sampling gemessen |
| [D7 Early Pruning](#d7) | Ranking | D6 funktioniert |

---

<a name="d1"></a>
## D1 — Heuristische konditionierte Quelle

**Kategorie A (FM-spezifisch).** Aufwand: gering. Keine trainierbaren Parameter.

### Der Punkt, der beim Schreiben von EXP-101 auffiel

Eine Quelle wird bei `t = 0` gezogen. Zu diesem Zeitpunkt haben die Fragmente
noch **keine Position** — die wird ja gerade erst erzeugt. Eine „fragmentweise
konditionierte Quelle" kann sich also nicht auf die Umgebung des Fragments
beziehen. Konditionierbar ist nur auf

```
(Fragmentidentität, Ligandtopologie, Tasche als Ganzes)
```

Das ist deutlich schwächer als es klingt und begrenzt den erreichbaren Gewinn.
Alle Kandidaten unten respektieren diese Grenze.

### Kandidaten

| | Eingaben | Definition | Äquivariant? | Leakage |
|---|---|---|---|---|
| **H1 Hauptachsen global** | Konformer-Punktwolke, Taschen-Punktwolke | `Q = V_pocket V_ligᵀ` aus den Trägheitshauptachsen, eine Rotation für alle Fragmente | ✅ beide Frames rotieren mit | ✅ keine |
| **H2 Anisotropie-gewichtet** | wie H1 plus Eigenwerte | wie H1, aber Konzentration `κ` skaliert mit der Anisotropie der Tasche: kugelförmige Tasche → nahe Haar | ✅ | ✅ |
| **H3 Taschenöffnungs-Achse** | Hauptachse des Hohlraums, Längsachse des Liganden | rotiert die Ligand-Längsachse auf die Hohlraum-Längsachse, Restfreiheitsgrad um diese Achse uniform | ✅ | ✅ |
| **H4 Fragment-Anker-Richtung** | Vektor Fragment-COM → Anker | ~~richtet den Ankervektor aus~~ | ✅ | ❌ **verworfen** — braucht die Fragmentposition, die es bei `t=0` nicht gibt |

**H4 ist ausdrücklich abgelehnt.** Es klingt am gezieltesten und ist genau
deshalb die gefährlichste Option: es setzt Wissen voraus, das der Inferenzpfad
nicht hat.

### Vorzeichenfixierung — die Falle

Eigenvektoren sind nur bis aufs Vorzeichen bestimmt. Vier gleichwertige
rechtshändige Frames sind möglich. Ohne Fixierung wäre die „Heuristik" in
Wahrheit gleichverteilt über vier Orientierungen — und würde als plausibles
Ergebnis getarnt kaum besser als Haar abschneiden. `exp101_distance_audit.py`
fixiert das dritte Moment (Schiefe positiv) und erzwingt `det = +1`.

### Als konzentrierte Verteilung statt Punktschätzung

```
R_0 ~ IG-SO(3)(Q, κ)        bzw.  R_0 = Q · exp(hat(ε)),  ε ~ N(0, σ² I₃)
```

`σ → ∞` reproduziert Haar, `σ → 0` die deterministische Heuristik. **σ ist der
Ablationsparameter**, mit dem sich die Frage „hilft Information oder hilft nur
weniger Entropie?" trennen lässt.

### Abbruchkriterium

`EXP-101` misst `d_SO(3)(R_0, R_1)` gegen die Haar-Referenz (Median 132.3°).
Unter 10° Gewinn wird D1 nicht gebaut.

*Erste Messung auf 9 Dummy-Komplexen (44 Fragmente): H1 erreicht 119.0° gegen
Haar 138.1°, also 19° Gewinn. Ermutigend, aber n ist viel zu klein — die
belastbare Messung läuft auf PoseBusters.*

---

<a name="d2"></a>
## D2 — Gelernte konditionierte Quelle `q_φ(R_0 | P, L)`

**Kategorie A.** Aufwand: hoch. **Nur nach positivem D1.**

| Frage | Entscheidung |
|---|---|
| Eingaben | Taschen-Atomwolke (Typ + Koordinaten), Ligandgraph, **keine** Fragmentpositionen |
| Architektur | ein kleiner äquivarianter Encoder, ≤ 5 % der Parameter von EquiformerV2 |
| Ausgabe | Mischung `Σ_k π_k · IG-SO(3)(R_k, κ_k)`, `k ≤ 4` |
| Warum multimodal | Docking ist multimodal; eine deterministische Abbildung `R_0 = f_φ(P,L)` kann das nicht darstellen und kollabiert auf den Modus-Mittelwert — dieselbe Schrumpfung, die wir beim Vektorfeld vermuten |
| Ziehen | `k ~ Cat(π)`, dann `R = R_k exp(hat(ε))`, `ε ~ N(0, κ_k⁻¹ I)` |
| Trainingsziel | NLL von `R_1` unter `q_φ`; getrennt von `v_θ` trainiert |
| Äquivarianz | `q_φ(QR | QP, QL) = q_φ(R | P, L)` — als Test, nicht als Annahme |
| Kapazitätsgrenze | bewusst klein, siehe Risiko unten |

### Das Risiko, das den ganzen Strang entwerten kann

Löst `q_φ` bereits fast das vollständige Docking, ist das System faktisch
*Docking-Modell → Docking-Modell* und der FM-Beitrag nicht mehr isolierbar.

**Pflichtkontrolle:** `q_φ` **allein** auswerten (ohne Flow). Liegt sein bester
von K Zügen schon nahe am Ziel, ist das Ergebnis nicht interpretierbar.

### Diagnosen, die mitlaufen müssen

- **Modus-Kollaps:** mittlere paarweise `d_SO(3)` zwischen K Zügen; nahe null = kollabiert
- **Entropie:** mittlere Entropie von `q_φ` gegen die Haar-Entropie
- **Coverage@K:** `min_k d(R_0^(k), R_1)`
- **Leakage:** `q_φ` auf permutierten Protein-Ligand-Paaren — muss deutlich schlechter werden

---

<a name="d3"></a>
## D3 — Direkter äquivarianter Rotationskopf

**Kategorie C (allgemein, nicht FM).** Aufwand: mittel.
**Nur bei Diagnose = DIRECTION-NOISE-LIKE oder BOTH.**

### Der aktuelle Pfad, exakt

```
model.py:407      force_block: SO2EquivariantGraphAttention
model.py:718-722  forces = force_block(...).embedding.narrow(1,1,3).view(-1,3)   [N,3]
                  ↓  EIN Output-Block, ein l=1-Vektor je Atom
sigma_flow_generator.py:726   total_force  = Σ_i f_i                    → ΔT = F/m
sigma_flow_generator.py:731   total_torque = Σ_i (r_i−c) × f_i          → ω  = I⁻¹τ
```

Translation ist das **nullte**, Rotation das **erste Moment** desselben Feldes.
Gemessen auf 711 realen Fragmenten: der Rotationskanal verstärkt Rauschen
**8.8×** stärker und verträgt relativ zur eigenen Zielamplitude nur **0.25×**
so viel (bei 3–4-Atom-Fragmenten 0.14×).

### Vorschlag

Ein **zweiter** `l=1`-Output-Block, über die Fragmentatome gemittelt:

```
ω_F = mean_{i ∈ F} g_i          statt      ω_F = I⁻¹ Σ_i (r_i − c) × f_i
```

**Äquivarianz:** unter `Q ∈ SO(3)` gilt `g_i → Q g_i`, also `mean → Q · mean` —
dieselbe Konstruktion, die die Translation bereits benutzt. Da nur eigentliche
Rotationen auftreten (`det = +1`), spielt der Unterschied axialer/polarer Vektor
hier keine Rolle. *Bei Spiegelungen wäre er relevant; die kommen nicht vor.*

### Tensorformen

| Größe | Form |
|---|---|
| `g` (neuer Kopf) | `[N, 3]` |
| Fragmentzuordnung | `frag_idx_map`, `[N]` |
| `ω_F` | `[B·F, 3]` |
| `pred_u_t_R` nach Frame-Fix | `R_tᵀ hat(ω_F) R_t`, `[B·F, 3, 3]` |

### Parameterzuwachs

Ein zweiter `SO2EquivariantGraphAttention`-Block ≈ Größe des vorhandenen
`force_block`. Bei Produktionsdimensionen einige Prozent des Gesamtmodells.
**Vor der Ablation exakt auszählen** — sonst ist der Vergleich durch die
Kapazität konfundiert. Faire Gegenkontrolle: SigmaFlow-Minimal mit einem
gleich großen, aber ungenutzten Zusatzblock.

### Minimaler Diff

`net/model.py` (zweiter Block + Rückgabe als Tupel) · `_compute_forces` ·
`linear_mechanics` (Drehmoment entfällt) · `newton_maruyama` (`I⁻¹` entfällt) ·
`_predict_fragment_updates`. **Bricht Checkpoint-Kompatibilität.**

### Nötige Tests

Äquivarianz von `ω_F` unter globalem `Q`; Invarianz unter Translation;
Nulltest (konstantes `g` → konstantes `ω`); Gradientenfluss in beide Köpfe;
und die Kontrolle, dass die Translation **unverändert** bleibt.

---

<a name="d4"></a>
## D4 — Endpunkt-Parametrisierung `R̂_1`

**Kategorie B (FM-enabled).** Aufwand: mittel.
**Nur bei Diagnose = SHRINKAGE-LIKE.**

### Warum das die passende Antwort auf Schrumpfung ist

Mittelung von Rotations**vektoren** schrumpft gegen null. Mittelung von
Rotations**matrizen** mit anschließender Projektion auf SO(3) tut das **nicht** —
sie liefert eine gültige Rotation voller Magnitude. Ist die beobachtete
Nullwirkung Schrumpfung, ist das der strukturell passende Gegenentwurf.

Exakte Analogie zu `x_0`- statt `ε`-Vorhersage bei Diffusion — deshalb
Kategorie B, nicht A.

### Parametrisierungswahl

| Option | Bewertung |
|---|---|
| **6D + Gram-Schmidt** | **empfohlen.** Stetig, keine Singularitäten, Standard seit Zhou et al. 2019 |
| Quaternion | Doppelüberdeckung `q ~ −q`; Verlust muss das abfangen |
| Lie-Algebra `so(3)` | genau die Darstellung, die schrumpft — wäre selbstwidersprüchlich |
| 3×3 + SVD-Projektion | funktioniert, teurer, Gradient durch SVD heikel |

### Verlust und Anbindung

```
L_R = d_SO(3)(R̂_1, R_1)² = ‖log(R̂_1ᵀ R_1)‖²
u_t = log(R_tᵀ R̂_1) / (1 − t)
```

**Die numerische Falle:** der Faktor `1/(1−t)` verstärkt Fehler in `R̂_1` nahe
`t = 1` unbeschränkt. Gegenmittel: `t_max < 1` beim Training (es gibt bereits
`epsilon_t`), oder `u_t` clippen. Muss vor der Ablation entschieden werden.

**Auxiliär oder Ersatz?** Empfehlung: **Ersatz**, sonst mischt man zwei
Änderungen. Eine auxiliäre Variante wäre eine eigene, spätere Ablation.

---

<a name="d5"></a>
## D5 — Staged Probability Path

**Kategorie B.** Aufwand: mittel. **Priorität: niedrig.**

### Was die Mathematik sagt

Unter dem geodätischen Pfad gilt exakt

```
R_t = R_0 exp(t log(R_0ᵀR_1))  ⇒  u_t = log(R_tᵀR_1)/(1−t) = log(R_0ᵀR_1)
```

— **konstant in t.** Eine Umparametrisierung `s_R(t)` skaliert nur die
**Magnitude** (`u_t = s'_R(t) log Δ`), nicht die **Richtung**. Die Richtung ist
bei jedem `t` die volle Antwort. **Der Schwierigkeitskern bleibt also bestehen.**

### Der einzige Mechanismus, der trotzdem trägt

Friert man die Rotation für `t < t*` ein, ist bei `t > t*` die Translation
bereits gelöst, und die Orientierung muss in einem **korrekt platzierten**
Taschenkontext bestimmt werden statt gleichzeitig mit einer falschen
Platzierung. Das ist eine echte Konditionierungsverbesserung — aber sie kommt
aus der **Reihenfolge**, nicht aus dem Zeitplan.

### Bewertung der Varianten

| Variante | Urteil | Begründung |
|---|---|---|
| Standardpfad | Referenz | — |
| **Translation-first / Rotation-delayed** | **UNCLEAR** | echter Mechanismus, aber der Kern bleibt; und die globale Zeitgewichtung (Variante a) war bereits **wirkungslos** |
| Torsion-late | **NOT WORTH IT** | SigmaFlow hat gar keinen Torsionskanal — Torsion entsteht implizit aus der Starrfragment-Zerlegung |
| Gelernte Zeitpläne `s_R(t;φ)` | **NOT WORTH IT** (jetzt) | zusätzliche Parameter, Effekt nicht isolierbar |

### Wechselwirkung mit D1

Beide adressieren „das Ziel ist früh nicht bestimmbar", auf verschiedenen Wegen:
D1 verkleinert die Unsicherheit, D5 verschiebt den Zeitpunkt. Sie sind
**nicht additiv zu erwarten** und dürfen nicht gemeinsam eingeführt werden,
bevor jede einzeln gemessen ist.

### Bekannte Nebenwirkung

Ein gestaffelter Pfad ist nicht mehr die kürzeste Geodäte in `SE(3)`. Die
Pfadkrümmung steigt, was **mehr** NFE erfordern kann. Da die NFE-Kurve
aktuell flach ist (5→200), wäre das verkraftbar — aber es ist mitzumessen.

---

<a name="d6"></a>
## D6 — Confidence: der Datensatz

**Kategorie C (allgemein).** Aufwand: hoch. Kein Training in dieser Runde.

### Zielwahl — `P(RMSD < 2 Å)` ist vermutlich **nicht** das beste Ziel

Bei einer Positivrate von 4.4 % ist es stark unbalanciert und liefert kaum
Gradient. Empfohlene Reihenfolge:

1. **Regression auf `log RMSD`** — dichtes Signal, nutzt jeden Datenpunkt
2. **Paarweises Ranking innerhalb eines Komplexes** — das ist die tatsächliche
   Aufgabe und eliminiert komplexspezifische Schwierigkeitsoffsets
3. `P(RMSD < 2 Å)` als **Kalibrierungskopf obendrauf**, nicht als Hauptziel

PoseBusters-Validität als Ziel wäre falsch: sie korreliert nur schwach mit RMSD,
und beide Methoden versagen bei denselben Checks.

### Erste Version: nur Endposen

```
Eingabe:  (Protein, Ligand, ẑ_1)        Ausgabe: Skalar
```

Kein Trajektorienzustand, keine Zeitabhängigkeit. Das ist die einfachste
Version, die den Oracle-Top1-Abstand angreift.

### Datensatzaufbau

| | |
|---|---|
| Quelle | K Posen je Komplex aus dem trainierten Generator |
| K | 10 (vorhanden), später 40 |
| Label | RMSD zur nächstgelegenen Kristallkopie |
| **Split** | **auf Komplexebene**, nie auf Posenebene |
| Leckagegefahr | Posen desselben Komplexes in Train **und** Val ⇒ das Modell lernt den Komplex, nicht die Qualität |
| Zusätzlich | Split nach Proteinsequenzähnlichkeit, sonst leaken Homologe |
| Klassenbalance | 4.4 % positiv ⇒ Gewichtung oder Fokal-Verlust nötig |
| Format | Parquet/CSV mit `complex_id, pose_id, rmsd, split` + SDF-Pfad |

### Später: Zwischenzustände

Für `C_t` würden Zustände bei `t ∈ {0.1, …, 0.9}` gespeichert und **dasselbe
Endlabel** angehängt. Das ist die übliche Konstruktion, aber sie hat eine
Schwäche, die dokumentiert werden muss: bei kleinem `t` ist der Ausgang noch
nicht determiniert, das Label ist dort teilweise Rauschen. Als Zielgröße wäre
`P(Erfolg | z_t)` korrekt, das Label ist aber eine einzelne Realisierung.

---

<a name="d7"></a>
## D7 — Early Pruning

**Kategorie C.** Nur Spezifikation.

```
K Starts → N_early ODE-Schritte → C_t → beste M behalten → bis t=1 integrieren
```

Ersparnis: `K·N_total` → `K·N_early + M·(N_total − N_early)`.

### Was gemessen werden muss

| Größe | Warum |
|---|---|
| Gesamt-NFE mit und ohne Pruning | die eigentliche Ersparnis |
| Wall-clock | NFE ≠ Zeit, wenn Batching sich ändert |
| Erfolgsrate | darf nicht fallen |
| **Erhaltene Oracle-Coverage** | wie viele der guten Trajektorien wurden fälschlich verworfen |
| Pruning-Fehlerrate | Anteil der bei `t_early` verworfenen Läufe, die am Ende gut geworden wären |

Die vierte und fünfte Zeile sind der Kern: eine Compute-Ersparnis, die die
besten Trajektorien wegwirft, ist keine.

**Voraussetzung:** `C_t` muss bei kleinem `t` überhaupt trennscharf sein. Ist
sie es nicht, ist Pruning wertlos — und genau das ist zuerst zu messen, bevor
irgendetwas gebaut wird.

---

## DO NOT IMPLEMENT YET

Nichts davon wird gebaut, bevor die jeweilige Voraussetzung erfüllt ist:

| Notiz | Blockiert durch |
|---|---|
| D1 | EXP-101 Distance Audit auf PoseBusters |
| D2 | D1 positiv |
| D3 | Rotationsdiagnose = DIRECTION-NOISE-LIKE oder BOTH |
| D4 | Rotationsdiagnose = SHRINKAGE-LIKE |
| D5 | niedrige Priorität; nicht vor D3/D4 |
| D6 | Multi-Sampling-Messung auf dem 24h-Modell |
| D7 | D6 funktioniert und `C_t` ist früh trennscharf |

Der Grund für diese Strenge: jede vorgezogene Implementierung verzweigt die
Codebasis auf Basis einer Vermutung. Der Symmetrie-Fall hat gezeigt, wie teuer
das wäre — dort hätte eine plausible Hypothese („53 % Automorphismen") zu einer
Variante geführt, die nachweislich 0 von 467 Fehlern erklärt.
