# EXP-101 — Source Distance Audit

Status: **BEREIT** (Code geschrieben, Tests grün, wartet auf ARC).
Braucht **keinen Checkpoint und keine GPU**.

> **Wo der Code liegt:** `arc/exp101_distance_audit.py` (+ `arc/exp101_distance_audit.slurm`,
> Tests in `arc/test_exp101_distance_audit.py`). Dieser Ordner enthält nur diese
> Beschreibung — das Skript liegt bei den übrigen Jobs, weil es gegen die
> **EXP-100**-Codebasis läuft und nicht gegen einen eigenen Variantenbaum.

## Fragestellung

Die Quelle des Flow Matchings ist uninformiert:

```
trans_0 ~ N(0, I)          r3_flow_matcher.sample_init
R_0     ~ Haar auf SO(3)   so3_flow_matcher.sample_init
```

Bevor EXP-102/103 eine konditionierte Quelle bauen, muss gemessen sein, ob
überhaupt Struktur da ist, die eine Alternative ausnutzen könnte.

## Zwei Punkte, die schon vor dem Lauf aus dem Code beantwortet sind

1. **Die Koordinaten sind taschenzentriert.** `sigma_flow_generator.py:499`
   zieht `pocket_com` ab, `:501` teilt durch `DIMENSIONAL_SCALE`.
   `trans_0 ~ N(0, I)` sitzt also im Taschenzentrum. Die Translation hat damit
   bereits einen informativen Prior; die Rotation hat kein Gegenstück, weil
   Haar keinen Mittelwert hat. **Genau diese Asymmetrie ist die Motivation des
   ganzen Strangs.**

2. **`R_1 ≡ I` in SigmaFlow_Minimal.** `get_fragment_com_and_rot` gibt die
   Zielrotation unbedingt als Identität zurück (`rots.append(I3)`). Dort wäre
   jede Rotationsdistanzmessung tautologisch: das Karcher-Mittel lauter
   Identitäten ist die Identität, der Abstand überall 0, der „Gewinn" also die
   vollen 132° — das Gate riefe mit maximaler Zuversicht *„EXP-102 bauen"*.
   Das Skript bricht in Minimal deshalb per `assert` ab und läuft nur gegen
   **EXP-100**, wo `R_1` relativ zu einem inferenzverfügbaren Referenzkonformer
   definiert ist.

## Der konzeptionelle Kern

Eine Quelle wird bei `t = 0` gezogen. Zu diesem Zeitpunkt haben die Fragmente
noch **keine Position** — die wird ja gerade erst erzeugt. Eine
„fragmentweise konditionierte Quelle" kann sich also nicht auf die Umgebung
des Fragments beziehen. Konditioniert werden kann nur auf
*(Fragmentidentität, Ligandtopologie, Tasche als Ganzes)*.

Deshalb misst H1 eine **globale** Ausrichtung — eine Rotation für alle
Fragmente eines Liganden. Das ist das Stärkste, was ohne Positionswissen
sauber definierbar ist.

## Was gemessen wird

| Zeile | Was | zulässig als Quelle? |
|---|---|---|
| **H0** | Haar (uninformiert) — die Referenz | ja, der Status quo |
| **H1** | Hauptachsen Ligand → Tasche | **ja**, inferenzverfügbar |
| **Hc** | beste *konstante* Rotation (Karcher) | nein — obere Schranke für alles Nicht-Konditionierte |
| **Hid** | `R_0 = I` | nein — Kontrolle, zeigt den Abstand des Konformer-Eigenframes |

Zusätzlich Translation: `‖p_0 − p_1‖` für `N(0,I)` vs. `p_0 = 0`, und die
empirische Prüfung, ob `DIMENSIONAL_SCALE = 2.7 Å` wirklich die
Standardabweichung der Fragmentschwerpunkte ist (im Projekt bisher nirgends
nachgerechnet).

**Hc ist die eigentlich scharfe Referenz für H1.** Liegt H1 nicht klar unter
Hc, nutzt die Heuristik die Tasche nicht wirklich aus — sie findet dann nur
eine global häufige Orientierung wieder.

## Das Gate

| Gemessen | Folge |
|---|---|
| beste **zulässige** Heuristik senkt den Median um > 10° | EXP-102 motiviert |
| ≤ 10° | EXP-102 **nicht bauen** — der Engpass ist dann der Rotationskanal selbst, nicht die Quelle |

Die 10° sind eine Konvention dieses Audits, keine Größe aus einem Paper.

## Ausführen

```bash
sbatch arc/exp101_distance_audit.slurm
python arc/test_exp101_distance_audit.py     # 11 Checks, lokal
```

## Numerischer Befund aus der Testentwicklung

Die erste Fassung von `so3_angle_deg` benutzte `arccos((tr−1)/2)` und zeigte
bei float32 einen Selbstabstand von 2.3·10⁻² Grad. Das passte auffällig genau
zur Schranke `sqrt(2·eps32) = 2.8·10⁻²`, die aus der unendlichen Ableitung des
`arccos` bei ±1 folgt — und ließ sich bequem als „Grenze der Darstellung"
abhaken.

Das war falsch. Es war **Auslöschung in der Formel**: bei kleinen Winkeln ist
`tr ≈ 3`, und `6 − 2·tr` subtrahiert zwei fast gleiche große Zahlen. Direkt
als `‖R − I‖_F` gerechnet sinkt derselbe Wert auf 8·10⁻⁶ Grad, und der Fehler
wird **linear** in `eps` statt `sqrt(eps)`.

Für dieses Gate (Effekte ab 10°) war beides irrelevant. Festgehalten wird es,
weil der Denkfehler generalisiert: *ein Messfehler, der zu einer plausiblen
theoretischen Schranke passt, ist damit noch nicht erklärt.*
