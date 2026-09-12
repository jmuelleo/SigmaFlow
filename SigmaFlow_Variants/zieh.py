"""Top-1 bei K Ziehungen JE KOMPLEX -- der festgelegte 1000-Ziehungen-Schaetzer.

WARUM ES DIESE DATEI GIBT
    `zellen.top1` benutzt genau diese Methodik, liefert aber nur den
    Mittelwert ueber die Komplexe. Fuer gepaarte Tests braucht es die Werte je
    Komplex. `final_tables.rule` liefert zwar je Komplex, rechnet aber die
    GESCHLOSSENE FORM -- ein anderer Schaetzer derselben Groesse. In diesem
    Projekt ist der gezogene Schaetzer festgelegt; deshalb hier eine Fassung,
    die beides kann.

METHODIK, identisch zu zellen.top1
    Je Komplex N_ZIEHUNGEN Teilmengen der Groesse K aus den vorhandenen Posen,
    ohne Zuruecklegen. In jeder Teilmenge waehlt die Regel eine Pose; ob sie
    das Ziel erfuellt, ist eine Null oder Eins. Der Mittelwert dieser
    Ziehungen ist der Wert des Komplexes.

    Bei K >= n gibt es genau eine Teilmenge; dann wird direkt gerechnet.

    Die Posen werden EINMAL nach der Regel sortiert, beste zuerst. Danach ist
    die Gewinnerin einer Teilmenge die mit der kleinsten gezogenen Position.
"""
import numpy as np
import pandas as pd

from zellen import N_ZIEHUNGEN, ZIEH_SEED, _ordnung

# final_tables/plot_ranking benutzen englische Regelnamen, zellen.py deutsche.
# Beide werden akzeptiert, damit es nur EINE Implementierung gibt.
ALIAS = {"heuristic": "mixed", "vinardo": "gnina", "draw": "zug",
         "oracle": "orakel", "mixed": "mixed", "gnina": "gnina",
         "zug": "zug", "orakel": "orakel", "filter": "filter"}


def top1_je_komplex(m: pd.DataFrame, ziel: str, K: int, regel: str = "mixed",
                    n_ziehungen: int = N_ZIEHUNGEN,
                    seed: int = ZIEH_SEED) -> pd.Series:
    """Wert je Komplex in [0, 1]. `regel` wie in zellen.top1."""
    regel = ALIAS.get(regel, regel)
    rng = np.random.default_rng(seed)
    out = {}
    for code, g in m.groupby("complex"):
        n = len(g)
        k = min(K, n)
        if regel == "zug":
            out[code] = float(g[ziel].mean())
            continue
        if regel == "orakel":
            y = g[ziel].to_numpy(bool)
            if k >= n:
                out[code] = float(y.any())
            else:
                pick = rng.random((n_ziehungen, n)).argpartition(k - 1, axis=1)[:, :k]
                out[code] = float(y[pick].any(axis=1).mean())
            continue
        y = _ordnung(g, regel)[ziel].to_numpy(float)
        if k >= n:
            out[code] = float(y[0])
            continue
        pos = rng.random((n_ziehungen, n)).argpartition(k - 1, axis=1)[:, :k]
        out[code] = float(y[pos.min(axis=1)].mean())
    return pd.Series(out)
