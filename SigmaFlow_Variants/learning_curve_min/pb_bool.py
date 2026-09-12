"""Eine PoseBusters-Pruefspalte verlaesslich nach bool wandeln.

WARUM ES DIESE FUNKTION GIBT
    `bust_table` liefert NaN, wenn ein Modul ausfaellt -- etwa
    "Energy ratio module failed because UFF parameters missing". Sobald ein
    NaN in einer Spalte steht, kann Pandas sie nicht mehr als bool halten und
    macht float64 daraus. Beim Schreiben wird True dann zu "1.0", nicht zu
    "True".

    Das naheliegende `spalte.astype(str).str.lower() == "true"` zaehlt jede so
    kodierte bestandene Pruefung als DURCHGEFALLEN -- still, ohne Fehler, mit
    plausibel aussehendem Ergebnis. Am 2026-08-27 hat das eine monoton
    STEIGENDE Erfolgsquote (75,0 -> 77,9 -> 81,8 %) als Einbruch auf 44,3 %
    dargestellt und beinahe zu einer erfundenen Aussage ueber Ueberanpassung
    gefuehrt.

    Besonders tueckisch: bei den lokalen Ligand-CSVs mischen sich beide
    Kodierungen in EINER Datei, weil die Bloecke aus acht Prozessen kommen und
    nur manche ein NaN enthalten. Bei den ARC-Redock-CSVs schreibt ein Seed in
    einem Stueck -- dort kippt dann die ganze Spalte auf einmal.

KONVENTION FUER NaN
    NaN heisst "Pruefung nicht durchfuehrbar" und gilt als NICHT bestanden.
    So haelt es PoseBusters' eigenes CLI, und so sind die 12-h-Zahlen der
    Arbeit gerechnet. Weil die betroffenen Molekuele in allen Armen dieselben
    sind, verzerrt das keinen Vergleich.

UNBEKANNTE WERTE BRECHEN AB
    Ein stiller Zaehlfehler ist schlimmer als ein Abbruch. Alles, was weder
    eindeutig wahr noch eindeutig falsch ist, loest deshalb eine Ausnahme aus.
"""
import pandas as pd

WAHR = {"true", "1", "1.0"}
FALSCH = {"false", "0", "0.0", "nan", "none", ""}


def als_bool(spalte: pd.Series) -> pd.Series:
    # `.fillna("nan")` ist NICHT ueberfluessig. Unter dem neuen
    # Zeichenkettentyp (dtype "str", pandas >= 2.1 mit
    # future.infer_string bzw. pandas 3) propagieren die `.str`-Zugriffe
    # NaN, statt es wie bei dtype "object" in die Zeichenkette "nan" zu
    # wandeln. Ohne diese Zeile bleibt ein echter Fliesskomma-NaN uebrig,
    # der weder in WAHR noch in FALSCH steht und die Ausnahme ausloest --
    # so geschehen am 2026-09-05 auf ARC bei den PDBBind-Kristallposen.
    v = spalte.astype(str).str.strip().str.lower().fillna("nan")
    unbekannt = sorted(set(v.unique()) - WAHR - FALSCH)
    if unbekannt:
        raise ValueError(f"Spalte {spalte.name!r}: unerwartete Werte "
                         f"{unbekannt[:5]} -- nicht raten, nachsehen.")
    return v.isin(WAHR)


def _selbsttest():
    import numpy as np
    # gemischt, wie es real vorkommt: bool-Block und float-Block konkateniert
    s = pd.Series([True, False, 1.0, 0.0, np.nan, "True", "False"],
                  name="internal_energy")
    assert list(als_bool(s)) == [True, False, True, False, False, True, False]
    # reine float-Spalte: der Fall, der bei ARC eine ganze Datei kippen wuerde
    assert als_bool(pd.Series([1.0, 1.0, 0.0], name="x")).sum() == 2
    # reine bool-Spalte bleibt, wie sie war
    assert als_bool(pd.Series([True, True, False], name="x")).sum() == 2
    # neuer Zeichenkettentyp: dort ueberlebt NaN die .str-Kette
    try:
        s = pd.Series(["True", None, "False"], name="x", dtype="str")
    except TypeError:
        s = pd.Series(["True", np.nan, "False"], name="x", dtype=object)
    assert list(als_bool(s)) == [True, False, False]
    # Unfug bricht ab, statt still zu zaehlen
    try:
        als_bool(pd.Series(["ja"], name="x"))
    except ValueError:
        pass
    else:
        raise AssertionError("unbekannter Wert haette abbrechen muessen")
    print("pb_bool: 5 Selbsttests bestanden")


if __name__ == "__main__":
    _selbsttest()
