"""Tests fuer die Pfadpruefung des 72h-Sanity-Gates.

Der Anlass: die vorherige Fassung pruefte auf TEILZEICHENKETTEN und haette
jeden SigmaFlow-Lauf getoetet, weil die Repowurzel auf ARC
SigmaFlow_Development_JulianMueller heisst und damit "SigmaFlow_Development"
enthaelt. Aufgefallen ist das erst beim Hinzufuegen eines dritten Arms --
die beiden wartenden Jobs haetten es nach Tagen Queue-Zeit gezeigt.

Diese Datei prueft nur pruefe_pfad(): der Codeteil braucht die echten
Module und laeuft deshalb nur auf ARC.

AUFRUF   python arc/test_final_sanity_gate.py
"""
from __future__ import annotations

import sys

from final_sanity_gate import pruefe_pfad

WURZEL = "/data/stat-cadd/shug8458"
REPO = f"{WURZEL}/SigmaFlow_Development_JulianMueller/SigmaFlow"

# (Beschreibung, Pfad, erwartetes Segment, soll durchgehen)
FAELLE = [
    ("Minimal auf ARC -- die Repowurzel enthaelt 'SigmaFlow_Development'",
     f"{REPO}/SigmaFlow_Minimal/src/sigmadock/__init__.py",
     "SigmaFlow_Minimal", True),
    ("EXP-110 auf ARC",
     f"{REPO}/SigmaFlow_FM_Specific/EXP-110_two_head_vector_field/src/sigmadock/__init__.py",
     "EXP-110_two_head_vector_field", True),
    ("SigmaDock auf ARC",
     f"{WURZEL}/SigmaDock_Reproduction_JulianMueller/sigmadock/src_sigmadock/__init__.py",
     "SigmaDock_Reproduction_JulianMueller", True),
    ("Windows-Trennzeichen",
     r"C:\repo\SigmaFlow_Minimal\src\sigmadock\__init__.py",
     "SigmaFlow_Minimal", True),

    ("EXP-110 erwartet, aber Minimal geladen",
     f"{REPO}/SigmaFlow_Minimal/src/sigmadock/__init__.py",
     "EXP-110_two_head_vector_field", False),
    ("Minimal erwartet, aber EXP-110 geladen",
     f"{REPO}/SigmaFlow_FM_Specific/EXP-110_two_head_vector_field/src/sigmadock/__init__.py",
     "SigmaFlow_Minimal", False),
    ("global installierte Kopie in site-packages",
     f"{WURZEL}/sigmaflow_env/lib/python3.11/site-packages/sigmadock/__init__.py",
     "SigmaFlow_Minimal", False),
    ("Variantenbaum statt Hauptbaum",
     f"{REPO}/SigmaFlow_Variants/d_frame_fix/src/sigmadock/__init__.py",
     "SigmaFlow_Minimal", False),
    ("SigmaDock erwartet, aber SigmaFlow geladen",
     f"{REPO}/SigmaFlow_Minimal/src/sigmadock/__init__.py",
     "SigmaDock_Reproduction_JulianMueller", False),
]


def main() -> int:
    fehler = 0
    for beschreibung, pfad, erwartet, soll_ok in FAELLE:
        try:
            pruefe_pfad(pfad, erwartet)
            ist_ok = True
            meldung = ""
        except SystemExit as e:
            ist_ok = False
            meldung = str(e)
        if ist_ok == soll_ok:
            print(f"  OK   {beschreibung}")
        else:
            fehler += 1
            erwartung = "durchgehen" if soll_ok else "abgelehnt werden"
            print(f"  FEHL {beschreibung}")
            print(f"       sollte {erwartung}; {meldung or 'ging durch'}")
    print(f"\n{len(FAELLE) - fehler}/{len(FAELLE)} bestanden")
    return 1 if fehler else 0


if __name__ == "__main__":
    sys.exit(main())
