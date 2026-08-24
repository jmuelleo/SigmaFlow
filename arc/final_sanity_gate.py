"""Prueft vor dem 72h-Lauf, dass der RICHTIGE Codebaum geladen ist.

Faellt das hier durch, sind 72 GPU-Stunden gerettet. Faellt es faelschlich
durch, sind Tage Wartezeit in der Queue verloren -- beide Fehlerrichtungen
sind teuer, deshalb steht die Pruefung in einer eigenen, testbaren Datei
statt in einem Heredoc.

AUFRUF   python arc/final_sanity_gate.py <erwartetes_segment> <art>

  <erwartetes_segment>  Verzeichnisname, der als GANZES Pfadsegment in
                        sigmadock.__file__ vorkommen muss.
  <art>                 diffusion | flow | flow_twohead

WARUM SEGMENTE UND NICHT TEILZEICHENKETTEN
    Die Repowurzel auf ARC heisst SigmaFlow_Development_JulianMueller und
    enthaelt damit die Zeichenkette "SigmaFlow_Development". Eine
    Teilzeichenketten-Blacklist schlaegt darauf an und toetet jeden
    SigmaFlow-Lauf in der ersten Minute -- nach Tagen in der Warteschlange.
    Ein echter falscher Baum traegt den falschen Namen als ganzes Segment
    und wird weiterhin gefangen.

WARUM ZUSAETZLICH EINE CODEPRUEFUNG
    Ein richtiger Pfad beweist nicht, dass der Code darin der erwartete ist.
    Eine unvollstaendige Uebertragung laesst den Pfad stimmen, waehrend das
    Modul alt ist -- und der Lauf waere ein teurer No-Op.
"""
from __future__ import annotations

import inspect
import os
import sys

# Namen, die als ganzes Segment auf einen anderen Codebaum hindeuten.
BAEUME = {
    "SigmaDock_Reproduction_JulianMueller",
    "SigmaFlow_Minimal",
    "SigmaFlow_Development",
    "SigmaFlow_Variants",
    "archive",
    "EXP-100_shared_trunk",
    "EXP-102_heuristic_conditional_source",
    "EXP-105_confidence_ranking",
    "EXP-110_two_head_vector_field",
}

# Die Zeichenketten, an denen die jeweilige Rahmenkorrektur erkennbar ist.
# Als Fragmente zusammengesetzt, damit keine Anfuehrungszeichen im Quelltext
# mit denen der geprueften Quelle kollidieren.
Q = chr(34)
FRAME_KLASSISCH = "R_t.transpose(-1, -2) @ updates[" + Q + "omega" + Q + "] @ R_t"
FRAME_TWOHEAD = "R_t.transpose(-1, -2) @ omega_world @ R_t"
NEWTON_EULER = "updates[" + Q + "omega" + Q + "]"


def pruefe_pfad(pfad: str, erwartet: str) -> None:
    segmente = set(pfad.replace(os.sep, "/").split("/"))
    if erwartet not in segmente:
        raise SystemExit(
            f"SANITY GATE: FALSCHE Quelle -- Segment {erwartet!r} fehlt in {pfad}")
    treffer = sorted((BAEUME - {erwartet}) & segmente)
    if treffer:
        raise SystemExit(f"SANITY GATE: FALSCHE Quelle {treffer} in {pfad}")


def pruefe_code(art: str) -> None:
    if art == "diffusion":
        import sigmadock.diff.denoiser  # noqa: F401
        print("  Original-SigmaDock (Diffusions-Denoiser) verifiziert")
        return

    from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator
    src = inspect.getsource(SigmaFlowGenerator._compute_vector_field)

    if art == "flow_twohead":
        from sigmadock.net.model import EquiformerV2
        m = inspect.getsource(EquiformerV2)
        pruefungen = [
            (FRAME_TWOHEAD in src, "adjungierte Konjugation (Welt -> Koerper) FEHLT"),
            (NEWTON_EULER not in src, "alter Newton-Euler-Pfad noch aktiv"),
            ("self.trans_block" in m, "trans_block fehlt"),
            ("self.rot_block" in m, "rot_block fehlt"),
            ("self.force_block" not in m, "ALTER force_block noch vorhanden"),
            (hasattr(SigmaFlowGenerator, "pool_fragment_fields"),
             "pool_fragment_fields fehlt"),
        ]
        for ok, meldung in pruefungen:
            if not ok:
                raise SystemExit("SANITY GATE: " + meldung)
        print("  EXP-110 Zwei-Kopf verifiziert: Koepfe, Pooling, Rahmenwechsel")
        return

    if art == "flow":
        if FRAME_KLASSISCH not in src:
            raise SystemExit("SANITY GATE: FRAME FIX FEHLT")
        print("  Rahmenkorrektur verifiziert")
        return

    raise SystemExit(f"SANITY GATE: unbekannte Art {art!r}")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    erwartet, art = sys.argv[1], sys.argv[2]
    import sigmadock
    pfad = os.path.abspath(sigmadock.__file__)
    print("  sigmadock geladen aus:", pfad)
    pruefe_pfad(pfad, erwartet)
    pruefe_code(art)
    print("SANITY GATE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
