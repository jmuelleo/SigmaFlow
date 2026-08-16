"""
cleveref_type_check.py
======================

Regression test for the theorem-environment / cleveref fix in
``Texte/theory.tex``.

All theorem-like environments in the monograph deliberately share one
counter, so that numbering runs continuously inside a chapter.  Done naively
(``\newtheorem{definition}[theorem]{Definition}``) that makes cleveref print
"Theorem" for *every* cross-reference, because cleveref identifies a
reference by the counter it stepped.  The fix uses ``aliascnt`` to give each
environment its own counter name aliased to ``theorem``.

This script verifies the fix from the compiled ``theory.aux``: for every
label whose name carries a type prefix (``def:``, ``prop:``, ``thm:``,
``rem:``, ``caution:``, ``ex:``, ``lem:``, ``cor:``), it reads the type
cleveref actually recorded and checks that the two agree.

Run:  python audits/cleveref_type_check.py
"""

import re
import sys
from pathlib import Path

AUX = Path(__file__).resolve().parent.parent / "Texte" / "theory.aux"

# label prefix -> cleveref type that must be recorded
EXPECTED = {
    "def":       "definition",
    "prop":      "proposition",
    "thm":       "theorem",
    "rem":       "remark",
    "caution":   "caution",
    "ex":        "example",
    "lem":       "lemma",
    "cor":       "corollary",
    "not":       "notation",
}

# labels that legitimately point at something other than their prefix suggests
ALLOWED_EXCEPTIONS = {
    # (label, recorded type) pairs that are deliberate
}


def main():
    if not AUX.exists():
        sys.exit(f"aux file not found: {AUX}  (compile theory.tex first)")

    text = AUX.read_text(encoding="utf-8", errors="replace")

    # cleveref writes:  \newlabel{LABEL@cref}{{[type][num][...]num}{[...]page}}
    pat = re.compile(r"\\newlabel\{([^}]+)@cref\}\{\{\[([a-zA-Z]+)\]")
    recorded = {m.group(1): m.group(2) for m in pat.finditer(text)}
    print(f"labels with a cleveref type recorded: {len(recorded)}")

    checked = 0
    bad = []
    by_type = {}
    for label, typ in sorted(recorded.items()):
        by_type[typ] = by_type.get(typ, 0) + 1
        if ":" not in label:
            continue
        prefix = label.split(":", 1)[0]
        if prefix not in EXPECTED:
            continue
        checked += 1
        if typ != EXPECTED[prefix] and (label, typ) not in ALLOWED_EXCEPTIONS:
            bad.append((label, EXPECTED[prefix], typ))

    print("\nrecorded types, by frequency:")
    for typ, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {typ:<14} {n:>4}")

    print(f"\ntheorem-like labels checked: {checked}")
    if bad:
        print(f"\nMISMATCHES ({len(bad)}):")
        for label, want, got in bad[:40]:
            print(f"  {label:<45} expected {want:<12} got {got}")
        if len(bad) > 40:
            print(f"  ... and {len(bad)-40} more")
        sys.exit(1)

    # the specific symptom that motivated the fix
    offenders = [l for l, t in recorded.items()
                 if t == "theorem" and l.split(":", 1)[0] in
                 {"def", "rem", "caution", "ex", "prop", "lem", "cor", "not"}]
    if offenders:
        print(f"\nFAIL: {len(offenders)} non-theorem labels still typed 'theorem'")
        sys.exit(1)

    print("\nAll theorem-like cross-references carry the correct type.")


main()
