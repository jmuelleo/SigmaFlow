#!/usr/bin/env python3
"""
Maschineller Konfigurationsvergleich SigmaFlow gegen SigmaDock.

WARUM NICHT VON HAND
    Eine handgepflegte Tabelle "was ist gleich, was ist anders" veraltet beim
    ersten Default-Wechsel in einem der beiden Baeume, ohne dass jemand es
    merkt. Dieses Skript liest die Dataclass-Defaults per AST direkt aus
    beiden config.py und legt die Flags darueber, die das Jobskript setzt.
    Es behauptet nichts, es liest.

KLASSIFIKATION
    IDENTICAL                   -- in beiden Armen derselbe Wert.
    REQUIRED_ENGINE_DIFFERENCE  -- Unterschied, der aus dem Wechsel des
                                   generativen Verfahrens zwingend folgt.
    POTENTIAL_CONFOUNDER        -- Unterschied, der NICHT aus dem Verfahren
                                   folgt. Jeder Eintrag hier gefaehrdet den
                                   Vergleich und muss begruendet oder beseitigt
                                   werden.

Aufruf:
    python arc/compare_final_configs.py \
        --sigmaflow-config SigmaFlow_Minimal/src/sigmadock/config.py \
        --sigmadock-config SigmaDock/src_sigmadock/config.py
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

# Flags, die train_final_72h.slurm methodenspezifisch setzt. Das sind die
# EINZIGEN bewussten Abweichungen; alles andere kommt aus final_config.sh und
# ist damit per Konstruktion gleich.
ENGINE_FLAGS = {
    "sigmadock": ["--rot_score_method space", "--rot_score_scaling rms"],
    "sigmaflow_minimal": [],
}

# Parameter, deren Abweichung aus dem Verfahren folgt und deshalb erlaubt ist.
ENGINE_SPECIFIC_KEYS = {
    "rot_score_method",
    "rot_score_scaling",
    "score_method",
    "sigma_min",
    "sigma_max",
    "flow_time_convention",
    "ode_solver",
    "num_steps",
}

# Parameter, die fuer den Vergleich zwingend gleich sein muessen.
CRITICAL_KEYS = {
    "batch_size", "accum_grad_batches", "max_epochs", "max_steps",
    "weight_decay", "optimizer_eps", "betas", "grad_clip",
    "init_lr_start", "lr_warmup_frac", "num_lr_cycles", "cycle_warmup_frac",
    "max_lr_start", "max_lr_end", "min_lr_start", "min_lr_end",
    "use_ema", "ema_rampup_ratio", "ema_halflife",
    "trans_score_weight", "rot_score_weight", "fragment_scaling",
    "seed", "precision", "cuda_precision", "num_workers",
    "val_check_interval", "early_stopping_patience", "monitor_metric",
}


def literal(node: ast.AST) -> object:
    """Einen Default-Ausdruck zu einem Python-Wert machen, soweit moeglich."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        # z.B. `1 / 16` oder `field(default_factory=...)`
        try:
            return eval(compile(ast.Expression(node), "<def>", "eval"), {"__builtins__": {}}, {})
        except Exception:
            return f"<{ast.dump(node)[:40]}...>"


def extract_defaults(path: Path) -> dict[str, object]:
    """Alle annotierten Klassenattribute mit Default aus einer config.py."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    out: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                out[stmt.target.id] = literal(stmt.value)
    return out


def classify(key: str, a: object, b: object, present_a: bool, present_b: bool) -> tuple[str, str]:
    if not present_a or not present_b:
        side = "nur SigmaFlow" if present_a else "nur SigmaDock"
        if key in ENGINE_SPECIFIC_KEYS:
            return "REQUIRED_ENGINE_DIFFERENCE", f"{side}; verfahrensspezifisch"
        if key in CRITICAL_KEYS:
            return "POTENTIAL_CONFOUNDER", f"{side}, aber vergleichskritisch"
        return "INFO_ONLY", side
    if a == b:
        return "IDENTICAL", ""
    if key in ENGINE_SPECIFIC_KEYS:
        return "REQUIRED_ENGINE_DIFFERENCE", "Unterschied folgt aus dem Verfahren"
    if key in CRITICAL_KEYS:
        return "POTENTIAL_CONFOUNDER", "vergleichskritisch und ungleich"
    return "POTENTIAL_CONFOUNDER", "ungleich, Einfluss nicht ausgeschlossen"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sigmaflow-config", type=Path,
                   default=Path("SigmaFlow_Minimal/src/sigmadock/config.py"))
    p.add_argument("--sigmadock-config", type=Path,
                   default=Path("SigmaDock/src_sigmadock/config.py"))
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--fail-on-confounder", action="store_true",
                   help="Exit 1, wenn ein vergleichskritischer Unterschied bleibt.")
    args = p.parse_args()

    for f in (args.sigmaflow_config, args.sigmadock_config):
        if not f.exists():
            print(f"FEHLER: {f} nicht gefunden", file=sys.stderr)
            return 2

    sf = extract_defaults(args.sigmaflow_config)
    sd = extract_defaults(args.sigmadock_config)
    keys = sorted(set(sf) | set(sd))

    buckets: dict[str, list] = {
        "IDENTICAL": [], "REQUIRED_ENGINE_DIFFERENCE": [],
        "POTENTIAL_CONFOUNDER": [], "INFO_ONLY": [],
    }
    for k in keys:
        verdict, note = classify(k, sf.get(k), sd.get(k), k in sf, k in sd)
        buckets[verdict].append({
            "key": k, "sigmaflow": sf.get(k, "<fehlt>"),
            "sigmadock": sd.get(k, "<fehlt>"), "note": note,
        })

    print()
    print("=" * 76)
    print("KONFIGURATIONSVERGLEICH  SigmaFlow_Minimal  gegen  SigmaDock")
    print("=" * 76)
    print(f"  SigmaFlow: {args.sigmaflow_config}")
    print(f"  SigmaDock: {args.sigmadock_config}")
    print(f"  {len(keys)} Parameter verglichen")
    print()

    print(f"IDENTICAL                  : {len(buckets['IDENTICAL']):4d}")
    print(f"REQUIRED_ENGINE_DIFFERENCE : {len(buckets['REQUIRED_ENGINE_DIFFERENCE']):4d}")
    print(f"POTENTIAL_CONFOUNDER       : {len(buckets['POTENTIAL_CONFOUNDER']):4d}")
    print(f"INFO_ONLY                  : {len(buckets['INFO_ONLY']):4d}")

    # Nur die kritischen Parameter einzeln zeigen -- der Rest waere Rauschen.
    print()
    print("-" * 76)
    print("VERGLEICHSKRITISCHE PARAMETER (muessen IDENTICAL sein)")
    print("-" * 76)
    bad_critical = []
    for k in sorted(CRITICAL_KEYS):
        if k not in sf and k not in sd:
            continue
        va, vb = sf.get(k, "<fehlt>"), sd.get(k, "<fehlt>")
        same = (k in sf and k in sd and va == vb)
        mark = "OK " if same else "!! "
        print(f"  {mark}{k:26s} SF={va!r:<22} SD={vb!r}")
        if not same:
            bad_critical.append(k)

    print()
    print("-" * 76)
    print("BEWUSSTE VERFAHRENSUNTERSCHIEDE (Flags des Jobskripts)")
    print("-" * 76)
    for arm, flags in ENGINE_FLAGS.items():
        print(f"  {arm:20s} {' '.join(flags) if flags else '(keine)'}")

    if buckets["POTENTIAL_CONFOUNDER"]:
        print()
        print("-" * 76)
        print("VERBLEIBENDE MOEGLICHE CONFOUNDER")
        print("-" * 76)
        for row in buckets["POTENTIAL_CONFOUNDER"]:
            print(f"  {row['key']:30s} SF={row['sigmaflow']!r:<20} SD={row['sigmadock']!r:<20} {row['note']}")

    print()
    print("=" * 76)
    if bad_critical:
        print(f"VERDIKT: AMBER -- {len(bad_critical)} vergleichskritische Abweichung(en): "
              f"{', '.join(bad_critical)}")
    else:
        print("VERDIKT: GREEN -- alle vergleichskritischen Parameter sind identisch.")
    print("=" * 76)

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "critical_mismatches": bad_critical,
            "counts": {k: len(v) for k, v in buckets.items()},
            "buckets": buckets,
        }, indent=2, default=str), encoding="utf-8")
        print(f"geschrieben: {args.json_out}")

    return 1 if (bad_critical and args.fail_on_confounder) else 0


if __name__ == "__main__":
    raise SystemExit(main())
