"""Verify that a checkpoint really IS the intended SigmaFlow Frame-Fix run.

The path 0-08-11_18-00-41/.../last.ckpt was quoted from memory in an earlier
session. This script checks it against the checkpoint's own contents instead
of trusting the string, and prints everything needed to identify the run.

Usage:
    python verify_checkpoint.py <path/to/last.ckpt> [more.ckpt ...]
"""

import sys
from pathlib import Path

# The checkpoint pickles reference the `sigmadock` package, so it must be
# importable even when this script is run interactively without PYTHONPATH set.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402


def describe(p: Path) -> None:
    print("=" * 88)
    print(p)
    print("=" * 88)
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  NICHT LADBAR: {type(e).__name__}: {e}")
        return

    hp = ck.get("hyper_parameters", {}) or {}
    n_par = sum(v.numel() for v in ck.get("state_dict", {}).values())
    print(f"  epoch            : {ck.get('epoch')}")
    print(f"  global_step      : {ck.get('global_step')}")
    print(f"  Parameter        : {n_par/1e6:.2f} M")
    for k in ("trans_score_weight", "rot_score_weight", "sigma_min",
              "lr", "learning_rate", "max_epochs", "batch_size"):
        if k in hp:
            print(f"  {k:<17}: {hp[k]}")

    # Is this a FLOW-MATCHING checkpoint or a diffusion one?
    keys = list(ck.get("state_dict", {}).keys())
    joined = " ".join(keys[:400])
    print(f"  state_dict keys  : {len(keys)}")
    has_ema = any(k.startswith("ema_model") or "ema" in k.lower() for k in keys)
    print(f"  EMA-Gewichte     : {'ja' if has_ema else 'nein'}")

    # Diffusion-only hyper-parameters must be ABSENT in a SigmaFlow run.
    diffusion_markers = [k for k in ("rot_score_method", "rot_score_scaling",
                                     "noise_scale", "sigma_max", "schedule")
                         if k in hp]
    print(f"  Diffusions-Marker: {diffusion_markers if diffusion_markers else 'keine (gut)'}")

    cb = ck.get("callbacks", {})
    for k, v in (cb or {}).items():
        if isinstance(v, dict) and "best_model_score" in v:
            print(f"  best_model_score : {v.get('best_model_score')}")
            print(f"  best_model_path  : {v.get('best_model_path')}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for a in sys.argv[1:]:
        describe(Path(a))
    print("PRUEFEN SIE: global_step sollte im Bereich der 12h-Laeufe liegen")
    print("(SigmaFlow 12h endete laut STATUS.md bei global_step 13.750),")
    print("und es duerfen KEINE Diffusions-Marker auftauchen.")
