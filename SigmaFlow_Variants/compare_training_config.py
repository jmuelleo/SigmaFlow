"""Which settings of the reference training config our runs did not reproduce.

Reads the shipped SigmaDock training config, the dataclass defaults in
config.py, and the arguments our own training script passes, and reports for
each reference setting whether we matched it, overrode it, or silently fell
back to a code default.
"""
import pathlib
import re

import yaml

ROOT = pathlib.Path(".").resolve().parent
REF_YAML = ROOT / "SigmaDock" / "conf" / "training" / "slurm.yaml"
REF_CONFIG = ROOT / "SigmaDock" / "src_sigmadock" / "config.py"
OUR_SLURM = ROOT / "arc" / "train_final_72h.slurm"

ref = yaml.safe_load(REF_YAML.read_text(encoding="utf-8"))
src = REF_CONFIG.read_text(encoding="utf-8")
ours = OUR_SLURM.read_text(encoding="utf-8")

# dataclass defaults, e.g.  name: type = value  # comment
defaults = {}
for m in re.finditer(r"(?m)^\s{4}(\w+):\s*[^=\n]+=\s*([^#\n]+)", src):
    defaults.setdefault(m.group(1), m.group(2).strip().rstrip(","))

passed = set(re.findall(r"--([a-z_]+)", ours))

print(f"{'setting':<32}{'reference':>26}{'code default':>18}   status")
for key, val in ref.items():
    if key in ("data_dir",):
        continue
    d = defaults.get(key, "-")
    if key in passed:
        status = "passed by our script"
    elif str(val) == str(d):
        status = "default equals reference"
    else:
        status = "*** NOT SET, default differs ***"
    print(f"{key:<32}{str(val):>26}{str(d):>18}   {status}")
