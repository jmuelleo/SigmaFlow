#!/usr/bin/env python3
"""
Scheduler-Horizont-Tests fuer die finalen 72h-Laeufe.

Was hier abgesichert wird
-------------------------
1. STEP-SEMANTIK
   Gleiche effektive Batch  ->  gleiche Optimizer-Steps je Epoche
                            ->  gleicher Scheduler-Horizont,
   unabhaengig davon, wie die effektive Batch aus physischer Batch und
   Gradientenakkumulation zusammengesetzt ist.

2. DER BUG, DEN WIR UMGEHEN
   train.py:221 rechnet den Horizont ohne Division durch accum. Der Test
   zeigt numerisch, dass diese Formel mit dem Akkumulationsfaktor skaliert,
   und dokumentiert damit, warum --max_steps explizit gesetzt werden muss.

3. LERNRATEN-VERHALTEN AM ECHTEN SCHEDULER
   Nicht die Kommentare im Code, sondern die Klasse selbst wird befragt:
   - Wo endet die Lernrate bei vollstaendig durchlaufenem Horizont?
   - Was passiert JENSEITS des Horizonts?
   - Wie hoch steht die Lernrate, wenn der Horizont zu gross gewaehlt war?

Aufruf:  python arc/test_scheduler_horizon.py
Exit 0 = alle Checks bestanden.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "SigmaFlow_Minimal" / "src"))

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(condition), detail))
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  --  {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Die Formeln, exakt wie in arc/calculate_final_epochs.py.
# Bewusst hier dupliziert: ein Test, der die zu testende Implementierung
# importiert, kann eine falsche Formel nicht entdecken.
# ---------------------------------------------------------------------------
def steps_per_epoch_actual(n_train: int, b_phys: int, accum: int, world: int) -> int:
    """Was Lightning tatsaechlich an Optimizer-Steps je Epoche ausfuehrt."""
    # drop_last=False (data.py:736-747)  ->  aufrunden
    microbatches = math.ceil(n_train / (b_phys * world))
    # Lightning flusht den angebrochenen Akkumulationsrest am Epochenende
    return math.ceil(microbatches / accum)


def steps_per_epoch_safe(n_train: int, b_phys: int, accum: int, world: int) -> int:
    """Bewusste Unterschaetzung, die als Scheduler-Horizont benutzt wird."""
    return n_train // (b_phys * accum * world)


def buggy_horizon(max_epochs: int, n_train: int, b_phys: int, world: int) -> int:
    """train.py:221 -- ohne Division durch accum."""
    return max_epochs * n_train // (b_phys * world)


# ---------------------------------------------------------------------------
N_TRAIN = 19443
WORLD = 1
EPOCHS = 68

print()
print("=" * 70)
print("1. GLEICHE EFFEKTIVE BATCH -> GLEICHER HORIZONT")
print("=" * 70)

scenarios = [("A", 32, 1), ("B", 16, 2), ("C", 8, 4)]
safe_vals, actual_vals, eff_vals = [], [], []

for label, b_phys, accum in scenarios:
    eff = b_phys * accum * WORLD
    s_safe = steps_per_epoch_safe(N_TRAIN, b_phys, accum, WORLD)
    s_act = steps_per_epoch_actual(N_TRAIN, b_phys, accum, WORLD)
    eff_vals.append(eff)
    safe_vals.append(s_safe)
    actual_vals.append(s_act)
    print(f"  Szenario {label}: batch={b_phys:2d} accum={accum}  ->  "
          f"eff={eff}  steps/epoch sicher={s_safe}  tatsaechlich={s_act}")

check("alle drei Szenarien haben effektive Batch 32", len(set(eff_vals)) == 1 and eff_vals[0] == 32,
      f"eff={eff_vals}")
check("steps/epoch (sicher) identisch ueber alle Szenarien", len(set(safe_vals)) == 1,
      f"{safe_vals}")
check("steps/epoch (tatsaechlich) identisch ueber alle Szenarien", len(set(actual_vals)) == 1,
      f"{actual_vals}")
check("sicherer Wert unterschaetzt den tatsaechlichen nie",
      all(s <= a for s, a in zip(safe_vals, actual_vals)),
      f"safe={safe_vals[0]} <= actual={actual_vals[0]}")

horizons = [s * EPOCHS for s in safe_vals]
check("Scheduler-Horizont identisch ueber alle Szenarien", len(set(horizons)) == 1,
      f"max_steps={horizons[0]:,}")

print()
print("=" * 70)
print("2. ANDERE EFFEKTIVE BATCH -> ANDERER HORIZONT")
print("=" * 70)
s64 = steps_per_epoch_safe(N_TRAIN, 64, 1, WORLD)
print(f"  batch=64 accum=1  ->  steps/epoch={s64}, max_steps={s64 * EPOCHS:,}")
check("effektive Batch 64 ergibt anderen Horizont als 32", s64 * EPOCHS != horizons[0],
      f"{s64 * EPOCHS:,} != {horizons[0]:,}")
check("doppelte Batch halbiert die Steps je Epoche (bis auf Rundung)",
      abs(s64 * 2 - safe_vals[0]) <= 1, f"2*{s64} vs {safe_vals[0]}")

print()
print("=" * 70)
print("3. DER BUG IN train.py:221 -- Horizont skaliert mit accum")
print("=" * 70)
buggy = [buggy_horizon(EPOCHS, N_TRAIN, b, WORLD) for _, b, _ in scenarios]
for (label, b_phys, accum), h in zip(scenarios, buggy):
    ratio = h / horizons[0]
    print(f"  Szenario {label} (batch={b_phys:2d} accum={accum}): "
          f"fehlerhafter Horizont={h:,}  =  {ratio:.2f}x des korrekten")
check("fehlerhafte Formel ist bei accum=1 korrekt (bis auf Rundung)",
      abs(buggy[0] - horizons[0]) <= EPOCHS, f"{buggy[0]:,} vs {horizons[0]:,}")
check("fehlerhafte Formel ist bei accum=2 rund doppelt zu gross",
      1.9 < buggy[1] / horizons[0] < 2.1, f"Faktor {buggy[1] / horizons[0]:.2f}")
check("fehlerhafte Formel ist bei accum=4 rund vierfach zu gross",
      3.8 < buggy[2] / horizons[0] < 4.2, f"Faktor {buggy[2] / horizons[0]:.2f}")

print()
print("=" * 70)
print("4. ECHTES LERNRATENVERHALTEN (Klasse aus dem Repo, nicht nachgebaut)")
print("=" * 70)

try:
    import torch
    from torch import optim

    from sigmadock.core.misc import StepDecayExponentialCosineAnnealingWarmRestarts
except Exception as exc:  # pragma: no cover
    check("Scheduler importierbar", False, f"{type(exc).__name__}: {exc}")
else:
    # Repo-Defaults, config.py:135-143
    LR_CFG = dict(min_lr_start=1e-5, min_lr_end=1e-6, max_lr_start=1e-4, max_lr_end=1e-5)

    def lr_trace(horizon: int, probe_steps: list[int]) -> dict[int, float]:
        """Lernrate an bestimmten Schritten fuer einen gegebenen Horizont."""
        param = torch.nn.Parameter(torch.zeros(1))
        opt = optim.AdamW([param], lr=LR_CFG["max_lr_start"])
        sched = StepDecayExponentialCosineAnnealingWarmRestarts(
            opt, max_steps=horizon, n_cycles=1, warmup_frac=0.25, **LR_CFG
        )
        out, target = {}, max(probe_steps)
        for t in range(target + 1):
            if t in probe_steps:
                out[t] = opt.param_groups[0]["lr"]
            sched.step()
        return out

    H = 4000  # klein genug fuer einen schnellen Test, gross genug fuer die Form
    trace = lr_trace(H, [0, H // 2, H, int(H * 1.5), 2 * H])

    for t, lr in sorted(trace.items()):
        print(f"  t={t:6d}  ({t / H:4.2f} x Horizont)   lr={lr:.3e}")

    check("Start bei max_lr_start", math.isclose(trace[0], 1e-4, rel_tol=1e-6),
          f"{trace[0]:.3e}")
    check("Lernrate faellt monoton bis zum Horizont",
          trace[0] > trace[H // 2] > trace[H], "1.0 > 0.5 > 0.0 des Cosinus")
    check("am Horizont am unteren Ende angekommen", trace[H] <= 1.05e-5,
          f"lr(H)={trace[H]:.3e} <= min_lr_start")
    check("JENSEITS des Horizonts kein Wiederanstieg (Clamping)",
          math.isclose(trace[2 * H], trace[H], rel_tol=1e-9),
          f"lr(2H)={trace[2 * H]:.3e} == lr(H)={trace[H]:.3e}")

    # Die eigentliche Gefahr: Horizont zu gross gewaehlt, Lauf endet frueher.
    over = lr_trace(2 * H, [H])
    ratio = over[H] / trace[H]
    print()
    print(f"  Horizont 2H, Lauf endet bei H:  lr={over[H]:.3e}  "
          f"= {ratio:.1f}x der Lernrate eines vollstaendigen Anneals")
    check("zu grosser Horizont laesst den Lauf bei deutlich hoeherer LR enden",
          ratio > 3.0, f"Faktor {ratio:.1f}")
    check("zu KLEINER Horizont ist dagegen harmlos (LR clampt am Minimum)",
          math.isclose(trace[int(H * 1.5)], trace[H], rel_tol=1e-9),
          "lr(1.5H) == lr(H)")

print()
print("=" * 70)
passed = sum(1 for _, ok, _ in CHECKS if ok)
total = len(CHECKS)
print(f"ERGEBNIS: {passed}/{total} Checks bestanden")
print("=" * 70)
if passed != total:
    print("\nFehlgeschlagen:")
    for name, ok, detail in CHECKS:
        if not ok:
            print(f"  - {name}  ({detail})")
raise SystemExit(0 if passed == total else 1)
