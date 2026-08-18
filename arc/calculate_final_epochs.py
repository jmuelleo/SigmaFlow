#!/usr/bin/env python3
"""
Leitet FINAL_MAX_EPOCHS und den expliziten Scheduler-Horizont aus einer
GEMESSENEN Durchsatzzahl ab.

WARUM ES DIESES SKRIPT GIBT
    Der Cosine-Anneal wird auf `max_steps` kalibriert. Wird `max_epochs` zu
    hoch gesetzt, endet der Lauf mitten im Anneal bei hoher Lernrate -- genau
    der Fehler, der die Laeufe vom 2026-08-07 entwertet hat. Wird er zu
    niedrig gesetzt, laeuft der Anneal durch und das Training trainiert danach
    auf der abgesenkten Lernrate weiter. Der Fehler ist also ASYMMETRISCH,
    und deshalb darf die Zahl nicht im Kopf gerechnet werden.

DIE KETTE, DIE HIER GESCHLOSSEN WIRD
    samples/s  ->  max_epochs  ->  steps/epoch  ->  max_steps  ->  LR-Horizont

Aufruf:
    python arc/calculate_final_epochs.py \
        --samples-per-s 5.42 --n-train 19443 \
        --physical-batch 32 --accum 1 --world-size 1
    python arc/calculate_final_epochs.py --from-throughput-json <datei> ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass


# ---------------------------------------------------------------------------
# Die Step-Semantik. Jede Zeile hier entspricht einer im Code verifizierten
# Tatsache; die Fundstellen stehen daneben, damit die Herleitung spaeter
# nachpruefbar bleibt und nicht neu erraten werden muss.
# ---------------------------------------------------------------------------
#
# data.py:736-747   train_dataloader() ist ein map-style DataLoader OHNE
#                   drop_last  =>  drop_last=False  =>  der letzte,
#                   unvollstaendige Microbatch wird MITGENOMMEN.
#
#     n_microbatches_per_epoch = ceil(N_train / B_phys)
#
# Lightning fuehrt einen Optimizer-Step aus, wenn das Akkumulationsfenster
# voll ist ODER die Epoche endet (der Rest wird geflusht):
#
#     s_epoch_actual = ceil(n_microbatches_per_epoch / A)
#
# trainer.py:158-161  lr_scheduler_config = {"interval": "step"}
#                   =>  der Scheduler zaehlt OPTIMIZER-Steps, nicht Microbatches.
#
# train.py:221      max_steps = max_epochs * len(train_datafront)
#                                // (batch_size * world_size)
#                   =>  KEINE Division durch accum. Genau hier entsteht der
#                       Fehler, den dieses Skript umgeht, indem es --max_steps
#                       explizit setzt (train.py:218-220 bevorzugt den
#                       uebergebenen Wert).
#
# WARUM WIR BEWUSST UNTERSCHAETZEN
# core/misc.py:52-56  cycle_frac = min(t_in_cycle / cycle_length, 1.0)
#                   =>  jenseits von max_steps CLAMPT die Lernrate am unteren
#                       Ende des Cosine. Sie steigt NICHT wieder an.
#     Folge: mehr Steps als max_steps ist harmlos (Training auf min-LR),
#            weniger Steps als max_steps ist der Totalschaden.
#     Deshalb ist die sichere Richtung, max_steps zu UNTERschaetzen:
#
#         s_epoch_safe = floor(N_train / B_eff)  <=  s_epoch_actual
#
#     Beweis:  floor(N/(B*A)) <= N/(B*A) <= ceil(N/B)/A <= ceil(ceil(N/B)/A)


@dataclass
class EpochPlan:
    """Ein vollstaendig abgeleiteter Trainingsplan fuer einen Arm."""

    arm: str
    samples_per_s: float
    n_train: int
    physical_batch: int
    accum: int
    world_size: int
    effective_batch: int
    budget_hours: float
    continuous_epochs: float
    max_epochs: int
    steps_per_epoch_safe: int
    steps_per_epoch_actual: int
    max_steps: int
    expected_optimizer_steps: int
    expected_examples_seen: int
    expected_train_hours: float
    safety_margin_hours: float
    anneal_completes: bool


def derive_plan(
    arm: str,
    samples_per_s: float,
    n_train: int,
    physical_batch: int,
    accum: int,
    world_size: int,
    budget_hours: float,
    max_epochs_override: int | None = None,
) -> EpochPlan:
    if samples_per_s <= 0:
        raise ValueError("samples_per_s muss > 0 sein (gemessen, nicht geraten).")
    if n_train <= 0:
        raise ValueError("n_train muss > 0 sein (aus der echten Datafront).")

    effective_batch = physical_batch * accum * world_size

    # Wie viele Epochen passen in das Budget?
    continuous_epochs = budget_hours * 3600.0 * samples_per_s / n_train
    max_epochs = max_epochs_override if max_epochs_override is not None else int(math.floor(continuous_epochs))
    if max_epochs < 1:
        raise ValueError(
            f"Budget reicht fuer weniger als eine Epoche "
            f"({continuous_epochs:.3f}). Konfiguration pruefen."
        )

    # Sichere (unterschaetzende) und tatsaechliche Steps je Epoche.
    steps_per_epoch_safe = n_train // effective_batch
    n_microbatches = math.ceil(n_train / (physical_batch * world_size))
    steps_per_epoch_actual = math.ceil(n_microbatches / accum)

    max_steps = steps_per_epoch_safe * max_epochs
    expected_optimizer_steps = steps_per_epoch_actual * max_epochs
    expected_examples_seen = n_train * max_epochs
    expected_train_hours = expected_examples_seen / samples_per_s / 3600.0

    return EpochPlan(
        arm=arm,
        samples_per_s=samples_per_s,
        n_train=n_train,
        physical_batch=physical_batch,
        accum=accum,
        world_size=world_size,
        effective_batch=effective_batch,
        budget_hours=budget_hours,
        continuous_epochs=continuous_epochs,
        max_epochs=max_epochs,
        steps_per_epoch_safe=steps_per_epoch_safe,
        steps_per_epoch_actual=steps_per_epoch_actual,
        max_steps=max_steps,
        expected_optimizer_steps=expected_optimizer_steps,
        expected_examples_seen=expected_examples_seen,
        expected_train_hours=expected_train_hours,
        safety_margin_hours=budget_hours - expected_train_hours,
        # Der Anneal ist genau dann durch, wenn tatsaechlich mindestens so
        # viele Optimizer-Steps laufen, wie der Scheduler als Horizont kennt.
        anneal_completes=expected_optimizer_steps >= max_steps,
    )


def render(plan: EpochPlan) -> str:
    lines = [
        f"-- {plan.arm} " + "-" * max(0, 58 - len(plan.arm)),
        f"  Gemessener Durchsatz     : {plan.samples_per_s:.3f} Beispiele/s",
        f"  Trainingsbeispiele       : {plan.n_train:,}",
        f"  Trainingsbudget          : {plan.budget_hours:.1f} h",
        f"  Batch                    : {plan.physical_batch} physisch"
        f" x {plan.accum} accum x {plan.world_size} GPU = {plan.effective_batch} effektiv",
        "",
        f"  Kontinuierliche Schaetzung: {plan.continuous_epochs:.2f} Epochen",
        f"  EMPFOHLEN FINAL_MAX_EPOCHS: {plan.max_epochs}",
        f"  Steps/Epoche (sicher)     : {plan.steps_per_epoch_safe}"
        f"   [Scheduler-Basis, unterschaetzt bewusst]",
        f"  Steps/Epoche (tatsaechl.) : {plan.steps_per_epoch_actual}"
        f"   [was Lightning wirklich laeuft]",
        f"  EMPFOHLEN MAX_STEPS       : {plan.max_steps:,}",
        f"  Erwartete Optimizer-Steps : {plan.expected_optimizer_steps:,}",
        f"  Erwartete Beispiele       : {plan.expected_examples_seen:,}",
        f"  Erwartete Trainingsdauer  : {plan.expected_train_hours:.2f} h",
        f"  Sicherheitsmarge          : {plan.safety_margin_hours:.2f} h",
        "",
    ]
    if plan.anneal_completes:
        lines.append(
            f"  PASS: {plan.expected_optimizer_steps:,} tatsaechliche Steps >= "
            f"{plan.max_steps:,} Horizont -> Anneal laeuft vollstaendig durch."
        )
    else:
        lines.append(
            f"  FAIL: nur {plan.expected_optimizer_steps:,} Steps gegen "
            f"{plan.max_steps:,} Horizont -> Anneal UNVOLLSTAENDIG. NICHT starten."
        )
    if plan.safety_margin_hours < 0:
        lines.append(
            f"  FAIL: erwartete Dauer {plan.expected_train_hours:.2f} h "
            f"ueberschreitet das Budget {plan.budget_hours:.1f} h."
        )
    return "\n".join(lines)


def compare_arms(plans: list[EpochPlan], tolerance_pct: float) -> tuple[str, str]:
    """Beantwortet: gleiche Epochenzahl fuer beide Arme, oder je eigene?

    Rueckgabe: (empfohlene_politik, begruendungstext)
    """
    if len(plans) != 2:
        return "n/a", "Vergleich braucht genau zwei Arme."

    a, b = plans
    slower, faster = (a, b) if a.samples_per_s <= b.samples_per_s else (b, a)
    delta_pct = 100.0 * (faster.samples_per_s - slower.samples_per_s) / slower.samples_per_s

    head = (
        f"  {a.arm}: {a.samples_per_s:.3f} Beispiele/s -> {a.max_epochs} Epochen\n"
        f"  {b.arm}: {b.samples_per_s:.3f} Beispiele/s -> {b.max_epochs} Epochen\n"
        f"  Durchsatzunterschied: {delta_pct:.1f} %\n"
    )

    if abs(delta_pct) <= tolerance_pct:
        common = min(a.max_epochs, b.max_epochs)
        return (
            "same_epochs",
            head
            + f"\n  EMPFEHLUNG: GLEICHE Epochenzahl fuer beide Arme = {common}\n"
            f"  (Minimum, also aus dem LANGSAMEREN Arm '{slower.arm}'.)\n\n"
            f"  Begruendung: bei <= {tolerance_pct:.0f} % Durchsatzunterschied kostet die\n"
            f"  Angleichung dem schnelleren Arm nur wenig Rechenzeit, entfernt aber\n"
            f"  einen Confounder vollstaendig: beide Modelle sehen dann exakt gleich\n"
            f"  viele Beispiele und machen gleich viele Gradientenschritte. Der\n"
            f"  schnellere Arm ist frueher fertig; das ist zulaessig, weil ein\n"
            f"  vollstaendig durchlaufener Anneal das Guetekriterium ist, nicht die\n"
            f"  ausgeschoepfte Walltime.",
        )

    return (
        "matched_walltime",
        head
        + f"\n  EMPFEHLUNG: UNTERSCHIEDLICHE Epochenzahlen, gleiches Zeitbudget.\n"
        f"  {a.arm} = {a.max_epochs}, {b.arm} = {b.max_epochs}\n\n"
        f"  Begruendung: bei > {tolerance_pct:.0f} % Unterschied wuerde eine erzwungen\n"
        f"  gleiche Epochenzahl den schnelleren Arm kuenstlich bremsen. Das\n"
        f"  Studiendesign ist ein COMPUTE-gematchter Vergleich (gleiche GPU-Stunden),\n"
        f"  nicht ein epochengematchter. Beim Berichten MUSS dann aber beides\n"
        f"  genannt werden: gleiche Stunden, ungleiche Epochen und Beispiele.",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", default=None, help="Armname (mehrfach fuer Vergleich)")
    p.add_argument("--samples-per-s", action="append", type=float, required=True)
    p.add_argument("--n-train", type=int, required=True, help="ECHTE Laenge der Trainings-Datafront")
    p.add_argument("--physical-batch", type=int, required=True)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--budget-hours", type=float, default=70.0,
                   help="Reine Optimierungszeit; Default 70 h innerhalb eines 72h-Jobs.")
    p.add_argument("--tolerance-pct", type=float, default=5.0,
                   help="Bis zu diesem Durchsatzunterschied wird eine gemeinsame Epochenzahl empfohlen.")
    p.add_argument("--json-out", default=None)
    p.add_argument("--write-env", default=None,
                   help="Schreibt arc/final_horizon.env -- die einzige Quelle, aus der "
                        "final_config.sh den Horizont liest.")
    p.add_argument("--throughput-source", default="unspecified",
                   help="Woher die samples/s stammen (Job-ID, CSV-Pfad). Wird mitgeschrieben.")
    args = p.parse_args()

    arms = args.arm or [f"arm{i}" for i in range(len(args.samples_per_s))]
    if len(arms) != len(args.samples_per_s):
        print("FEHLER: Zahl der --arm und --samples-per-s muss uebereinstimmen.", file=sys.stderr)
        return 2

    plans = [
        derive_plan(
            arm=arm,
            samples_per_s=sps,
            n_train=args.n_train,
            physical_batch=args.physical_batch,
            accum=args.accum,
            world_size=args.world_size,
            budget_hours=args.budget_hours,
        )
        for arm, sps in zip(arms, args.samples_per_s)
    ]

    print()
    print("=" * 64)
    print("FINALE EPOCHEN- UND SCHEDULER-HERLEITUNG")
    print("=" * 64)
    for plan in plans:
        print(render(plan))

    policy = "single_arm"
    if len(plans) == 2:
        print("-" * 64)
        print("ARM-VERGLEICH")
        print("-" * 64)
        policy, text = compare_arms(plans, args.tolerance_pct)
        print(text)
        print()

    failed = [pl for pl in plans if not pl.anneal_completes or pl.safety_margin_hours < 0]
    verdict = "RED" if failed else "GREEN"
    print("=" * 64)
    print(f"VERDIKT: {verdict}")
    print("=" * 64)

    if args.json_out:
        payload = {"policy": policy, "verdict": verdict, "plans": [asdict(pl) for pl in plans]}
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"geschrieben: {args.json_out}")

    if args.write_env:
        if verdict != "GREEN":
            print("\nVERWEIGERT: --write-env schreibt keinen Horizont mit Verdikt "
                  f"{verdict}. Erst die Konfiguration korrigieren.", file=sys.stderr)
            return 1
        write_env(args.write_env, plans, policy, args)
        print(f"geschrieben: {args.write_env}")

    return 0 if verdict == "GREEN" else 1


def write_env(path: str, plans: list[EpochPlan], policy: str, args: argparse.Namespace) -> None:
    """Erzeugt arc/final_horizon.env.

    Bei policy == "same_epochs" bekommen ALLE Arme die Epochenzahl des
    langsameren. Das ist bewusst: eine gemeinsame Zahl entfernt einen
    Confounder, und der langsamere Arm ist der einzige, dessen Anneal sonst
    nicht ins Zeitbudget passt.
    """
    common_epochs = min(pl.max_epochs for pl in plans) if policy == "same_epochs" else None

    lines = [
        "# ERZEUGT von arc/calculate_final_epochs.py -- NICHT VON HAND EDITIEREN.",
        "# Diese Datei ist die einzige Quelle des Trainingshorizonts.",
        "# final_config.sh sourced sie; resolve_horizon_for_model() prueft sie",
        "# gegen die aktuelle Batch-Konfiguration und rechnet max_steps nach.",
        "#",
        f"# Politik            : {policy}",
        f"# Durchsatzquelle    : {args.throughput_source}",
        f"# Zeitbudget         : {args.budget_hours} h reine Optimierung",
        "",
        f"export FINAL_N_TRAIN={plans[0].n_train}",
        f"export FINAL_HORIZON_EFFECTIVE_BATCH={plans[0].effective_batch}",
        f"export FINAL_HORIZON_POLICY={policy}",
        f'export FINAL_HORIZON_THROUGHPUT_SOURCE="{args.throughput_source}"',
        "",
    ]

    for pl in plans:
        key = pl.arm.upper()
        epochs = common_epochs if common_epochs is not None else pl.max_epochs
        steps = pl.steps_per_epoch_safe * epochs
        lines += [
            f"# {pl.arm}: {pl.samples_per_s:.3f} Beispiele/s, "
            f"eigenstaendig waeren {pl.max_epochs} Epochen moeglich",
            f"export FINAL_MAX_EPOCHS_{key}={epochs}",
            f"export FINAL_MAX_STEPS_{key}={steps}",
            f"export FINAL_SAMPLES_PER_S_{key}={pl.samples_per_s:.4f}",
            "",
        ]

    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
