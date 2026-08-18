"""Baut das komplette Visualisierungspaket eines Komplexes.

Eingabe sind die kanonischen .npz-Dateien (eine je Modell). Alles andere
entsteht daraus. Das Skript beruehrt weder Modell noch Sampler und braucht
weder GPU noch Checkpoint -- die teure Arbeit ist bereits in der .npz.

    python -m visualization.build_case \\
        --complex 6VTA_AKN \\
        --sigmaflow  traj_sigmaflow_12h.npz \\
        --sigmadock  traj_sigmadock_12h.npz \\
        --receptor   receptor.pdb \\
        --out        visualisations/6VTA_AKN

Ergebnis:

    visualisations/<ID>/
        receptor.pdb                (kopiert, falls angegeben)
        crystal_ligand.pdb          (aus der .npz, falls enthalten)
        sigmaflow/  final_pose.pdb trajectory.pdb trajectory_state.npz
                    trajectory_metrics.csv trajectory_metrics_ligand.csv
        sigmadock/  dito
        view_static.pml
        view_trajectory_sigmaflow.pml
        view_trajectory_sigmadock.pml
        metrics.csv
        README.md
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from .plots import (
    plot_cumulative_rotation,
    plot_cumulative_translation,
    plot_rmsd,
    plot_rotation_error,
    plot_translation_error,
    plot_transport_asymmetry,
)
from .pymol_scripts import write_static_pml, write_trajectory_pml
from .trajectory import TrajectoryState
from .writers import write_metrics_csv, write_multistate_pdb, write_single_pdb


def build(complex_id: str, trajs: dict[str, TrajectoryState], out: Path,
          receptor: Path | None = None, stride: int = 1,
          make_plots: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    if receptor and receptor.exists():
        shutil.copy(receptor, out / "receptor.pdb")

    # Kristallpose einmal -- sie ist modellunabhaengig. Weichen zwei Modelle
    # darin ab, ist etwas grundsaetzlich falsch und wir sagen es.
    crystals = {k: t for k, t in trajs.items() if t.crystal is not None}
    if crystals:
        ref_label, ref = next(iter(crystals.items()))
        for lab, t in crystals.items():
            if t.crystal.shape != ref.crystal.shape:
                raise ValueError(
                    f"Kristallpose von {lab} hat andere Atomzahl als {ref_label} "
                    "-- die Modelle beziehen sich nicht auf dasselbe Molekuel")
        write_single_pdb(ref.crystal, ref.elements, ref.frag_ids,
                         out / "crystal_ligand.pdb",
                         remark=f"crystal ligand {complex_id}")

    for label, tr in trajs.items():
        tr.validate()
        sub = out / label
        sub.mkdir(parents=True, exist_ok=True)
        tr.save(sub / "trajectory_state.npz")
        write_multistate_pdb(tr, sub / "trajectory.pdb", stride=stride)
        write_single_pdb(tr.positions[-1], tr.elements, tr.frag_ids,
                         sub / "final_pose.pdb",
                         remark=f"{label} final pose {complex_id}")
        _, diag = write_metrics_csv(tr, sub / "trajectory_metrics.csv")
        summary[label] = diag

        write_trajectory_pml(
            out / f"view_trajectory_{label}.pml",
            receptor="receptor.pdb" if (out / "receptor.pdb").exists() else None,
            crystal="crystal_ligand.pdb" if (out / "crystal_ligand.pdb").exists() else None,
            trajectory=f"{label}/trajectory.pdb",
            final_pose=f"{label}/final_pose.pdb",
            n_fragments=tr.n_fragments,
            complex_id=complex_id,
            model=label,
        )

    write_static_pml(
        out / "view_static.pml",
        receptor="receptor.pdb" if (out / "receptor.pdb").exists() else None,
        crystal="crystal_ligand.pdb" if (out / "crystal_ligand.pdb").exists() else None,
        poses={lab: f"{lab}/final_pose.pdb" for lab in trajs},
        complex_id=complex_id,
    )

    # Zusammenfassung je Modell
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "steps", "fragments", "atoms", "time_kind",
                    "has_crystal", "degenerate_fragments", "final_rmsd",
                    "final_centroid_error", "final_median_rotation_error_deg"])
        for label, tr in trajs.items():
            import csv as _csv
            lig = list(_csv.DictReader(
                (out / label / "trajectory_metrics_ligand.csv").open(encoding="utf-8")))
            last = lig[-1] if lig else {}
            w.writerow([label, tr.n_steps, tr.n_fragments, tr.n_atoms,
                        tr.meta.get("time_kind", ""), tr.crystal is not None,
                        ";".join(map(str, summary[label]["degenerate_fragments"])) or "-",
                        last.get("rmsd_to_crystal", ""),
                        last.get("centroid_error", ""),
                        last.get("median_fragment_rotation_error_deg", "")])

    if make_plots:
        try:
            pdir = out / "plots"
            pdir.mkdir(exist_ok=True)
            plot_rotation_error(trajs, pdir / "A_rotation_error.png")
            plot_translation_error(trajs, pdir / "B_translation_error.png")
            plot_rmsd(trajs, pdir / "C_rmsd.png")
            for lab, tr in trajs.items():
                plot_cumulative_rotation(tr, pdir / f"D_cum_rotation_{lab}.png")
                plot_cumulative_translation(tr, pdir / f"E_cum_translation_{lab}.png")
                plot_transport_asymmetry(tr, pdir / f"F_asymmetry_{lab}.png",
                                         label=f"{complex_id} -- {lab}")
        except ImportError:
            print("[info] matplotlib fehlt, Plots uebersprungen")

    (out / "README.md").write_text(_readme(complex_id, trajs, summary),
                                   encoding="utf-8")
    return summary


def _readme(complex_id: str, trajs: dict, summary: dict) -> str:
    lines = [f"# {complex_id}", "",
             "Erzeugt von `visualization/build_case.py`.", "",
             "## Ansehen", "", "```bash",
             "pymol view_static.pml                    # Endposen gegen Kristall"]
    for lab in trajs:
        lines.append(f"pymol view_trajectory_{lab}.pml".ljust(41)
                     + f"# {lab}-Trajektorie")
    lines += ["```", "",
              "In PyMOL: `scrub`, `ghost`, `final`, `frag 3`, "
              "oder `scene scrub|ghost|final`.", "", "## Inhalt", ""]
    for lab, tr in trajs.items():
        deg = summary[lab]["degenerate_fragments"]
        lines += [f"### {lab}", "",
                  f"- Schritte: {tr.n_steps}, Atome: {tr.n_atoms}, "
                  f"Fragmente: {tr.n_fragments}",
                  f"- Zeitachse: `{tr.meta.get('time_kind')}` "
                  "(ODE-Zeit und Diffusionszeit sind NICHT dieselbe Groesse)",
                  f"- Checkpoint: `{tr.meta.get('checkpoint', 'unbekannt')}`",
                  f"- Kristallpose vorhanden: {tr.crystal is not None}"]
        if deg:
            lines.append(f"- **Entartete Fragmente** (Rotation nicht bestimmbar): "
                         + ", ".join(f"{k} ({v})" for k, v in deg.items()))
        lines.append("")
    lines += ["## Fragmentidentitaet", "",
              "| Feld | Wert | PyMOL |", "|---|---|---|",
              "| chain | A, B, C, ... | `chain C` |",
              "| resi | Fragmentindex+1 | `resi 3` |",
              "| B-Faktor | Fragmentindex | `spectrum b` |", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex", required=True)
    ap.add_argument("--sigmaflow", type=Path)
    ap.add_argument("--sigmadock", type=Path)
    ap.add_argument("--extra", type=Path, nargs="*", default=[],
                    help="weitere .npz, Label = Dateiname ohne Endung")
    ap.add_argument("--receptor", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    trajs: dict[str, TrajectoryState] = {}
    if a.sigmaflow:
        trajs["sigmaflow"] = TrajectoryState.load(a.sigmaflow)
    if a.sigmadock:
        trajs["sigmadock"] = TrajectoryState.load(a.sigmadock)
    for p in a.extra:
        trajs[p.stem] = TrajectoryState.load(p)
    if not trajs:
        print("mindestens eine Trajektorie angeben"); return 2

    summary = build(a.complex, trajs, a.out, a.receptor, a.stride,
                    make_plots=not a.no_plots)
    print(f"geschrieben nach {a.out}")
    for lab, d in summary.items():
        print(f"  {lab}: {d['n_steps']} Schritte, {d['n_fragments']} Fragmente"
              + (f", entartet: {d['degenerate_fragments']}"
                 if d["degenerate_fragments"] else ""))
    print(f"\n  pymol {a.out / 'view_static.pml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
