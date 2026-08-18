"""Schreibt eine TrajectoryState nach PDB und CSV.

PDB-MEHRZUSTANDSFORMAT

  Ein MODEL/ENDMDL-Block je Integrationsschritt. PyMOL laedt das als EIN
  Objekt mit K Zustaenden; der Frame-Regler wird damit zum Zeitregler, ohne
  dass ein Skript noetig waere.

  Die Atomreihenfolge MUSS ueber alle Zustaende identisch sein. Waere sie es
  nicht, verbaende PyMOL beim Umschalten verschiedene Atome miteinander und
  die Bewegung im Bild waere frei erfunden. `write_multistate_pdb` prueft das
  und schreibt lieber nichts.

FRAGMENTIDENTITAET -- dreifach kodiert

  | Feld       | Wert            | PyMOL-Selektion              |
  |------------|-----------------|------------------------------|
  | chain      | A, B, C, ...    | `chain C`                    |
  | resi       | Fragmentindex+1 | `resi 3`                     |
  | B-Faktor   | Fragmentindex   | `b > 2.5 and b < 3.5`        |

  Redundant mit Absicht. Chain und resi sind bequem zu tippen, der B-Faktor
  ueberlebt `spectrum b` und Formatkonvertierungen. Ab 62 Fragmenten gehen
  die Chain-Buchstaben aus -- hier irrelevant (Maximum im Datensatz: 11),
  wird aber gemeldet statt still umzubrechen.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .reconstruct import (
    kabsch,
    reconstruct_fragment_states,
    relative_rotation_angle_deg,
    rotation_angle_deg,
)
from .trajectory import TrajectoryState

_CHAINS = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
           "abcdefghijklmnopqrstuvwxyz"
           "0123456789")


def _chain_for(frag: int) -> str:
    return _CHAINS[frag] if frag < len(_CHAINS) else "z"


def _atom_line(serial: int, name: str, element: str, frag: int,
               xyz: np.ndarray) -> str:
    """Eine ATOM-Zeile im festen PDB-Spaltenformat."""
    # Atomnamen sind auf 4 Zeichen begrenzt; Elemente mit einem Buchstaben
    # beginnen konventionell in Spalte 14, nicht 13.
    nm = name[:4]
    nm = f" {nm:<3}" if len(element) == 1 and len(nm) < 4 else f"{nm:<4}"
    return (
        f"ATOM  {serial:5d} {nm}{'LIG':>4} {_chain_for(frag)}{frag + 1:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"{1.00:6.2f}{float(frag):6.2f}"
        f"{'':10}{element:>2}"
    )


def _atom_names(elements: np.ndarray) -> list[str]:
    """C1, C2, N1, ... -- je Element durchnummeriert, stabil ueber Zustaende."""
    counts: dict[str, int] = {}
    out = []
    for e in elements:
        e = str(e)
        counts[e] = counts.get(e, 0) + 1
        out.append(f"{e}{counts[e]}")
    return out


def write_multistate_pdb(traj: TrajectoryState, path: str | Path,
                         stride: int = 1) -> Path:
    """Ein MODEL je Schritt. `stride > 1` duennt lange Trajektorien aus."""
    traj.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if traj.n_fragments > len(_CHAINS):
        raise ValueError(f"{traj.n_fragments} Fragmente, aber nur "
                         f"{len(_CHAINS)} Chain-Bezeichner verfuegbar")

    names = _atom_names(traj.elements)
    steps = range(0, traj.n_steps, stride)
    lines: list[str] = [
        f"REMARK   1 SigmaFlow trajectory  complex={traj.meta.get('complex_id')}",
        f"REMARK   1 model={traj.meta.get('model')}  "
        f"time_kind={traj.meta.get('time_kind')}",
        f"REMARK   1 states={len(list(steps))}  atoms={traj.n_atoms}  "
        f"fragments={traj.n_fragments}",
        "REMARK   1 chain und resi kodieren den Fragmentindex, B-Faktor ebenso",
    ]
    for model_no, t in enumerate(range(0, traj.n_steps, stride), start=1):
        lines.append(f"MODEL     {model_no:4d}")
        for i in range(traj.n_atoms):
            lines.append(_atom_line(i + 1, names[i], str(traj.elements[i]),
                                    int(traj.frag_ids[i]), traj.positions[t, i]))
        lines.append("TER")
        lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def write_single_pdb(positions: np.ndarray, elements: np.ndarray,
                     frag_ids: np.ndarray, path: str | Path,
                     remark: str = "") -> Path:
    """Eine einzelne Pose (Endpose, Kristallpose)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = _atom_names(elements)
    lines = [f"REMARK   1 {remark}"] if remark else []
    for i in range(len(positions)):
        lines.append(_atom_line(i + 1, names[i], str(elements[i]),
                                int(frag_ids[i]), positions[i]))
    lines += ["TER", "END"]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def compute_metrics(traj: TrajectoryState) -> tuple[list[dict], list[dict], dict]:
    """(per-Fragment-Zeilen, Ligand-Zeilen, Diagnose).

    Fehlt die Kristallpose, bleiben die *_error_to_target-Spalten leer statt
    mit einem Platzhalter gefuellt zu werden -- eine 0 dort waere spaeter
    nicht von einem echten Nullfehler zu unterscheiden.
    """
    rot, trans, degenerate = reconstruct_fragment_states(
        traj.positions, traj.frag_ids)
    T, F = traj.n_steps, traj.n_fragments

    # Zielzustand je Fragment, falls die Kristallpose vorliegt
    tgt_R: np.ndarray | None = None
    tgt_c: np.ndarray | None = None
    if traj.crystal is not None:
        tgt_R = np.tile(np.eye(3), (F, 1, 1))
        tgt_c = np.zeros((F, 3))
        for f in range(F):
            idx = traj.fragment_atom_indices(f)
            tgt_c[f] = traj.crystal[idx].mean(0)
            if f not in degenerate:
                tgt_R[f] = kabsch(traj.positions[0][idx], traj.crystal[idx])

    step_dt = np.zeros((T, F))
    step_dr = np.zeros((T, F))
    step_dt[1:] = np.linalg.norm(np.diff(trans, axis=0), axis=-1)
    step_dr[1:] = relative_rotation_angle_deg(rot[:-1], rot[1:])

    frag_rows: list[dict] = []
    for t in range(T):
        for f in range(F):
            row = {
                "step": t,
                "time": float(traj.times[t]),
                "time_kind": traj.meta.get("time_kind", ""),
                "fragment": f,
                "degenerate": int(f in degenerate),
                "delta_translation": float(step_dt[t, f]),
                "cumulative_translation": float(step_dt[: t + 1, f].sum()),
                "delta_rotation_deg": float(step_dr[t, f]),
                "cumulative_rotation_deg": float(step_dr[: t + 1, f].sum()),
                "net_rotation_from_start_deg": float(rotation_angle_deg(rot[t, f])),
                "translation_error_to_target": "",
                "rotation_error_to_target_deg": "",
            }
            if tgt_c is not None:
                row["translation_error_to_target"] = float(
                    np.linalg.norm(trans[t, f] - tgt_c[f]))
                if f not in degenerate:
                    row["rotation_error_to_target_deg"] = float(
                        relative_rotation_angle_deg(rot[t, f], tgt_R[f]))
            frag_rows.append(row)

    lig_rows: list[dict] = []
    for t in range(T):
        row = {
            "step": t,
            "time": float(traj.times[t]),
            "rmsd_to_crystal": "",
            "centroid_error": "",
            "median_fragment_rotation_error_deg": "",
            "mean_fragment_rotation_error_deg": "",
            "max_fragment_rotation_error_deg": "",
        }
        if traj.crystal is not None:
            d = traj.positions[t] - traj.crystal
            row["rmsd_to_crystal"] = float(np.sqrt((d * d).sum(-1).mean()))
            row["centroid_error"] = float(np.linalg.norm(
                traj.positions[t].mean(0) - traj.crystal.mean(0)))
            errs = [relative_rotation_angle_deg(rot[t, f], tgt_R[f])
                    for f in range(F) if f not in degenerate]
            if errs:
                row["median_fragment_rotation_error_deg"] = float(np.median(errs))
                row["mean_fragment_rotation_error_deg"] = float(np.mean(errs))
                row["max_fragment_rotation_error_deg"] = float(np.max(errs))
        lig_rows.append(row)

    diag = {
        "n_steps": T, "n_fragments": F, "n_atoms": traj.n_atoms,
        "degenerate_fragments": degenerate,
        "has_crystal": traj.crystal is not None,
    }
    return frag_rows, lig_rows, diag


def write_metrics_csv(traj: TrajectoryState, path: str | Path) -> tuple[Path, dict]:
    frag_rows, lig_rows, diag = compute_metrics(traj)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(frag_rows[0].keys()))
        w.writeheader()
        w.writerows(frag_rows)
    lig_path = path.with_name(path.stem + "_ligand.csv")
    with lig_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lig_rows[0].keys()))
        w.writeheader()
        w.writerows(lig_rows)
    return path, diag
