"""ARC-Seite: aus einem Sampling-Lauf die kanonische .npz erzeugen.

⚠️  STATUS: IMPLEMENTIERT, ABER NICHT GEGEN ECHTE DATEN VALIDIERT.
    Diese Datei ist der einzige Teil der Visualisierungskette, der den
    echten Datensatz und ein Checkpoint braucht. Sie konnte lokal nicht
    ausgefuehrt werden. Alles, was aus der .npz entsteht, ist dagegen
    vollstaendig lokal getestet (visualization/tests/).

WAS DER SAMPLER LIEFERT (verifiziert durch Codeinspektion)

    sampler(batch=...) -> (batch, all_pos, all_losses)
        all_pos : [T, N_lig, 3]  torch.Tensor
                  SKALIERT und TASCHEN-ZENTRIERT, nicht in Angstroem.

  `sample.py:117` rechnet fuer die Endpose zurueck:

        x_angstrom = x * HPARAMS.general.dimensional_scale + pocket_com

  fuer `trajectory` aber NICHT. Diese Umrechnung passiert hier -- sie ist die
  haeufigste Fehlerquelle der ganzen Kette, weshalb `TrajectoryState.validate`
  die Ligandausdehnung prueft und bei vergessener Umrechnung Alarm schlaegt.

WAS ZUSAETZLICH GEBRAUCHT WIRD UND NICHT IM SAMPLER-OUTPUT STEHT

  - `batch.frag_idx_map`   Fragmentindex je Atom (Protein = -1)
  - Elementsymbole         aus dem RDKit-Molekuel (`batch.mol_info["original"]`)
  - `batch.ref_pos`        Kristallpose, fuer die Fehlerspalten
  - die Zeitachse          `timesteps` des Samplers

  Alle vier liegen im Batch bzw. im Sampler-Aufruf vor; keines erfordert eine
  Aenderung am Sampler.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .trajectory import TrajectoryState


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def build_trajectory_state(
    *,
    trajectory,                 # [T, N_lig, 3] skaliert+zentriert
    frag_idx_lig,               # [N_lig] Fragmentindex, bereits auf Ligand gefiltert
    elements,                   # [N_lig] Elementsymbole
    pocket_com,                 # [3]
    dimensional_scale: float,
    times,                      # [T]
    complex_id: str,
    model: str,
    time_kind: str,
    crystal_positions=None,     # [N_lig, 3] in Angstroem
    checkpoint: str = "",
    extra_meta: dict | None = None,
) -> TrajectoryState:
    """Baut die kanonische Darstellung. Rechnet nach Angstroem zurueck."""
    traj = _to_numpy(trajectory).astype(np.float64)
    if traj.ndim != 3 or traj.shape[2] != 3:
        raise ValueError(f"trajectory muss [T,N,3] sein, ist {traj.shape}")

    com = _to_numpy(pocket_com).astype(np.float64).reshape(3)
    positions = traj * float(dimensional_scale) + com[None, None, :]

    frag = _to_numpy(frag_idx_lig).astype(np.int64)
    if (frag < 0).any():
        raise ValueError("frag_idx enthaelt -1: Proteinatome wurden nicht "
                         "herausgefiltert (is_lig = frag_idx_map != -1)")
    # lueckenlos auf 0..F-1 umnummerieren
    uniq = np.unique(frag)
    remap = {int(u): i for i, u in enumerate(uniq)}
    frag = np.array([remap[int(v)] for v in frag], dtype=np.int32)

    els = np.array([str(e) for e in elements], dtype="U4")
    if len(els) != positions.shape[1]:
        raise ValueError(f"{len(els)} Elemente gegen {positions.shape[1]} Atome")

    crystal = None
    if crystal_positions is not None:
        crystal = _to_numpy(crystal_positions).astype(np.float64)
        if crystal.shape != positions.shape[1:]:
            raise ValueError(f"crystal {crystal.shape} passt nicht zu "
                             f"{positions.shape[1:]}")

    meta = {"complex_id": complex_id, "model": model, "time_kind": time_kind,
            "checkpoint": checkpoint,
            "dimensional_scale": float(dimensional_scale),
            "pocket_com": com.tolist()}
    if extra_meta:
        meta.update(extra_meta)

    st = TrajectoryState(
        positions=positions, times=np.asarray(times, dtype=np.float64),
        elements=els, frag_ids=frag, crystal=crystal, meta=meta)
    st.validate()
    return st


def elements_from_mol(mol) -> list[str]:
    """Elementsymbole aus einem RDKit-Molekuel, in Atomreihenfolge.

    Die Reihenfolge MUSS der der Graphknoten entsprechen. Trifft das nicht
    zu, sind alle Fragmentzuordnungen falsch -- deshalb wird die Laenge oben
    hart geprueft, statt still zuzuschneiden.
    """
    return [a.GetSymbol() for a in mol.GetAtoms()]


ARC_SNIPPET = r'''
# --------------------------------------------------------------------------
# Auf ARC, im Sampling-Skript oder in einer Python-Sitzung mit geladenem
# Modell. Erzeugt eine .npz je Komplex; danach ist nichts mehr GPU-abhaengig.
#
#   from visualization.extract_from_sampling import build_trajectory_state
#
# batch          -- der Batch NACH dem Sampling (enthaelt pos_t, frag_idx_map)
# all_pos        -- zweiter Rueckgabewert von sampler(...)
# timesteps      -- die im Sampler benutzte Zeitachse
# --------------------------------------------------------------------------
import numpy as np, torch
from sigmadock.config import HPARAMS
from visualization.extract_from_sampling import build_trajectory_state, elements_from_mol

is_lig = torch.where(batch.frag_idx_map != -1)[0]

for b in torch.unique(batch.batch[is_lig]):
    sel_global = is_lig[batch.batch[is_lig] == b]          # Atomindizes dieses Molekuels
    sel_local  = torch.where(batch.batch[is_lig] == b)[0]  # Spalten in all_pos

    mol = batch.mol_info["original"][int(b)]
    cid = "_".join(Path(batch.mol_info["pdb_path"][int(b)]).stem.split("_")[:2])
    scale = HPARAMS.general.dimensional_scale

    st = build_trajectory_state(
        trajectory      = all_pos[:, sel_local, :],
        frag_idx_lig    = batch.frag_idx_map[sel_global],
        elements        = elements_from_mol(mol),
        pocket_com      = batch.pocket_com[int(b)],
        dimensional_scale = scale,
        times           = np.asarray(timesteps, dtype=float),
        complex_id      = cid,
        model           = "sigmaflow",          # bzw. "sigmadock"
        time_kind       = "ode_t",              # bzw. "diffusion_t"
        crystal_positions = batch.ref_pos[sel_global],
        checkpoint      = str(CKPT_PATH),
    )
    st.save(f"trajectories/{cid}_sigmaflow_12h.npz")
'''


def print_arc_snippet() -> None:
    print(ARC_SNIPPET)


if __name__ == "__main__":
    print(__doc__)
    print_arc_snippet()
