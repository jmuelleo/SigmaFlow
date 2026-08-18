"""Trajektorien-Export und PyMOL-Visualisierung fuer SigmaFlow und SigmaDock.

Kanonischer Fluss:

    Sampling (ARC)  ->  trajectory_state.npz  ->  { PDB, CSV, PML, Plots }
                        ^^^^^^^^^^^^^^^^^^^^
                        einzige Schnittstelle

Alles rechts der .npz ist reine, lokal getestete Nachverarbeitung. Auf ARC
muss nur die .npz entstehen -- siehe `extract_from_sampling.py`.

Der Sampler selbst wird NICHT angefasst: R_t und trans_t werden per Kabsch
aus den Atompositionen zurueckgerechnet (`reconstruct.py`), weil die
Fragmente starr sind. Damit bleibt `SigmaFlow_Minimal` eingefroren.
"""

from .reconstruct import (  # noqa: F401
    kabsch,
    reconstruct_fragment_states,
    relative_rotation_angle_deg,
    rotation_angle_deg,
)
from .trajectory import TrajectoryState  # noqa: F401
from .writers import (  # noqa: F401
    compute_metrics,
    write_metrics_csv,
    write_multistate_pdb,
    write_single_pdb,
)

__all__ = [
    "TrajectoryState",
    "kabsch",
    "reconstruct_fragment_states",
    "rotation_angle_deg",
    "relative_rotation_angle_deg",
    "write_multistate_pdb",
    "write_single_pdb",
    "write_metrics_csv",
    "compute_metrics",
]
