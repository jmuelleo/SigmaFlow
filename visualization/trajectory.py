"""Kanonische Zwischendarstellung einer Sampling-Trajektorie.

WARUM EIN EIGENES FORMAT

  Aus einer Trajektorie sollen PDB, CSV, PyMOL-Skripte und Thesis-Plots
  entstehen. Wuerde jedes dieser Ziele direkt aus dem Sampler-Output lesen,
  haetten wir vier Stellen, an denen Koordinatenkonventionen falsch sein
  koennen. Stattdessen gibt es GENAU EINE: diese Klasse. Alles andere ist
  eine reine Funktion davon.

  Praktischer Nebeneffekt: auf ARC muss nur die .npz erzeugt werden. Der
  gesamte Rest laeuft lokal und ist hier getestet.

KOORDINATEN -- die Falle

  `sample.py` gibt `trajectory` in SKALIERTEN, taschen-zentrierten Koordinaten
  heraus (nur `x0`/`x0_hat` werden nach Angstroem zurueckgerechnet):

      x_angstrom = x_traj * dimensional_scale + pocket_com

  Diese Klasse speichert IMMER Angstroem. Der Produzent muss umrechnen; die
  Validierung unten faengt vergessene Umrechnungen an der Groessenordnung ab.

ZEIT -- nicht dieselbe Groesse bei beiden Modellen

  SigmaFlow: ODE-Integrationszeit t in [0,1], t=1 ist die Datenverteilung.
  SigmaDock: Rueckwaerts-Diffusionszeit, andere Semantik.
  `time_kind` haelt das fest. Die beiden duerfen nie stillschweigend als
  dieselbe physikalische Zeit behandelt werden.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TIME_KINDS = ("ode_t", "diffusion_t")


@dataclass
class TrajectoryState:
    """Eine Sampling-Trajektorie eines Komplexes.

    Attribute
    ---------
    positions      [T, N, 3] float64, Angstroem, Weltkoordinaten
    times          [T]       float64, monoton
    elements       [N]       str, chemisches Symbol je Atom
    frag_ids       [N]       int, Fragmentindex je Atom (>= 0)
    crystal        [N, 3] | None  Kristallpose, gleiche Atomreihenfolge
    rotations      [T, F, 3, 3] | None  je Fragment, aus Kabsch rekonstruiert
    translations   [T, F, 3]    | None  Fragmentschwerpunkte
    meta           frei; muss complex_id, model, time_kind enthalten
    """

    positions: np.ndarray
    times: np.ndarray
    elements: np.ndarray
    frag_ids: np.ndarray
    crystal: np.ndarray | None = None
    rotations: np.ndarray | None = None
    translations: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ shape
    @property
    def n_steps(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.positions.shape[1])

    @property
    def n_fragments(self) -> int:
        return int(self.frag_ids.max()) + 1 if self.frag_ids.size else 0

    def fragment_atom_indices(self, f: int) -> np.ndarray:
        return np.flatnonzero(self.frag_ids == f)

    # ------------------------------------------------------------- validation
    def validate(self, strict: bool = True) -> list[str]:
        """Gibt eine Liste von Problemen zurueck; leer heisst sauber.

        `strict=True` wirft stattdessen. Die Pruefungen sind bewusst
        paranoid: ein stiller Fehler hier vergiftet jede Abbildung, die
        spaeter daraus entsteht.
        """
        p: list[str] = []
        T, N = self.n_steps, self.n_atoms

        if self.positions.ndim != 3 or self.positions.shape[2] != 3:
            p.append(f"positions muss [T,N,3] sein, ist {self.positions.shape}")
        if T < 2:
            p.append(f"Trajektorie braucht mindestens 2 Schritte, hat {T}")
        if self.times.shape != (T,):
            p.append(f"times muss [{T}] sein, ist {self.times.shape}")
        elif not np.all(np.diff(self.times) > 0):
            p.append("times ist nicht streng monoton steigend")
        if self.elements.shape != (N,):
            p.append(f"elements muss [{N}] sein, ist {self.elements.shape}")
        if self.frag_ids.shape != (N,):
            p.append(f"frag_ids muss [{N}] sein, ist {self.frag_ids.shape}")
        elif N and (self.frag_ids.min() < 0):
            p.append("frag_ids enthaelt negative Werte (Protein nicht gefiltert?)")
        elif N and set(np.unique(self.frag_ids)) != set(range(self.n_fragments)):
            p.append("frag_ids sind nicht lueckenlos 0..F-1")

        if not np.isfinite(self.positions).all():
            p.append("positions enthaelt NaN/Inf")

        if self.crystal is not None and self.crystal.shape != (N, 3):
            p.append(f"crystal muss [{N},3] sein, ist {self.crystal.shape}")
        if self.rotations is not None:
            if self.rotations.shape != (T, self.n_fragments, 3, 3):
                p.append(f"rotations muss [{T},{self.n_fragments},3,3] sein, "
                         f"ist {self.rotations.shape}")
            else:
                RtR = self.rotations @ np.swapaxes(self.rotations, -1, -2)
                err = np.abs(RtR - np.eye(3)).max()
                if err > 1e-4:
                    p.append(f"rotations nicht orthogonal, max Abweichung {err:.2e}")
        if self.translations is not None and \
                self.translations.shape != (T, self.n_fragments, 3):
            p.append(f"translations muss [{T},{self.n_fragments},3] sein, "
                     f"ist {self.translations.shape}")

        # Groessenordnung: Angstroem, nicht skalierte Einheiten. Ein Ligand,
        # dessen Ausdehnung unter 1 A liegt, ist mit Sicherheit nicht
        # zurueckgerechnet worden.
        if N > 1 and np.isfinite(self.positions).all():
            extent = float(np.ptp(self.positions[-1], axis=0).max())
            if extent < 1.0:
                p.append(f"Ligand-Ausdehnung im Endzustand nur {extent:.3f} A -- "
                         "vermutlich fehlt die Umrechnung "
                         "x*dimensional_scale + pocket_com")

        for k in ("complex_id", "model", "time_kind"):
            if k not in self.meta:
                p.append(f"meta['{k}'] fehlt")
        if self.meta.get("time_kind") not in TIME_KINDS:
            p.append(f"meta['time_kind'] muss aus {TIME_KINDS} sein, "
                     f"ist {self.meta.get('time_kind')!r}")

        if strict and p:
            raise ValueError("TrajectoryState ungueltig:\n  - " + "\n  - ".join(p))
        return p

    # --------------------------------------------------------------- roundtrip
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "positions": self.positions.astype(np.float64),
            "times": self.times.astype(np.float64),
            "elements": self.elements.astype("U4"),
            "frag_ids": self.frag_ids.astype(np.int32),
            "meta_json": np.array(json.dumps(self.meta)),
        }
        for name in ("crystal", "rotations", "translations"):
            v = getattr(self, name)
            if v is not None:
                arrays[name] = v.astype(np.float64)
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> TrajectoryState:
        z = np.load(Path(path), allow_pickle=False)
        return cls(
            positions=z["positions"],
            times=z["times"],
            elements=z["elements"],
            frag_ids=z["frag_ids"],
            crystal=z["crystal"] if "crystal" in z else None,
            rotations=z["rotations"] if "rotations" in z else None,
            translations=z["translations"] if "translations" in z else None,
            meta=json.loads(str(z["meta_json"])),
        )

    def __repr__(self) -> str:
        return (f"TrajectoryState({self.meta.get('complex_id','?')}, "
                f"{self.meta.get('model','?')}, T={self.n_steps}, "
                f"N={self.n_atoms}, F={self.n_fragments}, "
                f"time={self.meta.get('time_kind','?')})")
