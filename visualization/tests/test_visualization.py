"""Tests der Visualisierungs-Werkzeuge.

Alles hier laeuft ohne ARC, ohne GPU, ohne Checkpoint. Getestet wird die
Nachverarbeitung -- also genau der Teil, der spaeter aus echten Daten die
Abbildungen erzeugt und der deshalb VOR den echten Daten stimmen muss.

    python visualization/tests/test_visualization.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from visualization.plots import HAAR_MEAN_DEG  # noqa: E402
from visualization.pymol_scripts import write_static_pml, write_trajectory_pml  # noqa: E402
from visualization.reconstruct import (  # noqa: E402
    fragment_is_identifiable,
    kabsch,
    reconstruct_fragment_states,
    relative_rotation_angle_deg,
    rotation_angle_deg,
)
from visualization.trajectory import TrajectoryState  # noqa: E402
from visualization.writers import (  # noqa: E402
    compute_metrics,
    write_metrics_csv,
    write_multistate_pdb,
    write_single_pdb,
)

FAILS: list[str] = []
rng = np.random.default_rng(20260817)


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def random_rotation():
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def make_traj(n_frag=3, atoms_per_frag=6, T=8, with_crystal=True,
              degenerate_first=False):
    """Synthetische Trajektorie mit BEKANNTER Rototranslation je Fragment."""
    elements, frag_ids, base = [], [], []
    for f in range(n_frag):
        n = 2 if (degenerate_first and f == 0) else atoms_per_frag
        # asymmetrisch, damit die Orientierung bestimmt ist
        local = rng.normal(size=(n, 3)) * np.array([1.6, 0.9, 0.5])
        base.append(local + np.array([6.0 * f, 0.0, 0.0]))
        elements += ["C", "N", "O", "C", "C", "S"][:n] if n <= 6 else ["C"] * n
        frag_ids += [f] * n
    base = np.vstack(base)
    frag_ids = np.array(frag_ids)
    elements = np.array(elements[:len(frag_ids)])

    # bekannte Bewegung: je Fragment eine feste Achse, linear in t
    axes = [random_rotation() for _ in range(n_frag)]
    shifts = rng.normal(size=(n_frag, 3)) * 3.0
    positions = np.zeros((T, len(frag_ids), 3))
    for t in range(T):
        frac = t / (T - 1)
        for f in range(n_frag):
            idx = np.flatnonzero(frag_ids == f)
            c = base[idx].mean(0)
            ang = frac * np.pi / 3
            R = axes[f] @ rot_x(ang) @ axes[f].T
            positions[t, idx] = (base[idx] - c) @ R.T + c + frac * shifts[f]

    crystal = positions[-1].copy() if with_crystal else None
    return TrajectoryState(
        positions=positions, times=np.linspace(0.0, 1.0, T),
        elements=elements, frag_ids=frag_ids, crystal=crystal,
        meta={"complex_id": "TEST_XYZ", "model": "sigmaflow",
              "time_kind": "ode_t", "checkpoint": "synthetic"})


def main() -> int:  # noqa: C901
    tmp = Path(tempfile.mkdtemp(prefix="vistest_"))

    print("\n1. Kabsch und Winkel")
    X = rng.normal(size=(10, 3))
    R_true = random_rotation()
    Y = X @ R_true.T + np.array([3.0, -1.0, 2.0])
    R_est = kabsch(X, Y)
    check("Kabsch findet die exakte Rotation",
          float(np.abs(R_est - R_true).max()) < 1e-10,
          f"max Abweichung {np.abs(R_est - R_true).max():.2e}")
    check("Kabsch liefert det = +1", abs(np.linalg.det(R_est) - 1) < 1e-10)
    check("90 Grad um x wird als 90 Grad gemessen",
          abs(float(rotation_angle_deg(rot_x(np.pi / 2))) - 90) < 1e-8)
    check("180 Grad wird als 180 Grad gemessen",
          abs(float(rotation_angle_deg(rot_x(np.pi))) - 180) < 1e-6)
    check("Identitaet gibt 0 Grad",
          float(rotation_angle_deg(np.eye(3))) < 1e-12)
    # Kleinwinkel: hier faellt die arccos-Form auseinander, die Chord-Form nicht
    small = rotation_angle_deg(rot_x(1e-7))
    check("Kleinwinkel bleibt genau (Chord-Form)",
          abs(float(small) - np.degrees(1e-7)) < 1e-9,
          f"{float(small):.3e} statt {np.degrees(1e-7):.3e}")
    A, B = random_rotation(), random_rotation()
    check("relativer Winkel ist symmetrisch",
          abs(float(relative_rotation_angle_deg(A, B))
              - float(relative_rotation_angle_deg(B, A))) < 1e-9)

    print("\n2. Identifizierbarkeit wird erkannt")
    check("1 Atom -> nicht identifizierbar",
          not fragment_is_identifiable(rng.normal(size=(1, 3)))[0])
    check("2 Atome -> nicht identifizierbar",
          not fragment_is_identifiable(rng.normal(size=(2, 3)))[0])
    col = np.linspace(0, 1, 5)[:, None] * np.array([1.0, 2.0, 3.0])
    check("kollinear -> nicht identifizierbar",
          not fragment_is_identifiable(col)[0],
          fragment_is_identifiable(col)[1])
    check("generische Wolke -> identifizierbar",
          fragment_is_identifiable(rng.normal(size=(5, 3)))[0])

    print("\n3. Rekonstruktion aus Atomkoordinaten")
    tr = make_traj()
    R, Tn, deg = reconstruct_fragment_states(tr.positions, tr.frag_ids)
    check("Formen stimmen", R.shape == (tr.n_steps, tr.n_fragments, 3, 3)
          and Tn.shape == (tr.n_steps, tr.n_fragments, 3))
    check("R_0 = I (Referenz ist der Anfangszustand)",
          float(np.abs(R[0] - np.eye(3)).max()) < 1e-9)
    check("alle R orthogonal",
          float(np.abs(R @ np.swapaxes(R, -1, -2) - np.eye(3)).max()) < 1e-9)
    # Kontrolle: rekonstruierte Bewegung reproduziert die Positionen exakt
    err = 0.0
    for t in range(tr.n_steps):
        for f in range(tr.n_fragments):
            idx = tr.fragment_atom_indices(f)
            c0 = tr.positions[0][idx].mean(0)
            pred = (tr.positions[0][idx] - c0) @ R[t, f].T + Tn[t, f]
            err = max(err, float(np.abs(pred - tr.positions[t, idx]).max()))
    check("Rekonstruktion reproduziert die Atompositionen exakt", err < 1e-9,
          f"max Fehler {err:.2e} A")
    check("keine falschen Entartungsmeldungen", deg == {}, str(deg))

    trd = make_traj(degenerate_first=True)
    _, _, deg2 = reconstruct_fragment_states(trd.positions, trd.frag_ids)
    check("2-Atom-Fragment wird als entartet gemeldet", 0 in deg2, str(deg2))

    print("\n4. Keine Mutation der Eingaben (statt Sampler-Instrumentierung)")
    before = tr.positions.copy()
    reconstruct_fragment_states(tr.positions, tr.frag_ids)
    compute_metrics(tr)
    write_multistate_pdb(tr, tmp / "m.pdb")
    check("positions unveraendert nach Rekonstruktion, Metriken und Export",
          np.array_equal(before, tr.positions))
    check("Rekonstruktion ist deterministisch",
          np.array_equal(reconstruct_fragment_states(tr.positions, tr.frag_ids)[0],
                         reconstruct_fragment_states(tr.positions, tr.frag_ids)[0]))

    print("\n5. Mehrzustands-PDB")
    p = write_multistate_pdb(tr, tmp / "traj.pdb")
    txt = p.read_text().splitlines()
    models = [i for i, ln in enumerate(txt) if ln.startswith("MODEL")]
    ends = [i for i, ln in enumerate(txt) if ln.startswith("ENDMDL")]
    check("ein MODEL je Schritt", len(models) == tr.n_steps,
          f"{len(models)} MODEL gegen {tr.n_steps} Schritte")
    check("jedes MODEL wird geschlossen", len(models) == len(ends))
    per_state = []
    for a, b in zip(models, ends):
        per_state.append([ln for ln in txt[a:b] if ln.startswith("ATOM")])
    check("Atomzahl in allen Zustaenden gleich",
          len({len(s) for s in per_state}) == 1,
          f"{sorted({len(s) for s in per_state})}")
    check("Atomzahl stimmt mit der Trajektorie ueberein",
          len(per_state[0]) == tr.n_atoms)
    names = [[ln[12:16] for ln in s] for s in per_state]
    check("Atomreihenfolge ueber alle Zustaende invariant",
          all(n == names[0] for n in names))
    # Koordinaten des letzten Zustands gegen das Array
    last = np.array([[float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
                     for ln in per_state[-1]])
    check("Koordinaten korrekt geschrieben",
          float(np.abs(last - tr.positions[-1]).max()) < 1e-3,
          f"max {np.abs(last - tr.positions[-1]).max():.2e} A (PDB hat 3 Nachkommastellen)")

    print("\n6. Fragmentidentitaet im PDB")
    chains = [ln[21] for ln in per_state[0]]
    resis = [int(ln[22:26]) for ln in per_state[0]]
    bfacs = [float(ln[60:66]) for ln in per_state[0]]
    ok_chain = all(chains[i] == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[tr.frag_ids[i]]
                   for i in range(tr.n_atoms))
    check("chain kodiert den Fragmentindex", ok_chain)
    check("resi = Fragmentindex + 1",
          all(resis[i] == tr.frag_ids[i] + 1 for i in range(tr.n_atoms)))
    check("B-Faktor = Fragmentindex",
          all(abs(bfacs[i] - tr.frag_ids[i]) < 1e-6 for i in range(tr.n_atoms)))
    check("Elementspalte gesetzt",
          all(ln[76:78].strip() == str(tr.elements[i])
              for i, ln in enumerate(per_state[0])))

    print("\n7. NPZ-Rundlauf")
    q = tr.save(tmp / "state.npz")
    back = TrajectoryState.load(q)
    check("positions identisch", np.array_equal(back.positions, tr.positions))
    check("frag_ids identisch", np.array_equal(back.frag_ids, tr.frag_ids))
    check("elements identisch", list(back.elements) == list(tr.elements))
    check("crystal identisch", np.array_equal(back.crystal, tr.crystal))
    check("meta identisch", back.meta == tr.meta)
    check("Formen wie dokumentiert",
          back.positions.shape == (tr.n_steps, tr.n_atoms, 3))

    print("\n8. Validierung faengt Fehler")
    def rejects(mut, label):
        t2 = make_traj()
        mut(t2)
        probs = t2.validate(strict=False)
        check(label, len(probs) > 0, probs[0] if probs else "nichts gemeldet")

    rejects(lambda t: setattr(t, "times", t.times[::-1].copy()),
            "nicht monotone Zeit wird abgelehnt")
    rejects(lambda t: setattr(t, "frag_ids", t.frag_ids + 1),
            "luecken in frag_ids werden abgelehnt")
    rejects(lambda t: t.meta.pop("time_kind"),
            "fehlendes time_kind wird abgelehnt")
    rejects(lambda t: t.meta.__setitem__("time_kind", "wallclock"),
            "unbekanntes time_kind wird abgelehnt")
    rejects(lambda t: setattr(t, "positions", t.positions * 0.001),
            "nicht zurueckgerechnete Koordinaten werden erkannt")
    def _inject_nan(t):
        # Nur EINEN Eintrag verderben, damit die Formpruefung nicht vorher
        # anschlaegt -- sonst testet der Fall nicht, was er behauptet.
        t.positions = t.positions.copy()
        t.positions[2, 0, 0] = np.nan

    rejects(_inject_nan, "NaN wird erkannt")

    print("\n9. Metriken")
    fr, lig, diag = compute_metrics(tr)
    check("Zeilenzahl = Schritte x Fragmente",
          len(fr) == tr.n_steps * tr.n_fragments)
    check("Ligandzeilen = Schritte", len(lig) == tr.n_steps)
    check("Endzustand hat RMSD 0 zur Kristallpose (per Konstruktion)",
          float(lig[-1]["rmsd_to_crystal"]) < 1e-9,
          f"{float(lig[-1]['rmsd_to_crystal']):.2e}")
    check("Rotationsfehler im Endzustand ist 0",
          float(lig[-1]["median_fragment_rotation_error_deg"]) < 1e-6)
    check("Rotationsfehler im Anfangszustand ist > 0",
          float(lig[0]["median_fragment_rotation_error_deg"]) > 1.0,
          f"{float(lig[0]['median_fragment_rotation_error_deg']):.1f} Grad")
    check("kumulierte Groessen sind monoton",
          all(fr[i]["cumulative_translation"] <= fr[i + tr.n_fragments]["cumulative_translation"] + 1e-12
              for i in range(len(fr) - tr.n_fragments)))
    # Bekannte Antwort: je Fragment 60 Grad Gesamtdrehung
    net_last = [r["net_rotation_from_start_deg"] for r in fr
                if r["step"] == tr.n_steps - 1]
    check("Gesamtdrehung entspricht der konstruierten (60 Grad)",
          all(abs(v - 60.0) < 1e-6 for v in net_last),
          f"{[round(v, 4) for v in net_last]}")

    tr_nc = make_traj(with_crystal=False)
    _, lig_nc, _ = compute_metrics(tr_nc)
    check("ohne Kristallpose bleiben Fehlerspalten LEER, nicht 0",
          lig_nc[0]["rmsd_to_crystal"] == ""
          and lig_nc[0]["centroid_error"] == "")

    csvp, _ = write_metrics_csv(tr, tmp / "metrics.csv")
    check("CSV und Ligand-CSV geschrieben",
          csvp.exists() and csvp.with_name("metrics_ligand.csv").exists())

    print("\n10. PyMOL-Skripte")
    write_single_pdb(tr.crystal, tr.elements, tr.frag_ids, tmp / "crystal.pdb")
    pml = write_trajectory_pml(
        tmp / "view_trajectory.pml", receptor=None, crystal="crystal.pdb",
        trajectory="traj.pdb", final_pose=None,
        n_fragments=tr.n_fragments, complex_id="TEST_XYZ", model="sigmaflow")
    s = pml.read_text()
    check("laedt die Trajektorie als ein Objekt", "load traj.pdb, traj" in s)
    check("definiert alle vier Ansichten",
          all(f"def {v}(" in s for v in ("scrub", "ghost", "final", "frag")))
    check("registriert die Befehle in PyMOL",
          all(f'cmd.extend("{v}"' in s for v in ("scrub", "ghost", "final", "frag")))
    check("legt Szenen an", s.count("scene ") >= 3)
    check("erzeugt je Fragment eine Selektion",
          all(f"traj_frag{f}" in s for f in range(tr.n_fragments)))
    check("python-Block ist geschlossen",
          s.count("\npython\n") == s.count("\npython end\n") == 1)
    sp = write_static_pml(tmp / "view_static.pml", receptor=None,
                          crystal="crystal.pdb",
                          poses={"sigmaflow": "a.pdb", "sigmadock": "b.pdb"},
                          complex_id="TEST_XYZ")
    st = sp.read_text()
    check("Statikskript laedt beide Endposen",
          "load a.pdb, sigmaflow" in st and "load b.pdb, sigmadock" in st)

    print("\n11. Randfaelle")
    one = make_traj(n_frag=1, atoms_per_frag=5)
    write_multistate_pdb(one, tmp / "one.pdb")
    check("Ein-Fragment-Ligand funktioniert", one.n_fragments == 1)
    big = make_traj(n_frag=11, atoms_per_frag=4, T=6)
    pb = write_multistate_pdb(big, tmp / "big.pdb")
    lines0 = [ln for ln in pb.read_text().splitlines() if ln.startswith("ATOM")]
    check("11 Fragmente (Datensatzmaximum) funktionieren",
          big.n_fragments == 11 and len(lines0) == big.n_atoms * big.n_steps)
    check("Chain-Bezeichner bis Fragment 11 eindeutig",
          len({ln[21] for ln in lines0}) == 11)
    _, ligb, _ = compute_metrics(big)
    check("Metriken auch bei 11 Fragmenten endlich",
          all(isinstance(r["rmsd_to_crystal"], float) for r in ligb))
    stride = write_multistate_pdb(tr, tmp / "stride.pdb", stride=3)
    n_models = stride.read_text().count("MODEL ")
    check("stride duennt korrekt aus",
          n_models == len(range(0, tr.n_steps, 3)), f"{n_models} MODEL")

    print("\n12. Haar-Referenz")
    check("analytischer Haar-Mittelwert stimmt",
          abs(HAAR_MEAN_DEG - np.degrees((np.pi**2 / 2 + 2) / np.pi)) < 0.05,
          f"{np.degrees((np.pi**2 / 2 + 2) / np.pi):.2f} Grad")

    print("\n" + "=" * 70)
    print(f"Testartefakte unter: {tmp}")
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        return 1
    print("Alle Checks bestanden.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
