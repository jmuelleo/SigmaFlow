"""EXP-100 End-to-End: die ANGEBUNDENE Pipeline, nicht nur die isolierte Mathematik.

`test_exp100.py` prueft den mathematischen Kern (Kabsch, Endpunkte, Gauge,
Aequivarianz, Oracle). Dieses Skript prueft, ob die Anbindung an Datenpipeline,
Generator und Sampler stimmt - also genau die Schicht, in der sich ein
Vorzeichen- oder Rahmenfehler noch verstecken koennte.

  T1  Toy End-to-End    bekanntes (R*, p*) durch die ECHTE Transformationskette
  T2  Real End-to-End   Rekonstruktionsfehler auf echten Komplexen
  T3  Batch             gemischte Groessen, Pointer, Fragment-Offsets
  T4  Leakage           ref_conf_pos haengt nicht an der Kristallpose
  T5  Minimal-Invarianz was EXP-100 NICHT aendern darf
  T6  Sampler           Zustandsinvariante ueber echte ODE-Schritte

    PYTHONPATH=src python tests/test_exp100_pipeline.py
"""

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from rdkit import RDLogger
from torch import nn
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
RDLogger.DisableLog("rdApp.*")

from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator  # noqa: E402
from sigmadock.diff.state_reparam import kabsch_rotation           # noqa: E402
from sigmadock.oracle import HPARAMS                               # noqa: E402
from validate_mapping import build_dataset                         # noqa: E402

SCALE = HPARAMS.general.dimensional_scale
LIG_VIRT = HPARAMS.get_node_idx("ligand_virtual")
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(f"{name}  {detail}")
    return ok


def make_generator() -> SigmaFlowGenerator:
    """Die geprueften Methoden fassen das Modell nicht an - ein Platzhalter reicht
    und haelt den Test schnell und deterministisch."""
    return SigmaFlowGenerator(model=nn.Identity(), sigma_min=0.0, verbose=False)


def identity_like(R: torch.Tensor) -> torch.Tensor:
    return torch.eye(3, device=R.device, dtype=R.dtype).expand_as(R)


def reconstruct_endpoint(gen: SigmaFlowGenerator, batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Baut den Zustand bei t=1 ueber die ECHTE Kette und gibt (Rekonstruktion,
    Zielpose) in normalisierten Koordinaten zurueck."""
    pos_0, trans_1, R_1, _ = gen._get_initial_states(batch)
    pos_1_hat = gen._apply_transformations(
        pos_0=pos_0, batch=batch, trans_1=trans_1,
        R_ref=identity_like(R_1), R_t=R_1, trans_t=trans_1,
    )
    if isinstance(batch, Batch):
        ptrs = batch.ptr[1:] - batch.ptr[:-1]
    else:
        ptrs = torch.tensor([batch.x.shape[0]])
    pos_1 = (batch.ref_pos - batch.pocket_com.repeat_interleave(ptrs, dim=0)) / SCALE
    return pos_1_hat, pos_1


def frag_rmsds(pos_hat, pos_true, batch) -> np.ndarray:
    """Rekonstruktions-RMSD je Fragment, nur ueber die Atome, die Kabsch benutzt."""
    fm, ne, m = batch.frag_idx_map, batch.node_entity, batch.mask
    valid = (fm >= 0) & (ne != LIG_VIRT) & m
    flat = SigmaFlowGenerator.get_flat_fragment_index(batch)
    out = []
    for f in range(int(flat.max()) + 1):
        sel = valid & (flat == f)
        if not sel.any():
            continue
        d = (pos_hat[sel] - pos_true[sel]) * SCALE
        out.append(float(torch.sqrt((d ** 2).sum(-1).mean())))
    return np.array(out)


# --------------------------------------------------------------- T1 Toy
def t1_toy() -> None:
    """Bekannte Rotation und Translation durch die echte Kette. Erwartet wird,
    dass R_1 GENAU R* rekonstruiert - nicht die Transponierte."""
    print("\n=== T1  Toy End-to-End (bekanntes R*, p*) ===")
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(7)

    n = 9
    C = torch.randn(n, 3, generator=g, dtype=torch.float64)
    C = C - C.mean(0)
    ang = torch.tensor(1.1, dtype=torch.float64)
    axis = torch.tensor([0.3, -0.7, 0.65], dtype=torch.float64)
    axis = axis / axis.norm()
    K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=torch.float64)
    R_star = torch.eye(3, dtype=torch.float64) + torch.sin(ang) * K + (1 - torch.cos(ang)) * (K @ K)
    p_star = torch.tensor([1.7, -0.4, 2.2], dtype=torch.float64)
    Y = C @ R_star.transpose(-1, -2) + p_star

    R_hat = kabsch_rotation(C, Y - Y.mean(0))
    ang_err = torch.rad2deg(torch.arccos(torch.clamp(((R_hat @ R_star.T).diagonal(dim1=-2, dim2=-1).sum() - 1) / 2, -1, 1)))
    ang_wrong = torch.rad2deg(torch.arccos(torch.clamp(((R_hat @ R_star).diagonal(dim1=-2, dim2=-1).sum() - 1) / 2, -1, 1)))
    check("R_1 == R*", float(ang_err) < 1e-5, f"{float(ang_err):.2e} Grad")
    check("Kontrollprobe: R_1 != R*^T (Test ist richtungssensitiv)",
          float(ang_wrong) > 1.0, f"Abstand zur Transponierten {float(ang_wrong):.1f} Grad")
    check("p_1 == p*", float((Y.mean(0) - p_star).abs().max()) < 1e-12,
          f"{float((Y.mean(0) - p_star).abs().max()):.2e}")
    rec = C @ R_hat.transpose(-1, -2) + Y.mean(0)
    check("Rekonstruktion RMSD ~ 0", float(torch.sqrt(((rec - Y) ** 2).sum(-1).mean())) < 1e-12,
          f"{float(torch.sqrt(((rec - Y) ** 2).sum(-1).mean())):.2e} A")

    # Dieselbe Kette, aber durch _apply_transformations wie im Produktionscode
    gen = make_generator()
    from torch_geometric.data import Data
    N = n + 4
    d = Data(
        x=torch.zeros(N, 1),
        frag_idx_map=torch.cat([torch.full((4,), -1, dtype=torch.long), torch.zeros(n, dtype=torch.long)]),
        node_entity=torch.cat([torch.zeros(4, dtype=torch.long),
                               torch.full((n,), HPARAMS.get_node_idx("ligand_atom"), dtype=torch.long)]),
        mask=torch.ones(N, dtype=torch.bool),
    )
    d.batch = torch.zeros(N, dtype=torch.long)
    pos_0 = torch.cat([torch.zeros(4, 3, dtype=torch.float64), C + p_star], 0)
    got = gen._apply_transformations(
        pos_0=pos_0, batch=d, trans_1=p_star.unsqueeze(0),
        R_ref=torch.eye(3, dtype=torch.float64).unsqueeze(0),
        R_t=R_star.unsqueeze(0), trans_t=p_star.unsqueeze(0),
    )
    err = float((got[4:] - Y).abs().max())
    check("_apply_transformations(R_ref=I, R_t=R*) reproduziert Y", err < 1e-12, f"max |d| = {err:.2e} A")

    # Und der Gegentest: mit R_ref=R* (der alte, falsche Griff) MUSS es brechen
    wrong = gen._apply_transformations(
        pos_0=pos_0, batch=d, trans_1=p_star.unsqueeze(0),
        R_ref=R_star.unsqueeze(0), R_t=R_star.unsqueeze(0), trans_t=p_star.unsqueeze(0),
    )
    werr = float((wrong[4:] - Y).abs().max())
    check("Gegenprobe: R_ref=R_1 waere falsch (Test wuerde das merken)", werr > 0.1,
          f"max |d| = {werr:.2f} A")


# --------------------------------------------------------------- T2 Real
def t2_real(ds, gen) -> np.ndarray:
    print("\n=== T2  Real End-to-End auf echten Komplexen ===")
    all_r, n_ok = [], 0
    for i in range(len(ds)):
        d = ds[i]
        if d is None:
            continue
        b = Batch.from_data_list([d])
        pos_hat, pos_true = reconstruct_endpoint(gen, b)
        check_finite = bool(torch.isfinite(pos_hat).all())
        if not check_finite:
            check(f"item {i}: endlich", False)
            continue
        r = frag_rmsds(pos_hat, pos_true, b)
        all_r.append(r)
        n_ok += 1
    a = np.concatenate(all_r)
    q = np.percentile(a, [50, 90, 95])
    print(f"  Fragmente: {len(a)} aus {n_ok} Komplexen")
    print(f"  Rekonstruktionsfehler [A]  median={q[0]:.3f}  p90={q[1]:.3f}  p95={q[2]:.3f}  max={a.max():.3f}")
    check("Median in der erwarteten Groessenordnung (< 0.15 A)", q[0] < 0.15, f"{q[0]:.3f} A")
    check("p95 in der erwarteten Groessenordnung (< 1.2 A)", q[2] < 1.2, f"{q[2]:.3f} A")
    check("alle Werte endlich", bool(np.isfinite(a).all()))
    return a


# --------------------------------------------------------------- T3 Batch
def t3_batch(ds, gen) -> None:
    print("\n=== T3  Batch mit gemischten Groessen ===")
    items = [ds[i] for i in range(len(ds))]
    items = [d for d in items if d is not None]
    if len(items) < 3:
        check("genug Komplexe fuer einen Batchtest", False, f"{len(items)}")
        return

    b = Batch.from_data_list(items)
    sizes = [int(d.x.shape[0]) for d in items]
    nfrags = [len(torch.unique(d.frag_idx_map[d.frag_idx_map >= 0])) for d in items]
    nvirt = [int((d.node_entity == LIG_VIRT).sum()) for d in items]
    print(f"  Graphgroessen  : {sizes}")
    print(f"  Fragmente      : {nfrags}")
    print(f"  virtuelle Kn.  : {nvirt}")
    check("Graphgroessen wirklich verschieden", len(set(sizes)) > 1)
    check("Fragmentzahlen wirklich verschieden", len(set(nfrags)) > 1)
    check("Anzahl virtueller Knoten wirklich verschieden", len(set(nvirt)) > 1)
    check("ref_conf_pos wurde korrekt gebatcht",
          b.ref_conf_pos.shape == b.ref_pos.shape, f"{tuple(b.ref_conf_pos.shape)}")

    pos_0, trans_1, R_1, num_frag = gen._get_initial_states(b)
    check("num_fragments je Graph stimmt", num_frag.tolist() == nfrags,
          f"{num_frag.tolist()} vs {nfrags}")
    check("trans_1 hat sum(F) Zeilen", trans_1.shape[0] == sum(nfrags))
    check("R_1 ist orthogonal", bool(torch.allclose(
        R_1 @ R_1.transpose(-1, -2),
        torch.eye(3, dtype=R_1.dtype).expand_as(R_1), atol=1e-5)))
    check("det R_1 == +1", float((torch.linalg.det(R_1) - 1).abs().max()) < 1e-5,
          f"max |det-1| = {float((torch.linalg.det(R_1) - 1).abs().max()):.2e}")

    pos_hat, pos_true = reconstruct_endpoint(gen, b)
    r_batch = frag_rmsds(pos_hat, pos_true, b)

    # Entscheidend: pro Graph einzeln muss dasselbe herauskommen.
    r_single = np.concatenate([
        frag_rmsds(*reconstruct_endpoint(gen, Batch.from_data_list([d])), Batch.from_data_list([d]))
        for d in items
    ])
    check("Batch == Einzelverarbeitung", r_batch.shape == r_single.shape
          and bool(np.allclose(r_batch, r_single, atol=1e-4)),
          f"max |d| = {np.abs(r_batch - r_single).max():.2e} A" if r_batch.shape == r_single.shape
          else f"{r_batch.shape} vs {r_single.shape}")

    # Protein darf sich nicht bewegen
    prot = b.frag_idx_map < 0
    check("Proteinatome unveraendert", float((pos_hat[prot] - pos_true[prot]).abs().max()) < 1e-6,
          f"max |d| = {float((pos_hat[prot] - pos_true[prot]).abs().max()):.2e}")

    # Virtuelle Knoten und maskierte Dummies muessen mitgewandert und endlich sein
    aux = (b.frag_idx_map >= 0) & ((b.node_entity == LIG_VIRT) | (~b.mask))
    check("virtuelle/maskierte Knoten vorhanden", bool(aux.any()), f"n={int(aux.sum())}")
    check("virtuelle/maskierte Knoten endlich", bool(torch.isfinite(pos_hat[aux]).all()))


# --------------------------------------------------------------- T4 Leakage
def scramble(mol):
    """Voellig andere 3D-Koordinaten fuer dasselbe Molekuel."""
    from rdkit.Geometry import Point3D
    m = deepcopy(mol)
    c = m.GetConformer()
    g = torch.Generator().manual_seed(99)
    for i in range(m.GetNumAtoms()):
        p = c.GetAtomPosition(i)
        j = torch.randn(3, generator=g).numpy() * 3.0
        c.SetAtomPosition(i, Point3D(float(p.x + j[0] + 25.0), float(p.y + j[1]), float(p.z + j[2] - 17.0)))
    return m


def t4_leakage(ds) -> None:
    """Haengt `ref_conf_pos` an der Kristallpose?

    Die praezise Aussage, die geprueft werden muss, ist die aus der Spezifikation:
    FUER DENSELBEN MOLEKUELGRAPHEN darf `ref_conf_pos` nicht von den
    Eingangskoordinaten abhaengen. Der Graph selbst (Fragmentierung, Anker,
    Dummies, virtuelle Knoten) wird im Training unveraendert aus der gebundenen
    Pose gebaut - das ist eine geerbte Eigenschaft von SigmaFlow-Minimal, die
    EXP-100 nicht anfasst. T4b misst diesen Anteil getrennt und benennt ihn.
    """
    print("\n=== T4a  ref_conf_pos bei FESTEM Graphen ===")
    from sigmadock.chem.processing import get_global_ligand_graph

    n = 0
    for i in range(len(ds)):
        lig_p, pdb_p, ref_p = ds.datafront[i]
        try:
            _, bound, _, _ = ds.parse_complex(sdf=lig_p, pdb=pdb_p, ref_sdf=ref_p)
            frag_mol, frag_td, frag_info = ds.fragment_and_annotate(mol=deepcopy(bound), idx=i)
            frag_graph = get_global_ligand_graph(frag_mol, **frag_td, **frag_info)
        except Exception:
            continue

        kw = dict(frag_mol=frag_mol, frag_torchdata=frag_td, frag_info=frag_info,
                  frag_graph=frag_graph, idx=i)
        a = ds.build_reference_conformer_block(bound_ligand=deepcopy(bound), **kw)
        b_mol = scramble(bound)
        moved = float(np.abs(np.asarray(bound.GetConformer().GetPositions())
                             - np.asarray(b_mol.GetConformer().GetPositions())).max())
        b = ds.build_reference_conformer_block(bound_ligand=b_mol, **kw)

        dev = float((a - b).abs().max())
        check(f"item {i}: Eingangskoordinaten wirklich veraendert", moved > 5.0, f"{moved:.1f} A")
        check(f"item {i}: ref_conf_pos BITGLEICH -> kein Leakage", dev == 0.0, f"max |d| = {dev:.2e} A")
        n += 1
        if n >= 5:
            break
    if n == 0:
        check("T4a lief", False)

    print("\n=== T4b  Wieviel haengt am GRAPHEN (geerbt von Minimal)? ===")
    n = 0
    for i in range(len(ds)):
        lig_p, pdb_p, ref_p = ds.datafront[i]
        try:
            _, bound, _, _ = ds.parse_complex(sdf=lig_p, pdb=pdb_p, ref_sdf=ref_p)
            fa, ta, ia = ds.fragment_and_annotate(mol=deepcopy(bound), idx=i)
            fb, tb, ib = ds.fragment_and_annotate(mol=scramble(bound), idx=i)
        except Exception:
            continue
        same_atoms = fa.GetNumAtoms() == fb.GetNumAtoms()
        same_bonds = ({tuple(sorted((x.GetBeginAtomIdx(), x.GetEndAtomIdx()))) for x in fa.GetBonds()}
                      == {tuple(sorted((x.GetBeginAtomIdx(), x.GetEndAtomIdx()))) for x in fb.GetBonds()})
        same_frags = ia["num_fragments"] == ib["num_fragments"]
        same_ids = ia["fragment_ids"] == ib["fragment_ids"]
        print(f"  item {i}: Atomzahl gleich={same_atoms}  Bindungen gleich={same_bonds}  "
              f"Fragmentzahl gleich={same_frags}  Fragmentzuordnung gleich={same_ids}")
        n += 1
        if n >= 5:
            break
    print("  -> Die Fragmentierung wird im Training aus der gebundenen Pose abgeleitet.")
    print("     Das ist SigmaFlow-Minimal-Verhalten und von EXP-100 unveraendert;")
    print("     es wird in EXPERIMENT.md unter 'Known limitations' gefuehrt.")


# --------------------------------------------------------------- T5 Invarianz
def t5_minimal_invariants(ds, gen) -> None:
    """Was EXP-100 ausdruecklich NICHT aendern darf."""
    print("\n=== T5  Invarianten gegenueber SigmaFlow-Minimal ===")
    d = next(ds[i] for i in range(len(ds)) if ds[i] is not None)
    b = Batch.from_data_list([d])

    ptrs = b.ptr[1:] - b.ptr[:-1]
    pos_1 = (b.ref_pos - b.pocket_com.repeat_interleave(ptrs, dim=0)) / SCALE

    # trans_1 muss bitgleich zur Minimal-Berechnung sein
    com_min, rot_min = SigmaFlowGenerator.get_fragment_com_and_rot(pos_1, b)
    _, trans_1, R_1, _ = gen._get_initial_states(b)
    check("trans_1 identisch zu SigmaFlow-Minimal",
          bool(torch.allclose(torch.cat(com_min, 0), trans_1, atol=0, rtol=0)),
          f"max |d| = {float((torch.cat(com_min, 0) - trans_1).abs().max()):.2e}")
    check("Minimal gibt tatsaechlich R_1 = I zurueck (Kontrast)",
          bool(torch.allclose(torch.cat(rot_min, 0), torch.eye(3).expand_as(torch.cat(rot_min, 0)))))

    ang = torch.rad2deg(torch.arccos(torch.clamp(
        (R_1.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2, -1, 1)))
    check("EXP-100 liefert echte, nicht-triviale Rotationen",
          float(ang.max()) > 20.0,
          f"Winkel zu I: median={float(ang.median()):.1f} max={float(ang.max()):.1f} Grad")

    # pos_0 darf am Protein nichts anfassen
    pos_0, _, _, _ = gen._get_initial_states(b)
    prot = b.frag_idx_map < 0
    check("pos_0 laesst das Protein unveraendert",
          float((pos_0[prot] - pos_1[prot]).abs().max()) < 1e-9)
    check("pos_0 hat den Liganden VERAENDERT (sonst waere EXP-100 wirkungslos)",
          float((pos_0[~prot] - pos_1[~prot]).abs().max()) > 1e-3,
          f"max |d| = {float((pos_0[~prot] - pos_1[~prot]).abs().max()) * SCALE:.2f} A")

    # Fragment-COM von pos_0 muss weiterhin trans_1 sein
    com_new, _ = SigmaFlowGenerator.get_fragment_com_and_rot(pos_0, b)
    check("COM von pos_0 ist weiterhin trans_1",
          float((torch.cat(com_new, 0) - trans_1).abs().max()) < 1e-5,
          f"max |d| = {float((torch.cat(com_new, 0) - trans_1).abs().max()):.2e}")


# --------------------------------------------------------------- T6 Sampler
def t6_sampler_invariant(ds, gen) -> None:
    """Die Zustandsinvariante pos_t = R_t C_F + trans_t muss ueber die
    inkrementellen Schritte des Samplers erhalten bleiben."""
    print("\n=== T6  Sampler-Zustandsinvariante ueber echte Schritte ===")
    d = next(ds[i] for i in range(len(ds)) if ds[i] is not None)
    b = Batch.from_data_list([d])
    pos_0, trans_1, R_1, num_frag = gen._get_initial_states(b)

    torch.manual_seed(3)
    init = gen.flow_matcher.sample_init(int(num_frag.sum()), b.x.device)
    trans_t, R_t = init["trans_0"], init["R_0"]

    # Initialzustand exakt wie in sampling.py
    pos_t = gen._apply_transformations(
        pos_0=pos_0, batch=b, trans_1=trans_1,
        R_ref=identity_like(R_1), R_t=R_t, trans_t=trans_t,
    )

    flat = SigmaFlowGenerator.get_flat_fragment_index(b)
    lig = b.frag_idx_map >= 0

    def invariant_error(pos, R, tr) -> float:
        """||pos - (R C_F + tr)||_inf, mit C_F = pos_0 - trans_1."""
        C = pos_0[lig] - trans_1[flat[lig]]
        want = torch.einsum("nij,nj->ni", R[flat[lig]], C) + tr[flat[lig]]
        return float((pos[lig] - want).abs().max())

    check("Initialzustand erfuellt pos = R_0 C_F + trans_0",
          invariant_error(pos_t, R_t, trans_t) < 1e-5,
          f"max |d| = {invariant_error(pos_t, R_t, trans_t):.2e}")

    # Inkrementelle Schritte, exakt wie im Sampler: jeder Schritt uebergibt den
    # AKTUELLEN Zustand als Referenz. Die Zwischenrotationen sind bewusst
    # willkuerlich - die Invariante darf davon nicht abhaengen.
    n_steps = 10
    torch.manual_seed(11)
    for k in range(n_steps):
        if k == n_steps - 1:
            trans_next, R_next = trans_1, R_1  # letzter Schritt: exakt auf das Ziel
        else:
            trans_next = trans_t + 0.1 * torch.randn_like(trans_t)
            R_next = gen.flow_matcher.sample_init(R_1.shape[0], b.x.device)["R_0"].to(R_1)
        pos_t = gen._apply_transformations(
            pos_0=pos_t, batch=b, trans_1=trans_t, R_ref=R_t,
            trans_t=trans_next, R_t=R_next,
        )
        trans_t, R_t = trans_next, R_next
        if k == 0:
            check("Invariante nach dem ersten Schritt erhalten",
                  invariant_error(pos_t, R_t, trans_t) < 1e-4,
                  f"max |d| = {invariant_error(pos_t, R_t, trans_t):.2e}")

    check(f"Invariante nach {n_steps} Schritten erhalten",
          invariant_error(pos_t, R_t, trans_t) < 1e-4,
          f"max |d| = {invariant_error(pos_t, R_t, trans_t):.2e}")

    # Der Endzustand muss die Zielpose sein
    ptrs = b.ptr[1:] - b.ptr[:-1]
    pos_1 = (b.ref_pos - b.pocket_com.repeat_interleave(ptrs, dim=0)) / SCALE
    r = frag_rmsds(pos_t, pos_1, b)
    check("Sampler-Endzustand == Zielpose (bis auf Starrkoerperrest)",
          float(np.median(r)) < 0.15, f"median RMSD = {np.median(r):.3f} A")


if __name__ == "__main__":
    ds = build_dataset()
    gen = make_generator()
    t1_toy()
    t2_real(ds, gen)
    t3_batch(ds, gen)
    t4_leakage(ds)
    t5_minimal_invariants(ds, gen)
    t6_sampler_invariant(ds, gen)

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"FEHLGESCHLAGEN: {len(FAILURES)}")
        for f in FAILURES:
            print("  - " + f)
        sys.exit(1)
    print("ALLE PIPELINE-TESTS BESTANDEN")
