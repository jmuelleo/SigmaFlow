"""Frame-Audit v2 - mit exakt aequivarianten Aktivierungen.

Lauf 1 zeigte Residuen um 1e-1, aber der Gauge-Test (gleicher Input, andere
Zufallsachse in init_edge_rot_mat) lag schon bei 9e-2. Damit war das
Rauschniveau zu hoch, um "exakt" von "approximativ" zu trennen.

Verdaechtig ist die S2-Aktivierung: sie geht ueber ein endliches Gitter und
ist nachweislich nur naeherungsweise aequivariant. Dieses Skript ersetzt sie
durch die Gate-Aktivierung (use_gate_act=True, use_sep_s2_act=False), die rein
aus l=0-Skalaren x l>0-Kanaelen besteht und exakt aequivariant ist.

Faellt das Residuum dann auf Maschinengenauigkeit, ist bewiesen:
  * der Head ist architektonisch exakt aequivariant,
  * die Abweichung in Lauf 1 kam allein aus der S2-Aktivierung,
  * die Gauge-Wahl ist irrelevant.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import RDLogger
from torch_geometric.data import Batch

EXP = Path(r"C:\Users\julia\Documents\SigmaFlow\SigmaFlow_FM_Specific\EXP-100_state_reparam")
MIN = Path(r"C:\Users\julia\Documents\SigmaFlow\SigmaFlow_Minimal")
sys.path.insert(0, str(MIN / "src"))
sys.path.insert(0, str(EXP / "tests"))
RDLogger.DisableLog("rdApp.*")
torch.set_default_dtype(torch.float64)

from sigmadock.chem.processing import get_lig_idxs                   # noqa: E402
from sigmadock.config import StructuralConfig                        # noqa: E402
from sigmadock.diff.se3_flow_matcher import SE3_FlowMatcher          # noqa: E402
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator   # noqa: E402
from sigmadock.net.model import EquiformerV2                         # noqa: E402
from sigmadock.oracle import HPARAMS                                 # noqa: E402
from validate_mapping import build_dataset                           # noqa: E402


def rand_rot(gen):
    A = torch.randn(3, 3, generator=gen)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def rel(a, b):
    d = (a - b).abs().max().item()
    s = max(a.abs().max().item(), b.abs().max().item(), 1e-30)
    return d / s


class TimeEmb32(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner.float()

    def forward(self, t):
        return self.inner(t.float()).double()


def build_model(lmax, mmax, gate, seed=0):
    torch.manual_seed(seed)
    m = EquiformerV2(
        atom_feature_dims=list(StructuralConfig.atom_feature_dims),
        edge_feature_dims=list(StructuralConfig.edge_feature_dims),
        average_degrees=HPARAMS.all_degrees,
        use_esm_embeddings=False,
        num_layers=2, num_heads=2,
        sphere_channels=16, edge_channels=8,
        attn_hidden_channels=8, attn_alpha_channels=8, attn_value_channels=8,
        ffn_hidden_channels=16,
        lmax_list=[lmax], mmax_list=[mmax],
        distance_expansion_dim=8, t_emb_dim=8,
        alpha_drop=0.0, drop_path_rate=0.0, proj_drop=0.0,
        use_gate_act=gate, use_sep_s2_act=not gate,
        zero_init_last=False,
    ).eval()
    m.time_embedder = TimeEmb32(m.time_embedder)
    return m


def main():
    ds = build_dataset()
    items = [d for d in (ds[i] for i in range(len(ds))) if d is not None][:2]
    batch = Batch.from_data_list(items)

    def to64(v):
        if torch.is_tensor(v):
            return v.double() if v.is_floating_point() else v
        if isinstance(v, dict):
            return {k: to64(x) for k, x in v.items()}
        return v

    for key in list(batch.keys()):
        batch[key] = to64(batch[key])

    N = batch.x.shape[0]
    pos = torch.randn(N, 3, generator=torch.Generator().manual_seed(11)) * 0.5
    t = torch.full((batch.num_graphs,), 0.37)
    P = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])

    def forces(model, p, seed):
        b = batch.clone()
        b["pos_t"] = p
        b["pos_0"] = p
        torch.manual_seed(seed)
        with torch.no_grad():
            return model(b, t)

    print("=" * 78)
    print("A. AKTIVIERUNG ISOLIEREN")
    print("=" * 78)
    print(f"{'Konfiguration':<34}{'Gauge':>12}{'Q f(x)':>12}{'f(x)':>12}{'PQP^T':>12}")
    print("-" * 78)
    for lmax, mmax, gate in [(3, 2, False), (3, 2, True), (3, 3, True), (1, 1, True)]:
        model = build_model(lmax, mmax, gate)
        gq = torch.Generator().manual_seed(4242)
        ga, ha, hb, hc = [], [], [], []
        for k in range(12):
            Q = rand_rot(gq)
            f0 = forces(model, pos, 100 + k)
            f0b = forces(model, pos, 900 + k)          # andere Gauge, gleicher Input
            fQ = forces(model, pos @ Q.T, 500 + k)
            ga.append(rel(f0, f0b))
            ha.append(rel(fQ, f0 @ Q.T))
            hb.append(rel(fQ, f0))
            hc.append(rel(fQ, f0 @ (P @ Q @ P.T).T))
        name = f"lmax={lmax} mmax={mmax} {'gate' if gate else 'S2  '}"
        print(f"{name:<34}{np.median(ga):>12.2e}{np.median(ha):>12.2e}"
              f"{np.median(hb):>12.2e}{np.median(hc):>12.2e}")

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("B. GRUPPENWIRKUNG AUF R_t - AUS DEM CODE ABGELEITET, NICHT GERATEN")
    print("=" * 78)
    print("_apply_transformations baut pos_t aus pos_0 und R_t.")
    print("Frage: welches R' erzeugt aus Q*pos_0 genau Q*pos_t?")
    gen0 = SigmaFlowGenerator(build_model(3, 2, True), sigma_min=0.0,
                              cutoff_complex_interactions=HPARAMS.get_edge_spec("inter_complex").r_max,
                              cutoff_fragment_interactions=HPARAMS.get_edge_spec("inter_fragments").r_max,
                              cutoff_complex_virtual=HPARAMS.get_edge_spec("complex_lv2pv").r_max)
    Fnum = int(gen0.get_flat_fragment_index(batch).max().item()) + 1
    gr = torch.Generator().manual_seed(7)
    R_t = torch.stack([rand_rot(gr) for _ in range(Fnum)])
    I = torch.eye(3).expand(Fnum, 3, 3).contiguous()
    trans, _ = gen0.get_fragment_com_and_rot(pos, batch)
    trans = torch.cat(trans, 0)

    d_left, d_conj = [], []
    gq = torch.Generator().manual_seed(555)
    for _ in range(12):
        Q = rand_rot(gq)
        pt = gen0._apply_transformations(pos, batch, trans, I, R_t, trans)
        # Referenz mitgedreht: Q*pos_0, Q*trans
        pl = gen0._apply_transformations(pos @ Q.T, batch, trans @ Q.T, I,
                                         Q @ R_t, trans @ Q.T)
        pc = gen0._apply_transformations(pos @ Q.T, batch, trans @ Q.T, I,
                                         Q @ R_t @ Q.T, trans @ Q.T)
        d_left.append(rel(pl, pt @ Q.T))
        d_conj.append(rel(pc, pt @ Q.T))
    print(f"   R_t -> Q R_t       ergibt Q*pos_t :  {np.median(d_left):.3e}")
    print(f"   R_t -> Q R_t Q^T   ergibt Q*pos_t :  {np.median(d_conj):.3e}")
    print("   -> Die korrekte Wirkung auf R_t ist die mit dem kleinen Residuum.")
    print(f"   R_1 ist im Code hart die Identitaet; Q I Q^T = I bleibt erhalten,")
    print(f"   Q I = Q nicht. Nur die Konjugation ist mit R_1=I vertraeglich.")

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("C. VOLLE KETTE MIT EXAKTER AKTIVIERUNG")
    print("=" * 78)
    model = build_model(3, 2, True)
    gen = SigmaFlowGenerator(model, sigma_min=0.0,
                             cutoff_complex_interactions=HPARAMS.get_edge_spec("inter_complex").r_max,
                             cutoff_fragment_interactions=HPARAMS.get_edge_spec("inter_fragments").r_max,
                             cutoff_complex_virtual=HPARAMS.get_edge_spec("complex_lv2pv").r_max)

    def pipeline(p, Rt, seed):
        b = batch.clone()
        b["pos_t"] = p
        b["pos_0"] = p
        torch.manual_seed(seed)
        with torch.no_grad():
            eps = model(b, t)
        idx = get_lig_idxs(b.node_entity, mask=b.mask)
        lig_f = -eps.index_select(0, idx)
        cont = gen.get_flat_fragment_index(b)
        coms = torch.cat(gen.get_fragment_com_and_rot(p, b)[0], 0)
        M, I_t = gen.get_fragment_mass_inertia(p, b)
        F, T = gen.linear_mechanics(p[idx], lig_f, cont[idx], coms)
        upd = gen.newton_maruyama(F, T, M, I_t)
        uR = gen._compute_vector_field({"R_t": Rt}, upd, None)["pred_u_t_R"]
        return dict(F=F, T=T, dW=upd["delta_W"], om=upd["omega"], uR=uR, I=I_t)

    rows = {k: [] for k in ("F", "T", "I", "dW", "om", "uR_inv", "uR_conj")}
    gq = torch.Generator().manual_seed(999)
    for k in range(12):
        Q = rand_rot(gq)
        a = pipeline(pos, R_t, 100 + k)
        b_ = pipeline(pos @ Q.T, Q @ R_t @ Q.T, 700 + k)
        rows["F"].append(rel(b_["F"], a["F"] @ Q.T))
        rows["T"].append(rel(b_["T"], a["T"] @ Q.T))
        rows["I"].append(rel(b_["I"], Q @ a["I"] @ Q.T))
        rows["dW"].append(rel(b_["dW"], a["dW"] @ Q.T))
        rows["om"].append(rel(b_["om"], Q @ a["om"] @ Q.T))
        rows["uR_inv"].append(rel(b_["uR"], a["uR"]))
        rows["uR_conj"].append(rel(b_["uR"], Q @ a["uR"] @ Q.T))

    lbl = {
        "F":  "total_force     F -> Q F           Weltvektor",
        "T":  "total_torque    tau -> Q tau       Weltvektor",
        "I":  "inertia         I -> Q I Q^T       Welt-Tensor Stufe 2",
        "dW": "delta_W         dW -> Q dW         Weltvektor",
        "om": "omega           w -> Q w Q^T       Welt-so(3)",
        "uR_inv":  "pred_u_t_R      gegen INVARIANT    (waere reines body)",
        "uR_conj": "pred_u_t_R      u -> Q u Q^T",
    }
    for k, v in rows.items():
        print(f"   {lbl[k]:<52} {np.median(v):.3e}")

    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("D. ZIEL vs. VORHERSAGE - ist der Loss rotationsinvariant?")
    print("=" * 78)
    fm = SE3_FlowMatcher(0.0)
    tb = torch.full((Fnum,), 0.37)
    z = torch.zeros(Fnum, 3)
    dt = []
    gq = torch.Generator().manual_seed(31337)
    for _ in range(12):
        Q = rand_rot(gq)
        u0 = fm.calc_vector_field(z, R_t, z, I, tb)["u_t_R"]
        uQ = fm.calc_vector_field(z, Q @ R_t @ Q.T, z, I, tb)["u_t_R"]
        dt.append(rel(uQ, Q @ u0 @ Q.T))
    print(f"   Ziel u_t_R      u -> Q u Q^T                         {np.median(dt):.3e}")
    print("   Vorhersage transformiert identisch (Zeile oben)")
    print("   -> ||pred - target||_F ist invariant, der Loss ist wohldefiniert.")
    print("=" * 78)


if __name__ == "__main__":
    main()
