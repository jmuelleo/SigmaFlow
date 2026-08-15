"""Wie gross ist der Aequivarianzbruch der S2-Aktivierung bei Produktionsgroesse?

Lauf 2 hat bewiesen: mit Gate-Aktivierung ist die Kette exakt aequivariant
(1e-15), mit S2-Aktivierung nicht (1e-1). Die Produktionsvorgabe ist
use_gate_act=False / use_sep_s2_act=True, also die S2-Variante.

Offene Frage: war 1e-1 ein Artefakt des winzigen Testnetzes (16 Kanaele,
2 Bloecke) oder bleibt es bei der echten Groesse (128 Kanaele, 6 Bloecke)?

Zusaetzlich: derselbe Defekt macht den Forward nicht-deterministisch, weil
init_edge_rot_mat die Achse senkrecht zur Kante zufaellig zieht. Das wird
mitgemessen - es betrifft die Reproduzierbarkeit beim Sampling.
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

from sigmadock.config import StructuralConfig, TrainingConfig        # noqa: E402
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
    return (a - b).abs().max().item() / max(a.abs().max().item(),
                                            b.abs().max().item(), 1e-30)


class TimeEmb32(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner.float()

    def forward(self, t):
        return self.inner(t.float()).double()


def main():
    C = TrainingConfig
    ds = build_dataset()
    items = [d for d in (ds[i] for i in range(len(ds))) if d is not None][:2]
    batch = Batch.from_data_list(items)

    def to64(v):
        if torch.is_tensor(v):
            return v.double() if v.is_floating_point() else v
        if isinstance(v, dict):
            return {k: to64(x) for k, x in v.items()}
        return v

    for k in list(batch.keys()):
        batch[k] = to64(batch[k])

    N = batch.x.shape[0]
    pos = torch.randn(N, 3, generator=torch.Generator().manual_seed(11)) * 0.5
    t = torch.full((batch.num_graphs,), 0.37)

    print("=" * 78)
    print("S2-AKTIVIERUNG BEI PRODUKTIONSGROESSE")
    print("=" * 78)
    print(f"TrainingConfig: sphere_channels={C.sphere_channels}, "
          f"num_layers={C.num_layers}, num_heads={C.num_heads},")
    print(f"                l_max={C.l_max_list}, m_max={C.m_max_list}, "
          f"ffn={C.ffn_hidden_channels}")
    print("EquiformerV2-Defaults: use_gate_act=False, use_sep_s2_act=True")
    print("-> die Produktionskonfiguration benutzt die S2-Aktivierung.\n")

    torch.manual_seed(0)
    m = EquiformerV2(
        atom_feature_dims=list(StructuralConfig.atom_feature_dims),
        edge_feature_dims=list(StructuralConfig.edge_feature_dims),
        average_degrees=HPARAMS.all_degrees,
        use_esm_embeddings=False,
        num_layers=C.num_layers, num_heads=C.num_heads,
        sphere_channels=C.sphere_channels, edge_channels=C.edge_channels,
        attn_hidden_channels=C.attn_hidden_channels,
        attn_alpha_channels=C.attn_alpha_channels,
        attn_value_channels=C.attn_value_channels,
        ffn_hidden_channels=C.ffn_hidden_channels,
        lmax_list=list(C.l_max_list), mmax_list=list(C.m_max_list),
        distance_expansion_dim=C.distance_expansion_dim, t_emb_dim=C.t_emb_dim,
        alpha_drop=0.0, drop_path_rate=0.0, proj_drop=0.0,
        zero_init_last=False,   # Default True gaebe exakt 0 -> Test blind
    ).eval()
    m.time_embedder = TimeEmb32(m.time_embedder)
    print(f"Parameter: {m.num_params/1e6:.2f} M\n")

    def fwd(p, seed):
        b = batch.clone()
        b["pos_t"] = p
        b["pos_0"] = p
        torch.manual_seed(seed)
        with torch.no_grad():
            return m(b, t)

    gq = torch.Generator().manual_seed(4242)
    gauge, equi = [], []
    for k in range(10):
        Q = rand_rot(gq)
        f0 = fwd(pos, 100 + k)
        f0b = fwd(pos, 900 + k)
        fQ = fwd(pos @ Q.T, 500 + k)
        gauge.append(rel(f0, f0b))
        equi.append(rel(fQ, f0 @ Q.T))

    print("relative Residuen, 10 zufaellige Rotationen, float64:")
    print(f"  Nicht-Determinismus  f(x) vs. f(x) anderer Gauge : "
          f"median {np.median(gauge):.3e}  max {max(gauge):.3e}")
    print(f"  Aequivarianz         f(Qx) vs. Q f(x)            : "
          f"median {np.median(equi):.3e}  max {max(equi):.3e}")
    print()
    print("Zum Vergleich (Lauf 2, 16 Kanaele / 2 Bloecke):")
    print("  S2   : Gauge 8.15e-02   Aequivarianz 1.39e-01")
    print("  Gate : Gauge 1.35e-15   Aequivarianz 5.28e-15")
    print("=" * 78)


if __name__ == "__main__":
    main()
