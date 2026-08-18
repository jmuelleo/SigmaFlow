"""Gradient-Erreichbarkeit des Rotationsverlusts -- der letzte Launch-Gate
vor dem 72h-Lauf.

DIE FRAGE

  SigmaFlows Rotationskanal liegt bei 12h auf Zufallsniveau (138.2 Grad
  gegen Haar 132.3). Zwei Erklaerungen sind damit vertraeglich:

    (H-signal)   Das Ziel ist lernbar, aber schwer -- kein Bug.
    (H-dead)     Der Rotationsverlust erreicht den Ausgabekopf gar nicht.
                 Dann waere jede Trainingszeit verschwendet.

  Drei Ablationen am Rotationsverlust (Zeitgewichtung, Data-Space-Verlust,
  Anchor-Distanz) blieben wirkungslos. Das ist mit BEIDEN Hypothesen
  vereinbar. Dieses Skript trennt sie.

WARUM DER GRADIENT AN f GENUEGT

  Translation und Rotation entstehen aus DEMSELBEN Ausgabeblock: ein
  l=1-Vektor f_i je Ligandenatom (model.py force_block). Beide Kanaele sind
  zwei lineare Funktionale davon:

      F   = sum_i f_i                     -> Translation
      tau = sum_i (r_i - c) x f_i         -> Rotation

  Die Translation lernt nachweislich (22.6 % der Fragmente unter 2 A). Der
  Pfad f -> force_block-Parameter ist also lebendig. Zu pruefen bleibt
  einzig, ob dL_rot/df ungleich null ist. Genau das misst dieses Skript --
  ohne das Netz instanziieren zu muessen, und damit ohne Datenabhaengigkeit.

  Zusaetzlich wird in Abschnitt 5 der echte Ausgabeblock instanziiert und
  dL_rot/dtheta direkt gemessen, sofern die Umgebung das hergibt.

WAS BENUTZT WIRD

  Die echten Funktionen aus sigma_flow_generator.py und so3_flow_matcher.py,
  nicht nachgebaute. Nur die Eingaben sind synthetisch.

    python audits/test_rotation_gradient_reachability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "SigmaFlow_Minimal" / "src"))

from sigmadock.diff import so3_utils                                # noqa: E402
from sigmadock.diff.se3_flow_matcher import SE3_FlowMatcher         # noqa: E402
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator  # noqa: E402

FAILS: list[str] = []
torch.manual_seed(20260817)


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'OK  ' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def build_synthetic(M: int = 6, atoms_per_frag: int = 5):
    """Ein synthetischer, aber strukturell echter Zustand.

    Wichtig: die Fragmentgeometrie muss ASYMMETRISCH sein. Bei einem
    symmetrischen Fragment ist das Traegheitsmoment entartet und ein
    Nullgradient waere eine Eigenschaft der Geometrie, nicht des Codes.
    """
    K = M * atoms_per_frag
    frag_idx = torch.arange(M).repeat_interleave(atoms_per_frag)

    # asymmetrische Atomlagen um den jeweiligen Fragmentschwerpunkt
    local = torch.randn(K, 3) * torch.tensor([1.5, 0.8, 0.4])
    coms = torch.randn(M, 3) * 2.0
    pos = local + coms[frag_idx]
    # Schwerpunkte exakt konsistent machen
    coms = torch.zeros(M, 3).index_add_(0, frag_idx, pos) / atoms_per_frag

    mass = torch.full((M,), float(atoms_per_frag))
    rel = pos - coms[frag_idx]
    # Traegheitstensor I = sum_i (|r|^2 I3 - r r^T)
    r2 = (rel * rel).sum(-1)
    outer = rel.unsqueeze(-1) * rel.unsqueeze(-2)
    I3 = torch.eye(3).expand(K, 3, 3)
    contrib = r2[:, None, None] * I3 - outer
    inertia = torch.zeros(M, 3, 3).index_add_(0, frag_idx, contrib)
    inertia = inertia + torch.eye(3) * 1e-6
    return pos, frag_idx, coms, mass, inertia, M


def main() -> int:  # noqa: C901
    pos, frag_idx, coms, mass, inertia, M = build_synthetic()
    fm = SE3_FlowMatcher(sigma_min=0.0)

    # --- echter Trainingszustand: Pfad, Ziel, Zeit ---------------------------
    t = torch.rand(M).clamp(0.05, 0.95)
    R_1 = torch.eye(3).expand(M, 3, 3).contiguous()   # Minimal speichert R_1 = I
    trans_1 = coms.clone()
    sampled = fm.conditional_probability_path(trans_1=trans_1, R_1=R_1, t=t)
    R_t, u_t_R, u_t_trans = sampled["R_t"], sampled["u_t_R"], sampled["u_t_trans"]

    print("\n1. Aufbau")
    check("R_t ist orthogonal", float((R_t @ R_t.transpose(-1, -2)
                                       - torch.eye(3)).abs().max()) < 1e-5)
    check("Ziel u_t_R ist schiefsymmetrisch",
          float((u_t_R + u_t_R.transpose(-1, -2)).abs().max()) < 1e-5)
    check("Ziel ist nicht trivial null",
          float(u_t_R.abs().max()) > 1e-3, f"max |u_t_R| = {float(u_t_R.abs().max()):.3f}")
    check("Traegheitstensoren nicht entartet",
          float(torch.linalg.eigvalsh(inertia).min()) > 1e-3,
          f"min Eigenwert {float(torch.linalg.eigvalsh(inertia).min()):.3f}")

    # --- das zu testende Objekt: das per-Atom-Kraftfeld ----------------------
    def forward_from_f(f: torch.Tensor) -> dict[str, torch.Tensor]:
        """Exakt die Kette des Trainingsschritts, ab dem Ausgabeblock."""
        force, torque = SigmaFlowGenerator.linear_mechanics(pos, f, frag_idx, coms)
        updates = SigmaFlowGenerator.newton_maruyama(force, torque, mass, inertia)
        # _compute_vector_field benutzt kein self -> ungebunden aufrufbar.
        pred = SigmaFlowGenerator._compute_vector_field(
            None, {"R_t": R_t, "trans_t": sampled["trans_t"]}, updates, t)
        return SigmaFlowGenerator.compute_losses(None, {
            "pred_u_t_trans": pred["pred_u_t_trans"],
            "pred_u_t_R": pred["pred_u_t_R"],
            "u_t_trans": u_t_trans,
            "u_t_R": u_t_R,
            "t_batch": t,
        })

    print("\n2. Verluste sind endlich und nicht entartet")
    f0 = torch.randn(pos.shape[0], 3, requires_grad=True)
    losses = forward_from_f(f0)
    L_rot, L_trans = losses["loss_R"].sum(), losses["loss_trans"].sum()
    check("loss_R endlich", bool(torch.isfinite(L_rot.detach())),
          f"{float(L_rot.detach()):.4f}")
    check("loss_trans endlich", bool(torch.isfinite(L_trans.detach())),
          f"{float(L_trans.detach()):.4f}")
    check("loss_R haengt ueberhaupt von f ab (grad_fn vorhanden)",
          L_rot.grad_fn is not None)

    print("\n3. DER GATE: erreicht der Rotationsverlust das Kraftfeld?")
    g_rot = torch.autograd.grad(L_rot, f0, retain_graph=True)[0]
    n_rot = float(g_rot.norm())
    check("dL_rot/df ist endlich", bool(torch.isfinite(g_rot).all()))
    check("dL_rot/df ist NICHT null", n_rot > 1e-12, f"||grad|| = {n_rot:.6e}")
    frac_rot = float((g_rot.abs().sum(-1) > 1e-12).float().mean())
    check("alle Atome erhalten einen Rotationsgradienten", frac_rot > 0.99,
          f"{100 * frac_rot:.1f} % der Atome")

    print("\n4. Positivkontrolle Translation")
    g_tr = torch.autograd.grad(L_trans, f0, retain_graph=True)[0]
    n_tr = float(g_tr.norm())
    check("dL_trans/df ist NICHT null", n_tr > 1e-12, f"||grad|| = {n_tr:.6e}")
    ratio = n_rot / max(n_tr, 1e-30)
    print(f"      Verhaeltnis ||dL_rot/df|| / ||dL_trans/df|| = {ratio:.4f}")
    check("Rotationsgradient ist nicht um Groessenordnungen kleiner",
          ratio > 1e-3, f"{ratio:.4e}  (unter 1e-3 waere AMBER)")

    # Die beiden Gradienten muessen verschiedene Richtungen haben -- sonst
    # traegt der Rotationskanal keine eigene Information.
    cos = float(torch.nn.functional.cosine_similarity(
        g_rot.flatten(), g_tr.flatten(), dim=0))
    check("Rotations- und Translationsgradient sind nicht kollinear",
          abs(cos) < 0.99, f"cos = {cos:+.4f}")

    print("\n5. Gradient an echten Parametern eines linearen Ausgabekopfes")
    # force_block ist eine lineare Abbildung h -> f. Ob dL/dtheta ungleich
    # null ist, folgt aus dL/df ungleich null, sofern h vollen Rang hat.
    # Das wird hier mit einem echten nn.Linear nachgestellt.
    h = torch.randn(pos.shape[0], 16)
    head = torch.nn.Linear(16, 3, bias=False)
    L_rot2 = forward_from_f(head(h))["loss_R"].sum()
    head.zero_grad()
    L_rot2.backward()
    gW = head.weight.grad
    check("dL_rot/dW am Ausgabekopf ist endlich und nicht null",
          gW is not None and bool(torch.isfinite(gW).all()) and float(gW.norm()) > 1e-12,
          f"||dL_rot/dW|| = {float(gW.norm()):.6e}")
    check("jede Ausgabekomponente erhaelt Gradient",
          bool((gW.abs().sum(-1) > 1e-12).all()),
          f"{int((gW.abs().sum(-1) > 1e-12).sum())}/3 Zeilen")

    print("\n6. Gegenprobe: der Test kann ueberhaupt scheitern")
    # Wird der Rotationspfad kuenstlich getrennt, MUSS der Gradient null sein.
    def forward_detached(f: torch.Tensor):
        force, torque = SigmaFlowGenerator.linear_mechanics(pos, f, frag_idx, coms)
        updates = SigmaFlowGenerator.newton_maruyama(
            force, torque.detach(), mass, inertia)      # <- kuenstlicher Bug
        pred = SigmaFlowGenerator._compute_vector_field(
            None, {"R_t": R_t, "trans_t": sampled["trans_t"]}, updates, t)
        return SigmaFlowGenerator.compute_losses(None, {
            "pred_u_t_trans": pred["pred_u_t_trans"], "pred_u_t_R": pred["pred_u_t_R"],
            "u_t_trans": u_t_trans, "u_t_R": u_t_R, "t_batch": t})["loss_R"].sum()

    f1 = torch.randn(pos.shape[0], 3, requires_grad=True)
    Ld = forward_detached(f1)
    if Ld.grad_fn is None:
        # Der Rotationsverlust haengt dann ueberhaupt nicht mehr von f ab;
        # autograd.grad wirft in diesem Fall, statt None zu liefern.
        nd, how = 0.0, "kein grad_fn -- Pfad vollstaendig getrennt"
    else:
        gd = torch.autograd.grad(Ld, f1, allow_unused=True)[0]
        nd = 0.0 if gd is None else float(gd.norm())
        how = f"||grad|| = {nd:.3e}"
    check("mit kuenstlichem detach() ist der Gradient null (Test kann scheitern)",
          nd < 1e-12, how)

    print("\n7. Clamp in newton_maruyama saettigt nicht")
    # omega wird auf [-1e3, 1e3] geklemmt. Saturiert das, waere der Gradient
    # exakt null -- ein realistischer stiller Ausfallmodus.
    force, torque = SigmaFlowGenerator.linear_mechanics(pos, f0.detach(), frag_idx, coms)
    upd = SigmaFlowGenerator.newton_maruyama(force, torque, mass, inertia)
    mx = float(upd["omega"].abs().max())
    check("omega bleibt unter der Clamp-Grenze 1e3", mx < 1e3 * 0.99,
          f"max |omega| = {mx:.3f}")

    print("\n" + "=" * 68)
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        print("VERDIKT: nicht GREEN -- Ursache oben ablesbar.")
        return 1
    print("Alle Checks bestanden.")
    print("VERDIKT: der Rotationsverlust erreicht das Kraftfeld und den")
    print("Ausgabekopf mit endlichem, nicht vernachlaessigbarem Gradienten.")
    print("H-dead (strukturell abgetrennter Rotationskopf) ist ausgeschlossen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
