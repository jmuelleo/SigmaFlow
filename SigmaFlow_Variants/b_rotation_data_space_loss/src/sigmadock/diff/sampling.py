from copy import deepcopy
from typing import Literal

import numpy as np
import torch
from pytorch_lightning import seed_everything

# from torch import nn
from torch_geometric.data import Batch
from tqdm import tqdm

from sigmadock.diff.se3_flow_matcher import SE3_FlowMatcher
from sigmadock.oracle import HPARAMS
from sigmadock.diff.sigma_flow_generator import SigmaFlowGenerator


# NOTE this function evaluates given true pose exists (ReDocking Scenario) -> Not generic (yet)
def sample_notebook(
    denoiser: SigmaFlowGenerator,
    batch: Batch,
    t_min: float = HPARAMS.general.epsilon_t,
    rho: float = 3.0,
    t_max: float = 1.0,
    num_steps: int = 18,
    noise_scale: float = 0.1,
    noise_decay: float = 2.0,
    solver: Literal["euler", "heun"] = "euler",
    discretization: Literal["power", "edm"] = "power",
    seed: int = 0,
    use_true_vector_field: bool = False,
    verbose: bool = False,
) -> tuple[Batch, list[np.ndarray]]:
    """Integrate the flow-matching ODE forward (t_min -> t_max, noise -> data) with
    SigmaFlowGenerator's predicted vector field, via SE3_FlowMatcher.euler_step.
    Args:
        denoiser: SigmaFlowGenerator instance whose predicted vector field drives the ODE.
        batch: Batch of data containing the initial states and other necessary information.
        num_steps: Number of steps in the forward ODE integration.
        solver: Solver type for the ODE. Choices are:
            - euler (1x inference)
            - heun (2x inference)
            defaults to "euler".
        rho: Exponent for the time step discretization.
        noise_scale: kept for notebook call-site compatibility; the ODE integration here is
            deterministic and does not currently inject any noise (see STATUS.md PAUSE-PUNKT #11).
        t_min: Minimum time step for the ODE integration. Defaults to oracle.py's epsilon_t,
            the same floor used when sampling t during training - the network was never trained
            on t below this value.
        t_max: Maximum time step for the ODE integration.
        seed: Random seed for reproducibility.
        use_true_vector_field: If True, uses the true vector field for the ODE step instead of the denoiser's predicted vector field.
    Returns:
        A tuple containing the updated batch and a list of positions at each step of the ODE integration.
    Raises:
        AssertionError: If noise_scale is not in the range [0, 1].
    """

    assert noise_scale >= 0, "Noise scale must be non-negative."
    assert noise_scale <= 1, "Noise scale must be less than or equal to 1."
    if isinstance(seed, int):
        assert seed >= 0, "Seed must be non-negative."

    # For each seed clone the batch and repeat the data items!
    # TODO. Currently outside loop.

    # Seed everything for reproducibility
    seed_everything(seed, workers=True, verbose=False)
    batch = denoiser._prepare_batch(batch)

    # Note this could be a random conformer in 3D space -> Typically we do this in inference.
    pos_0, trans_1, R_1, num_fragments = denoiser._get_initial_states(batch)
    sampled_init = denoiser.flow_matcher.sample_init(torch.sum(num_fragments), batch.x.device)
    trans_0, R_0 = sampled_init["trans_0"], sampled_init["R_0"]


    # Update roto-translations to ambient space (complex)
    pos_T = denoiser._apply_transformations(
        pos_0=pos_0,
        batch=batch,
        trans_1=trans_1,
        R_1=R_1,
        trans_t=trans_0,
        R_t=R_0,
    )

    step_indices = torch.arange(num_steps, device=batch.x.device)  # [N]
    if discretization == "power":
        timesteps = torch.linspace(t_min, t_max, num_steps, device=batch.x.device)
    else:
        raise ValueError(f"Unknown discretization {discretization}. Choose 'power' or 'edm'.")

    # Initialize the denoiser with the initial states
    batch = denoiser._update_batch(
        batch=batch,
        pos_0=pos_0,
        pos_t=pos_T,
    )
    pos_t = pos_T
    trans_t = trans_0
    R_t = R_0

    if verbose:
        print(
            f"Using {num_steps} steps for forward ODE integration with: \n \
            seed={seed} \n \
            rho={rho} \n \
            solver={solver} \n \
            noise_scale={noise_scale} \n \
            t_min={t_min} \n \
            "
        )

    @torch.no_grad
    def _predict_vector_field_step(
        batch: Batch,
        t: float,
        R_t: torch.Tensor,
        trans_t: torch.Tensor,
        R_0: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        t_batch = t.repeat_interleave(sum(num_fragments))  # [B x F]
        # Create discretization & Get time from discretization
        lig_pseudoforces, forces_idxs = denoiser._compute_forces(
            batch=batch,
            t=torch.tensor([t] * batch.num_graphs, device=pos_t.device),  # [B]
        )  # [B x F x A, 3], [B x F x A]
        # Linear mechanics: mass, inertia, force & torque
        force_per_fragment, torque_per_fragment, frag_mass, frag_inertia_t = denoiser._compute_fragment_dynamics(
            batch=batch,
            R_t=R_t,  # [B x F, 3, 3]
            trans_t=trans_t,  # [B x F, 3]
            # NOTE R_1 only exists as safety during training for I_t calc. not required strictly (IRL)
            R_1=R_0,  # [B x F, 3, 3]
            lig_forces=lig_pseudoforces,
            forces_idxs=forces_idxs,
        )  # [B x F, 3], [B x F, 3], [B,F], [B x F, 3, 3]

        # Compute total scaled forces/torques and predict updates (Newton-Maruyama)
        fragment_updates = denoiser._predict_fragment_updates(
            force_per_fragment=force_per_fragment,
            torque_per_fragment=torque_per_fragment,
            frag_mass=frag_mass,
            frag_inertia_t=frag_inertia_t,
            # t_batch=t_batch,
        )  # [B x F, 3], [B x F, 3, 3]

        # Compute [R3, so3] vector_field
        pred_vector_field = denoiser._compute_vector_field({"R_t": R_t, "trans_t": trans_t}, fragment_updates, t_batch)
        return pred_vector_field

    all_pos = [pos_t.cpu().numpy()]
    all_edges = [
        {
            "edge_index": batch.edge_index,
            "edge_attr": batch.edge_attr,
            "edge_entity": batch.edge_entity,
        }
    ]
    all_losses = []
    # Iterate forward across timesteps (t_min -> t_max, noise -> data)
    for i, t in tqdm(enumerate(timesteps[:-1])):
        dt = timesteps[i+1] - timesteps[i]
        t = torch.tensor(t, device=batch.x.device)  # [1]
        t_batch = t.repeat_interleave(sum(num_fragments))  # [B x F]

        true_vector_field = denoiser._compute_true_vector_field(
            trans_1=trans_1,
            R_1=R_1,
            Tt=trans_t,
            Rt=R_t,
            t_batch=t_batch,
        )

        grad_trans_t = true_vector_field["u_t_trans"]
        grad_R_t = true_vector_field["u_t_R"]
        # Use True or Predicted vector_field
        if not use_true_vector_field:
            # Deepcopy because this step modifies the batch in-place (removes masked edges)
            pred_vector_field = _predict_vector_field_step(deepcopy(batch), t, R_t=R_t, trans_t=trans_t, R_0=R_0)
            grad_T_p = pred_vector_field["pred_u_t_trans"]
            grad_R_p = pred_vector_field["pred_u_t_R"]
        else:
            grad_T_p = grad_trans_t
            grad_R_p = grad_R_t

        # Log the losses
        losses: dict[str, torch.Tensor] = denoiser.compute_losses(
            {
                "pred_u_t_trans": grad_T_p,  # [B x F, 3]
                "pred_u_t_R": grad_R_p,  # [B x F, 3, 3]
                "u_t_trans": grad_trans_t,  # [B x F, 3]
                "u_t_R": grad_R_t,  # [B x F, 3, 3]
                "t_batch": t_batch,  # [B x F]
                "R_t": R_t,  # [B x F, 3, 3]
                "R_1": R_1,  # [B x F, 3, 3]
            }
        )
        # ODE step (use discretization & solver)
        step_result = denoiser.flow_matcher.euler_step(
            trans_t=trans_t,
            R_t=R_t,
            v_t_trans=grad_T_p,
            v_t_R=grad_R_p,
            dt=dt
        )
        trans_next, R_next = step_result["trans_new"], step_result["R_new"]

        # Update positions according to transformations from this ODE step.
        pos_t = denoiser._apply_transformations(
            batch=batch,
            # Refererence
            pos_0=pos_t,
            trans_1=trans_t,
            R_1=R_t,
            # Transformation
            trans_t=trans_next,
            R_t=R_next,
        )
        # Advance the fragment roto-translations for the next step.
        trans_t = trans_next
        R_t = R_next
        # Update batch with new positions (pos_t) and remove prev local interactions
        batch = denoiser._update_batch(
            batch=batch,
            pos_0=pos_0,
            pos_t=pos_t,
        )

        all_pos.append(pos_t.cpu().numpy())
        all_edges.append(
            {
                "edge_index": batch.edge_index,
                "edge_attr": batch.edge_attr,
                "edge_entity": batch.edge_entity,
            }
        )
        all_losses.append(losses)

        # TODO Heun's method
        #  Will require us to look at derivatives for T and R and average the Func Evaluations at t, t-t'
        #  Remember last step must be Euler so NFE = 2 * (N - 1) + 1
        #  Truncation error: O(N) -> O(N**3) which might allow us to do less steps (but who cares tho).

    is_lig = torch.where(batch.frag_idx_map != -1)[0]
    ref_lig_pos = batch.ref_pos
    pred_lig_pos = batch.pos_t * HPARAMS.general.dimensional_scale + batch.pocket_com.repeat_interleave(
        torch.bincount(batch.batch), dim=0
    )
    dev = (ref_lig_pos[is_lig] - pred_lig_pos[is_lig]).norm(dim=-1)
    # print(f"Average Deviation {dev.mean()}")
    return batch, all_pos, all_edges, all_losses


# Used by scripts/sample.py.
def sampler(
    denoiser: SigmaFlowGenerator,
    batch: Batch,
    t_min: float = HPARAMS.general.epsilon_t,
    rho: float = 3.0,
    t_max: float = 1.0,
    num_steps: int = 18,
    solver: Literal["euler", "heun"] = "euler",
    discretization: Literal["power", "edm"] = "power",
    use_true_vector_field: bool = False,
    verbose: bool = False,
    debug_first_step: bool = False,
) -> tuple[Batch, list[np.ndarray]]:
    """Integrate the flow-matching ODE forward (t_min -> t_max, noise -> data) with
    SigmaFlowGenerator's predicted vector field, via SE3_FlowMatcher.euler_step.
    Args:
        denoiser: SigmaFlowGenerator instance whose predicted vector field drives the ODE.
        batch: Batch of data containing the initial states and other necessary information.
        num_steps: Number of steps in the forward ODE integration.
        solver: Solver type for the ODE. Choices are:
            - euler (1x inference)
            - heun (2x inference)
            defaults to "euler".
        rho: Exponent for the time step discretization (only used by discretization="edm").
        t_min: Minimum time step for the ODE integration. Defaults to oracle.py's epsilon_t,
            the same floor used when sampling t during training - the network was never trained
            on t below this value.
        t_max: Maximum time step for the ODE integration.
        use_true_vector_field: If True, uses the true vector field for the ODE step instead of the denoiser's predicted vector field.
        debug_first_step: If True, prints per-fragment rotation angle (R_0 vs R_1) and isnan/isinf
            checks on the predicted and true vector fields at i==0, to diagnose numerical blowups
            at the very start of sampling. No effect on the returned trajectory.
    Returns:
        A tuple containing the updated batch and a list of positions at each step of the ODE integration.
    """

    # TODO For each seed clone the batch and repeat the data items!
    batch = denoiser._prepare_batch(batch)

    # Note this could be a random conformer in 3D space -> Typically we do this in inference.
    pos_0, trans_1, R_1, num_fragments = denoiser._get_initial_states(batch)
    sampled_init = denoiser.flow_matcher.sample_init(torch.sum(num_fragments), batch.x.device)
    trans_0, R_0 = sampled_init["trans_0"], sampled_init["R_0"]

    # Update roto-translations to ambient space (complex)
    pos_T = denoiser._apply_transformations(
        pos_0=pos_0,
        batch=batch,
        trans_1=trans_1,
        R_1=R_1,
        trans_t=trans_0,
        R_t=R_0,
    )

    step_indices = torch.arange(num_steps, device=batch.x.device)  # [N]
    if discretization == "power":
        timesteps = torch.linspace(t_min, t_max, num_steps, device=batch.x.device)
    elif discretization == "edm":
        timesteps = (
            t_min ** (1 / rho) + step_indices / (num_steps - 1) * (t_max ** (1 / rho) - t_min ** (1 / rho))
        ) ** rho
    else:
        raise ValueError(f"Unknown discretization {discretization}. Choose 'power' or 'edm'.")

    # Initialize the denoiser with the initial states
    batch = denoiser._update_batch(
        batch=batch,
        pos_0=pos_0,
        pos_t=pos_T,
    )
    pos_t = pos_T
    trans_t = trans_0
    R_t = R_0

    if verbose:
        print(
            f"Using {num_steps} steps for forward ODE integration with: \n \
            rho={rho} \n \
            solver={solver} \n \
            t_min={t_min} \n \
            "
        )

    @torch.no_grad
    def _predict_vector_field_step(
        batch: Batch,
        t: float,
        R_t: torch.Tensor,
        trans_t: torch.Tensor,
        R_0: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        t_batch = t.repeat_interleave(sum(num_fragments))  # [B x F]
        # Create discretization & Get time from discretization
        lig_pseudoforces, forces_idxs = denoiser._compute_forces(
            batch=batch,
            t=torch.tensor([t] * batch.num_graphs, device=pos_t.device),  # [B]
        )  # [B x F x A, 3], [B x F x A]
        # Linear mechanics: mass, inertia, force & torque
        force_per_fragment, torque_per_fragment, frag_mass, frag_inertia_t = denoiser._compute_fragment_dynamics(
            batch=batch,
            R_t=R_t,  # [B x F, 3, 3]
            trans_t=trans_t,  # [B x F, 3]
            # NOTE R_1 only exists as safety during training for I_t calc. not required strictly (IRL)
            R_1=R_0,  # [B x F, 3, 3]
            lig_forces=lig_pseudoforces,
            forces_idxs=forces_idxs,
        )  # [B x F, 3], [B x F, 3], [B,F], [B x F, 3, 3]

        # Compute total scaled forces/torques and predict updates (Newton-Maruyama)
        fragment_updates = denoiser._predict_fragment_updates(
            force_per_fragment=force_per_fragment,
            torque_per_fragment=torque_per_fragment,
            frag_mass=frag_mass,
            frag_inertia_t=frag_inertia_t,
            # t_batch=t_batch,
        )  # [B x F, 3], [B x F, 3, 3]

        # Compute [R3, so3] vector_field
        pred_vector_field = denoiser._compute_vector_field({"R_t": R_t, "trans_t": trans_t}, fragment_updates, t_batch)
        return pred_vector_field

    all_losses = []

    # Get ligand indices
    is_lig = torch.where(batch.frag_idx_map != -1)[0]
    # Initialize with Stationary sample

    all_pos = [pos_t[is_lig]]
    # Iterate forward across timesteps (t_min -> t_max, noise -> data)
    for i, t in tqdm(enumerate(timesteps[:-1])):
        dt = timesteps[i+1] - timesteps[i]
        t = torch.tensor(t, device=batch.x.device)  # [1]
        t_batch = t.repeat_interleave(sum(num_fragments))  # [B x F]

        true_vector_field = denoiser._compute_true_vector_field(
            trans_1=trans_1,
            R_1=R_1,
            Tt=trans_t,
            Rt=R_t,
            t_batch=t_batch,
        )

        grad_trans_t = true_vector_field["u_t_trans"]
        grad_R_t = true_vector_field["u_t_R"]
        # Use True or Predicted vector_field
        if not use_true_vector_field:
            # Deepcopy because this step modifies the batch in-place (removes masked edges)
            pred_vector_field = _predict_vector_field_step(deepcopy(batch), t, R_t=R_t, trans_t=trans_t, R_0=R_0)
            grad_T_p = pred_vector_field["pred_u_t_trans"]
            grad_R_p = pred_vector_field["pred_u_t_R"]
        else:
            grad_T_p = grad_trans_t
            grad_R_p = grad_R_t

        if debug_first_step and i == 0:
            tr = R_t.diagonal(dim1=-2, dim2=-1).sum(-1)
            angle_deg = torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1.0, 1.0)))
            print(f"[debug_first_step] t={t.item():.6f}")
            for f in range(R_t.shape[0]):
                print(f"  fragment {f}: angle(R_t, R_1) = {angle_deg[f].item():.4f} deg")
            for name, x in [
                ("grad_trans_t (true)", grad_trans_t),
                ("grad_R_t (true)", grad_R_t),
                ("grad_T_p (pred, drives trajectory)", grad_T_p),
                ("grad_R_p (pred, drives trajectory)", grad_R_p),
            ]:
                nan_mask = torch.isnan(x).reshape(x.shape[0], -1).any(dim=-1)
                inf_mask = torch.isinf(x).reshape(x.shape[0], -1).any(dim=-1)
                bad = torch.nonzero(nan_mask | inf_mask).flatten().tolist()
                print(f"  {name}: nan_count={int(nan_mask.sum())} inf_count={int(inf_mask.sum())} bad_fragments={bad}")

        # Log the losses
        losses: dict[str, torch.Tensor] = denoiser.compute_losses(
            {
                "pred_u_t_trans": grad_T_p,  # [B x F, 3]
                "pred_u_t_R": grad_R_p,  # [B x F, 3, 3]
                "u_t_trans": grad_trans_t,  # [B x F, 3]
                "u_t_R": grad_R_t,  # [B x F, 3, 3]
                "t_batch": t_batch,  # [B x F]
                "R_t": R_t,  # [B x F, 3, 3]
                "R_1": R_1,  # [B x F, 3, 3]
            }
        )
        # ODE step (use discretization & solver)
        step_result = denoiser.flow_matcher.euler_step(
            trans_t=trans_t,
            R_t=R_t,
            v_t_trans=grad_T_p,
            v_t_R=grad_R_p,
            dt=dt)
        trans_next, R_next = step_result["trans_new"], step_result["R_new"]

        # Update positions according to transformations from this ODE step.
        pos_t = denoiser._apply_transformations(
            batch=batch,
            # Refererence
            pos_0=pos_t,
            trans_1=trans_t,
            R_1=R_t,
            # Transformation
            trans_t=trans_next,
            R_t=R_next,
        )
        # Advance the fragment roto-translations for the next step.
        trans_t = trans_next
        R_t = R_next
        # Update batch with new positions (pos_t) and remove prev local interactions
        batch = denoiser._update_batch(
            batch=batch,
            pos_0=pos_0,
            pos_t=pos_t,
        )

        all_pos.append(pos_t[is_lig])
        all_losses.append(losses)

        # TODO Heun's method
        #  Will require us to look at derivatives for T and R and average the Func Evaluations at t, t-t'
        #  Remember last step must be Euler so NFE = 2 * (N - 1) + 1
        #  Truncation error: O(N) -> O(N**3) which might allow us to do less steps (but who cares tho).

    ref_lig_pos = batch.ref_pos
    pred_lig_pos = batch.pos_t * HPARAMS.general.dimensional_scale + batch.pocket_com.repeat_interleave(
        torch.bincount(batch.batch), dim=0
    )
    dev = (ref_lig_pos[is_lig] - pred_lig_pos[is_lig]).norm(dim=-1)
    # print(f"Average Deviation {dev.mean()}")
    return batch, torch.stack(all_pos, dim = 0), all_losses
