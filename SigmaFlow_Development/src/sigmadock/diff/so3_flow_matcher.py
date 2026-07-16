import torch
from sigmadock.diff import so3_utils


# We accept the convention used in so3_utils.expmap that we use righttrivialisation (work in the local tangent space instead of global tangent space)
class SO3_FlowMatcher:


    def __init__(self):
        pass # pass as we currently don't need any constructor arguments

    def sample_init(self, n: int, device: str) -> torch.Tensor:
        R_0 = so3_utils.sample_uniform(n).to(device, dtype = torch.float32) #  so3_utils.sample_uniform uses numpy (standard float64)
        return R_0
    
    def conditional_probability_path(self, R_1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = R_1.shape[0]
        device = R_1.device

        R_0 = self.sample_init(n, device) # Sample R_0 from the initial distribution (Uniform over SO(3))
        Delta = R_0.transpose(-1, -2) @ R_1 #.transpose(-1,-2) transposes a matrix (if the last 2 dimensions of the tensor are rows and columns of the matrix)
        log_Delta = so3_utils.log(Delta)
        u_t = log_Delta # Equivalent parametrisation to log(...)/(1-t) <- this is numerically more stable for t near 1
        R_t = R_0 @ so3_utils.exp(t[:, None, None] * log_Delta) #Geodesic Interpolation
        return R_t, u_t
    
    # Note: the original SigmaDock so3_utils.py had a numerical bug here:
    # Omega() clamped arccos to [-0.99, 0.99], giving wrong angles for true
    # angles between 172-180 degrees (~9% of uniformly sampled rotations).
    # Fine for diffusion (never needs an exact exp(log(R))==R round trip),
    # but breaks our exact geodesic interpolation. Fixed in this so3_utils.py
    # (clamp narrowed to [-1+1e-7, 1-1e-7]) - see SigmaDock/ for the original.

    def euler_step(self, R_t: torch.Tensor, v_t: torch.Tensor, dt: float) -> torch.Tensor:
        R_next = R_t @ so3_utils.exp(v_t*dt)
        return R_next
    


    


    







