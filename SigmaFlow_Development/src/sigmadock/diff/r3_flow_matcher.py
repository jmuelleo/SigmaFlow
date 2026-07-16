import torch

# Define class R3_FlowMatcher
class R3_FlowMatcher:
    def __init__(self, sigma_min: float):
        """
        sigma_min: reserved for future noised OT path, currently unused - the linear path below is deterministc
        """
        self.sigma_min = sigma_min # Keep for later implementation with small randomness to prevent degenerate distribution - for now not used

    # Define sampling initial distribution as method:
    def sample_init(self, n: int, device: str) -> torch.Tensor: 
        return torch.randn(n, 3, device = device) # tensor [n, 3] on device
    
    # Define conditional probability distribuiton implicitly via interpolation
    def conditional_probability_path(self, x_1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_1: [n,3] Data
        t: [n] time (in [0,1]) one value per fragment
        Returns:
        x_t: [n, 3] interpolated position at time t
        u_t: [n, 3] conditional vector field (if we use linear interpolation constant over t)
        """
        n = x_1.shape[0]
        device = x_1.device
        x_0 = self.sample_init(n, device)

        x_t = (1 - t[:, None])*x_0 + t[:, None]*x_1 # Linear Interpolation
        u_t = x_1 - x_0 # Conditional Vector Field (dX_t/dt = u_t(x|z))

        return x_t, u_t
    
    # Euler Integration Scheme <- Due to the Linear Interpolation path this is exact for u_t
    def euler_step(self, x_t: torch.Tensor, v_t: torch.Tensor, dt: float) -> torch.Tensor:
        x_next = x_t + v_t * dt
        return x_next
    
    
    



