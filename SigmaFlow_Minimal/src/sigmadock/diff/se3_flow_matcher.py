import torch

from sigmadock.diff.r3_flow_matcher import R3_FlowMatcher
from sigmadock.diff.so3_flow_matcher import SO3_FlowMatcher

class SE3_FlowMatcher:

    def __init__(self, sigma_min: float):
        self._r3_flow_matcher = R3_FlowMatcher(sigma_min = sigma_min)
        self._so3_flow_matcher = SO3_FlowMatcher()

    def sample_init(self, n: int, device: str) -> dict[str, torch.Tensor]:
        trans_0 = self._r3_flow_matcher.sample_init(n, device)
        R_0 = self._so3_flow_matcher.sample_init(n, device)

        result = {"trans_0": trans_0,
                  "R_0": R_0}
        
        return result
    
    def conditional_probability_path(self, trans_1: torch.Tensor, R_1: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        trans_t, u_t_trans = self._r3_flow_matcher.conditional_probability_path(trans_1, t)
        R_t, u_t_R = self._so3_flow_matcher.conditional_probability_path(R_1, t)

        result = {"trans_t": trans_t,
                  "R_t": R_t,
                  "u_t_trans": u_t_trans,
                  "u_t_R": u_t_R}
        
        return result
    
    def euler_step(self, trans_t: torch.Tensor, R_t: torch.Tensor, v_t_trans: torch.Tensor, v_t_R: torch.Tensor, dt: float) -> dict[str, torch.Tensor]:
        trans_new = self._r3_flow_matcher.euler_step(trans_t, v_t_trans, dt)
        R_new = self._so3_flow_matcher.euler_step(R_t, v_t_R, dt)

        result = {"trans_new": trans_new,
                  "R_new": R_new}
        
        return result
    
    def calc_vector_field(self, trans_t: torch.Tensor, R_t: torch.Tensor, trans_1: torch.Tensor, R_1: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        brings calc_trans_vector_field and calc_rot_vector_field together to a SE(3) vector_field
        """
        u_t_trans = self._r3_flow_matcher.calc_trans_vector_field(trans_t,trans_1, t)
        u_t_R = self._so3_flow_matcher.calc_rot_vector_field(R_t,R_1, t)

        result = {"u_t_trans": u_t_trans,
                  "u_t_R": u_t_R}
        return result
    
        



    


    



    


