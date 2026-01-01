from __future__ import annotations

from dataclasses import dataclass
import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

@dataclass(frozen=True)
class Parameters:
    tau_e: float
    tau_i: float
    tau_s: float
    tau_elig: float

@dataclass(frozen=True)
class Weights:
    w_ee: np.ndarray
    w_ei: np.ndarray
    w_ie: np.ndarray
    w_ii: np.ndarray

def rhs(
    r_e: np.ndarray,
    r_i: np.ndarray,
    s_e: np.ndarray,
    e_e: np.ndarray,
    u_e: np.ndarray,
    u_i: np.ndarray,
    b_e: np.ndarray,
    b_i: np.ndarray,
    W: Weights,
    P: Parameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Continuous-time dynamics (ODE right-hand side) for the excitatory and inhibitory populations.
    
    State:
    r_e : Excitatory firing rates (N_E,)
    r_i : Inhibitory firing rates (N_I,)
    s_e :  filtered excitatory trace (N_E,)

    Inputs: 
    u_e, u_i: external inputs at a current time
    b_e, b_i: constant biases 

    Returns:
    dr_e_dt : time derivative of excitatory rates
    dr_i_dt : time derivative of inhibitory rates
    ds_e_dt : time derivative of filtered excitatory trace
    """

    # Synaptic trace dynamics 
    ds_e_dt = (-s_e + r_e) / P.tau_s

    #Currents (before nonlinearity)
    x_e = (W.w_ee @ s_e) - (W.w_ei @ r_i) + u_e + b_e
    x_i = (W.w_ie @ r_e) - (W.w_ii @ r_i) + u_i + b_i

    # Firing rate dynamics
    dr_e_dt = (-r_e + relu(x_e)) / P.tau_e
    dr_i_dt = (-r_i + relu(x_i)) / P.tau_i
    ds_e_dt = (-e_e + r_e) / P.tau_elig

    return dr_e_dt, dr_i_dt, ds_e_dt