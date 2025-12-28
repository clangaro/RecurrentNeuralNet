import numpy as np

from timedpred.dynamics import Params, Weights
from timedpred.simulate import SimConfig, make_trial_inputs_minimal, simulate_trial


def main():
    # Smallest reasonable sizes for a first run
    n_e = 20
    n_i = 10

    # Parameters (seconds)
    P = Params(tau_e=0.020, tau_i=0.010, tau_s=0.200)

    # Random weights (very small magnitudes to avoid blow-up)
    rng = np.random.default_rng(0)
    W = Weights(
        w_ee=np.abs(rng.normal(0.0, 0.05, size=(n_e, n_e))),
        w_ei=np.abs(rng.normal(0.0, 0.05, size=(n_e, n_i))),
        w_ie=np.abs(rng.normal(0.0, 0.05, size=(n_i, n_e))),
        w_ii=np.abs(rng.normal(0.0, 0.05, size=(n_i, n_i))),
    )

    # Biases (baseline excitability)
    b_e = np.zeros(n_e)
    b_i = np.zeros(n_i)

    # Simulation config
    C = SimConfig(dt=0.001, t_max=0.6, record_every=1)

    # Inputs: CS then US
    inputs = make_trial_inputs_minimal(n_e=n_e, n_i=n_i, dt=C.dt, t_max=C.t_max)

    # Initial state
    r_e0 = np.zeros(n_e)
    r_i0 = np.zeros(n_i)
    s_e0 = np.zeros(n_e)

    traj = simulate_trial(r_e0, r_i0, s_e0, inputs, b_e, b_i, W, P, C)

    print("Simulation complete.")
    print("t shape:", traj.t.shape)
    print("r_e shape:", traj.r_e.shape)
    print("r_i shape:", traj.r_i.shape)
    print("s_e shape:", traj.s_e.shape)
    print("max r_e:", np.max(traj.r_e))
    print("max r_i:", np.max(traj.r_i))


if __name__ == "__main__":
    main()
