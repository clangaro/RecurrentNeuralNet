import numpy as np

from timedpred.dynamics import Parameters, Weights
from timedpred.simulate import SimConfig, make_trial_inputs_minimal, simulate_trial


def main():
    # Smallest reasonable sizes for a first run
    n_e = 20
    n_i = 10

    # Parameters (seconds)
    P = Parameters(tau_e=0.020, tau_i=0.010, tau_s=0.200)

    # Random weights (very small magnitudes to avoid blow-up)
    rng = np.random.default_rng(0)
    # W.w_ee structure:
    rows = postsynaptic, columns = presynaptic
    
    Indices:
    0 : n_a   # → E_A neurons
    n_a : n_e # → E_B neurons
    
    Block meanings in W_EE:
    W_EE[:n_a, :n_a]  # → E_A → E_A
    W_EE[:n_a, n_a:]  # → E_B → E_A
    W_EE[n_a:, :n_a]  # → E_A → E_B  (this will get a weak bias)
    W_EE[n_a:, n_a:]  # → E_B → E_B

    W = Weights(
        w_ee=np.abs(rng.normal(0.0, 0.05, size=(n_e, n_e))),
        w_ei=np.abs(rng.normal(0.0, 0.05, size=(n_e, n_i))),
        w_ie=np.abs(rng.normal(0.0, 0.05, size=(n_i, n_e))),
        w_ii=np.abs(rng.normal(0.0, 0.05, size=(n_i, n_i))),
    )

    #Weak bias from E_A to E_B (anatomical prior, not learned)
    epsilon = 0.01
    W.w_ee[n_a:, :n_a] += epsilon

    # Biases (baseline excitability)
    b_e = np.zeros(n_e)
    b_i = np.zeros(n_i)

    # Simulation config
    C = SimConfig(dt=0.001, max_time=0.6, record_every=1)

    # Inputs: CS then US
    inputs = make_trial_inputs_minimal(n_e=n_e, n_i=n_i, dt=C.dt, max_time=C.max_time)

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
