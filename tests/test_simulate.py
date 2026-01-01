import numpy as np

from timedpred.dynamics import Weights, Parameters  # keep your class names
from timedpred.simulate import SimConfig, make_trial_inputs_minimal, simulate_trial


def test_simulate_trial_shapes_and_finiteness():
    rng = np.random.default_rng(0)

    n_e = 20
    n_i = 10
    fraction_a = 0.5
    n_a = int(np.round(fraction_a * n_e))

    # Parameters (seconds)
    P = Parameters(tau_e=0.020, tau_i=0.010, tau_s=0.200, tau_elig=0.2)

    # Weights: small random positive magnitudes
    W = Weights(
        w_ee=np.abs(rng.normal(0.0, 0.05, size=(n_e, n_e))),
        w_ei=np.abs(rng.normal(0.0, 0.05, size=(n_e, n_i))),
        w_ie=np.abs(rng.normal(0.0, 0.05, size=(n_i, n_e))),
        w_ii=np.abs(rng.normal(0.0, 0.05, size=(n_i, n_i))),
    )

    # Weak E_A -> E_B bias
    epsilon = 0.01
    W.w_ee[n_a:, :n_a] += epsilon

    # Biases
    b_e = np.zeros(n_e)
    b_i = np.zeros(n_i)

    # Simulation config
    C = SimConfig(dt=0.001, max_time=0.6, record_every=1)

    # Inputs
    inputs = make_trial_inputs_minimal(
        n_e=n_e, n_i=n_i, dt=C.dt, max_time=C.max_time, fraction_a=fraction_a
    )

    # Initial state
    r_e0 = np.zeros(n_e)
    r_i0 = np.zeros(n_i)
    s_e0 = np.zeros(n_e)

    traj = simulate_trial(r_e0, r_i0, s_e0, inputs, b_e, b_i, W, P, C)

    # Shape checks
    assert traj.t.shape == (600,)
    assert traj.r_e.shape == (600, n_e)
    assert traj.r_i.shape == (600, n_i)
    assert traj.s_e.shape == (600, n_e)

    # Finite checks
    assert np.all(np.isfinite(traj.r_e))
    assert np.all(np.isfinite(traj.r_i))
    assert np.all(np.isfinite(traj.s_e))
