from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .dynamics import Parameters, Weights, rhs


@dataclass(frozen=True)
class SimConfig:
    dt: float          # seconds
    max_time: float    # seconds
    record_every: int  # record every n integration steps


@dataclass(frozen=True)
class TrialInputs:
    u_e: np.ndarray  # shape (T, N_E)
    u_i: np.ndarray  # shape (T, N_I)


@dataclass(frozen=True)
class Trajectory:
    t: np.ndarray
    r_e: np.ndarray
    r_i: np.ndarray
    s_e: np.ndarray
    e_e: np.ndarray


def simulate_trial(
    r_e0: np.ndarray,
    r_i0: np.ndarray,
    s_e0: np.ndarray,
    e_e0: np.ndarray,
    inputs: TrialInputs,
    b_e: np.ndarray,
    b_i: np.ndarray,
    W: Weights,
    P: Parameters,
    C: SimConfig,
) -> Trajectory:
    """
    Simulate a single trial of the network dynamics using forward Euler integration.

    Returns recorded time series for r_e, r_i, s_e.
    """

    # Basic shape checks (fail fast)
    T, n_e = inputs.u_e.shape
    T2, n_i = inputs.u_i.shape

    assert T == T2, "u_e and u_i must have same number of time steps"
    assert r_e0.shape == (n_e,)
    assert r_i0.shape == (n_i,)
    assert s_e0.shape == (n_e,)
    assert b_e.shape == (n_e,)
    assert b_i.shape == (n_i,)

    n_steps = T

    # number of recorded samples
    n_recorded = (n_steps + C.record_every - 1) // C.record_every

    t_recorded = np.empty(n_recorded, dtype=float)
    r_e_recorded = np.empty((n_recorded, n_e), dtype=float)
    r_i_recorded = np.empty((n_recorded, n_i), dtype=float)
    s_e_recorded = np.empty((n_recorded, n_e), dtype=float)
    e_e_recorded = np.empty((n_recorded, n_e), dtype=float)

    # State
    r_e = r_e0.astype(float).copy()
    r_i = r_i0.astype(float).copy()
    s_e = s_e0.astype(float).copy()
    e_e = e_e0.astype(float).copy()

    rec_idx = 0

    for k in range(n_steps):
        # current external inputs (already time discretized)
        u_e = inputs.u_e[k]
        u_i = inputs.u_i[k]

        # RHS derivatives
        # IMPORTANT: use keyword args so ordering cannot break silently
        dr_e_dt, dr_i_dt, ds_e_dt, de_e_dt = rhs(
            r_e=r_e,
            r_i=r_i,
            s_e=s_e,
            e_e=e_e,
            u_e=u_e,
            u_i=u_i,
            b_e=b_e,
            b_i=b_i,
            W=W,
            P=P,
        )

        # Euler step
        r_e = r_e + C.dt * dr_e_dt
        r_i = r_i + C.dt * dr_i_dt
        s_e = s_e + C.dt * ds_e_dt
        e_e = e_e + C.dt * de_e_dt

        # Compute weight increment 
        dW = P.eta_ee * np.outer(r_e, e_e)
        W.w_ee += dW

        # Enforce max weight constraint
        W.w_ee = np.clip(W.w_ee, 0.0, P.w_ee_max)

        # check for non-finite values
        if not (np.all(np.isfinite(r_e)) and np.all(np.isfinite(r_i)) and np.all(np.isfinite(s_e))):
            raise RuntimeError(f"Non-finite state detected at step {k}")

        # Record
        if k % C.record_every == 0:
            t_recorded[rec_idx] = k * C.dt
            r_e_recorded[rec_idx] = r_e
            r_i_recorded[rec_idx] = r_i
            s_e_recorded[rec_idx] = s_e
            e_e_recorded[rec_idx] = e_e
            rec_idx += 1

    return Trajectory(
        t=t_recorded,
        r_e=r_e_recorded,
        r_i=r_i_recorded,
        s_e=s_e_recorded,
        e_e=e_e_recorded
    )


def make_pulse_train(
    T: int,
    dt: float,
    onset: float,
    duration: float,
    amplitude: float,
) -> np.ndarray:
    """
    Create a square pulse input vector in discrete time.

    Returns array of shape (T,) with values 0 or amplitude.
    """
    t = np.arange(T) * dt
    mask = (t >= onset) & (t < onset + duration)
    out = np.zeros(T, dtype=float)
    out[mask] = amplitude
    return out


def make_trial_inputs_minimal(
    n_e: int,
    n_i: int,
    dt: float,
    max_time: float,
    cs_onset: float = 0.050,
    cs_duration: float = 0.020,
    cs_amp: float = 1.0,
    us_onset: float = 0.250,
    us_duration: float = 0.020,
    us_amp_e: float = 1.0,
    us_amp_i: float = 1.0,
    fraction_a: float = 0.5,
) -> TrialInputs:
    """
    Minimal trial inputs: CS pulse drives E_A, US pulse drives E_B, and US also drives I.

    Convention: excitatory units are ordered [E_A..., E_B...].
    """
    T = int(np.round(max_time / dt))
    n_a = int(np.round(fraction_a * n_e))
    n_b = n_e - n_a  # not used directly, but kept for clarity

    u_e = np.zeros((T, n_e), dtype=float)
    u_i = np.zeros((T, n_i), dtype=float)

    cs = make_pulse_train(T, dt, cs_onset, cs_duration, cs_amp)
    us_e = make_pulse_train(T, dt, us_onset, us_duration, us_amp_e)
    us_i = make_pulse_train(T, dt, us_onset, us_duration, us_amp_i)

    # CS drives E_A
    u_e[:, :n_a] += cs[:, None]

    # US drives E_B
    u_e[:, n_a:] += us_e[:, None]

    # US also drives inhibition (feedforward inhibition placeholder)
    u_i[:, :] += us_i[:, None]

    return TrialInputs(u_e=u_e, u_i=u_i)
