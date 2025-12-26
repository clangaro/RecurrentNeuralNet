# Timed Prediction RNN (minimal, dynamical-systems first)

1 Project Goal
The goal of this project is to construct a minimal, biologically motivated recurrent neural network
in which:
• Timed predictions emerge intrinsically from network dynamics,
• Timed prediction errors (including omission responses exceeding presence responses) arise
without explicit clock variables or dedicated error units,
• Learning depends on local, interpretable plasticity rules,
• Network behaviour can be directly mapped onto experimental findings reported by Liu &
Buonomano (2025).
The emphasis is on dynamical systems and circuit mechanisms rather than performance-optimised
machine learning.
2 Minimal Network Architecture
2.1 Populations
The network consists of:
• Two excitatory populations, EA and EB, corresponding to ensembles preferentially driven by
temporally ordered inputs (e.g. CS-like and US-like stimuli),
• A single inhibitory population I, providing global stabilisation.
This represents the minimal circuit motif capable of expressing learned sequential activation,
asymmetric coupling, and inhibition-stabilised dynamics.

2.2 Rate Dynamics
Population firing rates obey standard continuous-time dynamics:
τE ˙rE = −rE + ϕ (WEExE −WEIrI + u(t)) , (1)
τI ˙rI = −rI + ϕI (WIErE −WIIrI ) , (2)
where rE = [rA; rB] and ϕ(·) is a saturating nonlinearity (e.g. softplus or threshold-linear).
2.3 Stability Regime
Initial synaptic weights are chosen such that:
• Excitatory coupling is weak and largely symmetric,
• Inhibitory feedback is sufficiently strong to ensure stability,
• The network operates in an inhibition-stabilised or damped-transient regime.
3 Emergent Timing Mechanism
Timing is not encoded explicitly. Instead, intrinsic timescales arise from synaptic dynamics.
Two biologically motivated options are considered:
• Short-term synaptic plasticity (facilitation/depression) on E → E synapses,
• Slow synaptic currents (e.g. NMDA-like filtering):
τs ˙sE = −sE + rE, (3)
with recurrent input mediated via sE.
Both mechanisms provide internal temporal structure without delay lines or clocks.
4 Learning Rules
4.1 Asymmetric Excitatory Plasticity
Excitatory synapses follow a temporally asymmetric Hebbian rule implemented via eligibility traces:
τpree˙j = −ej + rj(t), (4)
Δwij ∝ ri(t)ej(t) − λrj(t)ei(t). (5)
This rule strengthens synapses when presynaptic activity reliably precedes postsynaptic activity,
promoting directional coupling from EA to EB during training.
Weights obey Dale’s law and are constrained within fixed bounds.

4.2 Inhibitory Homeostatic Plasticity
Inhibitory synapses onto excitatory neurons adapt to stabilise firing rates:
ΔwEI ki ∝ rIk(t) (rEi (t) − r∗), 
where r∗ is a target excitatory rate.
This prevents runaway excitation during learning and maintains balanced dynamics.
5 Training Paradigm
Training consists of repeated trials in which:
• A brief input (CS) drives EA,
• After a fixed interval Δ, a second input (US) drives EB.
Through plasticity, asymmetric coupling from EA to EB emerges, allowing CS input alone to
evoke a delayed response in EB.
6 Emergent Timed Prediction Error
Prediction errors arise from circuit interactions rather than explicit error computation.
A key mechanism is feedforward inhibition recruited by the US:
• When the US occurs, it drives both EB and I, suppressing the internally generated predicted
response,
• When the US is omitted, this inhibitory suppression is absent, revealing a larger delayed
response.
Thus, omission responses exceed presence responses due to disinhibition of predicted activity,
producing a timed prediction error signal at the population and single-neuron level.
7 Expected Outcomes
After learning:
• CS alone evokes a delayed, temporally precise response in EB (timed prediction),
• CS+US trials show reduced late responses,
• CS-only (omission) trials show enhanced late responses.
These behaviours correspond directly to key experimental observations in Liu & Buonomano
(2025).

8 Project Scope
This model prioritises:
• Interpretability over optimisation,
• Minimal circuit complexity,
• Direct correspondence between model components and biological mechanisms.
The final deliverables include a clean simulation, mechanistic analyses, and a concise written
explanation suitable for PhD applications or direct communication with the Buonomano lab.


## Structure
- `src/timedpred/` core model code
- `scripts/` runnable entrypoints
- `tests/` pytest
- `notebooks/` exploratory (kept minimal)
- `docs/` short writeups for a PhD application

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
