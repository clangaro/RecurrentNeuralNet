# Project Goal

The goal of this project is to construct a minimal, biologically motivated recurrent neural network in which:

- Timed predictions emerge intrinsically from network dynamics  
- Timed prediction errors (including omission responses exceeding presence responses) arise without explicit clock variables or dedicated error units  
- Learning depends on local, interpretable plasticity rules  
- Network behaviour maps directly onto experimental findings reported by **Liu & Buonomano (2025)**  

The emphasis is on **dynamical systems and circuit mechanisms**, not performance-optimised machine learning.

---

# Minimal Network Architecture

## Populations

The network consists of:

- Two excitatory populations, **EA** and **EB**, corresponding to ensembles preferentially driven by temporally ordered inputs (e.g. CS-like and US-like stimuli)  
- A single inhibitory population, **I**, providing global stabilisation  

This represents the minimal circuit motif capable of expressing learned sequential activation, asymmetric coupling, and inhibition-stabilised dynamics.

---

## Rate Dynamics

Population firing rates obey standard continuous-time rate equations:

```math
\tau_E \, \dot{\mathbf{r}}_E
= -\mathbf{r}_E
+ \phi\!\left(
W_{EE} \mathbf{x}_E
- W_{EI} r_I
+ \mathbf{u}(t)
\right)
```

```math
\tau_I \, \dot{r}_I
= -r_I
+ \phi_I\!\left(
W_{IE} \mathbf{r}_E
- W_{II} r_I
\right)
```

where:

- **rE = [rA, rB]ᵀ**  
- **φ(·)** is a saturating nonlinearity (e.g. softplus or threshold-linear)

---

## Stability Regime

Initial synaptic weights are chosen such that:

- Excitatory coupling is weak and approximately symmetric  
- Inhibitory feedback is sufficiently strong to ensure stability  
- The network operates in an **inhibition-stabilised or damped-transient regime**

---

# Emergent Timing Mechanism

Timing is not encoded explicitly. Instead, intrinsic timescales arise from synaptic dynamics.

Two biologically motivated mechanisms are considered:

- Short-term synaptic plasticity (facilitation or depression) on **E → E** synapses  
- Slow synaptic currents (e.g. NMDA-like filtering):

```math
\tau_s \, \dot{\mathbf{s}}_E
= -\mathbf{s}_E + \mathbf{r}_E
```

Recurrent input is mediated via **sE**.

Both mechanisms generate internal temporal structure **without delay lines or clocks**.

---

# Learning Rules

## Asymmetric Excitatory Plasticity

Excitatory synapses follow a temporally asymmetric Hebbian rule implemented via eligibility traces:

```math
\tau_{\text{pre}} \, \dot{e}_j
= -e_j + r_j(t)
```

```math
\Delta w_{ij}
\propto r_i(t)\, e_j(t)
- \lambda\, r_j(t)\, e_i(t)
```

This rule strengthens synapses when presynaptic activity reliably precedes postsynaptic activity, promoting directional coupling from **EA → EB** during training.

Weights obey **Dale’s law** and are constrained within fixed bounds.

---

## Inhibitory Homeostatic Plasticity

Inhibitory synapses onto excitatory neurons adapt to stabilise firing rates:

```math
\Delta w^{EI}_{k i}
\propto r_I^{(k)}(t)\,
\bigl( r_E^{(i)}(t) - r^\ast \bigr)
```

where **r\*** is a target excitatory firing rate.

This prevents runaway excitation during learning and maintains balanced dynamics.

---

# Training Paradigm

Training consists of repeated trials in which:

- A brief input (**CS**) drives **EA**  
- After a fixed interval **Δ**, a second input (**US**) drives **EB**

Through plasticity, asymmetric coupling from **EA → EB** emerges, allowing CS input alone to evoke a delayed response in **EB**.

---

# Emergent Timed Prediction Error

Prediction errors arise from circuit interactions rather than explicit error computation.

A key mechanism is feedforward inhibition recruited by the US:

- When the US occurs, it drives both **EB** and **I**, suppressing the internally generated predicted response  
- When the US is omitted, this inhibitory suppression is absent, revealing a larger delayed response  

Thus, **omission responses exceed presence responses** due to disinhibition of predicted activity, producing a timed prediction-error signal at both population and single-neuron levels.

---

# Expected Outcomes

After learning:

- CS alone evokes a delayed, temporally precise response in **EB** (timed prediction)  
- CS+US trials show reduced late responses  
- CS-only (omission) trials show enhanced late responses  

These behaviours correspond directly to key experimental observations in **Liu & Buonomano (2025)**.

---

# Project Scope

This model prioritises:

- Interpretability over optimisation  
- Minimal circuit complexity  
- Direct correspondence between model components and biological mechanisms  

Deliverables include a clean simulation, mechanistic analyses, and a concise explanation suitable for **PhD applications** or direct communication with the **Buonomano lab**.
