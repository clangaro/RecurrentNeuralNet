1. Project Goal
The goal of this project is to construct a minimal, biologically motivated recurrent neural network in which:
Timed predictions emerge intrinsically from network dynamics
Timed prediction errors (including omission responses exceeding presence responses) arise without explicit clock variables or dedicated error units
Learning depends on local, interpretable plasticity rules
Network behaviour maps directly onto experimental findings reported by Liu & Buonomano (2025)
The emphasis is on dynamical systems and circuit mechanisms, not performance-optimised machine learning.
2. Minimal Network Architecture
2.1 Populations
The network consists of:
Two excitatory populations, $E_A$ and $E_B$, corresponding to ensembles preferentially driven by temporally ordered inputs (e.g. CS-like and US-like stimuli)
A single inhibitory population $I$, providing global stabilisation
This represents the minimal circuit motif capable of expressing learned sequential activation, asymmetric coupling, and inhibition-stabilised dynamics.
2.2 Rate Dynamics
Population firing rates obey standard continuous-time rate equations:
τ
E
 
r
˙
E
=
−
r
E
+
ϕ
 ⁣
(
W
E
E
x
E
−
W
E
I
r
I
+
u
(
t
)
)
(1) 
τ 
E
​	
  
r
˙
  
E
​	
 =−r 
E
​	
 +ϕ(W 
EE
​	
 x 
E
​	
 −W 
EI
​	
 r 
I
​	
 +u(t))(1)
τ
I
 
r
˙
I
=
−
r
I
+
ϕ
I
 ⁣
(
W
I
E
r
E
−
W
I
I
r
I
)
(2) 
τ 
I
​	
  
r
˙
  
I
​	
 =−r 
I
​	
 +ϕ 
I
​	
 (W 
IE
​	
 r 
E
​	
 −W 
II
​	
 r 
I
​	
 )(2)
where
$\mathbf{r}_E = \begin{bmatrix} r_A \ r_B \end{bmatrix}$
$\phi(\cdot)$ is a saturating nonlinearity (e.g. softplus or threshold-linear)
2.3 Stability Regime
Initial synaptic weights are chosen such that:
Excitatory coupling is weak and approximately symmetric
Inhibitory feedback is sufficiently strong to ensure stability
The network operates in an inhibition-stabilised or damped-transient regime
3. Emergent Timing Mechanism
Timing is not encoded explicitly. Instead, intrinsic timescales arise from synaptic dynamics.
Two biologically motivated mechanisms are considered:
Short-term synaptic plasticity (facilitation or depression) on $E \rightarrow E$ synapses
Slow synaptic currents (e.g. NMDA-like filtering):
τ
s
 
s
˙
E
=
−
s
E
+
r
E
(3) 
τ 
s
​	
  
s
˙
  
E
​	
 =−s 
E
​	
 +r 
E
​	
 (3)
Recurrent input is mediated via $\mathbf{s}_E$.
Both mechanisms generate internal temporal structure without delay lines or clocks.
4. Learning Rules
4.1 Asymmetric Excitatory Plasticity
Excitatory synapses follow a temporally asymmetric Hebbian rule implemented via eligibility traces:
τ
pre
 
e
˙
j
=
−
e
j
+
r
j
(
t
)
(4) 
τ 
pre
​	
  
e
˙
  
j
​	
 =−e 
j
​	
 +r 
j
​	
 (t)(4)
Δ
w
i
j
∝
r
i
(
t
)
e
j
(
t
)
−
λ
 
r
j
(
t
)
e
i
(
t
)
(5) 
Δw 
ij
​	
 ∝r 
i
​	
 (t)e 
j
​	
 (t)−λr 
j
​	
 (t)e 
i
​	
 (t)(5)
This rule strengthens synapses when presynaptic activity reliably precedes postsynaptic activity, promoting directional coupling from $E_A$ to $E_B$ during training.
Weights obey Dale’s law and are constrained within fixed bounds.
4.2 Inhibitory Homeostatic Plasticity
Inhibitory synapses onto excitatory neurons adapt to stabilise firing rates:
Δ
w
k
i
E
I
∝
r
I
(
k
)
(
t
)
 
(
r
E
(
i
)
(
t
)
−
r
∗
)
Δw 
ki
EI
​	
 ∝r 
I
(k)
​	
 (t)(r 
E
(i)
​	
 (t)−r 
∗
 )
where $r^\ast$ is a target excitatory firing rate.
This prevents runaway excitation during learning and maintains balanced dynamics.
5. Training Paradigm
Training consists of repeated trials in which:
A brief input (CS) drives $E_A$
After a fixed interval $\Delta$, a second input (US) drives $E_B$
Through plasticity, asymmetric coupling from $E_A$ to $E_B$ emerges, allowing CS input alone to evoke a delayed response in $E_B$.
6. Emergent Timed Prediction Error
Prediction errors arise from circuit interactions rather than explicit error computation.
A key mechanism is feedforward inhibition recruited by the US:
When the US occurs, it drives both $E_B$ and $I$, suppressing the internally generated predicted response
When the US is omitted, this inhibitory suppression is absent, revealing a larger delayed response
Thus, omission responses exceed presence responses due to disinhibition of predicted activity, producing a timed prediction-error signal at both population and single-neuron levels.
7. Expected Outcomes
After learning:
CS alone evokes a delayed, temporally precise response in $E_B$ (timed prediction)
CS+US trials show reduced late responses
CS-only (omission) trials show enhanced late responses
These behaviours correspond directly to key experimental observations in Liu & Buonomano (2025).
8. Project Scope
This model prioritises:
Interpretability over optimisation
Minimal circuit complexity
Direct correspondence between model components and biological mechanisms
Deliverables include a clean simulation, mechanistic analyses, and a concise explanation suitable for PhD applications or direct communication with the Buonomano lab.