# Reinforcement Learning Fundamentals

A professional repository dedicated to the clean implementation, mathematical profiling, and visualization of fundamental Reinforcement Learning (RL) algorithms. All code aligns with the theory presented in Sutton & Barto's *Reinforcement Learning: An Introduction*.

---

## 1. Multi-Armed Bandits (Stationary $k$-Armed Testbed)

### Theoretical Background
The Multi-Armed Bandit problem isolates the fundamental **Exploration-Exploitation trade-off**. Unlike full MDPs, it evaluates decision-making within a single stationary state. 

The agent balances picking the historical optimal action (exploitation) versus discovering superior un-sampled actions (exploration) via a biased-coin $\epsilon$-greedy policy:

$$A_t = \begin{cases} \arg\max_a Q_t(a) & \text{with probability } 1 - \epsilon \\ \text{a random action} & \text{with probability } \epsilon \end{cases}$$

To preserve memory efficiency, action-value estimates ($Q$) are optimized incrementally without maintaining a trailing database of historical rewards:

$$Q_{n+1} = Q_n + \frac{1}{n} [R_n - Q_n]$$

Where:
* $Q_n$: Current value estimate for the action.
* $R_n$: The immediate reward signal received.
* $n$: The sequential count of how many times this specific arm has been pulled.

---

### Performance & Convergence Analysis

Below is the experimental learning curve capturing the average reward over 1,000 algorithmic timesteps with an exploration factor of $\epsilon = 0.1$.

![Epsilon-Greedy Learning Curve](Multi-Armed-Bandits/bandit_epsilon_01.png)

**Key Insights:**
* **The Learning Curve:** Between steps 0 and 200, a rapid steep climb signifies active learning as the agent identifies higher-paying arms.
* **Steady-State Convergence:** As $t \to \infty$, the incremental step-size factor $\frac{1}{n}$ diminishes incoming variance. The system flattens onto an optimal reward ceiling, verifying that the estimated values successfully converged with the hidden true action values.

---

## 2. Upper Confidence Bound (UCB) Algorithm

### Theoretical Background

While $\epsilon$-greedy explores *randomly*, UCB explores *intelligently* using the principle of **Optimism in the Face of Uncertainty**. It selects the action that maximizes an upper confidence bound on the estimated value:

$$A_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]$$

Where:
* $Q_t(a)$: Current estimated value of arm $a$.
* $N_t(a)$: Number of times arm $a$ has been pulled.
* $t$: Total number of timesteps elapsed.
* $c$: Exploration parameter controlling the width of the confidence bound.

The exploration bonus $c \sqrt{\frac{\ln t}{N_t(a)}}$ naturally **grows** for under-explored arms and **shrinks** for frequently-pulled arms — directing exploration where uncertainty is highest, without any randomness.

**Initialization:** Each arm is pulled once during a warm-up phase ($t \leq k$) to avoid division by zero in $N_t(a)$.

---

### Performance & Convergence Analysis

#### Single UCB Run (c = 2)

![UCB Learning Curve](Multi-Armed-Bandits/UCB_Bandit_Learning_curve.png)

#### UCB vs $\epsilon$-Greedy Comparison (200 runs averaged)

![UCB vs Epsilon-Greedy](Multi-Armed-Bandits/UCB_vs_epsilon_greedy.png)

**Key Insights:**
* UCB consistently outperforms $\epsilon$-greedy ($\epsilon = 0.1$) because it allocates exploration budget to arms that are *promising but under-sampled*, rather than exploring uniformly at random.
* UCB converges faster in early steps due to its deterministic, information-driven exploration.

#### Effect of Exploration Parameter $c$

![UCB Effect of c](Multi-Armed-Bandits/UCB_exploration_parameter_c.png)

**Key Insights:**
* **Low $c$ (0.5):** Near-greedy behavior — fast initial convergence but risks locking onto a suboptimal arm.
* **Moderate $c$ (1–2):** Optimal balance — sufficient exploration to find the best arm without excessive waste.
* **High $c$ (5):** Over-exploration — wastes steps on inferior arms due to an inflated confidence bonus.

---

## Environment Setup
* Python 3.12 (Miniconda Environment)
* NumPy
* Matplotlib