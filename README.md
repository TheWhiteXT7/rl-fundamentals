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

## Environment Setup
* Python 3.12 (Miniconda Environment)
* NumPy
* Matplotlib