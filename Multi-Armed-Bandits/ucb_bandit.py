import numpy as np
import matplotlib.pyplot as plt

#Upper Confidence Bound Alogirithm

#setting up the environment
k = 10 #number of arms
steps = 1000

#reward structure
true_action_values = np.random.normal(0,1, k) #hidden reward
# Agent memory
q_values = np.zeros(k)
action_counts = np.zeros(k)

#tracking performance
rewards_history = np.zeros(steps)
c = 2 #exploration parameter

for t in range(1, steps + 1):
    if t <= k:
        action = t - 1
    else:
        ucb_values = q_values + c * np.sqrt(np.log(t)/action_counts)
        action = np.argmax(ucb_values)
    
    #pulling the arm and observe reward
    reward = np.random.normal(true_action_values[action], 1)
    #update agent memory
    action_counts[action] += 1
    n = action_counts[action]
    q_values[action] = q_values[action] + (reward - q_values[action]) / n
    #record reward
    rewards_history[t-1] = reward

#cumulative average for plotting
cumulative_average = np.cumsum(rewards_history)/(np.arange(1, steps + 1))

#print results
print("True action values:", true_action_values)
print("Estimated action values:", q_values)
print("Action counts:", action_counts)

# Plotting Graph
plt.figure(figsize=(10, 6))
plt.plot(cumulative_average, label=f"UCB (c = {c})")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("UCB Bandit Learning Curve")
plt.legend()
plt.grid(True)
plt.show()

#Simulation 2: UCB vs Epsilon-Greedy comparison
num_runs = 200 #average over multiple runs for smoother curves

ucb_avg_rewards = np.zeros(steps)
egreedy_avg_rewards = np.zeros(steps)

for run in range(num_runs):
    #shared reward structure for fair comparison
    true_vals = np.random.normal(0, 1, k)

    #--- UCB run ---
    q_ucb = np.zeros(k)
    n_ucb = np.zeros(k)
    c_val = 2
    for t in range(1, steps + 1):
        if t <= k:
            action = t - 1
        else:
            ucb_vals = q_ucb + c_val * np.sqrt(np.log(t) / n_ucb)
            action = np.argmax(ucb_vals)
        reward = np.random.normal(true_vals[action], 1)
        n_ucb[action] += 1
        q_ucb[action] += (reward - q_ucb[action]) / n_ucb[action]
        ucb_avg_rewards[t-1] += reward

    #--- Epsilon-Greedy run ---
    q_eg = np.zeros(k)
    n_eg = np.zeros(k)
    epsilon = 0.1
    for t in range(steps):
        if np.random.rand() < epsilon:
            action = np.random.randint(0, k)
        else:
            action = np.argmax(q_eg)
        reward = np.random.normal(true_vals[action], 1)
        n_eg[action] += 1
        q_eg[action] += (reward - q_eg[action]) / n_eg[action]
        egreedy_avg_rewards[t] += reward

ucb_avg_rewards /= num_runs
egreedy_avg_rewards /= num_runs

#cumulative averages
ucb_cumavg = np.cumsum(ucb_avg_rewards) / (np.arange(1, steps + 1))
eg_cumavg = np.cumsum(egreedy_avg_rewards) / (np.arange(1, steps + 1))

plt.figure(figsize=(10, 6))
plt.plot(ucb_cumavg, label="UCB (c = 2)")
plt.plot(eg_cumavg, label="ε-Greedy (ε = 0.1)")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("UCB vs Epsilon-Greedy Comparison")
plt.legend()
plt.grid(True)
plt.show()

#Simulation 3: UCB with different c values
c_values = [0.5, 1, 2, 5]
results = {}

for c_val in c_values:
    avg_rewards = np.zeros(steps)
    for run in range(num_runs):
        true_vals = np.random.normal(0, 1, k)
        q = np.zeros(k)
        n = np.zeros(k)
        for t in range(1, steps + 1):
            if t <= k:
                action = t - 1
            else:
                ucb_vals = q + c_val * np.sqrt(np.log(t) / n)
                action = np.argmax(ucb_vals)
            reward = np.random.normal(true_vals[action], 1)
            n[action] += 1
            q[action] += (reward - q[action]) / n[action]
            avg_rewards[t-1] += reward
    results[c_val] = avg_rewards / num_runs

plt.figure(figsize=(10, 6))
for c_val in c_values:
    cumavg = np.cumsum(results[c_val]) / (np.arange(1, steps + 1))
    plt.plot(cumavg, label=f"c = {c_val}")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("UCB: Effect of Exploration Parameter c")
plt.legend()
plt.grid(True)
plt.show()
