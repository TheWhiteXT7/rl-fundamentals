import numpy as np 
import matplotlib.pyplot as plt

#setting up environment
k = 10 #number of arms 
steps = 1000
epsilon = 0.1
#generate the true hidden probabilities / rewards signal for each arm
true_action_values = np.random.normal(0,1, k)
#q value analysis (agent's memeory)
q_values = np.zeros(k) #value of each arm
action_counts = np.zeros(k) # Number of times each arm was pulled

# Reward tracking
rewards_history = np.zeros(steps)

for i in range (steps):
    #tast 1 : Action selection (epsilon-greedy).
    if np.random.rand() < epsilon:
        action = np.random.randint(0,k) #exploration(10%)
    else:
        action = np.argmax(q_values) #exploitation(90%)
    
    #Task 2: pull the arm and observe reward
    reward = np.random.normal(true_action_values[action],1) 

    #Task 3: Update agent's memory (Q-values)
    action_counts[action] += 1 
    n = action_counts[action]
    #incremental update formula 
    old_value = q_values[action]
    new_value = old_value + (1/n) * (reward-old_value)
    q_values[action] = new_value 
    # Reward tracking
    rewards_history[i] = reward 

#Using avg revards for better plotting
cumulative_average = np.cumsum(rewards_history)/(np.arange(1,steps+1))
# Print results
print("True action values:", true_action_values)
print("Estimated action values:", q_values)

# Task 4: Plotting Graph
# Graph plotting
plt.figure(figsize=(10, 6))
plt.plot(cumulative_average, label=f"e = {epsilon}")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Multi-Armed Bandit Learning Curve")
plt.legend()
plt.grid(True)
plt.show()