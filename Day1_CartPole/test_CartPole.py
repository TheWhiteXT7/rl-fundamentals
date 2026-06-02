import gymnasium as gym
import time

# 1. Initialize the Environment
# The 'render_mode' tells Gymnasium to show us a visual window
env = gym.make("CartPole-v1", render_mode="human")

# 2. Reset to start state
# observation contains 4 numbers: [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
observation, info = env.reset()

print(f"Initial State Observation: {observation}")

for _ in range(500):
    # 3. The "Brain" (currently a random guesser)
    #action = env.action_space.sample() 
    #Human logic (Heuristic Agent)
    if observation[2] > 0:
        action = 1
    else:
        action = 0
    # 4. Interact with the environment
    observation, reward, terminated, truncated, info = env.step(action)

    # 5. Check if we failed
    if terminated or truncated:
        print("Pole fell! Resetting...")
        observation, info = env.reset()
    
    time.sleep(0.01) # Slow it down so you can actually see it

env.close()