import gymnasium as gym
import numpy as np
import time

# ---------- 1. Setup the Environment ----------
# is_slippery=False makes the environment deterministic (the agent goes exactly where it steps)
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="ansi")

# ---------- 2. Initialize the Q-Table ----------
# The environment has 16 states (4x4 grid) and 4 actions (Left, Down, Right, Up)
state_space_size = env.observation_space.n
action_space_size = env.action_space.n
q_table = np.zeros((state_space_size, action_space_size))

print(f"State space: {state_space_size}, Action space: {action_space_size}")
print("Initial Q-Table (all zeros):\n", q_table)

# ---------- 3. Hyperparameters ----------
num_episodes = 1000      # How many times we play the game
max_steps_per_episode = 100
learning_rate = 0.1      # How much we update our Q-values
discount_rate = 0.99     # How much we care about FUTURE rewards vs immediate ones
exploration_rate = 1.0   # Probability of taking a random action (exploring)
max_exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# ---------- 4. Training Loop ----------
rewards_all_episodes = []

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    rewards_current_episode = 0
    
    for step in range(max_steps_per_episode):
        # Exploration-Exploitation Tradeoff
        # Roll the dice: do we explore (random) or exploit (use Q-table)?
        exploration_rate_threshold = np.random.uniform(0, 1)
        
        if exploration_rate_threshold > exploration_rate:
            # Exploit: pick the action with the highest Q-value
            action = np.argmax(q_table[state, :])
        else:
            # Explore: pick a random action
            action = env.action_space.sample()
            
        # Take the action
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Update the Q-Table (The Bellman Equation)
        # Q(s,a) = Q(s,a) + learning_rate * (reward + discount_rate * max(Q(s',a')) - Q(s,a))
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
            
        state = new_state
        rewards_current_episode += reward
        
        if done:
            break
            
    # Decay the exploration rate (slowly stop guessing and start using the table)
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
        
    rewards_all_episodes.append(rewards_current_episode)

# ---------- 5. Analyze the Results ----------
print("\n--- Training Finished ---")
print("Final Q-Table:\n", q_table)

# Calculate success rate over the last 100 episodes
success_rate = np.mean(rewards_all_episodes[-100:]) * 100
print(f"\nSuccess rate over last 100 episodes: {success_rate:.1f}%")

# ---------- 6. Watch the Agent Play ----------
print("\n--- Watching the Trained Agent (Exploiting only) ---")
state, info = env.reset()
done = False

for step in range(max_steps_per_episode):
    # Render the environment to the terminal
    print(env.render())
    time.sleep(0.5)
    
    # ALWAYS exploit (no random actions)
    action = np.argmax(q_table[state, :])
    new_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = new_state
    
    if done:
        print(env.render())
        if reward == 1:
            print("\n🎉 The agent reached the Goal!")
        else:
            print("\n💀 The agent fell in a hole.")
        break