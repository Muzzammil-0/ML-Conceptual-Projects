import numpy as np
import random

# ---------- 1. Define the Custom Environment (MDP) ----------
class GridWorld:
    def __init__(self, size=3):
        self.size = size
        # States 0-8 (3x3 grid). Start at 0, Goal at 8.
        self.state = 0
        self.goal = 8
        self.hole = 5 # Middle-right is a trap
        
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = [0, 1, 2, 3]
        self.state_space = list(range(size * size))

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        row, col = divmod(self.state, self.size)
        
        # State transition logic (The "P" in MDP)
        if action == 0:   row = max(0, row - 1) # Up
        elif action == 1: row = min(self.size - 1, row + 1) # Down
        elif action == 2: col = max(0, col - 1) # Left
        elif action == 3: col = min(self.size - 1, col + 1) # Right
        
        self.state = row * self.size + col
        
        # Reward logic (The "R" in MDP)
        reward = -0.1 # Small penalty for every step (encourages shortest path)
        done = False
        
        if self.state == self.goal:
            reward = 1.0
            done = True
        elif self.state == self.hole:
            reward = -1.0
            done = True
            
        return self.state, reward, done

# ---------- 2. Train Q-Learning on the Custom Env ----------
env = GridWorld(size=3)
q_table = np.zeros((9, 4)) # 9 states, 4 actions

# Hyperparameters
episodes = 500
lr = 0.1
gamma = 0.9
epsilon = 1.0

for ep in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Epsilon-greedy action
        if random.uniform(0, 1) < epsilon:
            action = random.choice(env.action_space)
        else:
            action = np.argmax(q_table[state, :])
            
        next_state, reward, done = env.step(action)
        
        # Bellman Equation
        q_table[state, action] = q_table[state, action] + lr * (
            reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action]
        )
        
        state = next_state
        
    epsilon = max(0.01, epsilon * 0.99) # Decay exploration

print("Training complete. Final Q-Table:\n", np.round(q_table, 2))

# ---------- 3. Test the Agent ----------
print("\n--- Testing Agent ---")
state = env.reset()
done = False
steps = 0
while not done and steps < 10:
    action = np.argmax(q_table[state, :])
    next_state, reward, done = env.step(action)
    print(f"State {state} -> Action {action} -> New State {next_state} (Reward: {reward})")
    state = next_state
    steps += 1
    
if reward == 1.0:
    print("Reached the Goal!")
else:
    print("Failed.")