# Reinforcement Learning — Tabular Q-Learning (Frozen Lake)

Implementing a foundational Q-Learning agent from scratch to solve the `FrozenLake` environment using the Bellman Equation.

## The Paradigm
Unlike Supervised Learning (labeled data) or Unsupervised Learning (pattern finding), Reinforcement Learning trains an **Agent** to take **Actions** in an **Environment** to maximize a cumulative **Reward** over time.

## The Algorithm (Q-Learning)
Q-Learning builds a "Q-Table" (a matrix of States x Actions). It stores the expected future reward of taking a specific action in a specific state.
- **The Bellman Equation:** `Q(s,a) = Q(s,a) + lr * (reward + discount * max(Q(s',a')) - Q(s,a))`
- This equation updates the value of an action by looking at the immediate reward plus the best possible reward from the next state.

## Environment
- OpenAI Gymnasium `FrozenLake-v1`
- 16 states (4x4 grid), 4 actions (Left, Down, Right, Up).
- The agent must reach the Goal without falling into a Hole.

## Results
- **Final Success Rate:** 58% (during training), 100% (during pure exploitation).
- **Final Q-Table:** The agent successfully mapped the positive values (0.99) to the safe paths and left the holes at 0.0. 

## Key Observations
- **Exploration vs. Exploitation:** The agent used an epsilon-greedy strategy. Initially, `exploration_rate=1.0` (pure random guessing). Over 1,000 episodes, this decayed to `0.01`. The 58% training success rate is entirely due to this 1% random chance of falling into a hole during training. When forced to exploit (`argmax`), it reaches the goal perfectly.
- **Sparse Rewards:** The agent receives `+1` only when it reaches the goal. It has to propagate this single reward backward through the Q-table to learn the entire path.
- **No Neural Networks:** This is "Tabular Q-Learning." Deep Q-Learning (DQN) replaces the matrix with a neural network, which is what powers modern RL (like AlphaGo or self-driving cars).

## Stack
`gymnasium` · `numpy` · `python`