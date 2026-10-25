# Custom Markov Decision Process (MDP) Environment

Building a Reinforcement Learning environment from scratch in pure Python, demonstrating the core mechanics of states, actions, transition probabilities, and sparse rewards.

## The Concept
A Markov Decision Process (MDP) is the mathematical foundation of Reinforcement Learning. It is defined by:
- **States (S):** The agent's current position (e.g., 3x3 grid coordinates).
- **Actions (A):** Up, Down, Left, Right.
- **Transitions (P):** The rule that dictates where the agent moves based on the action.
- **Rewards (R):** +1 for the goal, -1 for a hole, -0.1 for every step.

## Approach
1. Build a Python class `GridWorld` with `reset()` and `step(action)` methods.
2. Implement the row/col transition logic.
3. Run Tabular Q-Learning on the custom environment for 500 episodes.

## Results
- The agent learned the optimal path: `State 0 -> 1 -> 4 -> 7 -> 8` (Right, Down, Down, Right).
- The Q-table correctly assigned a value of `1.0` to the final action leading to the goal, and a value of `0.0` to the hole state (State 5).

## Key Observations
- **Sparse Rewards.** The agent receives `-0.1` for walking. This forces it to find the shortest path to the goal, rather than wandering in circles.
- **Learning the Topology.** We didn't code the Q-table. The agent learned the boundaries of the 3x3 grid (e.g., trying to move Left from State 0 just keeps it at State 0) and the location of the penalty hole.
- **Gymnasium under the hood.** Libraries like OpenAI Gym use this exact same architecture. `env.step(action)` returns the new state, reward, and a `done` flag.