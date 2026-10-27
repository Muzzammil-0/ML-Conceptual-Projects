# Federated Learning — Simulated FedAvg

Simulating a Federated Learning architecture from scratch to train a Logistic Regression model across three decentralized "hospitals" without sharing raw patient data.

## The Problem
In domains like healthcare, privacy laws prevent hospitals from pooling their data into a centralized server. Training a model on a single hospital's data results in a biased, poor-performing model.

## The Solution (Federated Averaging)
1. A central server initializes a global model.
2. The server sends the model weights to each hospital (client).
3. Each hospital trains the model locally on its private data.
4. Each hospital sends only the **mathematical weight updates** back to the server.
5. The server averages the weights (weighted by the number of samples per hospital).

## Dataset
- Breast Cancer Wisconsin (569 rows)
- Simulated 3 Hospitals with uneven IID data splits (227, 137, 91 samples).
- 5 Federated Communication Rounds.

## Results
- **Centralized Model Accuracy:** 0.9737
- **Federated Model Accuracy:** 0.9737 (Reached in Round 1)

## Key Observations
- **Instant Convergence (Convexity & IID).** The federated model reached the exact centralized accuracy in a single round. This happens because Logistic Regression has a single global minimum (convex), and the randomly split data is IID (every hospital sees a similar distribution). The weighted average of their local minimums lands perfectly on the global minimum.
- **Real-world complexity.** In production (e.g., EEG or imaging), Deep Neural Networks are non-convex, and hospital data is often Non-IID (different demographics, different machines). Under those conditions, convergence takes dozens of communication rounds.
- **Privacy Preservation.** The server aggregates gradients/weights, not data. A hospital's patient records never leave the local network.
- **Algorithm Compatibility.** Algorithms with explicit mathematical weights (Logistic Regression, Neural Networks) are easy to federate. Tree-based models (Random Forest, XGBoost) are difficult to federate because you cannot easily "average" decision trees.

## Stack
`scikit-learn` · `numpy` · `pandas`