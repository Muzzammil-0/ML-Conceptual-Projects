import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ---------- 1. Load and Split Data ----------
data = load_breast_cancer()
X, y = data.data, data.target

# Separate into Test (20%), Labeled Pool (start with 10), Unlabeled Pool (the rest)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale features
scaler = StandardScaler()
X_temp_scaled = scaler.fit_transform(X_temp)
X_test_scaled = scaler.transform(X_test)

# ---------- 2. The Active Learning Loop ----------
def run_active_learning(strategy='random', n_iterations=20, samples_per_iter=10):
    # Reset the pools for each run
    indices = np.arange(len(X_temp_scaled))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    # Start with 10 random labeled samples
    labeled_idx = list(indices[:10])
    unlabeled_idx = list(indices[10:])
    
    accuracies = []
    
    for i in range(n_iterations):
        # Train model on currently labeled data
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_temp_scaled[labeled_idx], y_temp[labeled_idx])
        
        # Evaluate on the unseen test set
        acc = accuracy_score(y_test, model.predict(X_test_scaled))
        accuracies.append(acc)
        
        # Stop if we run out of unlabeled data
        if len(unlabeled_idx) < samples_per_iter:
            break
            
        # --- Selection Strategy ---
        if strategy == 'random':
            # Pick randomly
            chosen = np.random.choice(unlabeled_idx, size=samples_per_iter, replace=False)
        else:
            # Uncertainty Sampling: Pick points where model is most confused
            probs = model.predict_proba(X_temp_scaled[unlabeled_idx])
            # Uncertainty is highest when prob is close to 0.5. 
            # We calculate the absolute difference between the two class probabilities.
            # Sort by lowest difference (most uncertain) to highest.
            uncertainty = np.abs(probs[:, 0] - probs[:, 1])
            chosen_pos = np.argsort(uncertainty)[:samples_per_iter]
            chosen = [unlabeled_idx[pos] for pos in chosen_pos]
            
        # "Human" labels the chosen points and moves them to the labeled pool
        labeled_idx.extend(chosen)
        unlabeled_idx = [idx for idx in unlabeled_idx if idx not in chosen]

    return accuracies

# ---------- 3. Run Simulations ----------
print("Running Active Learning simulation...")
print("Strategy 1: Random Sampling")
acc_random = run_active_learning(strategy='random')

print("Strategy 2: Uncertainty Sampling")
acc_uncertainty = run_active_learning(strategy='uncertainty')

# ---------- 4. Plot the Results ----------
x_axis = [10 + i*10 for i in range(len(acc_random))]

plt.figure(figsize=(10, 6))
plt.plot(x_axis, acc_random, marker='o', label='Random Sampling', color='red', linestyle='--')
plt.plot(x_axis, acc_uncertainty, marker='s', label='Active Learning (Uncertainty)', color='blue')

plt.title('Active Learning vs. Random Sampling')
plt.xlabel('Number of Labeled Samples')
plt.ylabel('Test Accuracy')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Print final stats
print("\n=== FINAL RESULTS ===")
print(f"After {x_axis[-1]} labels:")
print(f"Random Sampling accuracy:     {acc_random[-1]:.4f}")
print(f"Active Learning accuracy:     {acc_uncertainty[-1]:.4f}")