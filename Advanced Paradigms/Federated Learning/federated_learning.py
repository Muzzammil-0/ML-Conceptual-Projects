import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import copy

# ---------- 1. Load and Split Data (Simulating 3 Hospitals) ----------
data = load_breast_cancer()
X, y = data.data, data.target

# Global Test set (to evaluate the final models)
X_train_pool, X_test, y_train_pool, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale globally (in reality, clients scale their own data)
scaler = StandardScaler()
X_train_pool = scaler.fit_transform(X_train_pool)
X_test = scaler.transform(X_test)

# Split the training pool into 3 hospitals (uneven sizes to be realistic)
# Hospital 1: 50%, Hospital 2: 30%, Hospital 3: 20%
split1 = int(len(X_train_pool) * 0.5)
split2 = int(len(X_train_pool) * 0.8)

X_h1, y_h1 = X_train_pool[:split1], y_train_pool[:split1]
X_h2, y_h2 = X_train_pool[split1:split2], y_train_pool[split1:split2]
X_h3, y_h3 = X_train_pool[split2:], y_train_pool[split2:]

hospitals = [(X_h1, y_h1), (X_h2, y_h2), (X_h3, y_h3)]
print(f"Hospital 1: {len(X_h1)} samples")
print(f"Hospital 2: {len(X_h2)} samples")
print(f"Hospital 3: {len(X_h3)} samples\n")

# ---------- 2. Baseline: Centralized Learning ----------
# (What you did in your EEG project - bringing all data to one place)
central_model = LogisticRegression(max_iter=1000)
central_model.fit(X_train_pool, y_train_pool)
central_acc = accuracy_score(y_test, central_model.predict(X_test))
print(f"Centralized Model Accuracy: {central_acc:.4f}")

# ---------- 3. Federated Learning (FedAvg) ----------
def federated_averaging(global_model, hospitals, rounds=5):
    global_weights = global_model.coef_
    global_intercept = global_model.intercept_
    
    for r in range(rounds):
        client_weights = []
        client_intercepts = []
        client_sizes = []
        
        # --- Client Update Phase ---
        for X_client, y_client in hospitals:
            # 1. Server sends global weights to the client
            local_model = LogisticRegression(max_iter=1000)
            local_model.coef_ = copy.deepcopy(global_weights)
            local_model.intercept_ = copy.deepcopy(global_intercept)
            local_model.classes_ = np.array([0, 1]) # required for sklearn
            
            # 2. Client trains on its LOCAL data
            local_model.fit(X_client, y_client)
            
            # 3. Client sends weights back to server
            client_weights.append(local_model.coef_)
            client_intercepts.append(local_model.intercept_)
            client_sizes.append(len(X_client))
            
        # --- Server Aggregation Phase (FedAvg) ---
        total_samples = sum(client_sizes)
        new_global_weights = np.zeros_like(global_weights)
        new_global_intercept = np.zeros_like(global_intercept)
        
        # Weighted average based on how much data each hospital has
        for w, b, size in zip(client_weights, client_intercepts, client_sizes):
            weight_factor = size / total_samples
            new_global_weights += w * weight_factor
            new_global_intercept += b * weight_factor
            
        # Update global model for next round
        global_weights = new_global_weights
        global_intercept = new_global_intercept
        
        # Evaluate global model at the end of the round
        global_model.coef_ = global_weights
        global_model.intercept_ = global_intercept
        global_model.classes_ = np.array([0, 1])
        acc = accuracy_score(y_test, global_model.predict(X_test))
        print(f"Round {r+1}: Global Model Accuracy = {acc:.4f}")

    return global_model

# Initialize the global model
global_model = LogisticRegression(max_iter=1000)
# Fake fit just to initialize the weights shape
global_model.fit(X_train_pool[:10], y_train_pool[:10])

print("\n--- Starting Federated Learning ---")
fl_model = federated_averaging(global_model, hospitals, rounds=5)