# Semi-Supervised Learning — Pseudo-Labeling

Using `SelfTrainingClassifier` to train a model on a dataset with only 10% labeled data, demonstrating the conditions under which pseudo-labeling succeeds or fails.

## The Problem
In real-world ML, acquiring labels is expensive. You might have a massive dataset but only a fraction of it is labeled by human experts. Throwing away the unlabeled data is a waste of information.

## The Solution
**Pseudo-Labeling (Self-Training):**
1. Train the model on the small labeled dataset (10%).
2. Predict labels for the unlabeled dataset (90%).
3. Take the predictions where the model is highly confident (> 90% probability) and treat them as "real" labels.
4. Add these pseudo-labeled samples to the training set and repeat the process.

## Dataset
- Breast Cancer Wisconsin (569 rows)
- **Labeled set:** 45 samples (10% of training)
- **Unlabeled set:** 410 samples (masked with `-1`)
- **Test set:** 114 samples (20% holdout)

## Results

| Model | Data Used | Accuracy |
| :--- | :--- | :--- |
| **Supervised (Baseline)** | 10% Labeled | **0.9561** |
| Semi-Supervised (Self-Training) | 10% + 90% Unlabeled | 0.9474 |
| Supervised (Oracle Ceiling) | 100% Labeled | 0.9561 |

## Key Observations
- **The "Easy Dataset" Limitation.** The 10% baseline model achieved the exact same accuracy as the 100% Oracle model (0.9561). The Breast Cancer dataset is highly separable, meaning 45 well-chosen samples were already enough for the Random Forest to find the perfect decision boundary.
- **Confirmation Bias.** Because the base model was already at peak performance, the pseudo-labeled data added in iteration 1 (264 samples) merely reinforced the model's existing beliefs. This slight overfitting to its own predictions caused a microscopic drop in test accuracy.
- **The Convergence.** Look at the self-training iterations: 264 labels added in Iteration 1, dropping to 43, 25, 7, and finally 2. This shows the model successfully exhausting its "high confidence" predictions.
- **When SSL shines.** Semi-Supervised Learning yields massive improvements on **complex, unstructured data** (images, text, audio) where 10% of the data is nowhere near enough to find the boundary. For simple, low-dimensional tabular data, supervised models are already too efficient.

## Stack
`scikit-learn` · `pandas` · `numpy`