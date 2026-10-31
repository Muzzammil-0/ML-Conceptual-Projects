# Active Learning — Uncertainty Sampling

Simulating a human-in-the-loop Active Learning pipeline where the model selectively queries the most informative unlabeled data points, drastically reducing the number of labels required to achieve high accuracy.

## Dataset
- Breast Cancer Wisconsin (569 rows)
- Test Set: 20% holdout
- Initial Labeled Pool: 10 samples
- Query Size: 10 samples per iteration

## Approach
We compare two sampling strategies:
1. **Random Sampling:** The model randomly selects 10 unlabeled points, requests their labels, and retrains.
2. **Active Learning (Uncertainty):** The model predicts probabilities for all unlabeled points. It calculates uncertainty as `abs(P(class_0) - P(class_1))`. It requests the labels of the 10 points with the *lowest* uncertainty (i.e., probabilities closest to 50/50). These are the points sitting right on the decision boundary.

## Results
- **Active Learning:** Reached ~95.6% accuracy using only ~75 labels.
- **Random Sampling:** Reached ~92.9% accuracy after 200 labels. 
- Active Learning saved over 60% of the labeling budget while achieving a higher final accuracy.

## Key Observations
- **Querying the boundary.** By explicitly seeking out points where `P(Malignant) ≈ P(Benign)`, the model learns the decision boundary much faster than if it were shown easy, obvious examples.
- **The "Outlier Trap".** Look at the blue line between 75 and 125 labels: it dips. Once the model learned the main boundary, the remaining "uncertain" points were mostly noise/outliers. Querying these confused the model slightly. In production, a mix of uncertainty and random sampling is used to prevent this.
- **Cost efficiency.** Active Learning assumes human labeling is expensive. By reducing the required labeled data from 200 to 75 samples, it saves 62% of the annotation budget.

## Stack
`scikit-learn` · `matplotlib` · `numpy`