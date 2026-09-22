# Naive Bayes — Text Classification

Application of Multinomial Naive Bayes to the 20 Newsgroups dataset for binary text classification (`sci.space` vs. `talk.religion.misc`).

## Dataset
- 20 Newsgroups (scikit-learn built-in)
- Training docs: 970; Test docs: 645
- Features: Raw text converted to a 10,000-dimensional TF-IDF matrix

## Approach
1. Fetch text documents for two categories.
2. Build a pipeline: `TfidfVectorizer(stop_words='english', max_features=10000)` -> `MultinomialNB(alpha=0.1)`.
3. Train and evaluate using a confusion matrix and classification report.
4. Extract the top 10 most predictive words for each class.

## Results
- **Accuracy:** 0.98
- **Top Space Words:** 'moon', 'gov', 'nasa', 'space', 'edu', 'com', 'toronto', 'alaska', 'henry'
- **Top Religion Words:** 'morality', 'christian', 'people', 'jesus', 'god', 'writes', 'com', 'edu', 'kent'

## Key Observations
- **The "Naive" assumption exposed.** The model treats all words independently and has no context. It picked up on noise words like `com`, `edu` (email headers), and `toronto`, `henry` (likely specific user signatures or names in this small subset) as predictive features.
- **Laplace Smoothing:** `alpha=0.1` prevents zero-probability errors when encountering words in the test set that were not present in the training set.
- **High-Dimensionality:** Unlike tabular data where we fear the curse of dimensionality, Naive Bayes thrives in high dimensions (10,000 features) as long as the features are sparse (word counts).
- **Production King:** Despite its flaws, it is used in production for spam filtering and basic text categorization because of its millisecond training time and low memory footprint.

## Stack
`scikit-learn` · `numpy`