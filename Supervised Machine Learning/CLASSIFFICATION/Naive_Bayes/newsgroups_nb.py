import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

# ---------- 1. Load Text Data ----------
# We only grab two categories to keep it a binary classification problem
categories = ['sci.space', 'talk.religion.misc']

# Fetch training data
train_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
# Fetch test data
test_data = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

X_train, y_train = train_data.data, train_data.target
X_test, y_test = test_data.data, test_data.target

print(f"Training documents: {len(X_train)}")
print(f"Test documents: {len(X_test)}")
print(f"Classes: {train_data.target_names}")

# ---------- 2. Build the Pipeline ----------
# TfidfVectorizer converts raw text into a matrix of numbers (TF-IDF)
# MultinomialNB is the Naive Bayes classifier for text (word counts)
model = make_pipeline(
    TfidfVectorizer(stop_words='english', max_features=10000),
    MultinomialNB(alpha=0.1) # alpha is Laplace smoothing
)

# ---------- 3. Train and Evaluate ----------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=train_data.target_names))

print("=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

# ---------- 4. Inspect the "Naive" Assumption ----------
# Let's see which words Naive Bayes thinks are the most important for each class
vectorizer = model.named_steps['tfidfvectorizer']
classifier = model.named_steps['multinomialnb']
feature_names = vectorizer.get_feature_names_out()

print("\nTop 10 words for 'sci.space':")
# The classifier gives log probabilities. Sort them descending.
space_indices = np.argsort(classifier.feature_log_prob_[0])[-10:]
print([feature_names[i] for i in space_indices])

print("\nTop 10 words for 'talk.religion.misc':")
religion_indices = np.argsort(classifier.feature_log_prob_[1])[-10:]
print([feature_names[i] for i in religion_indices])