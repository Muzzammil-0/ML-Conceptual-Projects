# Collaborative Filtering Book Recommender using Truncated SVD

A lightweight, machine learning-based recommendation system built using Python, `pandas`, `numpy`, and `scikit-learn`. This project demonstrates how to leverage **Matrix Factorization (Truncated SVD)** for collaborative filtering to recommend books to users based on historical rating patterns.

---

## 📌 Project Overview

In collaborative filtering, the goal is to predict missing ratings or rank items for a user based on the preferences of similar users. This script:
1. **Loads and samples** rating data from the popular `goodbooks-10k` dataset.
2. **Constructs a User-Item Matrix** where rows represent users, columns represent books, and entries are ratings (with unrated books filled with zeros).
3. **Applies Truncated SVD** (Singular Value Decomposition) for dimensionality reduction and latent factor extraction.
4. **Reconstructs the Rating Matrix** to estimate predicted scores for unread books.
5. **Generates Top-N Recommendations** for a specific user, filtering out books they have already interacted with.

---

## 🛠️ Prerequisites & Dependencies

Make sure you have Python installed along with the following required libraries:

```bash
pip install pandas numpy scikit-learn