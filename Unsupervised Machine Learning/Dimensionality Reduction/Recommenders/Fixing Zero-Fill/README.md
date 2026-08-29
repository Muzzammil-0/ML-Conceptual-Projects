# Recommenders — Mean-Centered SVD

Fixing the "Zero-Fill" limitation of standard Matrix Factorization by centering user ratings around their mean, resulting in realistic predicted star ratings.

## The Problem
In standard SVD, missing values in the User-Item matrix are filled with `0`. Because most users have not rated most books, the matrix is mostly zeros. The algorithm learns that "0" is a valid rating, which drags down all predicted scores to nonsense numbers (e.g., predicting 0.31 stars instead of 4.5 stars).

## The Solution (Mean Centering)
1. Calculate the average rating for every user (e.g., User 55 gives an average of 3.8 stars).
2. Subtract this mean from all their rated books (e.g., a 5-star book becomes +1.2, a 2-star book becomes -1.8).
3. Factorize the centered matrix using `TruncatedSVD`.
4. Reconstruct the matrix and **add the user's mean back** to the predictions.

## Results
For User 55, the naive zero-fill SVD predicted a top score of **0.31 stars**. 
The mean-centered SVD predicts:
- Book 516: **4.41 stars**
- Book 1783: **4.40 stars**
- Book 46: **4.39 stars**

## Key Observations
- **Realistic Scores.** Mean-centering shifts the baseline so that the model predicts whether a user will like a book *more or less than their average*, rather than fighting the gravity of a thousand zeros.
- **Ranking remains identical.** While the *scores* became realistic, the *order* of the recommended books (the ranking) remained exactly the same as the zero-fill method. 
- **Business value.** Showing a user a recommendation with a predicted score of "4.41 stars" is much more convincing than showing them "0.31 stars," even if the items are the same.

## Stack
`scikit-learn` · `pandas` · `numpy`