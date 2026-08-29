# Recommenders — Implicit ALS (Industry Standard)

Using Alternating Least Squares (ALS) for Collaborative Filtering on implicit feedback data, the modern industry standard used by Spotify, Netflix, and YouTube.

## The Problem
SVD is designed for **explicit** feedback (1-5 star ratings). But in the real world, 99% of user data is **implicit** (clicks, views, purchases, time spent). If a user hasn't clicked a book, it doesn't mean they rate it "0 stars"—it just means they haven't seen it yet. Zero-filling an implicit matrix destroys the math.

## The Solution (Implicit ALS)
1. Use the `implicit` library, which implements the Alternating Least Squares algorithm.
2. Convert the User-Item interactions into a **sparse matrix**. Unclicked items are simply absent (null), not filled with zeros.
3. ALS treats the data as "confidence" (e.g., if you clicked a book 1 time, we are 40x more confident you like it than if you clicked it 0 times).

## Results
For User 55, ALS generated entirely different recommendations than SVD:
- Book ID 47 (Score: 0.01)
- Book ID 75 (Score: 0.01)

*(Note: The scores are small because ALS outputs a confidence weight, not a 1-5 star rating).*

## Key Observations
- **Handles Cold Start better.** Because ALS treats missing data as "unknown" rather than "zero," it handles sparse matrices much better than SVD.
- **Built for scale.** ALS is easily parallelized across CPU cores and GPUs, making it capable of handling billions of interactions (which is why Spotify uses it for music recommendations).
- **Confidence vs. Rating.** ALS does not try to predict what rating you will give a movie. It tries to predict the *probability* that you will interact with it.

## Stack
`implicit` · `scipy` · `pandas` · `numpy`