import pandas as pd
import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares

# ---------- 1. Load Data ----------
url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv"
df = pd.read_csv(url).sample(n=20000, random_state=42)

# Convert IDs to categorical codes (0 to N-1) because ALS requires indices
df['user_id'] = df['user_id'].astype('category').cat.codes
df['book_id'] = df['book_id'].astype('category').cat.codes

# ---------- 2. Create a Sparse Matrix ----------
# ALS is designed for sparse matrices. Unrated books are simply absent (not 0).
# We binarize the ratings (if they rated it > 0, they interacted with it)
sparse_matrix = sparse.csr_matrix(
    (df['rating'].astype(float), (df['user_id'], df['book_id']))
)

# ---------- 3. Train ALS ----------
# factors = number of latent traits
model = AlternatingLeastSquares(factors=20, iterations=15, random_state=42)
model.fit(sparse_matrix)

# ---------- 4. Recommend ----------
user_id = 55
# ALS .recommend() method automatically excludes items the user has already interacted with
# Returns (item_id, score) tuples
recommendations = model.recommend(user_id, sparse_matrix[user_id], N=5)

print(f"=== IMPLICIT ALS RECOMMENDATIONS (User {user_id}) ===")
for book_id, score in zip(recommendations[0], recommendations[1]):
    print(f"Book ID {book_id}: Recommendation Score = {score:.2f}")