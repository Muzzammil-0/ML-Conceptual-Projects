import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# ---------- 1. Load Data ----------
url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv"
df = pd.read_csv(url).sample(n=20000, random_state=42)
user_item_matrix = df.pivot(index='user_id', columns='book_id', values='rating').fillna(0)

# ---------- 2. SVD (Standard Zero-Fill) ----------
svd = TruncatedSVD(n_components=20, random_state=42)
user_factors = svd.fit_transform(user_item_matrix)
item_factors = svd.components_
predicted_df = pd.DataFrame(np.dot(user_factors, item_factors), 
                            index=user_item_matrix.index, columns=user_item_matrix.columns)

# ---------- 3. Recommend (Ranking Only) ----------
user_id = 55
read_books = user_item_matrix.loc[user_id]
read_books = read_books[read_books > 0].index

user_predictions = predicted_df.loc[user_id].drop(read_books)

# We don't print the score. We just print the rank (1st, 2nd, 3rd...)
top_5_books = user_predictions.sort_values(ascending=False).head(5).index

print(f"=== TOP 5 RANKED RECOMMENDATIONS (User {user_id}) ===")
for rank, book_id in enumerate(top_5_books, 1):
    print(f"  Rank {rank}: Book ID {book_id}")