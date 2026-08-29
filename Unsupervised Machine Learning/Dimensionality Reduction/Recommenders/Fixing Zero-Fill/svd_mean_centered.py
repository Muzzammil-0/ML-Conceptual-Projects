import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# ---------- 1. Load Data ----------
url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv"
df = pd.read_csv(url).sample(n=20000, random_state=42)

user_item_matrix = df.pivot(index='user_id', columns='book_id', values='rating')

# ---------- 2. The Fix: Mean Centering ----------
# Calculate the average rating of every user
user_means = user_item_matrix.mean(axis=1)

# Subtract the user mean from their ratings. 
# (Missing values remain NaN, we will fill them with 0 AFTER centering)
centered_matrix = user_item_matrix.sub(user_means, axis=0).fillna(0)

# ---------- 3. SVD ----------
svd = TruncatedSVD(n_components=20, random_state=42)
user_factors = svd.fit_transform(centered_matrix)
item_factors = svd.components_

# Reconstruct the matrix (predicted ratings relative to user mean)
predicted_centered = np.dot(user_factors, item_factors)

# Add the user means back to get actual star predictions!
predicted_df = pd.DataFrame(predicted_centered, index=centered_matrix.index, columns=centered_matrix.columns)
predicted_df = predicted_df.add(user_means, axis=0)

# ---------- 4. Recommend ----------
user_id = 55
read_books = user_item_matrix.loc[user_id].dropna().index
user_predictions = predicted_df.loc[user_id].drop(read_books)
top_5 = user_predictions.sort_values(ascending=False).head(5)

print(f"=== MEAN-CENTERED SVD (User {user_id}) ===")
for book, score in top_5.items():
    print(f"Book {book}: Predicted Rating = {score:.2f} stars")