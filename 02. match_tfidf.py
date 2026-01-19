# --- TF-IDF Text-Based Matching System ---
# This script compares investor profiles and startup profiles using text similarity.
# TF-IDF converts text into numeric vectors, then we use cosine similarity to score matches.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CSV files
startups = pd.read_csv("startups.csv").fillna("")
investors = pd.read_csv("investors.csv").fillna("")

# Build a text representation for each startup
def make_text_startup(row):
    return " ".join([
        str(row['name']),
        row['description'],
        row['sectors'],
        row['tech_stack']
    ])

# Build a text representation for each investor
def make_text_investor(row):
    return " ".join([
        row['name'],
        row['thesis_keywords'],
        row['preferred_sectors']
    ])

# Create startup and investor text lists
startup_texts = startups.apply(make_text_startup, axis=1)
investor_texts = investors.apply(make_text_investor, axis=1)

# Combine into a single corpus so TF-IDF learns a shared vocabulary
corpus = pd.concat([startup_texts, investor_texts])

# Create the TF-IDF model
vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
tfidf = vectorizer.fit_transform(corpus)

# Split back into startup vectors and investor vectors
n_start = len(startup_texts)
startup_vecs = tfidf[:n_start]
investor_vecs = tfidf[n_start:]

# Compute cosine similarity (higher = more similar)
sim_matrix = cosine_similarity(investor_vecs, startup_vecs)

# Example: get top matches for the first investor
inv_idx = 0
scores = list(enumerate(sim_matrix[inv_idx]))

# Sort by descending similarity score
scores.sort(key=lambda x: x[1], reverse=True)

# Print top 5
for idx, score in scores[:5]:
    s = startups.iloc[idx]
    print(f"{s['id']} {s['name']} — score {score:.3f}")
