# --- Clustering-Based Matching System (K-Means) ---
# This script groups startups into clusters based on text similarity,
# then assigns each investor to the nearest cluster and recommends startups inside it.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load CSVs
startups = pd.read_csv("startups.csv").fillna("")
investors = pd.read_csv("investors.csv").fillna("")

# Text functions
def make_text_startup(row):
    return " ".join([
        str(row['name']),
        row['description'],
        row['sectors'],
        row['tech_stack']
    ])

def make_text_investor(row):
    return " ".join([
        row['name'],
        row['thesis_keywords'],
        row['preferred_sectors']
    ])

# Generate text lists
startup_texts = startups.apply(make_text_startup, axis=1)
investor_texts = investors.apply(make_text_investor, axis=1)

# Build TF-IDF vectors
vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
all_texts = list(startup_texts) + list(investor_texts)
tfidf = vectorizer.fit_transform(all_texts)

# Split vectors
n_start = len(startup_texts)
startup_vecs = tfidf[:n_start]
investor_vecs = tfidf[n_start:]

# Choose number of clusters (experiment with 5–20)
K = 3

# Train K-means clustering model on startup vectors
km = KMeans(n_clusters=K, random_state=42)
cluster_labels = km.fit_predict(startup_vecs)

# Add cluster labels to the dataframe
startups['cluster'] = cluster_labels

# Get all cluster centroids
centroids = km.cluster_centers_()

# Assign investors to their closest centroid (cluster)
investor_cluster = []
for v in investor_vecs:
    sim_to_centroids = cosine_similarity(v, centroids)[0]
    best_cluster = int(sim_to_centroids.argmax())
    investor_cluster.append(best_cluster)

# Add assigned clusters to investor dataframe
investors['assigned_cluster'] = investor_cluster

# Match investors to startups in their cluster
for i, inv in investors.iterrows():
    c = inv['assigned_cluster']

    print(f"Investor {inv['name']} is placed in cluster {c}\n")

    # Select startups in same cluster
    candidates_idx = startups[startups['cluster'] == c].index.tolist()
    if not candidates_idx:
        print("No startups in this cluster.\n")
        continue

    # Get similarity scores inside the cluster
    cand_vecs = startup_vecs[candidates_idx]
    sim_scores = cosine_similarity(investor_vecs[i], cand_vecs)[0]

    # Rank startups by similarity
    ranked = sorted(zip(candidates_idx, sim_scores), key=lambda x: x[1], reverse=True)[:10]

    print("Top matches:")
    for idx, s in ranked:
      	print("  ", startups.loc[idx, 'name'], f"{s:.3f}")
    print()
