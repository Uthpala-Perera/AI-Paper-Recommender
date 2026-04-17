import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

df = pd.read_csv("cleaned_papers.csv")
embeddings = np.load("embeddings.npy")

model = SentenceTransformer("all-MiniLM-L6-v2")

def recommend_papers(query, top_k=5):
    query_embedding = model.encode([query])

    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "title": df.iloc[idx]["title"],
            "abstract": df.iloc[idx]["abstract"],
            "link": df.iloc[idx]["link"]
        })

    return results
