import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load everything once
df = pd.read_csv("cleaned_papers.csv")
index = faiss.read_index("faiss_index.index")
model = SentenceTransformer("all-MiniLM-L6-v2")

def recommend_papers(query, top_k=5):
    query_embedding = model.encode([query])
    
    distances, indices = index.search(np.array(query_embedding), top_k)

    results = []

    for idx in indices[0]:
        results.append({
            "title": df.iloc[idx]["title"],
            "abstract": df.iloc[idx]["abstract"],
            "link": df.iloc[idx]["link"]
        })

    return results