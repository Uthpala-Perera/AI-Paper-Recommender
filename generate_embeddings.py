from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

df = pd.read_csv("cleaned_papers.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

np.save("embeddings.npy", embeddings)

print("✅ Embeddings generated")