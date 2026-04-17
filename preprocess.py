import pandas as pd

df = pd.read_csv("papers.csv")

# Combine title + abstract
df["text"] = df["title"] + " " + df["abstract"]

# Basic cleaning
df["text"] = df["text"].str.lower()

df.to_csv("cleaned_papers.csv", index=False)

print("✅ Preprocessing complete")