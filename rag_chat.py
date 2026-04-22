from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Load ONCE (important)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

def chunk_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def answer_question(context, question):
    chunks = chunk_text(context)

    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    query_embedding = embed_model.encode([question])
    _, indices = index.search(np.array(query_embedding), k=2)

    relevant_chunks = [chunks[i] for i in indices[0]]
    context_text = " ".join(relevant_chunks)

    prompt = f"Answer the question based on the context:\n{context_text}\nQuestion: {question}\nAnswer:"

    result = qa_pipeline(prompt, max_length=150)

    return result[0]["generated_text"]
