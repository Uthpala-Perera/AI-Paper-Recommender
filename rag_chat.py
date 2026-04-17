from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

def chunk_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def answer_question(context, question):
    chunks = chunk_text(context)

    embeddings = embed_model.encode(chunks)
    query_embedding = embed_model.encode([question])

    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-2:][::-1]

    context_text = " ".join([chunks[i] for i in top_indices])

    prompt = f"""
    Answer the question based only on the context below.

    Context:
    {context_text}

    Question: {question}
    Answer:
    """

    result = qa_pipeline(prompt, max_length=150)
    return result[0]["generated_text"]
