from transformers import pipeline

# Load once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    # Limit input size (important!)
    text = text[:1000]

    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)

    return summary[0]["summary_text"]