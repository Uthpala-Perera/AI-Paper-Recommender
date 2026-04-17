import streamlit as st
from recommender import recommend_papers
from summarizer import summarize_text
from rag_chat import answer_question
import pandas as pd
import plotly.express as px

df = pd.read_csv("cleaned_papers.csv")

st.set_page_config(page_title="AI Research Assistant", layout="wide")

# ---------- UI ----------
st.markdown("""
<style>
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
.summary {
    background: rgba(0,0,0,0.3);
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 AI Research Assistant")
st.write("Search, summarize, and chat with research papers")

if "saved" not in st.session_state:
    st.session_state.saved = {}

top_k = st.sidebar.slider("Results", 1, 10, 5)
show_summary = st.sidebar.toggle("Summaries", True)

query = st.text_input("🔍 Enter topic")

tab1, tab2, tab3 = st.tabs(["Results", "Analytics", "Saved"])

# ---------- RESULTS ----------
with tab1:
    if query:
        results = recommend_papers(query, top_k)

        for i, paper in enumerate(results):
            st.markdown("<div class='card'>", unsafe_allow_html=True)

            st.subheader(paper["title"])

            if show_summary:
                try:
                    summary = summarize_text(paper["abstract"])
                    st.markdown("<div class='summary'>", unsafe_allow_html=True)
                    st.write(summary)
                    st.markdown("</div>", unsafe_allow_html=True)
                except:
                    st.write(paper["abstract"][:300])

            with st.expander("Full Abstract"):
                st.write(paper["abstract"])

            question = st.text_input("Ask about this paper", key=f"q_{i}")

            if question:
                answer = answer_question(paper["abstract"], question)
                st.write(answer)

            st.markdown(f"[Read Paper]({paper['link']})")

            if st.button("Save", key=f"s_{i}"):
                st.session_state.saved[paper["title"]] = paper

            st.markdown("</div>", unsafe_allow_html=True)

# ---------- ANALYTICS ----------
with tab2:
    df["length"] = df["text"].apply(len)
    fig = px.histogram(df, x="length")
    st.plotly_chart(fig)

# ---------- SAVED ----------
with tab3:
    for paper in st.session_state.saved.values():
        st.subheader(paper["title"])
        st.write(paper["abstract"][:300])
        st.markdown(f"[Read Paper]({paper['link']})")
