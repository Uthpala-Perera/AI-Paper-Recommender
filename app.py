import streamlit as st
from recommender import recommend_papers
from summarizer import summarize_text
from rag_chat import answer_question
import pandas as pd
import plotly.express as px

# Load dataset
df = pd.read_csv("cleaned_papers.csv")

st.set_page_config(page_title="AI Paper Recommender", layout="wide")

# ---------- UI STYLE ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}
.summary {
    background: rgba(0,0,0,0.3);
    padding: 12px;
    border-radius: 10px;
}
.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: white;
}
.subtitle {
    text-align: center;
    color: #cfcfcf;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<div class='title'>🚀 AI Research Explorer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Semantic Search • AI Summaries • Ask Questions</div>", unsafe_allow_html=True)

# ---------- SESSION ----------
if "saved" not in st.session_state:
    st.session_state.saved = {}

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Controls")
top_k = st.sidebar.slider("Results", 1, 10, 5)
show_summary = st.sidebar.toggle("Enable AI Summaries", True)

# ---------- SEARCH ----------
query = st.text_input("🔍 What do you want to research?")

# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(["📄 Results", "📊 Analytics", "⭐ Saved"])

# =======================
# 📄 RESULTS TAB
# =======================
with tab1:
    if query:
        with st.spinner("🔎 Searching papers..."):
            results = recommend_papers(query, top_k)

        for i, paper in enumerate(results):
            st.markdown("<div class='card'>", unsafe_allow_html=True)

            st.markdown(f"### {i+1}. {paper['title']}")

            # ---------- SUMMARY ----------
            if show_summary:
                try:
                    summary = summarize_text(paper["abstract"])
                    st.markdown("<div class='summary'>", unsafe_allow_html=True)
                    st.write("🧠 AI Summary")
                    st.write(summary)
                    st.markdown("</div>", unsafe_allow_html=True)
                except:
                    st.write(paper["abstract"][:300] + "...")

            # ---------- ABSTRACT ----------
            with st.expander("📖 Full Abstract"):
                st.write(paper["abstract"])

            # ---------- RAG CHAT ----------
            user_question = st.text_input(
                f"💬 Ask a question about this paper:",
                key=f"question_{i}"
            )

            if user_question:
                with st.spinner("🤖 Thinking..."):
                    answer = answer_question(paper["abstract"], user_question)

                st.markdown("**🤖 Answer:**")
                st.write(answer)

            # ---------- ACTION BUTTONS ----------
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"[🔗 Read Paper]({paper['link']})")

            with col2:
                if st.button("⭐ Save", key=f"save_{i}"):
                    st.session_state.saved[paper["title"]] = paper
                    st.success("Saved!")

            st.markdown("</div>", unsafe_allow_html=True)

# =======================
# 📊 ANALYTICS TAB
# =======================
with tab2:
    st.subheader("📊 Dataset Insights")

    df["length"] = df["text"].apply(len)

    fig = px.histogram(df, x="length", nbins=50, title="Paper Length Distribution")
    st.plotly_chart(fig, use_container_width=True)

# =======================
# ⭐ SAVED TAB
# =======================
with tab3:
    st.subheader("⭐ Saved Papers")

    if len(st.session_state.saved) == 0:
        st.info("No saved papers yet.")
    else:
        for paper in st.session_state.saved.values():
            st.markdown("<div class='card'>", unsafe_allow_html=True)

            st.markdown(f"### {paper['title']}")
            st.write(paper["abstract"][:300] + "...")
            st.markdown(f"[🔗 Read Paper]({paper['link']})")

            st.markdown("</div>", unsafe_allow_html=True)