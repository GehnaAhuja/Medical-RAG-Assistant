import streamlit as st
from src.rag_pipeline import RAGPipeline

# ---------------------------
# Initialize pipeline once
# ---------------------------
@st.cache_resource
def load_pipeline():
    return RAGPipeline(data_root="data", log_dir="logs", device="cpu")

rag = load_pipeline()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Medical RAG Assistant", page_icon="⚕️")

st.title("⚕️ Medical RAG Assistant")
st.write("Ask a health-related question. Sources: clinical docs, patient forums, medical blogs.")
st.info("⚠️ Disclaimer: This system is for educational purposes only. Not medical advice.")

query = st.text_input("Enter your medical question:")

if st.button("Ask") or query:
    if query.strip():
        with st.spinner("Retrieving and analyzing..."):
            result = rag.answer(query, top_k=5)

        st.subheader("Answer")
        st.write(result["response"])

        st.subheader("Citations")
        for cite in result["citations"]:
            st.write(f"- **{cite['source']}** → `{cite['doc']}` (chunk: {cite['chunk']})")

        st.caption(f"Log file saved: {result['log_file']}")
