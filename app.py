import streamlit as st
from src.rag_pipeline import RAGPipeline

# ---------------------------
# Initialize pipeline
# ---------------------------
@st.cache_resource
def load_pipeline():
    return RAGPipeline(data_root="data", log_dir="logs", device="cpu")

rag = load_pipeline()

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Medical RAG Assistant", page_icon="⚕️", layout="wide")

st.title("⚕️ Medical RAG Assistant")
st.caption("Ask health-related questions. Sources: clinical docs, patient forums, blogs.")
st.info("⚠️ Disclaimer: This demo is for educational purposes only. Not medical advice.")

# ---------------------------
# Session State for Chat
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# Display chat messages
# ---------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            # Show citations if available
            if "citations" in msg:
                with st.expander("📚 Citations"):
                    for cite in msg["citations"]:
                        st.write(f"- **{cite['source']}** → `{cite['doc']}` (chunk: {cite['chunk']})")
            if "log_file" in msg:
                st.caption(f"Log file: {msg['log_file']}")

# ---------------------------
# Chat input
# ---------------------------
if user_input := st.chat_input("Type your question here..."):
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Run pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag.answer(user_input, top_k=5)
            response = result["response"]

        st.markdown(response)
        with st.expander("📚 Citations"):
            for cite in result["citations"]:
                st.write(f"- **{cite['source']}** → `{cite['doc']}` (chunk: {cite['chunk']})")
        st.caption(f"Log file: {result['log_file']}")

    # Save assistant message with metadata
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "citations": result["citations"],
        "log_file": result["log_file"]
    })
