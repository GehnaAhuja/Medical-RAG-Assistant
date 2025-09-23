# ⚕️ Medical RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers health-related questions by combining **clinical guidelines, patient forums, and medical blogs**.  
This project demonstrates **multi-source retrieval, reranking, contradiction handling, LLM synthesis, and conversational memory** via a Streamlit app.

⚠️ **Disclaimer:** This project is for **educational purposes only** and does **not provide medical advice**.

---

## ⚡ Quickstart (4 Commands)

```
git clone https://github.com/yourusername/medical-rag-assistant.git
cd medical-rag-assistant
pip install -r requirements.txt
streamlit run app.py --server.fileWatcherType none
## To Evaluate
python -m src.evaluate --root data --device cpu
```
