# app.py
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------------- Load model ----------------
st.title("Local Embeddings with SentenceTransformers")

st.write("This app computes embeddings for your input text using the local model `all-MiniLM-L6-v2`.")

# Load model (done once)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- Input text ----------------
text = st.text_area("Enter text to embed")

if st.button("Compute Embedding"):
    if text.strip() == "":
        st.warning("Please enter some text!")
    else:
        emb = model.encode(text)
        st.success("Embedding computed!")
        st.write(emb)
        st.write(f"Embedding dimension: {len(emb)}")

# ---------------- Example: document embeddings ----------------
st.markdown("---")
st.write("Example: Compute embeddings for multiple documents")
docs = ["Policy A text...", "Policy B text..."]
if st.button("Compute Document Embeddings"):
    doc_embs = model.encode(docs)
    for i, d in enumerate(docs):
        st.write(f"Document {i} embedding: {doc_embs[i]}")
        st.write(f"Dimension: {len(doc_embs[i])}")
