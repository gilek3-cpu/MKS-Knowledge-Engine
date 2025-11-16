import streamlit as st
import pandas as pd
from openai import OpenAI

st.set_page_config(page_title="Silnik Wiedzy MKS", page_icon="🔎")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Wczytywanie CSV z polskimi znakami ---
@st.cache_data
def load_knowledge():
    return pd.read_csv("knowledge.csv", encoding="utf-8")

df = load_knowledge()

# --- Tworzenie embeddingów (UTF-8 fix) ---
@st.cache_resource
def embed_texts(texts):
    clean_texts = [t.encode("utf-8", errors="ignore").decode("utf-8") for t in texts]
    emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=clean_texts
    )
    return [e.embedding for e in emb.data]

DOCUMENT_EMB = embed_texts(df["content"].tolist())

# --- Funkcja wyszukująca najlepszą odpowiedź ---
import numpy as np

def search(query):
    q_emb = embed_texts([query])[0]
    scores = np.dot(DOCUMENT_EMB, q_emb)
    idx = np.argmax(scores)
    return df.iloc[idx]

# --- UI ---
st.title("🔎 Silnik Wiedzy MKS")

query = st.text_input("Zadaj pytanie:")

if query:
    result = search(query)
    st.subheader("🔹 Najtrafniejsza odpowiedź:")
    st.write(result["content"])

    st.caption(f"Kategoria: {result['category']} | Tag: {result['tags']}")

