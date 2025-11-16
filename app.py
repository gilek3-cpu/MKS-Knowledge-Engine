import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import unicodedata
import json

st.set_page_config(page_title="Silnik Wiedzy MKS", page_icon="📘")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Normalizacja znaków (pełny fix) ---
def normalize(text):
    if isinstance(text, str):
        return unicodedata.normalize("NFC", text)
    return text

# --- Wczytywanie danych ---
@st.cache_data
def load_knowledge():
    df = pd.read_csv("knowledge.csv", encoding="utf-8")
    df["content"] = df["content"].apply(normalize)
    return df

df = load_knowledge()

# --- Embeddingi ---
@st.cache_resource
def embed_texts(texts):
    clean = [normalize(t) for t in texts]

    # API chce czyste UTF-8 → pakujemy do JSON ręcznie (fix)
    payload = {"model": "text-embedding-3-large", "input": clean}
    payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    emb = client._client.post(
        "/v1/embeddings",
        data=payload,
        headers={
            "Content-Type": "application/json; charset=utf-8"
        }
    )

    vectors = [d["embedding"] for d in emb["data"]]
    return vectors

DOCUMENT_EMB = embed_texts(df["content"].tolist())

# --- Wyszukiwanie ---
def search(query):
    q_emb = embed_texts([query])[0]
    scores = np.dot(DOCUMENT_EMB, q_emb)
    idx = np.argmax(scores)
    return df.iloc[idx]

# --- UI ---
st.title("📘 Silnik Wiedzy MKS")
query = st.text_input("Zadaj pytanie:")

if query:
    answer = search(query)
    st.subheader("Najtrafniejsza odpowiedź:")
    st.write(answer["content"])
