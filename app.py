import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ==========================
#   USTAWIENIA
# ==========================

st.set_page_config(page_title="MKS Knowledge Engine", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ==========================
#   ŁADOWANIE DANYCH
# ==========================

@st.cache_data
def load_data():
    df = pd.read_csv("knowledge.csv")
    df["id"] = df.index
    return df

df = load_data()

# ==========================
#   TWORZENIE EMBEDDINGÓW
# ==========================

@st.cache_resource
def compute_embeddings(texts):
    """
    Generuje embeddings dla listy tekstów.
    """
    batch = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    vectors = np.array([d.embedding for d in batch.data])
    return vectors

# Przygotowanie tekstów do embeddingu
combined_texts = (
    df["category"].astype(str) + " "
    + df["tags"].astype(str) + " "
    + df["content"].astype(str)
).tolist()

emb_matrix = compute_embeddings(combined_texts)


# ==========================
#   FUNKCJA WYSZUKIWANIA
# ==========================

def semantic_search(query, top_k=10):
    """
    Zwraca najlepsze dopasowania do zapytania użytkownika.
    """
    q_emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    ).data[0].embedding

    sims = cosine_similarity([q_emb], emb_matrix)[0]
    top_idx = sims.argsort()[::-1][:top_k]

    results = df.iloc[top_idx].copy()
    results["similarity"] = sims[top_idx]
    return results


# ==========================
#   UI
# ==========================

st.title("🧠 MKS Knowledge Engine")
st.write("Zaawansowana wyszukiwarka wiedzy
