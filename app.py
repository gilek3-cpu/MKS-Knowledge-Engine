import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import json
from typing import List

st.set_page_config(page_title="Silnik Wiedzy MKS", page_icon="🔎")

# klient OpenAI (klucz bierze ze Streamlit secrets)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Wczytywanie CSV z polskimi znakami ---
@st.cache_data
def load_knowledge(path: str = "knowledge.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", sep=",")
    # upewnij się że kolumna content jest typu str
    df["content"] = df["content"].astype(str)
    return df

df = load_knowledge()

# --- Tworzenie embeddingów przez API (zwraca listę wektorów) ---
@st.cache_resource
def compute_embeddings(texts: List[str]) -> List[List[float]]:
    # API klienta openai (nowe SDK) potrafi samo poprawnie kodować UTF-8
    # Używamy metody client.embeddings.create(...)
    # Model stosujemy mniejszy do oszczędności (możesz użyć większego jeśli chcesz)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    # response.data to lista obiektów z polem "embedding"
    embeddings = [item.embedding for item in response.data]
    return embeddings

# przygotuj dokumenty i embeddingi (raz, cache)
DOCUMENT_TEXTS = df["content"].tolist()
DOCUMENT_EMB = compute_embeddings(DOCUMENT_TEXTS)

# --- funkcja wyszukująca najlepszy dokument ---
def semantic_search(query: str):
    if not query:
        return None
    # oblicz embedding zapytania
    q_resp = client.embeddings.create(model="text-embedding-3-small", input=[query])
    q_emb = q_resp.data[0].embedding
    emb_matrix = np.array(DOCUMENT_EMB)
    q_vec = np.array(q_emb)
    # kosinusowa podobieństwo
    # normalizujemy aby uniknąć problemów
    emb_norm = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    q_norm = q_vec / np.linalg.norm(q_vec)
    scores = emb_norm.dot(q_norm)
    top_idx = int(np.argmax(scores))
    return {
        "index": top_idx,
        "score": float(scores[top_idx]),
        "category": df.iloc[top_idx]["category"],
        "tags": df.iloc[top_idx]["tags"],
        "content": df.iloc[top_idx]["content"],
    }

# ---- UI ----
st.title("🔎 Silnik Wiedzy MKS")

query = st.text_input("Zadaj pytanie (np. 'góry', 'wytrzymałość'):")

if st.button("Szukaj") or query:
    with st.spinner("Szukam najlepszej odpowiedzi..."):
        try:
            result = semantic_search(query)
        except Exception as e:
            st.error(f"Wystąpił błąd przy wywołaniu OpenAI: {e}")
            result = None

    if result is None:
        st.info("Wpisz zapytanie i kliknij Szukaj.")
    else:
        st.subheader("Wynik:")
        st.markdown(f"**Kategoria:** {result['category']}")
        st.markdown(f"**Tagi:** {result['tags']}")
        st.write(result["content"])
        st.write(f"**Podobieństwo:** {result['score']:.3f}")

st.write("---")
st.markdown("Plik `knowledge.csv` ładowany jest z katalogu aplikacji. "
            "Aby dodać/zmienić wpisy, edytuj ten plik w repozytorium.")

