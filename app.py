import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# ===========================
#      USTAWIENIA
# ===========================

st.set_page_config(page_title="Silnik wiedzy MKS", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ===========================
#      ŁADOWANIE DANYCH
# ===========================

@st.cache_data
def load_data():
    df = pd.read_csv("knowledge.csv")
    df["id"] = df.index
    return df

df = load_data()

# ===========================
#      TWORZENIE EMBEDDINGÓW
# ===========================

@st.cache_resource
def compute_embeddings(texts):
    """
    Generuje embeddingi dla listy tekstów.
    """
    batch = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    vectors = np.array([item.embedding for item in batch.data])
    return vectors

# Tekst do porównania = połączenie kategorii + tagów + treści
combined_texts = (
    df["category"].fillna("") + " | " +
    df["tags"].fillna("") + " | " +
    df["content"].fillna("")
).tolist()

emb_matrix = compute_embeddings(combined_texts)

# ===========================
#      WYSZUKIWANIE
# ===========================

def semantic_search(query, top_k=5):
    query_emb = compute_embeddings([query])[0]
    similarities = cosine_similarity([query_emb], emb_matrix)[0]

    idx = similarities.argsort()[::-1][:top_k]

    results = []
    for i in idx:
        results.append({
            "score": float(similarities[i]),
            "category": df.iloc[i]["category"],
            "tags": df.iloc[i]["tags"],
            "content": df.iloc[i]["content"]
        })
    return results

# ===========================
#      INTERFEJS STREAMLIT
# ===========================

st.title("🔎 Silnik Wiedzy MKS – wyszukiwarka semantyczna")

query = st.text_input("Wpisz czego szukasz:", "")

if query:
    results = semantic_search(query, top_k=5)

    st.subheader("Wyniki wyszukiwania:")

    for r in results:
        st.markdown(f"""
        ### 📌 {r['category']}
        **Tagi:** {r['tags']}  
        **Dopasowanie:** {round(r['score'], 3)}
        
        {r['content']}
        ---
        """)

