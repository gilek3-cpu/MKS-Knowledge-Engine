import streamlit as st
import pandas as pd
from openai import OpenAI

st.set_page_config(page_title="Silnik Wiedzy MKS", page_icon="🧠")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

@st.cache_resource
def load_docs():
    # Wymuszamy UTF-8 i ignorujemy ewentualne złe znaki
    return pd.read_csv("knowledge.csv", encoding="utf-8", on_bad_lines="skip")

@st.cache_resource
def embed_texts(texts):
    embeds = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return [e.embedding for e in embeds.data]

# -----------------------------
# ŁADOWANIE BAZY
# -----------------------------
docs = load_docs()
texts = docs["content"].astype(str).tolist()

# embeddings dokumentów
DOCUMENT_EMB = embed_texts(texts)

# -----------------------------
# INTERFEJS
# -----------------------------
st.title("🔍 Silnik Wiedzy MKS")

user_input = st.text_input("Zadaj pytanie:")

if user_input:
    # embedding pytania
    query_emb = embed_texts([user_input])[0]

    # liczymy podobieństwo kosinusowe
    import numpy as np

    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sims = [cos_sim(query_emb, e) for e in DOCUMENT_EMB]

    best_idx = int(np.argmax(sims))
    best_row = docs.iloc[best_idx]

    # generujemy odpowiedź
    prompt = f"""
Użyj tej wiedzy:
Kategoriа: {best_row['category']}
Tagi: {best_row['tags']}
Treść: {best_row['content']}

Pytanie użytkownika: {user_input}

Odpowiedz krótko i konkretnie.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    st.markdown("### 🧠 Odpowiedź:")
    st.write(response.choices[0].message.content)

    st.markdown("---")
    st.markdown("### 📚 Znaleziono w bazie:")
    st.write(best_row["content"])
