import sys
sys.stdout.reconfigure(encoding="utf-8")

import streamlit as st
import numpy as np
from openai import OpenAI

# --- OpenAI client ---
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    default_headers={"Content-Type": "application/json; charset=utf-8"}
)

# --- Dokumenty bazowe ---
DOCUMENT_TEXTS = [
    "Procedura reklamacji - klient zgłasza problem przez formularz online.",
    "Harmonogram pracy magazynu: poniedziałek-piątek 08:00-16:00.",
    "Zasady zwrotu towaru - do 14 dni od daty zakupu, wymagany paragon.",
    "Instrukcja obsługi systemu MKS - logowanie, panel klienta, faktury."
]

# --- Funkcja generująca embeddingi dokumentów ---
@st.cache_data(show_spinner=False)
def compute_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([item.embedding for item in response.data])

# Generujemy embeddingi dokumentów JEDEN RAZ
DOCUMENT_EMB = compute_embeddings(DOCUMENT_TEXTS)

# --- UI ---
st.title("🔍 Silnik Wiedzy MKS – wyszukiwarka semantyczna")

query = st.text_input("Zadaj pytanie:", placeholder="np. 'Jak zgłosić reklamację?'")

if st.button("Szukaj") and query:
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding

    q_emb = np.array(q_emb)

    sims = DOCUMENT_EMB @ q_emb / (
        np.linalg.norm(DOCUMENT_EMB, axis=1) * np.linalg.norm(q_emb)
    )

    best_idx = int(np.argmax(sims))

    st.subheader("📄 Najbardziej pasująca odpowiedź:")
    st.write(DOCUMENT_TEXTS[best_idx])

    st.caption(f"Similarity score: {sims[best_idx]:.4f}")
