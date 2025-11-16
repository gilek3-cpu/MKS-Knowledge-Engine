# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

import streamlit as st
import numpy as np
from openai import OpenAI

# --- OpenAI klient ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Dokumenty bazowe ---
DOCUMENT_TEXTS = [
    "Procedura reklamacji: klient zgłasza problem przez formularz online.",
    "Harmonogram pracy magazynu: poniedziałek–piątek 08:00–16:00.",
    "Zasady zwrotu towaru: do 14 dni od daty zakupu, wymagany paragon.",
    "Instrukcja obsługi systemu MKS – logowanie, panel klienta, faktury."
]

# --- Funkcja generująca embeddingi ---
@st.cache_data(show_spinner=False)
def compute_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[str(t) for t in texts]
    )
    return np.array([item.embedding for item in response.data])

# Wygeneruj embeddingi dokumentów JEDEN RAZ
DOCUMENT_EMB = compute_embeddings(DOCUMENT_TEXTS)

# --- UI ---
st.title("🧠 Silnik Wiedzy MKS – wyszukiwarka semantyczna")

query = st.text_input("Zadaj pytanie:", placeholder="np. 'Jak zgłosić reklamację?'")

if st.button("Szukaj") and query:
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding

    # Oblicz podobieństwa (cosine similarity)
    similarities = np.dot(DOCUMENT_EMB, q_emb)

    best_idx = int(np.argmax(similarities))
    best_doc = DOCUMENT_TEXTS[best_idx]

    st.subheader("🔍 Najbardziej pasujący dokument:")
    st.write(best_doc)

    st.caption(f"Podobieństwo: {similarities[best_idx]:.4f}")
