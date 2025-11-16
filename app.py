# -*- coding: utf-8 -*-
import os
import streamlit as st
import numpy as np
from openai import OpenAI

# ---- FIX: blokada błędu Unicode przy wysyłaniu nagłówków HTTP ----
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["OPENAI_USER_AGENT"] = "MKS-Knowledge-Engine/1.0"   # <--- KLUCZOWA LINIA

# ---- OpenAI client ----
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---- Dokumenty ----
DOCUMENT_TEXTS = [
    "Procedura reklamacji: klient zgłasza problem przez formularz online.",
    "Harmonogram pracy magazynu: poniedziałek–piątek 08:00–16:00.",
    "Zasady zwrotu towaru: do 14 dni od daty zakupu, wymagany paragon.",
    "Instrukcja obsługi systemu MKS – logowanie, panel klienta, faktury."
]

# ---- Embeddingi ----
@st.cache_data
def compute_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[str(t) for t in texts]
    )
    return np.array([item.embedding for item in response.data])

DOCUMENT_EMB = compute_embeddings(DOCUMENT_TEXTS)

# ---- UI ----
st.title("🧠 Silnik Wiedzy MKS – wyszukiwarka semantyczna")

query = st.text_input("Wpisz pytanie")

if st.button("Szukaj") and query:
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding

    similarities = np.dot(DOCUMENT_EMB, q_emb)

    best_idx = int(np.argmax(similarities))
    best_doc = DOCUMENT_TEXTS[best_idx]

    st.subheader("🔍 Najbardziej trafny dokument:")
    st.write(best_doc)

    st.caption(f"Podobieństwo: {similarities[best_idx]:.4f}")

