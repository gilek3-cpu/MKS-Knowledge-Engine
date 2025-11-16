import streamlit as st
import pandas as pd
from openai import OpenAI
import numpy as np

# --- OpenAI client with UTF-8 fix ---
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    default_headers={"Content-Type": "application/json; charset=utf-8"}
)

DOCUMENT_TEXTS = [
    "Procedura reklamacji – klient zgłasza problem przez formularz online.",
    "Harmonogram pracy magazynu: poniedziałek–piątek 8:00–16:00.",
    "Zasady zwrotów towaru – do 14 dni od daty zakupu, wymagany paragon.",
    "Instrukcja obsługi systemu MKS – logowanie, panel klienta, faktury.",
]

@st.cache_data
def compute_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([item.embedding for item in response.data])

DOCUMENT_EMB = compute_embeddings(DOCUMENT_TEXTS)

st.title("🔎 Silnik Wiedzy MKS – wyszukiwarka semantyczna")

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

    st.subheader("📌 Najlepsza odpowiedź:")
    st.write(DOCUMENT_TEXTS[best_idx])

    st.caption(f"Podobieństwo: {sims[best_idx]:.3f}")
