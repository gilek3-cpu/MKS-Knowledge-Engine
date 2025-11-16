# -*- coding: utf-8 -*-
import json
import numpy as np
import streamlit as st
import requests

# --- Konfiguracja ---
# Upewnij się, że masz w Secrets: OPENAI_API_KEY
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)
if not OPENAI_API_KEY:
    st.error("Brakuje OPENAI_API_KEY w secrets (Settings -> Secrets).")
    st.stop()

EMBEDDING_URL = "https://api.openai.com/v1/embeddings"
HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
    "User-Agent": "MKS-Knowledge-Engine/1.0"
}

# --- Dokumenty bazowe (przykład) ---
DOCUMENT_TEXTS = [
    "Procedura reklamacji: klient zgłasza problem przez formularz online.",
    "Harmonogram pracy magazynu: poniedziałek–piątek 08:00–16:00.",
    "Zasady zwrotu towaru: do 14 dni od daty zakupu, wymagany paragon.",
    "Instrukcja obsługi systemu MKS – logowanie, panel klienta, faktury."
]

# --- Funkcja do pobierania embeddingów przez REST (requests) ---
def get_embeddings_via_requests(texts, model="text-embedding-3-small"):
    payload = {"model": model, "input": list(map(str, texts))}
    # requests zajmie się kodowaniem JSON jako UTF-8
    resp = requests.post(EMBEDDING_URL, headers=HEADERS, json=payload, timeout=30)
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        st.error(f"Błąd HTTP przy wywołaniu OpenAI: {e}\nKod odpowiedzi: {resp.status_code}")
        # pokazuj surową odpowiedź dla debugowania
        st.write(resp.text)
        raise
    data = resp.json()
    # data["data"] to lista elementów z polem "embedding"
    embeddings = [item["embedding"] for item in data["data"]]
    return np.array(embeddings, dtype=np.float32)

# --- Cache'ujemy embeddingi (raz) ---
@st.cache_data(show_spinner=False)
def compute_document_embeddings():
    return get_embeddings_via_requests(DOCUMENT_TEXTS)

DOCUMENT_EMB = compute_document_embeddings()

# --- UI ---
st.title("🔎 Silnik Wiedzy MKS — wyszukiwarka semantyczna")

query = st.text_input("Zadaj pytanie", placeholder="np. Jak zgłosić reklamację?")

if st.button("Szukaj") and query:
    with st.spinner("Generuję embedding zapytania..."):
        q_emb_arr = get_embeddings_via_requests([query])  # zwraca (1, dim)
        q_emb = q_emb_arr[0]

    # Liczymy kosinusowe podobieństwo (przyspieszony sposób)
    # normalization
    doc_norms = np.linalg.norm(DOCUMENT_EMB, axis=1)
    q_norm = np.linalg.norm(q_emb)
    # unikamy dzielenia przez zero
    if q_norm == 0 or np.any(doc_norms == 0):
        st.error("Błąd: wektor o zerowej długości.")
    else:
        sims = (DOCUMENT_EMB @ q_emb) / (doc_norms * q_norm)
        idx = int(np.argmax(sims))
        st.subheader("Najbardziej pasujący dokument:")
        st.write(DOCUMENT_TEXTS[idx])
        st.caption(f"Similarity score: {sims[idx]:.4f}")
