# -*- coding: utf-8 -*-
import os
import numpy as np
import streamlit as st
import httpx
from openai import OpenAI

# ---- FIX: całkowity override transportu HTTP ----
class AsciiTransport(httpx.HTTPTransport):
    def handle_request(self, request):
        clean_headers = {}
        for k, v in request.headers.items():
            try:
                clean_headers[k] = v.encode("ascii", "ignore").decode("ascii")
            except:
                clean_headers[k] = ""
        request.headers = clean_headers
        return super().handle_request(request)

http_client = httpx.Client(
    transport=AsciiTransport(),
    headers={"User-Agent": "MKS-Engine"},
    timeout=30.0
)

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    http_client=http_client,   # ← to jest klucz
)

# ---- Dokumenty ----
DOCUMENT_TEXTS = [
    "Procedura reklamacji: klient zgłasza problem przez formularz online.",
    "Harmonogram pracy magazynu: poniedziałek–piątek 08:00–16:00.",
    "Zasady zwrotu towaru: do 14 dni od daty zakupu, wymagany paragon.",
    "Instrukcja obsługi systemu MKS – logowanie, panel klienta, faktury."
]

# ---- Cache ----
@st.cache_data
def compute_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[str(t) for t in texts]
    )
    return np.array([x.embedding for x in response.data])

DOCUMENT_EMB = compute_embeddings(DOCUMENT_TEXTS)

# ---- UI ----
st.title("🧠 Silnik Wiedzy MKS – wyszukiwarka semantyczna")

query = st.text_input("Wpisz pytanie")

if st.button("Szukaj") and query:
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding

    sims = np.dot(DOCUMENT_EMB, q_emb)
    idx = int(np.argmax(sims))

    st.subheader("🔍 Najbardziej trafny dokument:")
    st.write(DOCUMENT_TEXTS[idx])

    st.caption(f"Podobieństwo: {sims[idx]:.4f}")
