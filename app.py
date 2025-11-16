import streamlit as st
import requests
import json
from groq import Groq

# Load Groq API Key
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# ------------------------------
# EMBEDDINGS via Groq
# ------------------------------
def compute_embeddings(texts):
    embeddings = []
    for t in texts:
        response = client.embeddings.create(
            model="nomic-embed-text",
            input=t
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

# ------------------------------
# LLM Response (using Groq)
# ------------------------------
def ask_llm(prompt):
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return completion.choices[0].message["content"]

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🧠 Silnik Wiedzy — Groq Edition 🚀")

st.write("Embeddings + LLM działają teraz **w 100% na darmowym Groq API**.")

# Example documents
DOCUMENT_TEXTS = [
    "Python jest językiem programowania używanym do analizy danych.",
    "Streamlit to framework do budowy aplikacji webowych w Pythonie.",
    "Groq oferuje bardzo szybkie darmowe modele AI dla programistów.",
]

# Cache embeddings
@st.cache_data
def load_document_embeddings():
    return compute_embeddings(DOCUMENT_TEXTS)

DOCUMENT_EMB = load_document_embeddings()

# ------------------------------
# Simple semantic search
# ------------------------------
import numpy as np

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query):
    query_emb = compute_embeddings([query])[0]
    sims = [cosine_similarity(query_emb, emb) for emb in DOCUMENT_EMB]
    best = np.argmax(sims)
    return DOCUMENT_TEXTS[best], sims[best]

# ------------------------------
# UI Input
# ------------------------------
query = st.text_input("Zadaj pytanie:")

if query:
    with st.spinner("Szukam..."):
        best_doc, score = search(query)
        st.subheader("Najbardziej pasujący dokument:")
        st.write(best_doc)

        # Ask LLM to answer using the found context
        final_prompt = f"""
Użyj poniższego fragmentu wiedzy aby odpowiedzieć na pytanie użytkownika.

Pytanie:
{query}

Kontekst:
{best_doc}

Odpowiedź:
"""

        answer = ask_llm(final_prompt)
        st.subheader("Odpowiedź modelu:")
        st.write(answer)
