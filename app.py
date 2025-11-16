import streamlit as st
import pandas as pd
import numpy as np
from groq import Groq
import os

# Load API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ------------------------------
# Compute Embeddings (Groq)
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
# LLM Response (Groq)
# ------------------------------
def ask_llm(prompt):
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return completion.choices[0].message.content

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🧠 Silnik Wiedzy – Groq Edition 🚀")
st.write("Embeddings + LLM działają teraz **w 100% na darmowym Groq API**.")

# Load your CSV knowledge
df = pd.read_csv("knowledge.csv")
DOCUMENT_TEXTS = df["text"].tolist()

# Cache embeddings
@st.cache_data
def load_document_embeddings():
    return compute_embeddings(DOCUMENT_TEXTS)

DOCUMENT_EMB = load_document_embeddings()

# ------------------------------
# Semantic Search
# ------------------------------
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
        st.subheader("Najbardziej pasujący fragment wiedzy:")
        st.write(best_doc)

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
