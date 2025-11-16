import streamlit as st
import pandas as pd
from openai import OpenAI
import numpy as np

st.set_page_config(page_title="Silnik Wiedzy MKS", page_icon="🔍")

st.title("🔍 Silnik Wiedzy MKS")

# -----------------------------
# Ładowanie dokumentów
# -----------------------------
@st.cache_data
def load_docs():
    df = pd.read_csv("knowledge.csv")
    df["full_text"] = df["category"] + " | " + df["tags"] + " | " + df["content"]
    return df

docs = load_docs()

# -----------------------------
# OpenAI klient
# -----------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# Embeddingi
# -----------------------------
@st.cache_resource
def embed_documents(texts):
    emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return np.array([e.embedding for e in emb.data])

DOCUMENT_EMB = embed_documents(docs["full_text"].tolist())

# -----------------------------
# Szukanie podobieństwa
# -----------------------------
def semantic_search(query, top_k=5):
    q_emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=query
    ).data[0].embedding

    sims = DOCUMENT_EMB @ np.array(q_emb)

    idx = sims.argsort()[::-1][:top_k]
    return docs.iloc[idx], sims[idx]


# -----------------------------
# ChatGPT odpowiedź
# -----------------------------
def ask_gpt(context, question):
    prompt = f"""
Użyj poniższego kontekstu i odpowiedz zwięźle i konkretnie:

KONTEKST:
{context}

PYTANIE:
{question}
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message["content"]


# -----------------------------
# UI
# -----------------------------
user_query = st.text_input("Zadaj pytanie:")

if user_query:
    st.subheader("🔎 Najbardziej pasujące fragmenty:")

    results, scores = semantic_search(user_query)

    context_block = ""

    for i, row in results.iterrows():
        st.markdown(f"**• {row['category']}** — _{row['tags']}_\n\n{row['content']}")
        st.markdown("---")
        context_block += row["content"] + "\n"

    st.subheader("💬 Odpowiedź:")
    answer = ask_gpt(context_block, user_query)
    st.write(answer)
