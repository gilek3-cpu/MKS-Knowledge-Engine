import streamlit as st
import numpy as np
import json
from groq import Groq
from openai import OpenAI
from openai import APIError
import pandas as pd

# --- KLUCZE I INICJALIZACJA ---
# Keys are retrieved from Streamlit Secrets (secrets.toml)

# Check Groq key (for LLM)
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Configuration error: Missing 'GROQ_API_KEY' in Streamlit Secrets. Required for LLM (Llama 3).")
    st.stop() 

# Check and initialize OpenAI client (for Embeddings)
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    if not OPENAI_API_KEY:
        st.error("Error: 'OPENAI_API_KEY' value is required in Streamlit Secrets. We use it for vectorization.")
        st.stop()
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except KeyError:
    st.error("Error: Missing 'OPENAI_API_KEY' in Streamlit Secrets. It is REQUIRED for embeddings.")
    st.stop()

# Initialize Groq client (for LLM)
try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Groq client initialization error: {e}")
    st.stop()


# ------------------------------
# EMBEDDINGS (Vectorization) - EXCLUSIVELY OpenAI
# ------------------------------
@st.cache_data
def compute_embeddings(texts):
    """
    Generates embeddings for a list of texts using the OpenAI model (text-embedding-3-small).
    """
    try:
        st.info("Using OpenAI (text-embedding-3-small) for embeddings...")
        # Required format is a list of strings
        response = openai_client.embeddings.create(
            model="text-embedding-3-small", 
            input=texts
        )
        # Retrieve vectors from the response
        embeddings = [data.embedding for data in response.data]
        st.success("OpenAI Embeddings success!")
        return embeddings

    except APIError as e:
        # Handle authorization/Quota errors
        st.error(f"Critical OpenAI API Error (Embeddings): {e}. Check if the OPENAI_API_KEY is correct and you have sufficient credits.")
        raise RuntimeError("Vectorization error: OpenAI key/credit verification failed.")
    except Exception as e:
        st.error(f"Unexpected error during OpenAI embeddings generation: {e}")
        raise RuntimeError("Vectorization error: Unknown error.")


# ------------------------------
# LLM Response (using Groq)
# ------------------------------
def ask_llm(prompt):
    """
    Generates an LLM response based on the prompt, using the Llama 3 70B model (fast).
    """
    try:
        completion = client.chat.completions.create(
            model="llama-3-8b-8192", # Fast Groq model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error during Groq LLM call: {e}")
        return "I apologize, an error occurred in communication with the LLM model."

# ------------------------------
# Load data from CSV
# ------------------------------
@st.cache_data
def load_and_prepare_data():
    """Loads data from knowledge.csv and combines it into a single text."""
    try:
        # Assumes knowledge.csv exists and has 'Kategoria' and 'Opis' columns
        df = pd.read_csv("knowledge.csv")
        
        # Combine 'Kategoria' and 'Opis' columns into one string for each row
        document_texts = [
            f"Kategoria: {row['Kategoria']}. Opis: {row['Opis']}" 
            for index, row in df.iterrows()
        ]
        return document_texts
    except FileNotFoundError:
        st.error("Error: File 'knowledge.csv' not found. Ensure it is in the same directory.")
        return []
    except Exception as e:
        st.error(f"Error loading knowledge.csv: {e}")
        return []

# ------------------------------
# Simple semantic search (Cosine Similarity)
# ------------------------------
def cosine_similarity(a, b):
    """Calculates the cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    # Protection against division by zero
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return np.dot(a, b) / (norm_a * norm_b)

def search(query):
    """Searches for the most similar document to the query."""
    # 1. Generate embedding for the query
    try:
        # Uses compute_embeddings (OpenAI)
        query_emb_list = compute_embeddings([query])
    except RuntimeError:
        return "Error generating query vector.", 0.0 
        
    if not query_emb_list:
        return "Error generating query vector.", 0.0

    query_emb = query_emb_list[0]
    
    # 2. Calculate similarity
    sims = [cosine_similarity(query_emb, emb) for emb in DOCUMENT_EMB]
    best = np.argmax(sims)
    
    return DOCUMENT_TEXTS[best], sims[best]

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🧠 Silnik Wiedzy (RAG) — Stabilna Edycja 🚀")
st.markdown("LLM (Llama 3 70B) działa na Groq. Embeddingi działają na **stabilnym API OpenAI**.")
st.markdown("---")


# 1. PRZYGOTOWANIE BAZY WIEDZY
DOCUMENT_TEXTS = load_and_prepare_data()

# Loading and caching embeddings
@st.cache_resource
def load_document_embeddings(doc_texts):
    """Loads embeddings and ensures the application does not run if it fails."""
    if not doc_texts:
        st.warning("Knowledge base is empty. Check knowledge.csv file.")
        return []

    st.subheader("Faza 1: Wczytywanie i wektoryzacja bazy wiedzy")
    with st.spinner("Generowanie embeddingów dla dokumentów..."):
        try:
            emb = compute_embeddings(doc_texts)
        except RuntimeError:
            st.warning("Application stopped due to OpenAI API key error. Check logs.")
            st.stop()
            
        st.success(f"Baza wiedzy (zawierająca {len(emb)} wektorów) załadowana pomyślnie!")
        return emb

# Call the loading function - if it fails, the application stops
DOCUMENT_EMB = load_document_embeddings(DOCUMENT_TEXTS)

# ------------------------------
# Phase 2: UI Input and response generation
# ------------------------------
st.subheader("Faza 2: Zapytanie do Silnika Wiedzy")
query = st.text_input("Zadaj pytanie (np. Jak ćwiczyć dynamiczną siłę?):")

if query:
    if not DOCUMENT_EMB:
        st.warning("Cannot perform search because the knowledge base is empty.")
    else:
        with st.spinner("Szukam kontekstu i generuję odpowiedź..."):
            
            # Semantic search
            best_doc, score = search(query)

            st.markdown("### 🔎 Znaleziony Kontekst (RAG Retrieval)")
            st.write(f"**Podobieństwo (Cosine Score):** {score:.4f}")
            st.code(best_doc, language='text') 
    
            # Create RAG prompt for LLM
            final_prompt = f"""
            Jesteś ekspertem technicznym i wspinaczkowym. Użyj **wyłącznie** poniższego fragmentu wiedzy, 
            aby odpowiedzieć na pytanie użytkownika. Odpowiadaj zwięźle i precyzyjnie. 
            Jeśli kontekst nie zawiera odpowiedzi, odpowiedz: 'Brak wystarczających informacji w bazie wiedzy.'.
    
            Pytanie:
            {query}
    
            Kontekst:
            {best_doc}
    
            Odpowiedź:
            """
    
            # Call LLM (Groq)
            answer = ask_llm(final_prompt)
            st.markdown("### 🤖 Odpowiedź Modelu (Llama 3 70B - Groq)")
            st.info(answer)
