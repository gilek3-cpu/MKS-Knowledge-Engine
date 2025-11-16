import streamlit as st
import numpy as np
import json
import pandas as pd
from groq import Groq
from openai import OpenAI
from openai import APIError
from sklearn.metrics.pairwise import cosine_similarity 

# --- KONFIGURACJA KLUCZY I INICJALIZACJA ---

st.set_page_config(layout="centered", page_title="Silnik Wiedzy RAG")

# Sprawdzamy, czy klucze są dostępne w Streamlit Secrets
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Błąd konfiguracji: Brak klucza 'GROQ_API_KEY' w Streamlit Secrets. Jest wymagany dla LLM (Llama 3).")
    st.stop() 

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    if not OPENAI_API_KEY:
        st.error("Błąd: Wymagana wartość 'OPENAI_API_KEY' w Streamlit Secrets. Używamy go do wektoryzacji (Embeddingów).")
        st.stop()
    # Inicjalizacja klienta OpenAI (do embeddingów)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except KeyError:
    st.error("Błąd: Brak klucza 'OPENAI_API_KEY' w Streamlit Secrets. Jest WYMAGANY dla embeddingów.")
    st.stop()

# Inicjalizacja klienta Groq (dla LLM)
try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Błąd inicjalizacji klienta Groq: {e}")
    st.stop()


# ------------------------------
# EMBEDDINGS (Wektoryzacja) - WYŁĄCZNIE OpenAI
# ------------------------------
@st.cache_data(show_spinner=False)
def compute_embeddings(texts):
    """
    Generuje embeddingi dla listy tekstów używając modelu OpenAI
    (text-embedding-3-small).
    """
    try:
        # st.info("Using OpenAI (text-embedding-3-small) for embeddings...")
        response = openai_client.embeddings.create(
            model="text-embedding-3-small", 
            input=texts
        )
        # Pobieranie wektorów
        embeddings = [data.embedding for data in response.data]
        return embeddings

    except APIError as e:
        st.error(f"Krytyczny błąd API OpenAI (Embeddingi): {e}. Sprawdź, czy klucz OPENAI_API_KEY jest poprawny i czy masz wystarczającą ilość kredytów.")
        raise RuntimeError("Błąd wektoryzacji: Weryfikacja klucza OpenAI/kredytów.")
    except Exception as e:
        st.error(f"Nieoczekiwany błąd podczas generowania embeddingów OpenAI: {e}")
        raise RuntimeError("Błąd wektoryzacji: Nieznany błąd.")


# ------------------------------
# LLM Response (using Groq)
# ------------------------------
def ask_llm(prompt):
    """
    Generuje odpowiedź LLM na podstawie promptu, używając modelu Llama 3 70B (szybki).
    """
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192", # Szybki model od Groq
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Błąd podczas wywołania LLM Groq: {e}")
        return "Przepraszam, wystąpił błąd w komunikacji z modelem LLM."

# ------------------------------
# Load data from CSV
# ------------------------------
@st.cache_data
def load_and_prepare_data():
    """Loads data from knowledge.csv and combines it into a single text."""
    try:
        # Assumes knowledge.csv exists and has 'Źródło' and 'Opis' columns
        df = pd.read_csv("knowledge.csv")
        
        # Zakładamy, że kolumny to 'Opis' i 'Źródło' i zmieniamy nazwę na Kategoria
        df.columns = ['Opis', 'Kategoria'] 
        
        # Łączymy kolumny 'Kategoria' i 'Opis' w jeden ciąg dla każdego wiersza
        document_texts = [
            f"Kategoria: {row['Kategoria']}. Opis: {row['Opis']}" 
            for index, row in df.iterrows()
        ]
        return document_texts
    except FileNotFoundError:
        # W przypadku braku pliku używamy awaryjnej, wbudowanej bazy wiedzy
        st.warning("Nie znaleziono pliku 'knowledge.csv'. Używam wbudowanej, awaryjnej bazy wiedzy.")
        return [
            "Python jest językiem programowania używanym do analizy danych, uczenia maszynowego i tworzenia aplikacji webowych.",
            "Streamlit to darmowy framework do budowy interaktywnych aplikacji webowych w Pythonie bez znajomości HTML/CSS/JS.",
            "Groq oferuje bardzo szybkie i darmowe modele AI dla programistów, działające na specjalistycznych akceleratorach LPU (Language Processing Unit).",
            "Podobieństwo Kosinusowe mierzy kąt między dwoma wektorami w przestrzeni, określając podobieństwo semantyczne.",
            "Do wspinaczki sportowej niezbędna jest dynamika siły, którą można ćwiczyć poprzez Campus Board, skoki na chwytach oraz trening pliometryczny.",
        ]
    except Exception as e:
        st.error(f"Błąd ładowania knowledge.csv: {e}. Używam awaryjnej bazy wiedzy.")
        return [
            "Wystąpił błąd podczas parsowania danych. Skupmy się na Groq i Embeddingach.",
            "System RAG składa się z dwóch głównych etapów: Retrieval (wyszukiwanie kontekstu) i Generation (generowanie odpowiedzi).",
        ]

# ------------------------------
# Simple semantic search (Cosine Similarity)
# ------------------------------
def search(query):
    """Wyszukuje najbardziej podobny dokument do zapytania."""
    # 1. Generowanie embeddingu dla zapytania - Używa compute_embeddings (OpenAI)
    try:
        # Pamiętaj, że compute_embeddings rzuca Runtime Error, jeśli klucz OpenAI jest zły
        query_emb_list = compute_embeddings([query])
    except RuntimeError:
        return "Błąd generowania wektora zapytania.", 0.0 
        
    if not query_emb_list:
        return "Błąd generowania wektora zapytania.", 0.0

    query_emb = query_emb_list[0]
    
    # 2. Obliczanie podobieństwa
    # Funkcja z scikit-learn jest używana do szybkiego obliczania
    # Wymagane jest przekształcenie do numpy array i dopasowanie kształtów
    doc_embeddings_array = np.array(DOCUMENT_EMB).astype(np.float64)
    query_emb_array = np.array(query_emb).reshape(1, -1)
    
    similarities = cosine_similarity(query_emb_array, doc_embeddings_array)
    best = np.argmax(similarities)
    
    return DOCUMENT_TEXTS[best], similarities[0, best]

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🧠 Silnik Wiedzy (RAG) — Stabilna Edycja 🚀")

st.markdown("LLM (Llama 3 70B) działa na Groq. Embeddingi (wektoryzacja) działają na **stabilnym API OpenAI**.")
st.markdown("---")


# 1. PRZYGOTOWANIE BAZY WIEDZY
DOCUMENT_TEXTS = load_and_prepare_data()

# Ładowanie i buforowanie embeddingów (z zabezpieczeniem)
@st.cache_resource
def load_document_embeddings(doc_texts):
    """Wczytuje embeddingi i zapewnia, że aplikacja się nie uruchomi, jeśli to się nie powiedzie."""
    if not doc_texts:
        return []

    st.subheader("Faza 1: Wczytywanie i wektoryzacja bazy wiedzy")
    with st.spinner(f"Generowanie embeddingów dla {len(doc_texts)} dokumentów..."):
        try:
            emb = compute_embeddings(doc_texts)
        except RuntimeError:
            st.warning("Aplikacja została zatrzymana z powodu błędu klucza API OpenAI. Sprawdź logi.")
            st.stop()
            
        st.success(f"Baza wiedzy (zawierająca {len(emb)} wektorów) załadowana pomyślnie!")
        return emb

# Wywołanie funkcji ładowania - jeśli zawiedzie, aplikacja się zatrzyma
DOCUMENT_EMB = load_document_embeddings(DOCUMENT_TEXTS)

# ------------------------------
# Phase 2: UI Input and response generation
# ------------------------------
st.subheader("Faza 2: Zapytanie do Silnika Wiedzy")
query = st.text_input("Zadaj pytanie (np. Czym jest Streamlit i do czego służy?):")

if query:
    if not DOCUMENT_EMB:
        st.warning("Nie można wykonać wyszukiwania, ponieważ baza wiedzy jest pusta.")
    else:
        with st.spinner("Szukam kontekstu i generuję odpowiedź..."):
            
            # Wyszukiwanie semantyczne
            best_doc, score = search(query)

            st.markdown("### 🔎 Znaleziony Kontekst (RAG Retrieval)")
            st.write(f"**Podobieństwo (Cosine Score):** {score:.4f}")
            st.code(best_doc, language='text') 
    
            # Tworzenie promptu RAG
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
    
            # Wywołanie LLM (Groq)
            answer = ask_llm(final_prompt)
            st.markdown("### 🤖 Odpowiedź Modelu (Llama 3 70B - Groq)")
            st.info(answer)
