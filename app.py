import streamlit as st
import numpy as np
import json
import pandas as pd
# Importy dla Groq (LLM)
from groq import Groq
# Importy dla OpenAI (Embeddingi)
from openai import OpenAI
from openai import APIError
# Import dla podobieństwa kosinusowego
from sklearn.metrics.pairwise import cosine_similarity 

# --- KONFIGURACJA KLUCZY I INICJALIZACJA ---

st.set_page_config(layout="centered", page_title="Silnik Wiedzy RAG")

# 1. Sprawdzamy klucz Groq (dla LLM)
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Błąd konfiguracji: Brak klucza 'GROQ_API_KEY' w Streamlit Secrets. Jest wymagany dla LLM (Llama 3).")
    st.stop() 

# 2. Sprawdzamy i inicjalizujemy klienta OpenAI (dla Embeddingów)
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    if not OPENAI_API_KEY:
        st.error("Błąd: Wymagana wartość 'OPENAI_API_KEY' w Streamlit Secrets. Używamy go do wektoryzacji (Embeddingów).")
        st.stop()
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except KeyError:
    st.error("Błąd: Brak klucza 'OPENAI_API_KEY' w Streamlit Secrets. Jest WYMAGANY dla embeddingów.")
    st.stop()

# 3. Inicjalizacja klienta Groq (dla LLM)
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
        # Pobieranie wektorów i konwersja do numpy array
        embeddings = np.array([data.embedding for data in response.data])
        return embeddings

    except APIError as e:
        # Obsługa błędów autoryzacji/Quota
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
    Generuje odpowiedź LLM na podstawie promptu, używając modelu Llama 3 70B (Groq).
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
    """Wczytuje dane z knowledge.csv i łączy je w jeden ciąg tekstowy dla każdego dokumentu."""
    try:
        # Wczytywanie pliku CSV
        df = pd.read_csv("knowledge.csv")
        
        # Zakładamy, że kolumny to 'Opis' i 'Źródło'
        df.columns = ['Opis', 'Kategoria'] # Tymczasowa zmiana nazwy dla spójności
        
        # Łączymy kolumny, tworząc ustrukturyzowany dokument tekstowy
        document_texts = [
            f"Kategoria: {row['Kategoria']}. Opis: {row['Opis']}" 
            for index, row in df.iterrows()
        ]
        return document_texts
    except FileNotFoundError:
        st.error("Błąd: Nie znaleziono pliku 'knowledge.csv'. Upewnij się, że znajduje się w tym samym katalogu co aplikacja.")
        # W przypadku błędu zatrzymujemy aplikację, ponieważ ten plik nie ma fallbacku
        st.stop() 
    except Exception as e:
        st.error(f"Błąd ładowania knowledge.csv: {e}")
        st.stop()
        return []

# ------------------------------
# Simple semantic search (Cosine Similarity)
# ------------------------------
def search(query):
    """Wyszukuje najbardziej podobny dokument do zapytania za pomocą Podobieństwa Kosinusowego."""
    # 1. Generowanie embeddingu dla zapytania
    try:
        query_emb_list = compute_embeddings([query])
    except RuntimeError:
        return "Błąd generowania wektora zapytania.", 0.0 
        
    if query_emb_list.size == 0:
        return "Błąd generowania wektora zapytania.", 0.0

    query_emb = query_emb_list[0]
    
    # 2. Obliczanie podobieństwa
    # Używamy zaimplementowanej funkcji cosine_similarity z scikit-learn
    # Reshape jest konieczny, bo cosine_similarity oczekuje 2D tablic
    document_embeddings_np = DOCUMENT_EMB.astype(np.float64) # Upewnienie się co do typu
    similarities = cosine_similarity(query_emb.reshape(1, -1), document_embeddings_np)
    best = np.argmax(similarities)
    
    return DOCUMENT_TEXTS[best], similarities[0, best]

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🧠 Silnik Wiedzy (RAG) — Stabilna Edycja 🚀")

st.markdown("LLM (Llama 3 70B) działa na Groq. Embeddingi działają na **stabilnym API OpenAI**.")
st.markdown("Wersja wymaga pliku **`knowledge.csv`** do załadowania bazy wiedzy.")
st.markdown("---")


# 1. PRZYGOTOWANIE BAZY WIEDZY
DOCUMENT_TEXTS = load_and_prepare_data()

# Ładowanie i buforowanie embeddingów (z zabezpieczeniem)
@st.cache_resource
def load_document_embeddings(doc_texts):
    """Wczytuje embeddingi i zapewnia, że aplikacja się nie uruchomi, jeśli to się nie powiedzie."""
    if not doc_texts:
        return np.array([]) # Zwracamy pustą tablicę numpy

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
query = st.text_input("Zadaj pytanie (np. Czym jest RAG i dlaczego wymaga dwóch kluczy API?):")

if query:
    if DOCUMENT_EMB.size == 0:
        st.warning("Nie można wykonać wyszukiwania, ponieważ baza wiedzy jest pusta lub wystąpił błąd ładowania.")
    else:
        with st.spinner("Szukam kontekstu i generuję odpowiedź..."):
            
            # Wyszukiwanie semantyczne
            best_doc, score = search(query)

            st.markdown("### 🔎 Znaleziony Kontekst (RAG Retrieval)")
            st.write(f"**Podobieństwo (Cosine Score):** {score:.4f}")
            st.code(best_doc, language='text') 
    
            # Tworzenie promptu RAG
            final_prompt = f"""
            Jesteś ekspertem technicznym. Użyj **wyłącznie** poniższego fragmentu wiedzy, 
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


Teraz masz drugi plik, **`rag_engine_app.py`**, również osadzony bezpośrednio w czacie. Pamiętaj, że ten plik wymaga również pliku **`knowledge.csv`** w tym samym katalogu, aby móc działać poprawnie.

Jeśli potrzebujesz pliku `knowledge.csv` (zakładając, że go nie masz), oto jego treść:


http://googleusercontent.com/immersive_entry_chip/0
