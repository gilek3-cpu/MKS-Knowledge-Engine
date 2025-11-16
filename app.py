import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
from openai import APIError

# --- 1. KONFIGURACJA API (Pobieranie kluczy z Streamlit Secrets) ---

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Brak klucza GROQ_API_KEY w Streamlit Secrets.")
    st.stop()
    
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Brak klucza OPENAI_API_KEY w Streamlit Secrets.")
    st.stop()

# Inicjalizacja klientów
client_groq = Groq(api_key=GROQ_API_KEY)
client_openai = OpenAI(api_key=OPENAI_API_KEY)


# --- 2. FUNKCJE RAG (Embeddings i Czat) ---

@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    """Loads knowledge.csv and prepares data lists."""
    try:
        df = pd.read_csv('knowledge.csv', encoding='utf-8')
        if 'Opis' not in df.columns or 'Źródło' not in df.columns:
            st.error("Plik 'knowledge.csv' musi zawierać kolumny: 'Opis' i 'Źródło'.")
            st.stop()

        document_texts = df['Opis'].tolist()
        document_sources = df['Źródło'].tolist()

        if not document_texts:
             st.warning("Baza wiedzy (knowledge.csv) jest pusta.")
             return [], []

        return document_texts, document_sources
    
    except Exception as e:
        st.error(f"Krytyczny błąd ładowania knowledge.csv: {e}")
        st.stop()


@st.cache_data(show_spinner=False)
def compute_document_embeddings(texts):
    """Generates embeddings for texts using OpenAI."""
    if not texts: return np.array([])
        
    try:
        response = client_openai.embeddings.create( 
            model="text-embedding-3-small", 
            input=texts
        )
        embeddings = np.array([data.embedding for data in response.data])
        return embeddings
        
    except APIError as e:
        st.error(f"Krytyczny błąd API OpenAI Embeddings. Sprawdź klucz. Szczegóły: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Nieznany błąd podczas wektoryzacji (OpenAI): {e}")
        st.stop()


def ask_llama(prompt):
    """Generates the LLM response using Llama 3 8B (Groq)."""
    try:
        completion = client_groq.chat.completions.create(
            model="llama-3-8b-8192", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Błąd podczas wywołania LLM Groq: {e}")
        return "Przepraszam, wystąpił błąd w komunikacji z modelem LLM."


def search_best_context(query_embedding, document_embeddings, document_texts, document_sources):
    """Finds the best matching context in the knowledge base."""
    # Reshape the query embedding for cosine_similarity
    similarities = cosine_similarity(np.array(query_embedding).reshape(1, -1), document_embeddings)
    
    best_index = np.argmax(similarities)
    best_score = similarities[0, best_index]
    
    return document_texts[best_index], best_score, document_sources[best_index]


# --- 3. GŁÓWNA STRUKTURA APLIKACJI STREAMLIT ---

st.set_page_config(layout="centered")
st.title("🚀 Silnik Wiedzy (RAG) – Stabilna Edycja")

# Faza 1: Wczytywanie bazy wiedzy i generowanie wektorów
DOCUMENT_TEXTS, DOCUMENT_SOURCES = load_and_prepare_data()
DOCUMENT_EMBEDDINGS = compute_document_embeddings(DOCUMENT_TEXTS)
is_data_ready = DOCUMENT_EMBEDDINGS.size > 0


if is_data_ready:
    st.success("Baza wiedzy i wektoryzacja gotowe do użycia.")
else:
    st.warning("Aplikacja działa, ale baza wiedzy jest pusta. Uzupełnij plik knowledge.csv.")

# Faza 2: Zapytanie do Silnika Wiedzy (UI)
st.subheader("Faza 2: Zapytaj Silnik Wiedzy")

query = st.text_input(
    "Zadaj pytanie (np. Jak ćwiczyć dynamikę siły?)", 
    key="query_input", 
    disabled=not is_data_ready
)


if query and is_data_ready:
    with st.spinner("Przetwarzanie zapytania..."):
        
        # A. Generowanie embeddingu dla zapytania
        try:
            query_embedding = client_openai.embeddings.create(
                model="text-embedding-3-small", 
                input=[query]
            ).data[0].embedding
        except Exception as e:
            st.error(f"Błąd generowania wektora dla zapytania: {e}")
            st.stop()
            
        # B. Wyszukiwanie kontekstu
        best_doc, score, source = search_best_context(
            query_embedding, 
            DOCUMENT_EMBEDDINGS, 
            DOCUMENT_TEXTS, 
            DOCUMENT_SOURCES
        )
        
        # C. Budowanie finalnego prompta dla LLM
        final_prompt = f"""
        Jesteś ekspertem technicznym. Użyj "Kontekstu" poniżej, aby odpowiedzieć na "Pytanie".
        Nie dodawaj informacji, których nie ma w kontekście. 
        Jeśli kontekst nie zawiera odpowiedzi, odpowiedz: "Brak wystarczających informacji w bazie wiedzy."
        
        Pytanie: {query}
        Kontekst: {best_doc}
        Odpowiedź: 
        """
        
        # D. Generowanie odpowiedzi LLM (z Groq)
        answer = ask_llama(final_prompt)
        
        # E. Wyświetlanie wyników
        st.markdown("---")
        st.subheader("🤖 Odpowiedź Modelu (Llama 3 8B)")
        st.info(answer)
        
        # F. Wyświetlanie kontekstu i źródła
        st.subheader("🔍 Użyty Kontekst (RAG)")
        st.code(f"{best_doc}")
        st.markdown(f"**Źródło:** {source} | **Podobieństwo:** {score:.4f}")
        
        st.markdown("---")
