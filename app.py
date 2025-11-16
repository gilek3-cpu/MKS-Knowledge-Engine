import streamlit as st
import numpy as np
import json
from groq import Groq
from openai import OpenAI
from openai import APIError
from sklearn.metrics.pairwise import cosine_similarity 

# --- KONFIGURACJA KLUCZY I INICJALIZACJA ---
# Wymaga kluczy: GROQ_API_KEY i OPENAI_API_KEY w Streamlit Secrets

st.set_page_config(layout="centered", page_title="Silnik Wiedzy RAG")

try:
    # Weryfikacja klucza GROQ dla LLM
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Błąd: Brak klucza 'GROQ_API_KEY' w Streamlit Secrets. Jest wymagany dla LLM (Llama 3).")
    st.stop() 

try:
    # Weryfikacja klucza OPENAI dla embeddingów
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    if not OPENAI_API_KEY:
        st.error("Błąd: Wymagana wartość 'OPENAI_API_KEY' w Streamlit Secrets. Używamy go do wektoryzacji (Embeddingów).")
        st.stop()
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except KeyError:
    st.error("Błąd: Brak klucza 'OPENAI_API_KEY' w Streamlit Secrets. Jest WYMAGANY dla embeddingów.")
    st.stop()

try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Błąd inicjalizacji klienta Groq: {e}")
    st.stop()


# ------------------------------
# WBUDOWANA BAZA WIEDZY (Dla uproszczonego testu)
# ------------------------------
# Ta lista tekstów służy jako zastępcza baza wiedzy RAG.
DOCUMENT_TEXTS = [
    "Python jest językiem programowania używanym do analizy danych, uczenia maszynowego i tworzenia aplikacji webowych.",
    "Streamlit to darmowy framework do budowy interaktywnych aplikacji webowych w Pythonie.",
    "Groq oferuje bardzo szybkie modele AI dla programistów, działające na akceleratorach LPU (Language Processing Unit).",
    "Podobieństwo Kosinusowe (Cosine Similarity) mierzy kąt między dwoma wektorami w przestrzeni, określając podobieństwo semantyczne.",
    "RAG (Retrieval-Augmented Generation) to architektura AI, która wykorzystuje bazę wiedzy (retrieval) do ulepszania odpowiedzi LLM (generation).",
    "Do wspinaczki sportowej niezbędna jest lina dynamiczna, uprząż i ekspresy. Ważna jest technika wiązania ósemki.",
    "Wspinaczka tradycyjna wymaga umiejętności osadzania własnej asekuracji, np. kości i friendów. Jest to bardziej wymagające psychicznie.",
]


# ------------------------------
# EMBEDDINGS (Wektoryzacja) - WYŁĄCZNIE OpenAI
# ------------------------------
@st.cache_data(show_spinner=False)
def compute_embeddings(texts):
    """Generuje embeddingi dla listy tekstów używając modelu OpenAI (text-embedding-3-small)."""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small", 
            input=texts
        )
        # Ekstrakcja wektorów z odpowiedzi API
        embeddings = np.array([data.embedding for data in response.data])
        return embeddings

    except APIError as e:
        st.error(f"Krytyczny błąd API OpenAI (Embeddingi): {e}. Sprawdź, czy klucz OPENAI_API_KEY jest poprawny.")
        # Podnoszenie wyjątku w celu zatrzymania aplikacji Streamlit
        raise RuntimeError("Błąd wektoryzacji: Weryfikacja klucza OpenAI/kredytów.")
    except Exception as e:
        st.error(f"Nieoczekiwany błąd podczas generowania embeddingów OpenAI: {e}")
        raise RuntimeError("Błąd wektoryzacji: Nieznany błąd.")


# ------------------------------
# LLM Response (using Groq)
# ------------------------------
def ask_llm(prompt):
    """Generuje odpowiedź LLM na podstawie promptu, używając modelu Llama 3 70B (Groq)."""
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Błąd podczas wywołania LLM Groq: {e}")
        return "Przepraszam, wystąpił błąd w komunikacji z modelem LLM (Groq)."

# ------------------------------
# Simple semantic search (Cosine Similarity)
# ------------------------------
def search(query, doc_embeddings, doc_texts):
    """Wyszukuje najbardziej podobny dokument do zapytania za pomocą Podobieństwa Kosinusowego."""
    try:
        # Konwersja zapytania na wektor
        query_emb_list = compute_embeddings([query])
    except RuntimeError:
        return "Błąd generowania wektora zapytania.", 0.0 
        
    if query_emb_list.size == 0:
        return "Błąd generowania wektora zapytania.", 0.0

    query_emb = query_emb_list[0]
    
    # Obliczanie podobieństwa kosinusowego
    # Wymagane jest rzutowanie na float64 dla poprawności
    similarities = cosine_similarity(query_emb.reshape(1, -1), doc_embeddings.astype(np.float64))
    best = np.argmax(similarities)
    
    # Zwracanie najlepiej dopasowanego tekstu i jego wyniku
    return doc_texts[best], similarities[0, best]

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🧠 Silnik Wiedzy — Uproszczona Edycja RAG 🚀")

st.write("LLM (Llama 3 70B) działa na Groq. Embeddingi działają na **stabilnym API OpenAI**.")
st.write("Ta wersja używa **wbudowanej, małej bazy wiedzy** w kodzie Python, nie pliku CSV.")
st.markdown("---")


# Ładowanie i buforowanie embeddingów dokumentów
@st.cache_resource
def load_document_embeddings(texts):
    """Wczytuje embeddingi i zapewnia, że aplikacja się nie uruchomi, jeśli to się nie powiedzie."""
    st.subheader("Faza 1: Wczytywanie bazy wiedzy")
    with st.spinner(f"Generowanie embeddingów dla {len(texts)} dokumentów..."):
        try:
            emb = compute_embeddings(texts)
        except RuntimeError:
            st.warning("Aplikacja została zatrzymana. Sprawdź, czy klucze API są poprawne.")
            st.stop()
            
        st.success("Baza wiedzy załadowana pomyślnie!")
        return emb

DOCUMENT_EMB = load_document_embeddings(DOCUMENT_TEXTS)

# ------------------------------
# UI Input
# ------------------------------
st.subheader("Faza 2: Zapytanie do Silnika Wiedzy")
query = st.text_input("Zadaj pytanie (np. Czym jest RAG?):")

if query:
    if DOCUMENT_EMB.size == 0:
        st.warning("Nie można wykonać wyszukiwania, ponieważ baza wiedzy jest pusta lub wystąpił błąd ładowania.")
    else:
        with st.spinner("Szukam kontekstu i generuję odpowiedź..."):
            
            # Wyszukiwanie sematyczne
            best_doc, score = search(query, DOCUMENT_EMB, DOCUMENT_TEXTS)

            st.markdown("### 🔎 Znaleziony Kontekst (RAG Retrieval)")
            st.write(f"**Podobieństwo (Cosine Score):** {score:.4f}")
            st.code(best_doc) 
    
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
