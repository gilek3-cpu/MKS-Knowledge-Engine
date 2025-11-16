import streamlit as st
import numpy as np
import json
# Importujemy Groq dla LLM
from groq import Groq
# Importujemy OpenAI dla opcjonalnego embeddingu
from openai import OpenAI

# --- KONFIGURACJA KLUCZY ---
# 1. Klucz Groq (dla Llama 3)
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Błąd: Brak klucza 'GROQ_API_KEY' w Streamlit Secrets.")
    st.stop() 

# 2. Klucz OpenAI (dla embeddingów - do testowania problemu Groq)
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
except KeyError:
    st.warning("Uwaga: Brak klucza 'OPENAI_API_KEY'. Próba użycia Groq dla embeddingów.")
    openai_client = None

# Inicjalizacja klienta Groq
try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Błąd inicjalizacji klienta Groq: {e}")
    st.stop()


# ------------------------------
# EMBEDDINGS (Wektoryzacja) - Używamy OpenAI jako fallback
# ------------------------------
@st.cache_data
def compute_embeddings(texts):
    """
    Generuje embeddingi dla listy tekstów. Domyślnie używa Groq, ale używa OpenAI,
    jeśli Groq zwraca błąd (np. 404 model_not_found) lub jeśli klucz OpenAI jest dostępny
    i Groq jest skonfigurowany, by użyć fallbacku.
    """
    embeddings = []
    
    # --- PRÓBA UŻYCIA GROQ ---
    if not openai_client:
        try:
            # W tym try-except Groq jest używany jako główny mechanizm.
            st.info("Używam Groq (nomic-embed-text) dla embeddingów...")
            for t in texts:
                response = client.embeddings.create(
                    model="nomic-embed-text",
                    input=t
                )
                embeddings.append(response.data[0].embedding)
            st.success("Groq Embeddings sukces!")
            return embeddings
        
        except Exception as e:
            # Jeśli Groq zawiedzie (jak w Twoim przypadku) i nie mamy klucza OpenAI, to kończymy.
            st.error(f"Krytyczny błąd API Groq w compute_embeddings. Błąd: {e}")
            raise RuntimeError("Nie udało się wygenerować embeddingów. (Brak fallbacku OpenAI).")


    # --- UŻYCIE OPENAI JAKO FALLBACK (JEŚLI KLUCZ OPENAI JEST DOSTĘPNY) ---
    else:
        st.info("Używam OpenAI (text-embedding-3-small) dla embeddingów (jako fallback)...")
        try:
            # Wywołujemy API OpenAI
            response = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            embeddings = [data.embedding for data in response.data]
            st.success("OpenAI Embeddings sukces!")
            return embeddings

        except Exception as e:
            # Jeśli OpenAI też zawiedzie (co jest mało prawdopodobne), rzucamy błąd.
            st.error(f"Krytyczny błąd API OpenAI w compute_embeddings. Błąd: {e}")
            raise RuntimeError("Nie udało się wygenerować embeddingów (fallback nieudany).")


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
# Streamlit UI
# ------------------------------
st.title("🧠 Silnik Wiedzy — Groq Edition 🚀")

# Zmodyfikowany tekst UI, aby odzwierciedlić użycie OpenAI/Groq
if openai_client:
    st.write("Embeddingi (OpenAI) + LLM (Llama 3 70B) - **Test Fallbacku Embeddings**.")
else:
    st.write("Embeddingi (Nomic Embed) + LLM (Llama 3 70B) - **Wszystko na Groq API**.")

st.markdown("---")


# Example documents
DOCUMENT_TEXTS = [
    "Python jest językiem programowania używanym do analizy danych, uczenia maszynowego i tworzenia aplikacji webowych.",
    "Streamlit to darmowy framework do budowy interaktywnych aplikacji webowych w Pythonie bez znajomości HTML/CSS/JS.",
    "Groq oferuje bardzo szybkie i darmowe modele AI dla programistów, działające na specjalistycznych akceleratorach LPU (Language Processing Unit).",
    "Funkcja Cosine Similarity mierzy kąt między dwoma wektorami w przestrzeni, określając podobieństwo semantyczne.",
]

# Ładowanie i buforowanie embeddingów (z zabezpieczeniem)
@st.cache_resource
def load_document_embeddings():
    """Wczytuje embeddingi i zapewnia, że aplikacja się nie uruchomi, jeśli to się nie powiedzie."""
    st.subheader("Faza 1: Wczytywanie bazy wiedzy")
    with st.spinner("Generowanie embeddingów dla dokumentów..."):
        try:
            emb = compute_embeddings(DOCUMENT_TEXTS)
        except RuntimeError:
            # Wyświetla błąd rzucony przez compute_embeddings
            st.error("Nie udało się załadować bazy wiedzy. Sprawdź klucze API i logi błędów.")
            st.stop()
            
        st.success("Baza wiedzy załadowana pomyślnie!")
        return emb

DOCUMENT_EMB = load_document_embeddings()

# ------------------------------
# Simple semantic search (Cosine Similarity)
# ------------------------------
def cosine_similarity(a, b):
    """Oblicza podobieństwo cosinusowe między dwoma wektorami."""
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return np.dot(a, b) / (norm_a * norm_b)

def search(query):
    """Wyszukuje najbardziej podobny dokument do zapytania."""
    # 1. Generowanie embeddingu dla zapytania
    try:
        query_emb_list = compute_embeddings([query])
    except RuntimeError:
        return "Błąd generowania wektora zapytania.", 0.0

    query_emb = query_emb_list[0]
    
    # 2. Obliczanie podobieństwa
    sims = [cosine_similarity(query_emb, emb) for emb in DOCUMENT_EMB]
    best = np.argmax(sims)
    
    return DOCUMENT_TEXTS[best], sims[best]

# ------------------------------
# UI Input
# ------------------------------
st.subheader("Faza 2: Zapytanie do Silnika Wiedzy")
query = st.text_input("Zadaj pytanie (np. Czym jest Streamlit?):")

if query:
    if not DOCUMENT_EMB:
        st.warning("Nie można wykonać wyszukiwania, ponieważ baza wiedzy jest pusta.")
    else:
        with st.spinner("Szukam kontekstu i generuję odpowiedź..."):
            
            # Wyszukiwanie semantyczne
            best_doc, score = search(query)

            st.markdown("### 🔎 Znaleziony Kontekst")
            st.write(f"**Podobieństwo (Cosine Score):** {score:.4f}")
            st.code(best_doc) 
    
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
    
            # Wywołanie LLM
            answer = ask_llm(final_prompt)
            st.markdown("### 🤖 Odpowiedź Modelu (Llama 3 70B)")
            st.info(answer)
