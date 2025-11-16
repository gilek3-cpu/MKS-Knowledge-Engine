import streamlit as st
# Wymagane tylko podstawowe biblioteki i klient Groq
from groq import Groq
import numpy as np 

# --- KONFIGURACJA STRONY I KLUCZY ---

st.set_page_config(layout="centered", page_title="Silnik Wiedzy RAG (Minimalistyczny Groq)")

st.title("🧠 Silnik Wiedzy RAG — Minimalistyczna Edycja Groq 🚀")

try:
    # Klucz GROQ API musi być ustawiony w Streamlit Secrets
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("Błąd: Brak klucza 'GROQ_API_KEY' w Streamlit Secrets. Jest wymagany dla LLM (Llama 3).")
    st.stop() 

try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Błąd inicjalizacji klienta Groq: {e}")
    st.stop()


# ------------------------------
# WBUDOWANA BAZA WIEDZY (KONIEC Z knowledge.csv!)
# ------------------------------
# Ta lista tekstów służy jako prosta baza wiedzy RAG.
DOCUMENT_TEXTS = [
    "Python jest językiem programowania używanym do analizy danych, uczenia maszynowego i tworzenia aplikacji webowych.",
    "Streamlit to darmowy framework do budowy interaktywnych aplikacji webowych w Pythonie.",
    "Groq oferuje bardzo szybkie modele AI dla programistów, działające na akceleratorach LPU (Language Processing Unit).",
    "RAG (Retrieval-Augmented Generation) to architektura AI, która wykorzystuje bazę wiedzy (retrieval) do ulepszania odpowiedzi LLM (generation).",
    "Do wspinaczki sportowej niezbędna jest lina dynamiczna, uprząż, ekspresy i ósemka. Asekuracja odbywa się z góry lub z dołu.",
    "Wspinaczka tradycyjna wymaga umiejętności osadzania własnej asekuracji, np. kości i friendów. Jest to bardziej wymagające psychicznie i sprzętowo.",
    "Najczęstsze błędy w Streamlit to brak klucza API, nieużywanie st.cache_data/st.cache_resource oraz problemy z zależnościami w requirements.txt.",
]

# ------------------------------
# PROSTE WYSZUKIWANIE PO SŁOWACH KLUCZOWYCH (BEZ SKLEARN/EMBEDDINGS)
# ------------------------------
def simple_keyword_search(query, doc_texts):
    """
    Wyszukuje najbardziej pasujący dokument na podstawie liczby wspólnych słów kluczowych.
    Jest to ZASTĘPCZY RAG, który nie wymaga zewnętrznych bibliotek.
    """
    # Tokenizacja i normalizacja zapytania
    query_words = set(query.lower().split())
    
    best_doc = ""
    max_matches = 0

    for doc in doc_texts:
        # Tokenizacja i normalizacja dokumentu
        doc_words = set(doc.lower().split())
        
        # Obliczanie liczby wspólnych słów (prosta metryka dopasowania)
        matches = len(query_words.intersection(doc_words))
        
        if matches > max_matches:
            max_matches = matches
            best_doc = doc
            
    return best_doc, max_matches


# ------------------------------
# GENEROWANIE ODPOWIEDZI LLM (GROQ)
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
# INTERFEJS UŻYTKOWNIKA STREAMLIT
# ------------------------------

st.write("Ta wersja jest **całkowicie samowystarczalna**. Nie wymaga **knowledge.csv** ani zewnętrznych bibliotek (typu `sklearn`).")
st.write("Używa prostego wyszukiwania słów kluczowych (Groq-Only RAG).")
st.markdown("---")


# Faza 1: Baza Wiedzy
st.subheader("Faza 1: Baza wiedzy")
st.success(f"Baza wiedzy załadowana pomyślnie! ({len(DOCUMENT_TEXTS)} dokumentów wbudowanych w kod.)")


# Faza 2: Zapytanie
st.subheader("Faza 2: Zapytanie do Silnika Wiedzy")
query = st.text_input("Zadaj pytanie (np. Co to jest RAG?):")

if query:
    with st.spinner("Szukam kontekstu i generuję odpowiedź..."):
        
        # 1. Wyszukiwanie kluczowe
        best_doc, matches = simple_keyword_search(query, DOCUMENT_TEXTS)

        st.markdown("### 🔎 Znaleziony Kontekst")
        st.write(f"**Liczba pasujących słów kluczowych:** {matches}")
        
        if matches == 0:
            st.warning("Brak pasujących słów kluczowych. Model odpowie bez kontekstu.")
            # Prompt dla braku kontekstu (pytanie otwarte)
            final_prompt = f"""
            Jesteś ekspertem technicznym. Postaraj się odpowiedzieć na pytanie.
            Jeśli nie masz pewności, odpowiedz: 'Nie jestem w stanie precyzyjnie odpowiedzieć na to pytanie bez kontekstu w mojej bazie wiedzy.'
            Pytanie: {query}
            Odpowiedź:
            """
        else:
            st.code(best_doc) 
    
            # 2. Tworzenie promptu RAG z kontekstem
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
    
        # 3. Wywołanie LLM (Groq)
        answer = ask_llm(final_prompt)
        st.markdown("### 🤖 Odpowiedź Modelu (Llama 3 70B - Groq)")
        st.info(answer)
