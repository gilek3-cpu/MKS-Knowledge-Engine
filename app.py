import streamlit as st
from openai import OpenAI
import numpy as np

st.set_page_config(page_title="Silnik wiedzy MKS", layout="wide")
st.title("🔎 Silnik Wiedzy MKS")

# --- OpenAI Client ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Przykładowa baza wiedzy ---
DOCUMENTS = [
    "Regulamin szkoły określa zasady organizacyjne oraz obowiązki uczniów.",
    "Uczeń ma prawo do bezpiecznych warunków nauki oraz uzyskania wsparcia pedagoga.",
    "Sekretariat jest czynny od poniedziałku do piątku w godzinach 8:00–15:00.",
    "Oceny można sprawdzać w dzienniku elektronicznym Librus.",
    "Rodzic może umówić spotkanie z wychowawcą poprzez e-dziennik.",
    "Szkoła organizuje dodatkowe zajęcia wyrównawcze oraz koła zainteresowań.",
]

# --- Liczymy embeddingi dokumentów ---
@st.cache_data(show_spinner=False)
def compute_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

DOCUMENT_EMB = compute_embeddings(DOCUMENTS)

# --- Funkcja wyszukująca najlepszy dokument ---
def semantic_search(query, top_k=3):
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    scores = []
    for doc, emb in zip(DOCUMENTS, DOCUMENT_EMB):
        score = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
        scores.append((score, doc))

    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[:top_k]

# --- UI ---
query = st.text_input("🔍 Wpisz pytanie:", placeholder="Np. kiedy czynny jest sekretariat?")

if query:
    st.subheader("📄 Najtrafniejsze wyniki wyszukiwania:")
    results = semantic_search(query)

    for score, doc in results:
        st.write(f"**Wynik dopasowania:** {round(score, 3)}")
        st.write(doc)
        st.markdown("---")

    # --- Generowanie odpowiedzi na podstawie wyników ---
    context = "\n".join([doc for _, doc in results])

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Odpowiadaj krótko i rzeczowo, korzystając tylko z podanych informacji."
            },
            {"role": "user", "content": f"Pytanie: {query}\n\nInformacje:\n{context}"}
        ]
    )

    st.subheader("🤖 Odpowiedź AI:")
    st.write(completion.choices[0].message.content)

