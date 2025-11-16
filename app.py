import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ------------------------------------------------------------
# 1. KONFIGURACJA MODELU
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# ------------------------------------------------------------
# 2. BAZA WIEDZY MKS – kategorie, tagi, treści
# ------------------------------------------------------------

knowledge_base = [
    {
        "title": "Balans i środek ciężkości",
        "category": "Technika",
        "tags": ["balans", "technika", "podstawy"],
        "content": "Utrzymywanie środka ciężkości blisko ściany pozwala odciążyć ręce i wspinać się wydajniej."
    },
    {
        "title": "Praca nóg – precyzyjne stawianie stóp",
        "category": "Technika",
        "tags": ["nogi", "technika", "podstawy"],
        "content": "Najważniejsza część techniki wspinaczkowej. Precyzja stóp pozwala generować ruch bez siły w rękach."
    },
    {
        "title": "Trening obwodowy na wytrzymałość",
        "category": "Trening",
        "tags": ["wytrzymałość", "trening"],
        "content": "Obwody 6–10 min pracy poprawiają wytrzymałość tlenową i zdolność do pracy na drogach."
    },
    {
        "title": "Trening siły palców – hangboard",
        "category": "Trening",
        "tags": ["siła palców", "trening"],
        "content": "Regularny trening na chwytotablicy wzmacnia siłę chwytu i jest kluczowy w trudnych drogach."
    },
    {
        "title": "Analiza sekwencji – wizualizacja",
        "category": "Taktyka",
        "tags": ["analiza", "taktyka", "planowanie"],
        "content": "Wyobrażanie sobie ruchów przed startem zwiększa skuteczność i zmniejsza błędy."
    },
    {
        "title": "Pokonywanie strachu przed lotem",
        "category": "Mental",
        "tags": ["strach", "mental", "lęk"],
        "content": "Ekspozycja, kontrolowane loty i stopniowa adaptacja pomagają redukować lęk przed odpadnięciem."
    },
    {
        "title": "Regeneracja – sen i odżywianie",
        "category": "Regeneracja",
        "tags": ["regeneracja", "sen", "odżywianie"],
        "content": "Odpowiednia regeneracja wpływa na adaptację treningową i zapobiega kontuzjom."
    },
    {
        "title": "Zasada Perfekcyjnego Koła MKS",
        "category": "Filozofia",
        "tags": ["MKS", "rozwój", "progres"],
        "content": "Proces stałego doskonalenia zakłada analizę, planowanie, wykonanie i refleksję po każdym przejściu."
    },
]

# ------------------------------------------------------------
# 3. Embeddingi bazy wiedzy
# ------------------------------------------------------------
@st.cache_resource
def embed_knowledge(knowledge):
    texts = [item["content"] for item in knowledge]
    return model.encode(texts, convert_to_tensor=True)

knowledge_embeddings = embed_knowledge(knowledge_base)

# ------------------------------------------------------------
# 4. WYSZUKIWANIE SEMANTYCZNE
# ------------------------------------------------------------
def search(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, knowledge_embeddings)[0]

    top_results = scores.topk(top_k)
    result_indices = top_results.indices.cpu().tolist()
    result_scores = top_results.values.cpu().tolist()

    results = []
    for idx, score in zip(result_indices, result_scores):
        item = knowledge_base[idx]
        results.append({
            "title": item["title"],
            "category": item["category"],
            "tags": item["tags"],
            "content": item["content"],
            "score": float(score)
        })
    return results

# ------------------------------------------------------------
# 5. INTERFEJS STREAMLIT
# ------------------------------------------------------------

st.set_page_config(page_title="MKS Knowledge Engine", page_icon="🧗", layout="wide")

st.title("🧠 MKS Knowledge Engine")
st.caption("Zaawansowana wyszukiwarka wiedzy wspinaczkowej oparta o Perfekcyjne Koło MKS")

query = st.text_input("🔍 Wpisz dowolne słowo lub zdanie:")

if query:
    results = search(query)

    st.markdown("---")
    st.subheader("Wyniki:")

    for r in results:
        with st.container():
            st.markdown(f"### **{r['title']}**")
            st.markdown(f"**Kategoria:** {r['category']}")
            st.markdown(f"**Tagi:** {', '.join(r['tags'])}")
            st.markdown(f"📄 {r['content']}")
            st.markdown(f"<small>Trafność: {r['score']:.3f}</small>", unsafe_allow_html=True)
            st.markdown("---")

