import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================
# 1. ŁADOWANIE MODELU
# ============================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ============================================================
# 2. BAZA WIEDZY
# ============================================================
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

# ============================================================
# 3. EMBEDDINGI DLA BAZY
# ============================================================
@st.cache_resource
def embed_knowledge(data):
    texts = [item["content"] for item in data]
    vectors = model.encode(texts)
    return np.array(vectors)

knowledge_embeddings = embed_knowledge(knowledge_base)

# ============================================================
# 4. RĘCZNY KOSINUS — BEZ TORCHA
# ============================================================
def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return np.dot(a_norm, b_norm)

# ============================================================
# 5. WYSZUKIWANIE
# ============================================================
def search(query, category=None, tags=None, top_k=5):
    query_emb = model.encode(query)

    scores = []
    for i, item in enumerate(knowledge_base):

        # filtr kategorii
        if category and item["category"] != category:
            continue

        # filtr tagów
        if tags:
            if not any(tag in item["tags"] for tag in tags):
                continue

        score = cosine_similarity(query_emb, knowledge_embeddings[i])
        scores.append((i, score))

    # Sortowanie po trafności
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for idx, score in scores:
        item = knowledge_base[idx]
        results.append({
            "title": item["title"],
            "category": item["category"],
            "tags": item["tags"],
            "content": item["content"],
            "score": float(score)
        })

    return results

# ============================================================
# 6. INTERFEJS STREAMLIT
# ============================================================
st.set_page_config(page_title="MKS Knowledge Engine", page_icon="🧗", layout="wide")

st.title("🧠 MKS Knowledge Engine")
st.caption("Zaawansowana wyszukiwarka wiedzy wspinaczkowej oparta o Perfekcyjne Koło MKS")

# pola wyboru
category_filter = st.selectbox(
    "📂 Filtr kategorii (opcjonalnie):",
    [""] + sorted(list(set([item["category"] for item in knowledge_base])))
)

tag_filter = st.multiselect(
    "🏷️ Filtr tagów (opcjonalnie):",
    sorted(list(set(tag for item in knowledge_base for tag in item["tags"])))
)

query = st.text_input("🔍 Wpisz dowolne słowo lub zdanie:")

if query:
    results = search(
        query,
        category=category_filter if category_filter else None,
        tags=tag_filter if tag_filter else None
    )

    st.markdown("---")
    st.subheader("Wyniki:")

    if not results:
        st.info("Brak wyników spełniających kryteria filtrowania.")
    else:
        for r in results:
            st.markdown(f"### **{r['title']}**")
            st.markdown(f"**Kategoria:** {r['category']}")
            st.markdown(f"**Tagi:** {', '.join(r['tags'])}")
            st.markdown(f"📄 {r['content']}")
            st.markdown(f"<small>Trafność: {r['score']:.3f}</small>", unsafe_allow_html=True)
            st.markdown("---")
