import streamlit as st
import pandas as pd

st.set_page_config(page_title="MKS Knowledge Engine", layout="wide")

st.title("🧠 MKS Knowledge Engine")
st.write("Wyszukiwarka wiedzy wspinaczkowej oparta o Perfekcyjne Koło MKS")

# Ładujemy dane
@st.cache_data
def load_data():
    data = [
        {"type": "Cel", "query": "Wejście na Mont Blanc", "answer": "Plan treningowy, aklimatyzacja, sprzęt."},
        {"type": "Problem", "query": "Co zrobić przy odmrożeniach?", "answer": "Zejście z wysokości, ogrzewanie pasywne, szpital."},
        {"type": "Wyjaśnienie", "query": "Dlaczego spada kondycja na wysokości?", "answer": "Niższe ciśnienie, mniejsza dostępność tlenu."},
        {"type": "Wybór", "query": "Jaki namiot wybrać zimą?", "answer": "Namiot ekspedycyjny, 4-sezonowy."}
    ]
    return pd.DataFrame(data)

df = load_data()

search = st.text_input("🔍 Wyszukaj dowolne słowo lub zdanie")

if search:
    results = df[df.apply(lambda row: search.lower() in row.astype(str).str.lower().to_string(), axis=1)]
    st.subheader("Wyniki:")
    st.table(results)
