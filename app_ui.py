import requests
import streamlit as st

st.set_page_config(page_title="Semantic Search", layout="wide")
st.title("🔎 Semantic Search (Azure AI Search + Embeddings)")

API = st.sidebar.text_input("Backend URL", "http://localhost:8000/search")
q = st.text_input("Query", "")
top_k = st.slider("Page size", 1, 20, 5)

if st.button("Search") and q.strip():
    r = requests.get(API, params={"query": q, "page": 1, "page_size": top_k}, timeout=60)
    r.raise_for_status()
    data = r.json()

    st.write(f"Total results: {data.get('total_results', 0)}")

    for i, item in enumerate(data.get("results", []), start=1):
        meta = item.get("meta") or {}
        st.subheader(f"{i}. {meta.get('title', item.get('id'))}")

        url = meta.get("url")
        if url:
            st.write(f"Source: {url}")

        st.caption(f"Score: {item.get('score')} | DocID: {item.get('id')}")

        passages = item.get("passages") or []
        for p in passages:
            st.markdown("**Passage:**")
            st.write(p.get("text", ""))
            st.caption(f"{p.get('retriever')} score: {p.get('score')} | chunk: {p.get('chunk_id')}")
            st.divider()
