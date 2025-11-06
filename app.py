import streamlit as st
import pandas as pd
import joblib
import numpy as np

# set seed untuk hasil yang konsisten
RSEED = 42
np.random.seed(RSEED)

@st.cache_resource
def load_artifacts():
    tfidf = joblib.load('artifacts/tfidf_vectorizer.joblib')
    nn = joblib.load('artifacts/nn_model.joblib')
    products = pd.read_csv('artifacts/products.csv')
    return tfidf, nn, products

tfidf, nn, products = load_artifacts()

st.title("🛍️ Sistem Rekomendasi Produk E-Commerce")
st.write("Aplikasi ini memberikan rekomendasi produk serupa berdasarkan kesamaan teks (TF-IDF + Cosine Similarity).")

# Pilih produk
product_id = st.selectbox("Pilih ID produk:", products['product_id'].astype(str).tolist())

def recommend(product_id, topn=10):
    idxs = products.index[products['product_id'] == product_id].tolist()
    if not idxs:
        st.warning("Produk tidak ditemukan.")
        return []
    idx = idxs[0]
    text = products.loc[idx, 'text']
    vec = tfidf.transform([text])
    distances, indices = nn.kneighbors(vec, n_neighbors=topn+1)
    recs = []
    for dist, ind in zip(distances[0][1:], indices[0][1:]):
        recs.append({
            'product_id': str(products.loc[ind, 'product_id']),
            'title': products.loc[ind, 'title'],
            'similarity': float(1 - dist)
        })
    return recs

if st.button("Tampilkan Rekomendasi"):
    results = recommend(product_id)
    st.subheader("Hasil Rekomendasi:")
    for r in results:
        st.write(f"- **{r['title']}** (ID: {r['product_id']}) — Similarity: {r['similarity']:.3f}")
