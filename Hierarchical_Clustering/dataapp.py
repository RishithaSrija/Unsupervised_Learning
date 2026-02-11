import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


# -----------------------------
# ULTRA MODERN CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg,#eef2ff,#f8fafc);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1e1e2f,#2b2b45);
}

section[data-testid="stSidebar"] * {
    color:white !important;
}

h1 {
    font-weight:600;
    background: linear-gradient(90deg,#7c3aed,#6366f1);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

div.stButton > button {
    background: linear-gradient(90deg,#7c3aed,#6366f1);
    color:white;
    border-radius:10px;
    border:none;
    padding:10px 20px;
    font-weight:600;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# TITLE
# -----------------------------

st.title('🟣 News Topic Discovery Dashboard')

st.write("""
This system uses Hierarchical Clustering to automatically group similar news articles.
Discover hidden themes without defining categories upfront.
""")


# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------

st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

max_features = st.sidebar.slider("Max TF-IDF Features", 100, 2000, 1000)

use_stopwords = st.sidebar.checkbox("Use English Stopwords", value=True)

ngram_option = st.sidebar.selectbox(
    "N-gram Range",
    ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
)

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"]
)

dendro_size = st.sidebar.slider(
    "Number of Articles for Dendrogram",
    20, 200, 100
)


# -----------------------------
# LOAD DATA SAFELY
# -----------------------------

def load_file(file):

    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']

    for enc in encodings:
        try:
            return pd.read_csv(file, header=None, encoding=enc)
        except:
            continue

    return None


if uploaded_file:

    df = load_file(uploaded_file)

    if df is None:
        st.error("Could not read file encoding.")
        st.stop()

    # auto detect text column
    text_col = df.apply(lambda col: col.astype(str).str.len().mean()).idxmax()

    text_data = df[text_col].dropna().astype(str)

    st.success(f"Detected text column: {text_col}")

    # -----------------------------
    # TF-IDF
    # -----------------------------

    if ngram_option == "Unigrams":
        ngram_range = (1,1)
    elif ngram_option == "Bigrams":
        ngram_range = (2,2)
    else:
        ngram_range = (1,2)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english' if use_stopwords else None,
        ngram_range=ngram_range
    )

    X = vectorizer.fit_transform(text_data)

    # -----------------------------
    # DENDROGRAM
    # -----------------------------

    if st.button("🟦 Generate Dendrogram"):

        sample = X[:dendro_size].toarray()

        Z = linkage(sample, method=linkage_method)

        fig = plt.figure(figsize=(10,5))
        dendrogram(Z)
        plt.title("Dendrogram")
        plt.ylabel("Distance")

        st.pyplot(fig)

    # -----------------------------
    # CLUSTERING
    # -----------------------------

    cluster_num = st.number_input(
        "Number of Clusters",
        min_value=2,
        max_value=20,
        value=4
    )

    if st.button("🟩 Apply Clustering"):

        model = AgglomerativeClustering(
            n_clusters=cluster_num,
            linkage=linkage_method
        )

        # convert to dense carefully
        X_dense = X.toarray()

        labels = model.fit_predict(X_dense)

        # -----------------------------
        # PCA VISUALIZATION
        # -----------------------------

        st.subheader("Cluster Visualization (PCA)")

        pca = PCA(n_components=2)

        X_reduced = pca.fit_transform(X_dense)

        fig = px.scatter(
            x=X_reduced[:,0],
            y=X_reduced[:,1],
            color=labels.astype(str),
            hover_name=text_data.iloc[:len(labels)]
        )

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # CLUSTER SUMMARY
        # -----------------------------

        st.subheader("Cluster Summary")

        terms = vectorizer.get_feature_names_out()

        summary = []

        for i in range(cluster_num):

            cluster_idx = np.where(labels == i)[0]

            if len(cluster_idx) == 0:
                continue

            cluster_vectors = X[cluster_idx]

            mean_tfidf = np.asarray(cluster_vectors.mean(axis=0)).flatten()

            top_terms = [terms[j] for j in mean_tfidf.argsort()[-10:]]

            snippet = text_data.iloc[cluster_idx[0]][:120]

            summary.append([i, len(cluster_idx), ", ".join(top_terms), snippet])

        summary_df = pd.DataFrame(
            summary,
            columns=["Cluster ID", "Number of Articles", "Top Keywords", "Example Article"]
        )

        st.dataframe(summary_df)

        # -----------------------------
        # SILHOUETTE SCORE
        # -----------------------------

        score = silhouette_score(X_dense, labels)

        st.subheader("📊 Silhouette Score")

        st.write(f"Score: {score:.3f}")

        st.info("""
Close to 1 → well-separated clusters  
Close to 0 → overlapping clusters  
Negative → poor clustering
""")

        # -----------------------------
        # BUSINESS INTERPRETATION
        # -----------------------------

        st.subheader("Business Interpretation")

        for row in summary:
            st.write(f"🟣 Cluster {row[0]}: Articles related to {row[2][:60]}...")

        st.success("""
Articles grouped in the same cluster share similar vocabulary and themes.
These clusters can be used for automatic tagging, recommendations, and content organization.
""")
