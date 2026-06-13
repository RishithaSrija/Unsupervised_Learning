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

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Main App */
.stApp{
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e293b 50%,
        #111827 100%
    );
    color: #f8fafc;
}

/* Main text */
p, li, label {
    color: #f8fafc !important;
}

[data-testid="stMarkdownContainer"] {
    color: #f8fafc !important;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background: linear-gradient(
        180deg,
        #111827,
        #1e293b
    );
    border-right: 1px solid rgba(255,255,255,0.1);
}

section[data-testid="stSidebar"] *{
    color: white !important;
}

/* Title */
h1{
    color:#ffffff !important;
    font-size:42px !important;
    font-weight:700 !important;
    text-align:center;
    margin-bottom:20px;
}

/* Headers */
h2,h3{
    color:#c4b5fd !important;
    font-weight:600 !important;
}

/* Cards */
[data-testid="stVerticalBlock"] > div{
    background: rgba(255,255,255,0.04);
    border-radius:18px;
    padding:18px;
    border:1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
}

/* Buttons */
.stButton > button{
    width:100%;
    border:none;
    border-radius:12px;
    padding:12px;
    font-weight:600;
    color:white;
    background:linear-gradient(
        90deg,
        #7c3aed,
        #6366f1
    );
    transition:0.3s;
}

.stButton > button:hover{
    transform:translateY(-2px);
    box-shadow:0 8px 25px rgba(124,58,237,0.5);
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stSelectbox div,
.stMultiSelect div{
    background:#1f2937 !important;
    color:white !important;
    border-radius:10px !important;
}

/* Sliders */
.stSlider{
    padding-top:10px;
}

/* DataFrame */
[data-testid="stDataFrame"]{
    border-radius:15px;
    overflow:hidden;
}

/* Metric boxes */
[data-testid="metric-container"]{
    background:rgba(255,255,255,0.05);
    border-radius:15px;
    padding:10px;
}
div[data-testid="stAlert"]{
    border-radius:15px;
    border:none;
    padding:16px;
}

/* Success Box */
.stSuccess{
    background:#064e3b !important;
    color:white !important;
    border-radius:12px;
}

/* Info Box */
.stInfo{
    background:#1e3a8a !important;
    color:white !important;
    border-radius:12px;
}

/* Warning */
.stWarning{
    border-radius:12px;
}

/* Plotly chart container */
.js-plotly-plot{
    border-radius:15px;
    overflow:hidden;
}

/* Scrollbar */
::-webkit-scrollbar{
    width:10px;
}

::-webkit-scrollbar-track{
    background:#111827;
}

::-webkit-scrollbar-thumb{
    background:#6366f1;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# TITLE
# -----------------------------

st.markdown("""
<h1>
📰 News Topic Discovery Dashboard
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
background:linear-gradient(135deg,#7c3aed,#4f46e5);
padding:25px;
border-radius:20px;
margin-bottom:20px;
color:white;
text-align:center;
">
<h2 style="color:white;">AI Powered News Clustering</h2>
<p>
Discover hidden news topics automatically using
TF-IDF, Hierarchical Clustering, PCA Visualization,
and Business Insights.
</p>
</div>
""", unsafe_allow_html=True)


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
        fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
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

        st.dataframe(
                summary_df,use_container_width=True,height=350)

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
