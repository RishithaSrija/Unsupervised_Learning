import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="🛒 Customer Segmentation – KMeans App", layout="wide")


# -------------------------------
# BEAUTIFUL CSS
# -------------------------------
st.markdown("""
<style>
.main {
    background: radial-gradient(circle at top, #f8fafc 0%, #eef2ff 45%, #f8fafc 100%);
}
h1 {
    color: #0f172a;
    font-weight: 950;
}
h2, h3 {
    color: #1e293b;
    font-weight: 850;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b);
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: white !important;
    font-weight: 700;
}

.card {
    background: rgba(255,255,255,0.94);
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0px 8px 22px rgba(0,0,0,0.08);
    border: 1px solid rgba(148,163,184,0.25);
    margin-bottom: 18px;
}

.badge {
    display: inline-block;
    padding: 8px 12px;
    border-radius: 999px;
    font-weight: 900;
    font-size: 13px;
    background: linear-gradient(90deg, #dcfce7, #bbf7d0);
    color: #065f46;
    border: 1px solid rgba(34,197,94,0.2);
}

.kpi-wrap {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    margin-top: 10px;
}
.kpi {
    flex: 1;
    min-width: 220px;
    padding: 16px;
    border-radius: 16px;
    background: linear-gradient(120deg, #eff6ff, #ffffff);
    border: 1px solid rgba(37, 99, 235, 0.15);
    box-shadow: 0px 6px 18px rgba(0,0,0,0.06);
}
.kpi-title {
    font-size: 14px;
    font-weight: 800;
    color: #334155;
}
.kpi-value {
    font-size: 26px;
    font-weight: 950;
    color: #0f172a;
    margin-top: 6px;
}

div.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    font-weight: 950;
    border-radius: 14px;
    padding: 12px 22px;
    border: none;
    transition: 0.25s ease;
    font-size: 16px;
    box-shadow: 0px 12px 24px rgba(37,99,235,0.22);
}
div.stButton > button:hover {
    transform: translateY(-2px) scale(1.02);
    background: linear-gradient(90deg, #1d4ed8, #6d28d9);
}

[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(148,163,184,0.35) !important;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# TITLE
# -------------------------------
st.title("🛒 Wholesale Customer Segmentation using K-Means")
st.write("✅ Select settings in sidebar → click **Run Clustering** → get all results in one place.")
st.divider()


# -------------------------------
# SIDEBAR - DATA UPLOAD
# -------------------------------
st.sidebar.header("📂 Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
default_path = "Wholesale customers data.csv"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Dataset uploaded!")
elif os.path.exists(default_path):
    df = pd.read_csv(default_path)
    st.sidebar.info("✅ Loaded dataset from folder")
else:
    st.warning("⚠️ Upload dataset to continue.")
    st.stop()


# -------------------------------
# SIDEBAR - FEATURE SELECTION
# -------------------------------
st.sidebar.header("🧾 Feature Selection")

default_features = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
available_cols = df.columns.tolist()

selected_features = [c for c in default_features if c in available_cols]
selected_features = st.sidebar.multiselect(
    "Select purchase/spending columns:",
    options=available_cols,
    default=selected_features
)

if len(selected_features) < 2:
    st.warning("⚠️ Select at least 2 spending features to cluster.")
    st.stop()


# -------------------------------
# SIDEBAR - SETTINGS
# -------------------------------
st.sidebar.header("⚙️ Model Settings")
k_val = st.sidebar.slider("Select K (No. of Clusters)", min_value=2, max_value=10, value=4)
rand_state = st.sidebar.number_input("Random State", min_value=0, max_value=999, value=42)
alt_state = st.sidebar.number_input("Alternative Random State (Stability)", min_value=0, max_value=999, value=99)

st.sidebar.header("📊 Visualization")
x_axis = st.sidebar.selectbox("X-axis Feature", selected_features, index=0)
y_axis = st.sidebar.selectbox("Y-axis Feature", selected_features, index=1)


# -------------------------------
# RUN BUTTON
# -------------------------------
run = st.sidebar.button("🚀 Run Clustering & Show Results")


# -------------------------------
# RESULTS AFTER CLICK
# -------------------------------
if run:

    # -------------------------------
    # Task 1: Exploration
    # -------------------------------
    st.subheader("🔹 Task 1: Data Exploration")
    st.markdown("<div class='card'>Dataset preview and missing value check.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.write("✅ Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

    with c2:
        st.write("✅ Dataset Info")
        st.markdown(f"<span class='badge'>Rows: {df.shape[0]}</span>  <span class='badge'>Columns: {df.shape[1]}</span>",
                    unsafe_allow_html=True)
        st.write("✅ Missing Values")
        st.dataframe(df.isnull().sum(), use_container_width=True)

    # -------------------------------
    # Task 2: Features
    # -------------------------------
    st.subheader("🔹 Task 2: Feature Selection")
    st.markdown("<div class='card'>Chosen features represent customer spending behavior.</div>", unsafe_allow_html=True)
    st.write("✅ Selected Spending Features:", selected_features)

    # -------------------------------
    # Task 3: Scaling
    # -------------------------------
    st.subheader("🔹 Task 3: Data Preparation (Scaling)")
    st.markdown("<div class='card'>Scaling ensures fair distance calculation in K-Means.</div>", unsafe_allow_html=True)

    X = df[selected_features].copy()
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.success("✅ Data scaled successfully using StandardScaler")

    # -------------------------------
    # Task 4 + 5: Elbow + Silhouette
    # -------------------------------
    st.subheader("🔹 Task 4 & 5: Optimal Cluster Identification")

    wcss = []
    sil_scores = []

    k_range = range(2, 11)

    # WCSS for K=1..10
    for k in range(1, 11):
        km = KMeans(n_clusters=k, init="k-means++", random_state=rand_state)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    # Silhouette for K=2..10
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", random_state=rand_state)
        labels_temp = km.fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, labels_temp))

    g1, g2 = st.columns(2)

    with g1:
        st.write("✅ Elbow Method (WCSS)")
        fig1 = plt.figure(figsize=(6, 4))
        plt.plot(range(1, 11), wcss, marker="o")
        plt.title("Elbow Method")
        plt.xlabel("K")
        plt.ylabel("WCSS")
        plt.grid(True)
        st.pyplot(fig1)

    with g2:
        st.write("✅ Silhouette Scores")
        fig2 = plt.figure(figsize=(6, 4))
        plt.plot(list(k_range), sil_scores, marker="o")
        plt.title("Silhouette Score")
        plt.xlabel("K")
        plt.ylabel("Score")
        plt.grid(True)
        st.pyplot(fig2)

    st.info("📌 A good K is chosen where Elbow bends and Silhouette score is higher.")

    # -------------------------------
    # Task 6: Train + Assign clusters
    # -------------------------------
    st.subheader("🔹 Task 6: Cluster Assignment")

    kmeans = KMeans(n_clusters=k_val, init="k-means++", random_state=rand_state)
    labels = kmeans.fit_predict(X_scaled)

    result_df = df.copy()
    result_df["Cluster"] = labels

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(f"<div class='kpi'><div class='kpi-title'>✅ Selected K</div><div class='kpi-value'>{k_val}</div></div>", unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"<div class='kpi'><div class='kpi-title'>🎯 Random State</div><div class='kpi-value'>{rand_state}</div></div>", unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"<div class='kpi'><div class='kpi-title'>👥 Customers</div><div class='kpi-value'>{len(result_df)}</div></div>", unsafe_allow_html=True)

    st.write("✅ Cluster Count:")
    st.dataframe(result_df["Cluster"].value_counts().sort_index(), use_container_width=True)

    # -------------------------------
    # Task 7: Visualization
    # -------------------------------
    st.subheader("🔹 Task 7: Cluster Visualization")
    st.markdown("<div class='card'>Visualization using selected two spending categories with cluster centers.</div>", unsafe_allow_html=True)

    centers_scaled = kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)

    x_idx = selected_features.index(x_axis)
    y_idx = selected_features.index(y_axis)

    fig3 = plt.figure(figsize=(10, 5))
    plt.scatter(result_df[x_axis], result_df[y_axis], c=result_df["Cluster"], cmap="viridis", alpha=0.7)
    plt.scatter(centers_original[:, x_idx], centers_original[:, y_idx],
                s=320, c="red", marker="X", label="Centers")
    plt.title(f"Customer Segments: {x_axis} vs {y_axis}")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.grid(True)
    st.pyplot(fig3)

    # -------------------------------
    # Task 8: Profiling
    # -------------------------------
    st.subheader("🔹 Task 8: Cluster Profiling")
    profile = result_df.groupby("Cluster")[selected_features].mean().round(2)
    st.write("✅ Average Spending per Cluster:")
    st.dataframe(profile, use_container_width=True)

    for c in sorted(result_df["Cluster"].unique()):
        dom = profile.loc[c].sort_values(ascending=False).index[0]
        st.markdown(f"""
        <div class="card">
            <h3>📌 Cluster {c}</h3>
            <b>Dominant Purchase Category:</b> {dom}<br>
            <b>Business Summary:</b> This customer segment mainly spends more on <b>{dom}</b>.
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------
    # Task 9: Business strategies
    # -------------------------------
    st.subheader("🔹 Task 9: Business Insight Generation")

    st.markdown("<div class='card'><b>✅ Suggested Strategy for Each Segment</b></div>", unsafe_allow_html=True)

    for c in sorted(result_df["Cluster"].unique()):
        dom = profile.loc[c].sort_values(ascending=False).index[0]
        st.success(f"Cluster {c}: Run targeted promotions and bulk discounts on **{dom}** products.")

    # -------------------------------
    # Task 10: Stability check
    # -------------------------------
    st.subheader("🔹 Task 10: Stability & Limitations")

    kmeans_alt = KMeans(n_clusters=k_val, init="k-means++", random_state=alt_state)
    labels_alt = kmeans_alt.fit_predict(X_scaled)

    same = (labels == labels_alt).sum()
    st.info(f"✅ Stability Check: Same assignments with random_state={alt_state}: **{same}/{len(labels)}**")

    st.warning("⚠️ Limitation: K-Means is sensitive to outliers and assumes clusters are spherical and evenly distributed.")

    # -------------------------------
    # Download result
    # -------------------------------
    st.subheader("⬇️ Download Clustered Dataset")
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV with Cluster Labels", data=csv, file_name="clustered_customers.csv", mime="text/csv")

else:
    st.info("👈 Select settings in sidebar and click **Run Clustering & Show Results**")
