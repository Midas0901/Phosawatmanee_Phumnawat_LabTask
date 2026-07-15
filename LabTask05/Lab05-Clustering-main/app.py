import streamlit as st
import json
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title("Business Location Explorer - Lab 5")

@st.cache_data
def load_data(path="business_locations.geojson"):
    with open(path) as f:
        geojson = json.load(f)
    rows = []
    for feat in geojson["features"]:
        props = feat["properties"]
        lon, lat = feat["geometry"]["coordinates"]
        rows.append({**props, "lon": lon, "lat": lat})
    return pd.DataFrame(rows)

df = load_data()

with st.expander("🔍 View Raw Data"):
    st.dataframe(df.head(20))
    st.write(f"Total Locations: {len(df)} | Neighborhoods: {df['Neighborhood'].nunique()}")

st.sidebar.header("1. Select Features")
NUMERIC_COLS = ["Floor_Area_sqm", "Daily_Foot_Traffic", "Community_Impact_Score", "Annual_Revenue_k"]
selected_features = st.sidebar.multiselect("Features", NUMERIC_COLS, default=NUMERIC_COLS)

if len(selected_features) < 2:
    st.warning("Please select at least 2 features")
    st.stop()

X = df[selected_features].to_numpy()
X_scaled = StandardScaler().fit_transform(X)

st.sidebar.header("2. Clustering Algorithm")
algo = st.sidebar.selectbox('Algorithm', ["Kmeans", "DBSCAN"])

# Clustering
if algo == "Kmeans":
    k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 4)
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X_scaled)
else:  # DBSCAN
    eps = st.sidebar.slider("eps (distance)", 0.1, 3.0, 0.5, step=0.05)
    min_samples = st.sidebar.slider("min_samples", 3, 30, 5)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)

df["cluster"] = labels.astype(str)

# Metrics
n_clusters = len(set(labels)) if -1 not in labels else len(set(labels)) - 1
noise_count = list(labels).count(-1)

st.metric("Number of Clusters", n_clusters, 
          delta=f"{noise_count} Noise Points" if noise_count > 0 else None)

# === Additional Graphs for DBSCAN ===
if algo == "DBSCAN":
    st.subheader("📊 DBSCAN Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Cluster Distribution**")
        cluster_counts = df["cluster"].value_counts().reset_index()
        cluster_counts.columns = ["Cluster", "Count"]
        fig_bar = px.bar(cluster_counts, x="Cluster", y="Count", 
                        color="Cluster", title="Points per Cluster")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        if noise_count > 0:
            st.info(f"⚠️ Found **{noise_count} Noise points** (labeled as -1)")

# Tabs
tab1, tab2 = st.tabs(["🗺️ Map", "📉 PCA"])

with tab1:
    fig_map = px.scatter_map(df, lat="lat", lon="lon", color="cluster",
                            zoom=10, height=600, map_style="carto-darkmatter",
                            hover_data=["Neighborhood", "Category", "Annual_Revenue_k"])
    st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(X_scaled)
    df["pca1"] = embedding[:,0]
    df["pca2"] = embedding[:,1]
    
    fig_pca = px.scatter(df, x="pca1", y="pca2", color="cluster", height=600,
                        hover_data=["Neighborhood"])
    st.plotly_chart(fig_pca, use_container_width=True)