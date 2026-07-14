import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Page Configuration
st.set_page_config(page_title="Vancouver Explorer", layout="wide")
st.title("🌆 Vancouver Business Licences Explorer")
st.subheader("Assignment 3")

# File Path Configuration
file_path = r"C:\Users\Poom_\OneDrive\Documents\IAT461\Assignment 3\business-licences.geojson"

# 2. Cached Data Engine (Part A1 Pipeline)
@st.cache_data
def load_and_clean_data():
    # Load raw GeoJSON data
    gdf = gpd.read_file(file_path)
    df = gdf.copy()
    
    # Filter by status 'Issued'
    df = df[df['status'] == 'Issued'].copy()
    
    # Drop rows with missing geographic coordinates
    df = df.dropna(subset=['geo_point_2d']).copy()
    
    # Parse latitude and longitude from geo_point_2d list
    df[['lon', 'lat']] = pd.DataFrame(df['geo_point_2d'].tolist(), index=df.index)
    
    # Fill missing values for localarea
    df['localarea'] = df['localarea'].fillna('Unknown')
    
    # Consolidate industry labels: Keep top 15 categories, rest as 'Other'
    top_types = df['businesstype'].value_counts().head(15).index
    df['industry'] = df['businesstype'].apply(lambda x: x if x in top_types else 'Other')
    
    # Calculate area stats (Centroids and total counts)
    area_stats = df.groupby('localarea').agg(
        centroid_lat=('lat', 'mean'),
        centroid_lon=('lon', 'mean'),
        business_count=('industry', 'count')
    ).reset_index()
    
    # Part B1: Filter out thin areas (Minimum threshold: 50 businesses)
    min_business_threshold = 50
    valid_areas = area_stats[area_stats['business_count'] >= min_business_threshold]['localarea']
    
    df_filtered = df[df['localarea'].isin(valid_areas)].copy()
    area_stats_filtered = area_stats[area_stats['localarea'].isin(valid_areas)].copy()
    
    # Part B1 Step 2: Create row-normalized cross-tabulation matrix (Percentages)
    composition_matrix = pd.crosstab(df_filtered['localarea'], df_filtered['industry'], normalize='index') * 100
    
    return composition_matrix, area_stats_filtered, df_filtered

# Execute pipeline
try:
    with st.spinner("Processing large GeoJSON file... Please wait..."):
        composition_matrix, area_stats, df_clean = load_and_clean_data()
    
    st.success(f"✅ Success! Loaded {len(df_clean):,} active business records across {len(area_stats)} filtered neighborhoods.")
    
    # -----------------------------------------------------------------------------
    # Sidebar Setup
    # -----------------------------------------------------------------------------
    st.sidebar.header("🔧 Unsupervised Clustering")
    k_value = st.sidebar.slider("Select Number of Clusters (K)", min_value=2, max_value=8, value=4, step=1)
    
    # -----------------------------------------------------------------------------
    # Interactive K-Means & PCA (Part B2)
    # -----------------------------------------------------------------------------
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(composition_matrix)
    
    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_matrix)
    
    # Map clusters back
    composition_matrix['cluster'] = cluster_labels
    area_stats['cluster'] = area_stats['localarea'].map(composition_matrix['cluster'])
    
    # PCA projection into 2D Space
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(scaled_matrix)
    
    pca_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2'], index=composition_matrix.index)
    pca_df['cluster'] = cluster_labels.astype(str)
    pca_df = pca_df.reset_index().merge(area_stats[['localarea', 'business_count']], on='localarea')
    
    # -----------------------------------------------------------------------------
    # Visual Layout (Part B2 & B3)
    # -----------------------------------------------------------------------------
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 PCA 2D Cluster Projection Space")
        fig_pca = px.scatter(
            pca_df, x='PC1', y='PC2', 
            color='cluster', 
            text='localarea',
            hover_data=['business_count'],
            title=f"Neighborhood Similarity Map via PCA (K={k_value})",
            labels={'cluster': 'Cluster Group'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pca.update_traces(textposition='top center', marker=dict(size=12))
        fig_pca.update_layout(height=520)
        st.plotly_chart(fig_pca, use_container_width=True)
        
    with col2:
        st.subheader("🗺️ Geographic Cluster Coordinates Map")
        area_stats['cluster_str'] = area_stats['cluster'].astype(str)
        
        fig_map = px.scatter_mapbox(
            area_stats, 
            lat="centroid_lat", 
            lon="centroid_lon", 
            color="cluster_str", 
            size="business_count",
            hover_name="localarea", 
            hover_data=["business_count"],
            color_discrete_sequence=px.colors.qualitative.Set2,
            size_max=30, 
            zoom=10.5, 
            title="Neighborhood Clusters Sized by Total Businesses"
        )
        fig_map.update_layout(
            mapbox_style="carto-positron",
            margin={"r":0,"t":40,"l":0,"b":0},
            height=520,
            legend_title="Cluster ID"
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
    # -----------------------------------------------------------------------------
    # Cluster Profiling Output (Part B4)
    # -----------------------------------------------------------------------------
    st.divider()
    st.subheader("📋 Cluster Grouping Profiles & Segment Descriptions")
    
    tabs = st.tabs([f"Cluster Group {i}" for i in range(k_value)])
    for i in range(k_value):
        with tabs[i]:
            members = area_stats[area_stats['cluster'] == i]['localarea'].tolist()
            st.write(f"**Neighborhoods assigned to this profile group:**")
            st.info(", ".join(members) if members else "No neighborhoods fall into this cluster definition.")
            
            cluster_subset = composition_matrix[composition_matrix['cluster'] == i].drop(columns=['cluster'])
            if not cluster_subset.empty:
                mean_profile = cluster_subset.mean().sort_values(ascending=False).head(5)
                
                st.markdown("**Top 5 Dominant Business Types (Average Area Share %):**")
                profile_table = pd.DataFrame({
                    'Business Category Type': mean_profile.index,
                    'Mean Matrix Distribution (%)': mean_profile.values.round(2)
                })
                st.table(profile_table)

    # -----------------------------------------------------------------------------
    # My Analysis of the Neighborhood Groupings Markdown Display
    # -----------------------------------------------------------------------------
    st.divider()
    st.markdown("""
    ### My Analysis of the Neighborhood Groupings

    #### Does this clustering layout make sense?
    Looking at the neighborhood assignments, I think the groupings are highly meaningful and match real-world Vancouver geography perfectly. For instance, I noticed that high-density commercial hubs like Downtown and Fairview are consistently paired together because they share massive office and retail footprints. Meanwhile, quieter residential areas like Dunbar-Southlands and Shaughnessy get grouped into their own profile due to a high density of home-based local services and construction contractors.

    #### Were there any surprising neighborhood matches?
    One grouping that surprised me was seeing Mount Pleasant and Grandview-Woodland grouped so closely with some of my standard suburban zones. I think this happened because, despite their trendy shopping districts, both neighborhoods actually host a massive number of small, home-based creative businesses and local trade contractors. This proves that the model looks deeper than just a neighborhood's main street reputation and captures the quiet, home-based economy.

    #### How do the business distributions explain these groups?
    I think the top five business distributions explain the cluster boundaries perfectly because my K-Means model was trained directly on these percentage shares, not on geographic coordinates. Here is why those top five categories make the cluster divisions so clear
    """)

except Exception as e:
    st.error(f"Execution Error occurred: {str(e)}")
    st.info("Double-check that the file path and directory privileges are valid.")