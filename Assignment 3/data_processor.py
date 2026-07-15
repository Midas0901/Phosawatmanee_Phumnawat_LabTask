import pandas as pd
import geopandas as gpd

def run_data_pipeline():
    print("Step 1: Loading raw GeoJSON dataset...")
    path = "business-licences.geojson"
    gdf = gpd.read_file(path)
    df = gdf.copy()
    
    print("Step 2: Cleaning records based on assignment specs...")
    # Filter only 'Issued' status and drop missing coordinates
    df = df[df['status'] == 'Issued'].copy()
    df = df.dropna(subset=['geo_point_2d']).copy()
    
    # Extract coordinate list into individual columns
    df[['lon', 'lat']] = pd.DataFrame(df['geo_point_2d'].tolist(), index=df.index)
    df['localarea'] = df['localarea'].fillna('Unknown')
    
    print("Step 3: Consolidating industry categories (Top 15 + Other)...")
    top_types = df['businesstype'].value_counts().head(15).index
    df['industry'] = df['businesstype'].apply(lambda x: x if x in top_types else 'Other')
    
    print("Step 4: Aggregating stats and filtering sparse areas...")
    area_stats = df.groupby('localarea').agg(
        centroid_lat=('lat', 'mean'),
        centroid_lon=('lon', 'mean'),
        business_count=('industry', 'count')
    ).reset_index()
    
    # Filter thin areas out (Minimum 50 businesses cutoff)
    min_business_threshold = 50
    valid_areas = area_stats[area_stats['business_count'] >= min_business_threshold]['localarea']
    df_filtered = df[df['localarea'].isin(valid_areas)].copy()
    area_stats_filtered = area_stats[area_stats['localarea'].isin(valid_areas)].copy()
    
    print("Step 5: Building row-normalized composition matrix...")
    # Create percentage cross-tabulation matrix
    composition_matrix = pd.crosstab(df_filtered['localarea'], df_filtered['industry'], normalize='index') * 100
    composition_matrix = composition_matrix.reset_index()
    
    print("Step 6: Saving optimized light-weight outputs...")
    composition_matrix.to_csv("processed_composition.csv", index=False)
    area_stats_filtered.to_csv("processed_area_stats.csv", index=False)
    print("🎉 All data processed successfully!")

if __name__ == "__main__":
    run_data_pipeline()