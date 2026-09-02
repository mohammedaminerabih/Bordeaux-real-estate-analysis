"""
Feature Engineering for Bordeaux Real Estate Analysis
Create safe features for predicting prix_m2 (price per square meter)
Strictly avoids data leakage: excludes valeur_fonciere, surface_reelle_bati,
and any transformations 
"""

import pandas as pd
import numpy as np
from pathlib import Path
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth in decimal degrees
    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    # Radius of earth
    r = 6371.0
    return c * r

def load_cleaned_data(filepath='data/processed/bordeaux_clean.csv'):
    """Load cleaned dataset"""
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / filepath
    df = pd.read_csv(csv_path)
    print(f"Loaded cleaned data with shape: {df.shape}")
    return df

def engineer_features(df):
    """Generate safe features from cleaned data"""
    print("Starting feature engineering...")
    # To avoid warnings
    df_features = df.copy()
    # Drop leakage columns that shouldnt be used as features
    # Keep prix_m2 as target variable
    df_features = df_features.drop(columns=['valeur_fonciere', 'surface_reelle_bati'], errors='ignore')

    # Ensure date_mutation is datetime
    df_features['date_mutation'] = pd.to_datetime(df_features['date_mutation'])

    # These are already in the dataset and safe to use
    # (excluded: valeur_fonciere, surface_reelle_bati, prix_m2)
    safe_originals = ['id_mutation', 'date_mutation', 'type_local',
                      'code_postal', 'nom_commune', 'nombre_pieces_principales',
                      'latitude', 'longitude']

    # TEMPORAL FEATURES from date_mutation
    df_features['date_year'] = df_features['date_mutation'].dt.year
    df_features['date_month'] = df_features['date_mutation'].dt.month
    df_features['date_quarter'] = df_features['date_mutation'].dt.quarter
    df_features['date_dayofweek'] = df_features['date_mutation'].dt.dayofweek  # Monday=0
    # Season: Spring (3-5), Summer (6-8), Fall (9-11), Winter (12,1,2)
    df_features['date_season'] = df_features['date_month'].apply(
        lambda m: 'winter' if m in [12,1,2] else
                  'spring' if m in [3,4,5] else
                  'summer' if m in [6,7,8] else 'fall'
    )

    # PROPERTY TYPE FEATURES
    # Binary encoding: Maison=1, Appartement=0
    df_features['type_local_encoded'] = (df_features['type_local'] == 'Maison').astype(int)

    # POSTAL CODE FEATURES
    # Frequency encoding: how common each postal code is
    postal_counts = df_features['code_postal'].value_counts()
    df_features['code_postal_frequency'] = df_features['code_postal'].map(postal_counts)

    # Top 5 postal codes binary feature
    top5_postals = postal_counts.head(5).index.tolist()
    for i, postal in enumerate(top5_postals, start=1):
        df_features[f'code_postal_top{i}'] = (df_features['code_postal'] == postal).astype(int)

    # Area group based on first three digits of postal code (all are 33xxx, so we use full code for grouping)
    # Simpler: assign to bins based on postal code ranges
    df_features['code_postal_area'] = pd.cut(df_features['code_postal'],
                                             bins=[0, 33099, 33199, 33299, 33399, 33499, 33599, 33699, 33799, 33899, 33999, np.inf],
                                             labels=['other','33000-33099','33100-33199','33200-33299','33300-33399',
                                                     '33400-33499','33500-33599','33600-33699','33700-33799','33800-33899','33900-33999'],
                                             right=False)
    # Optionnel - onvert to categorical codes for modeling 
    df_features['code_postal_area_code'] = df_features['code_postal_area'].cat.codes

    # GEOGRAPHICAL FEATURES
    # Keep raw latitude and longitude
    # Distance to Bordeaux city center (place de la bourse)
    BORDEAUX_CENTER_LAT = 44.8378
    BORDEAUX_CENTER_LON = -0.5792

    df_features['distance_to_center_km'] = df_features.apply(
        lambda row: haversine_distance(row['latitude'], row['longitude'],
                                       BORDEAUX_CENTER_LAT, BORDEAUX_CENTER_LON),
        axis=1
    )

    # Binned distance categories
    df_features['distance_to_center_bin'] = pd.cut(df_features['distance_to_center_km'],
                                                   bins=[0, 2, 5, 10, np.inf],
                                                   labels=['very_close','close','medium','far'],
                                                   right=False)
    df_features['distance_to_center_bin_code'] = df_features['distance_to_center_bin'].cat.codes

    # Simple latitude/longitude bins (alternative to raw)
    df_features['latitude_bin'] = pd.cut(df_features['latitude'],
                                         bins=[44.0, 44.5, 44.8, 45.0, np.inf],
                                         labels=['south','mid_south','mid_north','north'],
                                         right=False)
    df_features['latitude_bin_code'] = df_features['latitude_bin'].cat.codes

    df_features['longitude_bin'] = pd.cut(df_features['longitude'],
                                          bins=[-2.0, -1.0, -0.6, -0.2, 0.0, np.inf],
                                          labels=['west','mid_west','center','mid_east','east'],
                                          right=False)
    df_features['longitude_bin_code'] = df_features['longitude_bin'].cat.codes

    # PROPERTY CHARACTERISTICS (nombre_pieces_principales)
    # Frequency encoding
    pieces_counts = df_features['nombre_pieces_principales'].value_counts()
    df_features['nombre_pieces_frequency'] = df_features['nombre_pieces_principales'].map(pieces_counts)

    # Raw feature
    # Already have nombre_pieces_principales

    # Polynomial features
    df_features['nombre_pieces_squared'] = df_features['nombre_pieces_principales'] ** 2
    df_features['nombre_pieces_cubed'] = df_features['nombre_pieces_principales'] ** 3

    # Binned categories
    df_features['nombre_pieces_binned'] = pd.cut(df_features['nombre_pieces_principales'],
                                                 bins=[0, 1, 2, 3, 4, 5, 6, 10, np.inf],
                                                 labels=['studio','1_room','2_rooms','3_rooms','4_rooms','5_rooms','6_rooms','7+_rooms'],
                                                 right=False)
    df_features['nombre_pieces_binned_code'] = df_features['nombre_pieces_binned'].cat.codes

    # INTERACTION FEATURES (safe combinations) 
    # Interaction between property type and geographical features
    df_features['type_local_x_latitude'] = df_features['type_local_encoded'] * df_features['latitude']
    df_features['type_local_x_longitude'] = df_features['type_local_encoded'] * df_features['longitude']
    df_features['type_local_x_distance'] = df_features['type_local_encoded'] * df_features['distance_to_center_km']

    # Interaction between pieces and geographical features
    df_features['pieces_x_latitude'] = df_features['nombre_pieces_principales'] * df_features['latitude']
    df_features['pieces_x_longitude'] = df_features['nombre_pieces_principales'] * df_features['longitude']
    df_features['pieces_x_distance'] = df_features['nombre_pieces_principales'] * df_features['distance_to_center_km']

    # Interaction between property type and pieces
    df_features['type_local_x_pieces'] = df_features['type_local_encoded'] * df_features['nombre_pieces_principales']

    # FINAL FEATURE SET 
    # Identify all feature columns (exclude target and identifiers if needed for modeling)
    # We will keep all engineered features plus safe originals (except leakage ones)
    # For modeling, we will later exclude: id_mutation, date_mutation (original), nom_commune (maybe redundant)
    # But for the feature dataset we keep everything for transparency.

    # List of columns to exclude from modeling (non-predictive or leakage)
    # These will be identified later in train/test split script
    non_predictive = ['id_mutation', 'date_mutation', 'nom_commune',
                      'valeur_fonciere', 'surface_reelle_bati', 'prix_m2']
    # Note: valeur_fonciere, surface_reelle_bati, prix_m2 are not in df_features (we never added them)

    # Ensure we did noy accidentally create any leakage features
    leakage_check = ['valeur_fonciere', 'surface_reelle_bati',
                     'log_surface', 'log_valeur_fonciere', 'log_prix_m2',
                     'pieces_per_surface', 'surface_per_piece', 'price_per_room']
    unexpected_leakage = [col for col in leakage_check if col in df_features.columns]
    if unexpected_leakage:
        raise ValueError(f"Leakage features detected: {unexpected_leakage}")

    print(f"Feature engineering complete. Dataset shape: {df_features.shape}")
    print(f"Generated {df_features.shape[1] - len(safe_originals)} new features.")

    return df_features

def save_features(df, filename='data/processed/bordeaux_features.csv'):
    """Save feature-enriched dataset"""
    base_dir = Path(__file__).resolve().parent.parent
    output_path = base_dir / filename
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved feature-enriched dataset to: {output_path}")
    print(f"Shape: {df.shape}")

def main():
    """Main feature engineering pipeline"""
    print("=" * 60)
    print("   BORDEAUX REAL ESTATE ANALYSIS - PHASE 6: FEATURE ENGINEERING")
    print("=" * 60)
    print()

    # Load cleaned data
    df_clean = load_cleaned_data()

    # Engineer features
    df_features = engineer_features(df_clean)

    # Save results
    save_features(df_features)

    # Print summary of feature types
    print("\nFeature Summary:")
    print("- Temporal features: date_year, date_month, date_quarter, date_dayofweek, date_season")
    print("- Property type: type_local_encoded, type_local_frequency")
    print("- Postal code: frequency, top5 binaries, area groups, distance transformations")
    print("- Geographical: latitude, longitude, distance_to_center_km, binned versions")
    print("- Property characteristics: nombre_pieces_principale (raw, squared, cubed, frequency, binned)")
    print("- Interactions: type_local x geo, pieces x geo, type_local x pieces")
    print()
    print("All features are safe: none derived from valeur_fonciere or surface_reelle_bati")
    print("Ready for Phase 7: Train/test split + baseline modeling")

if __name__ == "__main__":
    main()