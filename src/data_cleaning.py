"""
Data Cleaning and Feature Engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path

def clean_data(input_file='data/processed/bordeaux_data.csv', output_file='data/processed/bordeaux_clean.csv'):
    # Determine base directory to function anywhere
    base_dir = Path(__file__).resolve().parent.parent
    input_path = base_dir / input_file
    output_path = base_dir / output_file

    print(f"Loading {input_path}.")
    try:
        df = pd.read_csv(input_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")
        return

    print(f"Original shape: {df.shape}")

    # 1) filter "Business"
    print("\n1. Applying filters ...")

    # Keep only 'Vente'
    df_clean = df[df['nature_mutation'] == 'Vente'].copy()
    print(f"   - After filtering 'Vente': {len(df_clean)}")

    # Keep only 'Maison' and 'Appartement'
    df_clean = df_clean[df_clean['type_local'].isin(['Maison', 'Appartement'])]
    print(f"   - After filtering Housing: {len(df_clean)}")

    # Drop missing prices
    df_clean = df_clean.dropna(subset=['valeur_fonciere'])
    print(f"   - After dropping missing prices: {len(df_clean)}")

    # Check the fields needed below instead of silently continuing with an incomplete input file.
    cols_to_keep = [
        'id_mutation', 'date_mutation', 'nature_mutation', 'valeur_fonciere',
        'type_local', 'code_postal', 'nom_commune',
        'surface_reelle_bati', 'nombre_pieces_principales',
        'latitude', 'longitude'
    ]
    missing_columns = [column for column in cols_to_keep if column not in df_clean.columns]
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")
    df_clean = df_clean[cols_to_keep]

    # DVF repeats the mutation value on its component rows. we don't choose one value if the source data disagrees within a mutation
    price_counts = df_clean.groupby('id_mutation')['valeur_fonciere'].nunique(dropna=True)
    inconsistent_prices = price_counts[price_counts > 1]
    if not inconsistent_prices.empty:
        raise ValueError(
            "Found mutations with multiple fonciere values; review the source "
            f"before aggregating: {inconsistent_prices.index[:10].tolist()}"
        )

    # 3) Deduplication Logic 
    # The modeling unit is one complete mutation (transaction)
    # The transaction price is repeated on its DVF rows
    # built surfaces and room counts are sumed across the selected housing
    # rows. A sale containing both houses and apartments is labeled "Mixte"

    print("\n2. Handling multi-row transactions...")

    # Define aggregation rules
    agg_rules = {
        'date_mutation': 'first',
        'valeur_fonciere': 'first',  # Price is repeated, take one
        'type_local': lambda values: (
            values.dropna().iloc[0]
            if values.dropna().nunique() == 1
            else 'Mixte'
        ),
        'code_postal': 'first',
        'nom_commune': 'first',
        'surface_reelle_bati': 'sum', # Sum surface parts if split
        'nombre_pieces_principales': 'sum',
        'latitude': 'mean',          # Approximate center for multi-parcel sales
        'longitude': 'mean'
    }

    # Aggregating by mutation
    df_grouped = df_clean.groupby('id_mutation').agg(agg_rules).reset_index()
    print(f"   - Grouped into unique transactions: {len(df_grouped)}")

    # 4) Feature Engineering
    print("\n3. Feature Engineering...")

    # Price per sqm & avoid division by 0
    df_grouped = df_grouped[df_grouped['surface_reelle_bati'] > 0]
    df_grouped['prix_m2'] = df_grouped['valeur_fonciere'] / df_grouped['surface_reelle_bati']

    print(f"   - Calculated Price/m² (removed 0 surface rows): {len(df_grouped)}")

    # 5) Remove outliers
    print("\n4. Removing Outliers...")
    print(f"   - Price/m² constraints: 500€ < p < 15,000€")

    initial_count = len(df_grouped)
    df_final = df_grouped[
        (df_grouped['prix_m2'] > 500) &
        (df_grouped['prix_m2'] < 15000)
    ]
    removed = initial_count - len(df_final)
    print(f"   - Removed {removed} outliers")

    # 5.5) Additional outlier filtering on surface area and room count
    print("\n4a. Additional outlier filtering...")
    # Remove extreme surface outliers (likly non residential properties)
    surface_before = len(df_final)
    df_final = df_final[df_final['surface_reelle_bati'] <= 500]  # <= 500 m²
    surface_removed = surface_before - len(df_final)
    print(f"   - Removed {surface_removed} extreme surface outliers (>500 m²)")

    # nombre_pieces_principales > 0 to prevent division by zero in features
    pieces_before = len(df_final)
    df_final = df_final[df_final['nombre_pieces_principales'] > 0]  # > 0 rooms
    pieces_removed = pieces_before - len(df_final)
    print(f"   - Removed {pieces_removed} rooms with zero or negative room count")

    # 6) Additional cleaning: drop rows with missing latitude & longitude
    print("\n5. Dropping missing latitude/longitude...")
    before = len(df_final)
    df_final = df_final.dropna(subset=['latitude', 'longitude'])
    after = len(df_final)
    print(f"   - Removed {before - after} rows with missing latitude/longitude")

    # 7) Save
    print(f"\nSaving to {output_path}...")
    df_final.to_csv(output_path, index=False)
    print(f"Done! Final dataset shape: {df_final.shape}")

    # Final check on stats
    print("\nFinal Stats (Price/m²):")
    print(df_final['prix_m2'].describe())

if __name__ == "__main__":
    clean_data()
