"""
Data Cleaning and Feature Engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path

def clean_data(input_file='data/processed/bordeaux_data.csv', output_file='data/processed/bordeaux_clean.csv'):
    # Determine base directory (parent of src)
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

    # 2) Select Columns
    cols_to_keep = [
        'id_mutation', 'date_mutation', 'valeur_fonciere',
        'type_local', 'code_postal', 'nom_commune',
        'surface_reelle_bati', 'nombre_pieces_principales',
        'latitude', 'longitude'
    ]
    # Ensure columns exist before selecting
    cols_to_keep = [c for c in cols_to_keep if c in df_clean.columns]
    df_clean = df_clean[cols_to_keep]

    # 3) Deduplication Logic 
    # We must be careful here. If a house has 2 rows (House + Garden), usually only the House row has surface_reelle_bati
    # The price is on both
    # Our Strategy is to group by mutation and sum surface, but take 1st of price since its repeated

    print("\n2. Handling multi-row transactions...")

    # Define aggregation rules
    agg_rules = {
        'date_mutation': 'first',
        'valeur_fonciere': 'first',  # Price is repeated, take one
        'type_local': 'first',       # To simplify take main type
        'code_postal': 'first',
        'nom_commune': 'first',
        'surface_reelle_bati': 'sum', # Sum surface parts if split
        'nombre_pieces_principales': 'sum',
        'latitude': 'first',
        'longitude': 'first'
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