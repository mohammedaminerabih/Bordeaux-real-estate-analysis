"""
Train/test split + baseline modeling

Approach to prevent data leakage:
1. Split cleaned data into train/test sets first (before any feature engineering)
2. Compute all aggregate-based features exclusively on the training set
3. Apply the same transformations to both training and test sets
4. Generate row independent features separately on each set
5. Train baseline linear regression model
6. Evaluate performance on both sets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def haversine_distance(lat1, lon1, lat2, lon2):
    """ Calculate the great circle distance between two points
        on the earth in decimal degrees
        Returns distance in kilometers"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversyne formula
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

def split_data(df, test_size=0.2, random_state=42):
    """Split data into training and test sets"""
    print(f"Splitting data into train/test sets (test_size={test_size}) ")

    # Separate features and target
    # We will keep all columns except the target for now,
    # and remove leakage columns and identifiers during feature engineering
    X = df.drop(columns=['prix_m2'])  # Features (all except target)
    y = df['prix_m2']                # Target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test

def engineer_features_train(X_train):
    """Engineer features on TRAINING SET ONLY
       Returns the feature-engineered training set and the transformation objects
    """
    print("Engineering features on training set ")

    # Start with a copy
    df_train = X_train.copy()

    # Drop leakage columns that should not be used as features
    # Keep prix_m2 as target variable (its not in X_train)
    df_train = df_train.drop(columns=['valeur_fonciere', 'surface_reelle_bati'], errors='ignore')

    # Ensure date_mutation is datetime
    df_train['date_mutation'] = pd.to_datetime(df_train['date_mutation'])

    # These are already in the dataset and safe to use as always: excluded: valeur_fonciere, surface_reelle_bati, prix_m2 - target is separate)
    safe_originals = ['id_mutation', 'date_mutation', 'type_local',
                      'code_postal', 'nom_commune', 'nombre_pieces_principales',
                      'latitude', 'longitude']

    #   TEMPORAL FEATURES 
    df_train['date_year'] = df_train['date_mutation'].dt.year
    df_train['date_month'] = df_train['date_mutation'].dt.month
    df_train['date_quarter'] = df_train['date_mutation'].dt.quarter
    df_train['date_dayofweek'] = df_train['date_mutation'].dt.dayofweek  # Monday=0
    # Season: Spring (3-5), Summer (6-8), Fall (9-11), Winter (12,1,2)
    df_train['date_season'] = df_train['date_month'].apply(
        lambda m: 'winter' if m in [12,1,2] else
                  'spring' if m in [3,4,5] else
                  'summer' if m in [6,7,8] else 'fall'
    )

    #   PROPERTY TYPE FEATURES  
    # Binary encoding: Maison=1, Appartement=0
    df_train['type_local_encoded'] = (df_train['type_local'] == 'Maison').astype(int)
    # Frequency encoding (computed on training set only)
    type_local_counts = df_train['type_local'].value_counts()
    df_train['type_local_frequency'] = df_train['type_local'].map(type_local_counts)

    #   POSTAL CODE FEATURES  
    # Frequency encoding: how common each postal code is (TRAINING SET ONLY)
    postal_counts = df_train['code_postal'].value_counts()
    df_train['code_postal_frequency'] = df_train['code_postal'].map(postal_counts)

    # Top 5 postal codes binary feature (based on TRAINING SET)
    top5_postals = postal_counts.head(5).index.tolist()
    for i, postal in enumerate(top5_postals, start=1):
        df_train[f'code_postal_top{i}'] = (df_train['code_postal'] == postal).astype(int)

    # Area group based on postal code ranges (based on TRAINING SET distribution)
    # Simpler: assign to bins based on postal code ranges
    df_train['code_postal_area'] = pd.cut(df_train['code_postal'],
                                         bins=[0, 33099, 33199, 33299, 33399, 33499, 33599, 33699, 33799, 33899, 33999, np.inf],
                                         labels=['other','33000-33099','33100-33199','33200-33299','33300-33399',
                                                 '33400-33499','33500-33599','33600-33699','33700-33799','33800-33899','33900-33999'],
                                         right=False)
    # Convert to categorical codes for modeling
    df_train['code_postal_area_code'] = df_train['code_postal_area'].cat.codes

    #   GEOGRAPHICAL FEATURES  
    # Keep raw latitude and longitude
    # Distance to Bordeaux city center (place de la bourse)
    BORDEAUX_CENTER_LAT = 44.8378
    BORDEAUX_CENTER_LON = -0.5792

    df_train['distance_to_center_km'] = df_train.apply(
        lambda row: haversine_distance(row['latitude'], row['longitude'],
                                       BORDEAUX_CENTER_LAT, BORDEAUX_CENTER_LON),
        axis=1
    )

    # Binned distance categories (based on TRAINING SET distribution)
    df_train['distance_to_center_bin'] = pd.cut(df_train['distance_to_center_km'],
                                               bins=[0, 2, 5, 10, np.inf],
                                               labels=['very_close','close','medium','far'],
                                               right=False)
    df_train['distance_to_center_bin_code'] = df_train['distance_to_center_bin'].cat.codes

    # Simple latitude/longitude bins (alternative to raw)
    df_train['latitude_bin'] = pd.cut(df_train['latitude'],
                                      bins=[44.0, 44.5, 44.8, 45.0, np.inf],
                                      labels=['south','mid_south','mid_north','north'],
                                      right=False)
    df_train['latitude_bin_code'] = df_train['latitude_bin'].cat.codes

    df_train['longitude_bin'] = pd.cut(df_train['longitude'],
                                       bins=[-2.0, -1.0, -0.6, -0.2, 0.0, np.inf],
                                       labels=['west','mid_west','center','mid_east','east'],
                                       right=False)
    df_train['longitude_bin_code'] = df_train['longitude_bin'].cat.codes

    #   PROPERTY CHARACTERISTICS (nombre_pieces_principales)  
    # Frequency encoding (TRAINING SET ONLY)
    pieces_counts = df_train['nombre_pieces_principales'].value_counts()
    df_train['nombre_pieces_frequency'] = df_train['nombre_pieces_principales'].map(pieces_counts)

    # Raw feature (keep)
    # Already have nombre_pieces_principales

    # Polynomial features
    df_train['nombre_pieces_squared'] = df_train['nombre_pieces_principales'] ** 2
    df_train['nombre_pieces_cubed'] = df_train['nombre_pieces_principales'] ** 3

    # Binned categories (based on TRAINING SET distribution)
    df_train['nombre_pieces_binned'] = pd.cut(df_train['nombre_pieces_principales'],
                                              bins=[0, 1, 2, 3, 4, 5, 6, 10, np.inf],
                                              labels=['studio','1_room','2_rooms','3_rooms','4_rooms','5_rooms','6_rooms','7+_rooms'],
                                              right=False)
    df_train['nombre_pieces_binned_code'] = df_train['nombre_pieces_binned'].cat.codes

    #   INTERACTION FEATURES (safe combinations)  
    # Interaction between property type and geographical features
    df_train['type_local_x_latitude'] = df_train['type_local_encoded'] * df_train['latitude']
    df_train['type_local_x_longitude'] = df_train['type_local_encoded'] * df_train['longitude']
    df_train['type_local_x_distance'] = df_train['type_local_encoded'] * df_train['distance_to_center_km']

    # Interaction between pieces and geographical features
    df_train['pieces_x_latitude'] = df_train['nombre_pieces_principales'] * df_train['latitude']
    df_train['pieces_x_longitude'] = df_train['nombre_pieces_principales'] * df_train['longitude']
    df_train['pieces_x_distance'] = df_train['nombre_pieces_principales'] * df_train['distance_to_center_km']

    # Interaction between property type and pieces
    df_train['type_local_x_pieces'] = df_train['type_local_encoded'] * df_train['nombre_pieces_principales']

    # Store transformation objects for applying to test set
    transform_objects = {
        'postal_counts': postal_counts,
        'top5_postals': top5_postals,
        'type_local_counts': type_local_counts,
        'code_postal_area_bins': [0, 33099, 33199, 33299, 33399, 33499, 33599, 33699, 33799, 33899, 33999, np.inf],
        'code_postal_area_labels': ['other','33000-33099','33100-33199','33200-33299','33300-33399',
                                    '33400-33499','33500-33599','33600-33699','33700-33799','33800-33899','33900-33999'],
        'distance_bins': [0, 2, 5, 10, np.inf],
        'distance_labels': ['very_close','close','medium','far'],
        'latitude_bins': [44.0, 44.5, 44.8, 45.0, np.inf],
        'latitude_labels': ['south','mid_south','mid_north','north'],
        'longitude_bins': [-2.0, -1.0, -0.6, -0.2, 0.0, np.inf],
        'longitude_labels': ['west','mid_west','center','mid_east','east'],
        'pieces_counts': pieces_counts,
        'pieces_bins': [0, 1, 2, 3, 4, 5, 6, 10, np.inf],
        'pieces_labels': ['studio','1_room','2_rooms','3_rooms','4_rooms','5_rooms','6_rooms','7+_rooms'],
        'BORDEAUX_CENTER_LAT': BORDEAUX_CENTER_LAT,
        'BORDEAUX_CENTER_LON': BORDEAUX_CENTER_LON
    }

    print(f"Feature engineering on training set complete. Shape: {df_train.shape}")

    return df_train, transform_objects

def engineer_features_test(X_test, transform_objects):
    """Engineer features on TEST SET using transformations derived from the training set
    Returns the feature-engineered test set"""
    print("Engineering features on test set using training derived transformations ")

    # Start with a copy
    df_test = X_test.copy()

    # Drop leakage columns that should not be used as features
    # Keep prix_m2 as target variable (it's not in X_test)
    df_test = df_test.drop(columns=['valeur_fonciere', 'surface_reelle_bati'], errors='ignore')

    # Ensure date_mutation is datetime
    df_test['date_mutation'] = pd.to_datetime(df_test['date_mutation'])

    #   TEMPORAL FEATURES  
    df_test['date_year'] = df_test['date_mutation'].dt.year
    df_test['date_month'] = df_test['date_mutation'].dt.month
    df_test['date_quarter'] = df_test['date_mutation'].dt.quarter
    df_test['date_dayofweek'] = df_test['date_mutation'].dt.dayofweek  # Monday=0
    # Season: Spring (3-5), Summer (6-8), Fall (9-11), Winter (12,1,2)
    df_test['date_season'] = df_test['date_month'].apply(
        lambda m: 'winter' if m in [12,1,2] else
                  'spring' if m in [3,4,5] else
                  'summer' if m in [6,7,8] else 'fall'
    )

    #   PROPERTY TYPE FEATURES  
    # Binary encoding: Maison=1, Appartement=0
    df_test['type_local_encoded'] = (df_test['type_local'] == 'Maison').astype(int)
    # Frequency encoding (using TRAINING SET mapping)
    df_test['type_local_frequency'] = df_test['type_local'].map(transform_objects['type_local_counts'])
    # Fill NaN for unseen categories with 0 (or could use min/avg frequency)
    df_test['type_local_frequency'] = df_test['type_local_frequency'].fillna(0)

    #   POSTAL CODE FEATURES  
    # Frequency encoding: using TRAINING SET mapping
    df_test['code_postal_frequency'] = df_test['code_postal'].map(transform_objects['postal_counts'])
    # Fill NaN for unseen postal codes with 0
    df_test['code_postal_frequency'] = df_test['code_postal_frequency'].fillna(0)

    # Top 5 postal codes binary feature (using TRAINING SET top 5)
    for i, postal in enumerate(transform_objects['top5_postals'], start=1):
        df_test[f'code_postal_top{i}'] = (df_test['code_postal'] == postal).astype(int)

    # Area group based on postal code ranges (using TRAINING SET bins)
    df_test['code_postal_area'] = pd.cut(df_test['code_postal'],
                                        bins=transform_objects['code_postal_area_bins'],
                                        labels=transform_objects['code_postal_area_labels'],
                                        right=False)
    # Convert to categorical codes for modeling
    df_test['code_postal_area_code'] = df_test['code_postal_area'].cat.codes

    #   GEOGRAPHICAL FEATURES  
    # Keep raw latitude and longitude
    # Distance to Bordeaux city center (place de la bourse)
    df_test['distance_to_center_km'] = df_test.apply(
        lambda row: haversine_distance(row['latitude'], row['longitude'],
                                       transform_objects['BORDEAUX_CENTER_LAT'],
                                       transform_objects['BORDEAUX_CENTER_LON']),
        axis=1
    )

    # Binned distance categories (using TRAINING SET bins)
    df_test['distance_to_center_bin'] = pd.cut(df_test['distance_to_center_km'],
                                              bins=transform_objects['distance_bins'],
                                              labels=transform_objects['distance_labels'],
                                              right=False)
    df_test['distance_to_center_bin_code'] = df_test['distance_to_center_bin'].cat.codes

    # Simple latitude/longitude bins (alternative to raw)
    df_test['latitude_bin'] = pd.cut(df_test['latitude'],
                                     bins=transform_objects['latitude_bins'],
                                     labels=transform_objects['latitude_labels'],
                                     right=False)
    df_test['latitude_bin_code'] = df_test['latitude_bin'].cat.codes

    df_test['longitude_bin'] = pd.cut(df_test['longitude'],
                                      bins=transform_objects['longitude_bins'],
                                      labels=transform_objects['longitude_labels'],
                                      right=False)
    df_test['longitude_bin_code'] = df_test['longitude_bin'].cat.codes

    #   PROPERTY CHARACTERISTICS (nombre_pieces_principales)  
    # Frequency encoding (using TRAINING SET mapping)
    df_test['nombre_pieces_frequency'] = df_test['nombre_pieces_principales'].map(transform_objects['pieces_counts'])
    # Fill NaN for unseen values with 0
    df_test['nombre_pieces_frequency'] = df_test['nombre_pieces_frequency'].fillna(0)

    # Raw feature (keep)
    # Already have nombre_pieces_principales

    # Polynomial features
    df_test['nombre_pieces_squared'] = df_test['nombre_pieces_principales'] ** 2
    df_test['nombre_pieces_cubed'] = df_test['nombre_pieces_principales'] ** 3

    # Binned categories (using TRAINING SET bins)
    df_test['nombre_pieces_binned'] = pd.cut(df_test['nombre_pieces_principales'],
                                             bins=transform_objects['pieces_bins'],
                                             labels=transform_objects['pieces_labels'],
                                             right=False)
    df_test['nombre_pieces_binned_code'] = df_test['nombre_pieces_binned'].cat.codes

    #   INTERACTION FEATURES (safe combinations)  
    # Interaction between property type and geographical features
    df_test['type_local_x_latitude'] = df_test['type_local_encoded'] * df_test['latitude']
    df_test['type_local_x_longitude'] = df_test['type_local_encoded'] * df_test['longitude']
    df_test['type_local_x_distance'] = df_test['type_local_encoded'] * df_test['distance_to_center_km']

    # Interaction between pieces and geographical features
    df_test['pieces_x_latitude'] = df_test['nombre_pieces_principales'] * df_test['latitude']
    df_test['pieces_x_longitude'] = df_test['nombre_pieces_principales'] * df_test['longitude']
    df_test['pieces_x_distance'] = df_test['nombre_pieces_principales'] * df_test['distance_to_center_km']

    # Interaction between property type and pieces
    df_test['type_local_x_pieces'] = df_test['type_local_encoded'] * df_test['nombre_pieces_principales']

    print(f"Feature engineering on test set complete. Shape: {df_test.shape}")

    return df_test

def prepare_modeling_data(df_features):
    """Prepare final modeling data by selecting features to use
    Excludes identifiers and non-predictive columns"""
    
    # Identify columns to exclude from modeling
    # These are either identifiers, dates (we use extracted features),
    # original categorical columns (we use encoded versions),
    # binned categorical columns (we use the _code versions),
    # or potentially redundant/non-predictive
    exclude_cols = [
        'id_mutation',           # Identifier
        'date_mutation',         # Original date (we use extracted features)
        'type_local',            # Original categorical (we use type_local_encoded)
        'code_postal',           # Original categorical (we use frequency and top5 features)
        'nom_commune',           # Likely redundant with code_postal
        'date_season',           # Categorical season (we use more granular temporal features)
        'code_postal_area',      # Categorical area (we use code_postal_area_code)
        'distance_to_center_bin', # Categorical distance bin (we use distance_to_center_bin_code)
        'latitude_bin',          # Categorical latitude bin (we use latitude_bin_code)
        'longitude_bin',         # Categorical longitude bin (we use longitude_bin_code)
        'nombre_pieces_binned',  # Categorical pieces bin (we use nombre_pieces_binned_code)
        # Note: valeur_fonciere and surface_reelle_bati are NOT in df_features
        # cause we droped them in the feature engineering functions
    ]

    # Select feature columns (all except excluded)
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]

    X = df_features[feature_cols]

    print(f"Selected {len(feature_cols)} features for modeling")
    print(f"Feature names: {list(feature_cols)}")

    return X, feature_cols

def train_baseline_model(X_train, y_train):
    """Train baseline linear regression model"""
    print("Training baseline Linear Regression model")

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Model training complete.")

    return model

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Baseline Linear Regression"):
    """Evaluate model performance on training and test sets"""
    print(f"\nEvaluating {model_name} ")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    # Print results
    print(f"{model_name} Performance:")
    print(f"  Training Set:")
    print(f"MAE: {train_mae:.2f} €/m²")
    print(f"RMSE: {train_rmse:.2f} €/m²")
    print(f"R²: {train_r2:.4f}")
    print(f"Test Set:")
    print(f"MAE: {test_mae:.2f} €/m²")
    print(f"RMSE: {test_rmse:.2f} €/m²")
    print(f"R²: {test_r2:.4f}")

    # Return metrics as dictionary
    metrics = {
        'model_name': model_name,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    }

    return metrics

def save_model_and_data(model, feature_names, X_train, X_test, y_train, y_test, metrics):
    """Save model, feature names, and processed data"""
    base_dir = Path(__file__).resolve().parent.parent

    # Create directories if they dont exist
    (base_dir / 'models').mkdir(parents=True, exist_ok=True)
    (base_dir / 'data' / 'processed').mkdir(parents=True, exist_ok=True)
    (base_dir / 'results').mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = base_dir / 'models' / 'baseline_linear_regression.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Save feature names
    feature_path = base_dir / 'models' / 'feature_names.joblib'
    joblib.dump(feature_names, feature_path)
    print(f"Feature names saved to: {feature_path}")

    # Save processed data (leakage-safe versions)
    # Save as combined datasets with features + target for easy loading
    train_df = X_train.copy()
    train_df['prix_m2'] = y_train
    test_df = X_test.copy()
    test_df['prix_m2'] = y_test

    train_path = base_dir / 'data' / 'processed' / 'train_set.csv'
    test_path = base_dir / 'data' / 'processed' / 'test_set.csv'

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Training set saved to: {train_path} (shape: {train_df.shape})")
    print(f"Test set saved to: {test_path} (shape: {test_df.shape})")

    # Also save the leakage-prone reference file with a clear name
    # This is just for reference - we wont use it for modeling
    # We could reload the original cleaned data and engineer features on full dataset
    # but let's skip that for now to avoid confusion - we'll just note it

    # Save metrics
    metrics_path = base_dir / 'results' / 'baseline_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Baseline Linear Regression Performance Metrics\n")
        f.write(f"=====\n\n")
        f.write(f"Training Set:\n")
        f.write(f"MAE:  {metrics['train_mae']:.2f} €/m²\n")
        f.write(f"RMSE: {metrics['train_rmse']:.2f} €/m²\n")
        f.write(f"R²:   {metrics['train_r2']:.4f}\n\n")
        f.write(f"Test Set:\n")
        f.write(f"MAE:  {metrics['test_mae']:.2f} €/m²\n")
        f.write(f"RMSE: {metrics['test_rmse']:.2f} €/m²\n")
        f.write(f"R²:   {metrics['test_r2']:.4f}\n\n")
        f.write(f"Feature Count: {len(feature_names)}\n")
        f.write(f"Training Samples: {X_train.shape[0]}\n")
        f.write(f"Test Samples: {X_test.shape[0]}\n")

    print(f"Metrics saved to: {metrics_path}")

    return model_path, feature_path, train_path, test_path, metrics_path

def main():
    """Main training pipeline"""
    print("=" * 70)
    print(" Implementing leakage-safe feature engineering and baseline modeling")
    print("=" * 70)
    print()

    # Load cleaned data
    df_clean = load_cleaned_data()

    # Split data FIRST (before any feature engineering) to prevent leakage
    X_train, X_test, y_train, y_test = split_data(df_clean, test_size=0.2, random_state=42)

    # Engineer features on TRAINING SET only (to prevent leakage)
    df_train_features, transform_objects = engineer_features_train(X_train)

    # Engineer features on TEST SET using training derived transformations
    df_test_features = engineer_features_test(X_test, transform_objects)

    # Prepare final modeling data (select features to use)
    X_train_model, feature_names = prepare_modeling_data(df_train_features)
    X_test_model, _ = prepare_modeling_data(df_test_features)  # Should get same feature names

    # Verify feature names match
    if list(X_train_model.columns) != list(X_test_model.columns):
        print("WARNING: Feature names don't match between train and test!")
        print(f"Train features: {list(X_train_model.columns)}")
        print(f"Test features:  {list(X_test_model.columns)}")
        # Use intersection to be safe
        common_features = list(set(X_train_model.columns) & set(X_test_model.columns))
        X_train_model = X_train_model[common_features]
        X_test_model = X_test_model[common_features]
        feature_names = common_features
        print(f"Using {len(feature_names)} common features")

    # Train baseline model
    model = train_baseline_model(X_train_model, y_train)

    # Evaluation
    metrics = evaluate_model(model, X_train_model, y_train, X_test_model, y_test)

    # Save model, data, and metrics
    save_model_and_data(model, feature_names, X_train_model, X_test_model, y_train, y_test, metrics)

    print()
    print("=" * 70)
    print("PHASE 7 COMPLETE: Baseline model trained and evaluated successfully")
    print("All feature engineering was leakage-safe:")
    print("  - Train/test split done BEFORE feature engineering")
    print("  - Aggregate features computed ONLY on training set")
    print("  - Same transformations applied to test set")
    print("  - No information from test set contaminated training")
    print("=" * 70)

if __name__ == "__main__":
    main()