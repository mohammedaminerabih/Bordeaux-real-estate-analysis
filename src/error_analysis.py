"""
Error Analysis for the best model (Random Forest Regressor) on the Bordeaux real estate dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Plot style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data_and_model():
    """Load the test set and the trained Random Forest model."""
    base_dir = Path(__file__).resolve().parent.parent

    # Load test set (with features and target)
    test_path = base_dir / 'data' / 'processed' / 'test_set_rf.csv'
    test_df = pd.read_csv(test_path)
    print(f"Loaded test set from: {test_path}")
    print(f"Test set shape: {test_df.shape}")

    # Separate features and target
    # The test set was saved with features + target, so we need to drop the target to get features
    # Note: the target is 'prix_m2'
    X_test = test_df.drop(columns=['prix_m2'])
    y_test = test_df['prix_m2']

    # Load the model
    model_path = base_dir / 'models' / 'random_forest_regressor.joblib'
    model = joblib.load(model_path)
    print(f"Loaded model from: {model_path}")

    # Load feature names to ensure we use the same features
    feature_path = base_dir / 'models' / 'rf_feature_names.joblib'
    feature_names = joblib.load(feature_path)
    print(f"Loaded feature names from: {feature_path}")
    print(f"Number of features: {len(feature_names)}")

    # Ensure the test set has the same features in the same order
    X_test = X_test[feature_names]

    return X_test, y_test, model

def make_predictions(X_test, model):
    """Make predictions using the trained model."""
    print("Making predictions...")
    y_pred = model.predict(X_test)
    return y_pred

def analyze_residuals(y_test, y_pred, X_test):
    """Analyze residuals and generate plots."""
    residuals = y_test - y_pred

    # Basic statistics
    print("\nResiduals Statistics:")
    print(f"Mean residual: {residuals.mean():.2f} €/m²")
    print(f"Std residual: {residuals.std():.2f} €/m²")
    print(f"Min residual: {residuals.min():.2f} €/m²")
    print(f"Max residual: {residuals.max():.2f} €/m²")

    # Create figures directory if it dont exist
    base_dir = Path(__file__).resolve().parent.parent
    figures_dir = base_dir / 'results' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Residuals histogram
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals (Actual - Predicted) €/m²')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='red', linestyle='--', label='Zero Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / 'residuals_histogram.png', dpi=300)
    plt.close()
    print(f"Saved residuals histogram to: {figures_dir / 'residuals_histogram.png'}")

    # 2. Residuals vs Predicted Values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Prix_m2 (€/m²)')
    plt.ylabel('Residuals (Actual - Predicted) €/m²')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(figures_dir / 'residuals_vs_predicted.png', dpi=300)
    plt.close()
    print(f"Saved residuals vs predicted to: {figures_dir / 'residuals_vs_predicted.png'}")

    # 3. Residuals vs Actual Values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, residuals, alpha=0.5)
    plt.title('Residuals vs Actual Values')
    plt.xlabel('Actual Prix_m2 (€/m²)')
    plt.ylabel('Residuals (Actual - Predicted) €/m²')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(figures_dir / 'residuals_vs_actual.png', dpi=300)
    plt.close()
    print(f"Saved residuals vs actual to: {figures_dir / 'residuals_vs_actual.png'}")

    # 4. Residuals vs Key Features (eg: surface_reelle_bati, latitude, longitude)
    # We need to load the original test set to get these features (they are in X_test but we have the feature names)
    # Since we have X_test as a dataframe with the feature names, we can use it

    # Select a few key features for analysis
    key_features = ['surface_reelle_bati', 'latitude', 'longitude']
    # Check if these features are in X_test
    available_features = [f for f in key_features if f in X_test.columns]

    if available_features:
        n_features = len(available_features)
        fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 5))
        if n_features == 1:
            axes = [axes]
        for i, feature in enumerate(available_features):
            axes[i].scatter(X_test[feature], residuals, alpha=0.5)
            axes[i].set_title(f'Residuals vs {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Residuals (Actual - Predicted) €/m²')
            axes[i].axhline(y=0, color='red', linestyle='--')
        plt.tight_layout()
        plt.savefig(figures_dir / 'residuals_vs_key_features.png', dpi=300)
        plt.close()
        print(f"Saved residuals vs key features to: {figures_dir / 'residuals_vs_key_features.png'}")
    else:
        print("Key features (surface_reelle_bati, latitude, longitude) not found in test set features.")

    return residuals

def main():
    """Main error analysis pipeline."""
    print("=" * 70)
    print(" Error Analysis for Random Forest Regressor")
    print("=" * 70)
    print()

    # Load data and model
    X_test, y_test, model = load_data_and_model()

    # Make predictions
    y_pred = make_predictions(X_test, model)

    # Analyse residuals
    residuals = analyze_residuals(y_test, y_pred, X_test)

    print()
    print("=" * 70)
    print("Error analysis complete. Plots saved in results/figures/")
    print("=" * 70)

if __name__ == "__main__":
    main()