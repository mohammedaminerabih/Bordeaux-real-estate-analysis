# Model Comparison for Bordeaux Real Estate Analysis

## Overview
This document compares the performance of three models trained on the Bordeaux real estate dataset:
1. Baseline Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

All models were trained using the same leakage safe pipeline:
- Train/test split performed before any feature engineering (80/20 split, random_state=42)
- Features engineered exclusively on the training set
- Same transformations applied to the test set
- Target variable: `prix_m2` (price per square meter)

## Performance Metrics

| Metric | Baseline Linear Regression | Random Forest Regressor | Gradient Boosting Regressor |
|--------|----------------------------|-------------------------|-----------------------------|
| **Training MAE (€/m²)** | 925.01 | 702.70 | 696.02 |
| **Training RMSE (€/m²)** | 1294.02 | 995.48 | 941.95 |
| **Training R²** | 0.2239 | 0.5407 | 0.5888 |
| **Test MAE (€/m²)** | 877.76 | 853.64 | 878.82 |
| **Test RMSE (€/m²)** | 1215.29 | 1203.61 | 1245.09 |
| **Test R²** | 0.1954 | 0.2108 | 0.1555 |

## Analysis

### Baseline Linear Regression
- Shows the highest bias (lowest training R²) but the smallest variance (smallest gap between training and test R²: 0.0285)
- Underfits the data, capturing only linear relationships

### Random Forest Regressor
- Achieves a good balance between bias and variance
- Training R²: 0.5407, Test R²: 0.2108 (gap: 0.3299)
- Outperforms the baseline on both training and test sets
- Shows moderate overfitting but reasonable generalization

### Gradient Boosting Regressor
- Shows the lowest bias (highest training R²: 0.5888) but the highest variance (largest gap: 0.4333)
- Overfits the training data significantly, leading to poorer generalization on the test set (Test R²: 0.1555)
- Requires hyperparameter tuning to reduce overfitting (e.g., decreasing `n_estimators`, increasing `min_samples_leaf`, or reducing `learning_rate`)

## Recommendation
Based on test set performance, the **Random Forest Regressor** is the best model for this dataset, offering the highest test R² (0.2108) and a reasonable balance between bias and variance

For production use, consider:
1. **Random Forest Regressor** as the final model (current hyperparameters: n_estimators=300, max_depth=20, max_features='sqrt', min_samples_leaf=5)
2. Alternatively, tune the Gradient Boosting Regressor to reduce overfitting and potentially achieve better performance

## Next Steps
- Perform error analysis on the best model (Random Forest) to understand prediction errors
- Consider feature importance interpretation from the Random Forest model
- Finalize documentation