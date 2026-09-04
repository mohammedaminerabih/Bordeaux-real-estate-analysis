# Bordeaux Real Estate Analysis

Real estate data analysis project based on DVF (Demandes de Valeurs Foncières) data from the French government, focusing on Bordeaux, France (Gironde, 33).

## Project Goals

- Analyse real estate price trends in Bordeaux
- Identify patterns by neighbourhood and property type
- Produce clear visualizations
- Practice the complete data science project cycle (from raw data to results)

## Data Source

Data comes from the official DVF database:

- Source: data.gouv.fr
- Scope: Gironde Department (33)
- Period: 2024

## Installation and Usage

### Prerequisites

- Python 3.10+
- Dependencies listed in `requirements.txt`

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download and Preprocess Data

The script `src/preprocessing.py` automatically downloads the data on first run. Alternatively:

1. Go to https://files.data.gouv.fr/geo-dvf/latest/csv/2024/departements/
2. Download `33.csv.gz` (Gironde department)
3. Place the file in data/raw/ as 33.csv.

### Run the preprocessing

To download and filter the raw DVF data for Bordeaux:

```bash
python src/preprocessing.py
```

This creates:
- data/processed/bordeaux_data.csv

### Run the data cleaning

To clean the filtered data, engineer features (prix_m2), handle multi-row transactions, remove outliers, and save cleaned data:

```bash
python src/data_cleaning.py
```

This creates:
- data/processed/bordeaux_clean.csv (final cleaned dataset)

### Run exploratory data analysis (EDA)

To perform exploratory data analysis and generate visualizations:

```bash
# Option 1: Run the notebook interactively
jupyter notebook notebooks/01_eda.ipynb

# Option 2: Run the notebook as a script
python notebooks/01_eda.ipynb
```

This generates figures in:
- results/figures/price_distribution.png
- results/figures/price_by_property_type.png
- results/figures/price_by_top10_postal_codes.png
- results/figures/correlation_heatmap.png

## Project Structure

```
Bordeaux-real-estate-analysis/
├── data/
│   ├── raw/
│   │   └── 33.csv
│   └── processed/
│       ├── bordeaux_data.csv
│       ├── bordeaux_clean.csv
│       └── inspection_results.txt (optional)
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── preprocessing.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── train_baseline.py
│   └── model_rf.py
├── results/
│   └── figures/
│       ├── price_distribution.png
│       ├── price_by_property_type.png
│       ├── price_by_top10_postal_codes.png
│       └── correlation_heatmap.png
├── models/
│   ├── baseline_linear_regression.joblib
│   ├── feature_names.joblib
│   ├── random_forest_regressor.joblib
│   └── rf_feature_names.joblib
├── README.md
├── requirements.txt
└── .gitignore
```

## Completed Tasks

✅ Data download and preprocessing (src/preprocessing.py)
✅ Data cleaning and feature engineering (prix_m2) (src/data_cleaning.py)
✅ Exploratory data analysis (EDA) (notebooks/01_eda.ipynb)
✅ Project restructuring and organization
✅ Documentation of all phases
✅ Feature engineering with leakage prevention (src/feature_engineering.py)
✅ Train/test split and baseline modeling (src/train_baseline.py) with data quality improvements
✅ Initial model evaluation and performance metrics
✅ Modèle 2 (Random Forest) (src/model_rf.py) with leakage-safe pipeline and performance metrics

## Upcoming Tasks
## Upcoming Tasks

- Model evaluation and comparison
- Error analysis
- Final documentation

## Author

Project conducted as part of the Master's in Artificial Intelligence (IA) at the University of Bordeaux.

## License

This project uses public data from the French government.

## Important Notes

- The EDA work in notebooks/01_eda.ipynb represents the foundation of the project.
- All modeling work (feature_engineering.py, train_baseline.py, model_rf.py) has been validated and is leakage-safe, with surface_reelle_bati correctly included as a legitimate feature.
- The Random Forest model has been retuned with regularization parameters (max_depth=20, min_samples_leaf=5) to address overfitting. The updated model shows improved generalization: Test R² increased from 0.16 to 0.21, while reducing overfitting (training-test R² gap reduced from 0.74 to 0.33).
