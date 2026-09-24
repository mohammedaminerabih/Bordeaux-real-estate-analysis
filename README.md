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

Open the notebook and run its cells from top to bottom:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

This generates figures in:
- results/figures/price_distribution.png
- results/figures/price_by_property_type.png
- results/figures/price_by_top10_postal_codes.png
- results/figures/correlation_heatmap.png

### Train and evaluate the models

Run these commands from the project root, after preprocessing and cleaning:

```bash
python src/train_baseline.py
python src/model_rf.py
python src/model_gb.py
python src/model_validation.py
python src/error_analysis.py
```

The first three scripts use the same 80/20 split. The baseline script also measures a median-only predictor. The validation script evaluates the fixed Random Forest settings on five shuffled folds and fits feature encodings separately inside each fold.

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
│   ├── model_rf.py
│   ├── model_gb.py
│   ├── model_validation.py
│   └── error_analysis.py
├── results/
│   ├── figures/
│   └── source_snapshot.md
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
✅ Exploratory feature export (src/feature_engineering.py; not used for model evaluation)
✅ Train/test split and baseline modeling (src/train_baseline.py) with data quality improvements
✅ Initial model evaluation and performance metrics
✅ Modèle 2 (Random Forest) (src/model_rf.py) with leakage-safe pipeline and performance metrics
✅ Modèle 3 (Gradient Boosting) (src/model_gb.py) with leakage safe pipeline and performance metrics
✅ Model evaluation and comparison (results/model_comparison.md)
✅ Error analysis (src/error_analysis.py and results/figures/)

## Remaining Work

- Evaluate on later years or other cities before making claims about future or wider-market performance.
- Consider spatial holdouts to measure performance in Bordeaux areas absent from training.
- Keep the raw data snapshot used for each published result; DVF source files can be updated.

## Author

Project conducted as part of the Master's in Artificial Intelligence (IA) at the University of Bordeaux.

## License

This project uses public data from the French government.

## Important Notes

- The cleaned file contains 3,923 transactions, not 3,923 individual homes. Rows are grouped by `id_mutation`; surfaces and room counts from house/apartment rows are summed for each sale. Mixed house/apartment sales are labeled `Mixte` rather than inheriting the type from whichever source row appears first.
- The target is `prix_m2 = full mutation value / selected residential built surface`. DVF's transaction value may also include associated items, such as dependencies, whose area is not counted in this denominator. It is not an allocated price for each individual apartment or house in a multi-property sale.
- `valeur_fonciere` and values derived from it (such as `log_valeur_fonciere`) are unavailable before a sale and must not be model inputs. `surface_reelle_bati` is kept because the built area is normally known from a listing; check that DVF's surface matches the listing definition for any real listing-time use.
- `src/feature_engineering.py` exports descriptive features over the complete dataset, including full-dataset frequency counts. Do not use its output to estimate model performance. The model scripts split first and compute their frequency mappings from training data only.
- The corrected Random Forest's held-out result is R²=0.2181 and MAE=847.00 €/m². These are from one random split and show modest performance within the 2024 Bordeaux sample, not proven performance in another year or location. See `results/rf_cross_validation.md` for the five-fold evaluation.
- A fresh clone does not contain the ignored `data/` files. Run preprocessing to download the source data before cleaning or modeling. The official DVF files are updated over time, so retain the exact source snapshot when reproducing a published result.
- `results/source_snapshot.md` records the SHA-256 of the source CSV used for the current metrics. Compare the downloaded file with that checksum when reproducing them.
