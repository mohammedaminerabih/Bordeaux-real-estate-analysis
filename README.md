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

Run the small feature-encoding regression tests with:

```bash
python -m unittest discover -s tests -v
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

The first three scripts use the same 80/20 split, the baseline script also measures a median-only predictor, validation script evaluates the fixed Random Forest settings on five shuffled folds and fits feature encodings separately inside each fold. 
These evaluations are random splits within Bordeaux in 2024, not tests of future years or unseen places. Date-of-sale-derived predictors are excluded because the project is framed as a listing-time estimate, postal areas use fixed 100-code intervals and ONE-HOT indicators rather than an artificial ordinal scale.

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
│   ├── feature_utils.py
│   ├── preprocessing.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── train_baseline.py
│   ├── model_rf.py
│   ├── model_gb.py
│   ├── model_validation.py
│   └── error_analysis.py
├── tests/
│   └── test_feature_utils.py
├── results/
│   ├── figures/
│   ├── baseline_metrics.txt
│   ├── rf_metrics.txt
│   ├── gb_metrics.txt
│   ├── rf_cross_validation.csv
│   ├── rf_cross_validation.md
│   ├── source_snapshot.md
│   ├── model_comparison.md
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

- Evaluating on later years or other cities before making claims about future or wider-market performance.
- Considering spatial holdouts to measure performance in Bordeaux areas absent from training.
- Keep the raw data snapshot used for each published result, DVF source files can be updated.

## Author

Project conducted as part of the Master's in Artificial Intelligence (IA) at the University of Bordeaux.

## License

This project uses public data from the French government.

## Important Notes

- The cleaned file contains 3,923 transactions, not 3,923 individual homes. Rows are grouped by `id_mutation`; surfaces and room counts from house/apartment rows are summed for each sale. Mixed house/apartment sales are labeled `Mixte` rather than inheriting the type from whichever source row appears first.
- The target is `prix_m2 = full mutation value / selected residential built surface`. DVF's transaction value may also include associated items, such as dependencies, whose area is not counted in this denominator. It is not an allocated price for each individual apartment or house in a multi-property sale.
- `valeur_fonciere` and values derived from it are unavailable before a sale and must not be model inputs. The scripts also exclude `date_mutation` and features derived from it because the deed date is not known when a listing-time estimate is made. `surface_reelle_bati` is kept because built area is normally listed, but check that DVF's definition matches the listing's.
- Postal-area features use exact 100-code intervals and one-hot encoding; they are not treated as ordinal numeric values.
- `src/feature_engineering.py` exports descriptive features over the complete dataset, including full-dataset frequency counts. Do not use its output to estimate model performance. The model scripts split first and compute their frequency mappings from training data only.
- Current results are recorded in `results/model_comparison.md`. Interpret them as transaction-level metrics for the stated 2024 Bordeaux sample, not as validated estimates for individual listings, other years, or unseen areas.
- A fresh clone does not contain the ignored `data/` files. Run preprocessing to download the source data before cleaning or modeling. The official DVF files are updated over time, so retain the exact source snapshot when reproducing a published result.
- `results/source_snapshot.md` records the SHA-256 of the source CSV used for the current metrics. Compare the downloaded file with that checksum when reproducing them.
