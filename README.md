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
- results. figs/correlation_heatmap.png

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
│   ├── inspect_data.py
│   └── eda.py
├── results/
│   └── figures/
│       ├── price_distribution.png
│       ├── price_by_property_type.png
│       ├── price_by_top10_postal_codes.png
│       └── correlation_heatmap.png
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

## Upcoming Tasks (Awaiting User Authorization)

- Feature engineering (Phase 6)
- Train/test split and baseline The user did not explicitly authorize moving to these phases, so they remain pending.
- Modeling (linear regression, random forest, gradient boosting) (Phase 7+)
- Model evaluation (MAE, RMSE, R²)
- Error analysis
- Final documentation

## Author

Project conducted as part of the Master's in Artificial Intelligence (IA) at the University of Bordeaux.

## License

This project uses public data from the French government.

## Important Notes

- All work beyond Phase 5 (EDA) has been reverted due to data leakage concerns
- Any future feature engineering or modeling must be explicitly authorized by the user
- The EDA work in notebooks/01_eda.ipynb represents the current state of the project