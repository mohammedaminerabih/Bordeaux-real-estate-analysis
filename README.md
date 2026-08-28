# Bordeaux Real Estate Analysis

Real estate data analysis project based on DVF (Demandes de Valeurs Foncières) data from the French government, focusing on Bordeaux, France (Gironde, 33).

## Project Goals

- Analyse real estate price trends in Bordeaux
- Identify patterns by neighbourhood and property type
* Produce clear visualizations
* Practice the complete data science project cycle (from raw data to results)

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

data/processed/bordeaux_data.csv

The next step is the exploratory data analysis in:

notebooks/01_eda.ipynb

## Project Structure

```
Bordeaux-real-estate-analysis/
├── data/
│   ├── raw/
│   │   └── 33.csv
│   └── processed/
│       └── bordeaux_data.csv
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── preprocessing.py
│   └── data_cleaning.py
├── results/
│   └── figures/
├── README.md
├── requirements.txt
└── .gitignore
```

## Upcoming Tasks

- Exploratory data analysis (EDA) in `notebooks/01_eda.ipynb`
- Data cleaning and preparation
- Feature engineering
- Train/test split and baseline model
- Modelling (linear regression, random forest, gradient boosting)
- Model evaluation (MAE, RMSE, R²)
- Error analysis
- Final documentation

## Author

Project conducted as part of the Master's in Artificial Intelligence (IA) at the University of Bordeaux.

## License

This project uses public data from the French government.
