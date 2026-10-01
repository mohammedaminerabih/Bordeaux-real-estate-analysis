# Model Comparison for Bordeaux Real Estate Analysis

## Task and data unit

The target is the average transaction price per built square metre:

`prix_m2 = valeur_fonciere / total surface_reelle_bati`

Each row represents one DVF mutation (`id_mutation`), the input is first limited to house and apartment rows, their surfaces and room counts are summed, while the repeated mutation price is counted once. 
Transactions containing both houses and apartments are labeled `Mixte`, the DVF mutation price can also include associated items such as dependencies that are not included in the residential surface total.
The resulting target is the full mutation price divided by the selected residential built area, it is not an individually allocated value for each dwelling in a multi-property sale.
The final cleaned dataset has 3,923 mutations, all three models use the same 80/20 random split (`random_state=42`): 3,138 training transactions and 785 test transactions. 
Derived training frequency mappings are applied to the test set. `valeur_fonciere` is excluded from the predictors, `surface_reelle_bati` is retained because it is available from a property listing.

DVF is produced by the French tax authority from notarial deeds and cadastral information, its geolocalized schema describes `id_mutation` as the identifier used to group the rows of a mutation. See [official DVF dataset](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres) and its [geolocalized field definitions](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees).

## Held-out test results

| Metric | Median Dummy | Linear Regression | Random Forest | Gradient Boosting |
|---|---:|---:|---:|---:|
| Train MAE (€/m²) | — | 923.97 | 721.44 | 702.79 |
| Train RMSE (€/m²) | — | 1,292.69 | 1,023.60 | 951.08 |
| Train R² | — | 0.2255 | 0.5144 | 0.5807 |
| Test MAE (€/m²) | 1,013.60 | 881.13 | **846.80** | 867.33 |
| Test RMSE (€/m²) | 1,358.95 | 1,218.05 | **1,202.21** | 1,223.82 |
| Test R² | -0.0060 | 0.1918 | **0.2126** | 0.1841 |

The median dummy predicts the training set median for every test transaction, it provides a simple reference with no location or property information.
All three trained models improve on this reference, the Random Forest performs best on this particular split, its test MAE is about 16.5% lower than the dummy's, but its R² is still only 0.2126.

## 5-fold Random Forest validation

The Random Forest settings were kept fixed. For every fold, frequency encodings were learned from that fold's training data and applied to its validation data.

| Metric | Random Forest mean ± std | Median Dummy mean ± std |
|---|---:|---:|
| MAE (€/m²) | 888.49 ± 27.82 | 1,067.14 ± 38.42 |
| RMSE (€/m²) | 1,250.37 ± 43.79 | 1,453.84 ± 72.53 |
| R² | 0.25 ± 0.04 | -0.01 ± 0.01 |

Per-fold scores are in `rf_cross_validation.csv`, see `rf_cross_validation.md` for the short report. The folds contains different transactions from the same 2024 Bordeaux dataset, so they assess stability within this dataset, not performance in another year or city

## Interpretation and limits

- The Random Forest is the strongest of the tested models, but its held-out R² of 0.2126 is modest.
- The split is random, so nearby transactions can appear in both training and test sets. A spatial holdout would provide a stricter test for predictions in unfamiliar neighborhoods.
- The dataset covers 2024 only, it doesn't show whether the model generalizes to later years.
- The target is aggregated at mutation level, a mutation can contain multiple housing components; the model doesn't estimate a separate price for each dwelling in such a sale.
- `surface_reelle_bati` is a legitimate predictor for a listing time estimate if its value is available before sale and matches the listing's surface definition, its importance score alone is not evidence that a model is free of leakage.

## Next work

1. Comparing against other held-out years and spatially held-out areas when suitable data is available.
2. Considering whether the intended product should predict transaction-level average price or the price per individual dwelling... DVF's mutation total cannot by itself allocate a package price among separate dwellings.
