# Model Comparison for Bordeaux Real Estate Analysis

## Task and data unit

The target is the average transaction price per built square metre:

`prix_m2 = valeur_fonciere / total surface_reelle_bati`

Each row represents one DVF mutation (`id_mutation`). The input is first limited to house and apartment rows. Their surfaces and room counts are summed, while the repeated mutation price is counted once. Transactions containing both houses and apartments are labeled `Mixte`. The DVF mutation price can also include associated items such as dependencies that are not included in the residential surface total. The resulting target is the full mutation price divided by the selected residential built area; it is not an individually allocated value for each dwelling in a multi-property sale.

The final cleaned dataset has 3,923 mutations. All three models use the same 80/20 random split (`random_state=42`): 3,138 training transactions and 785 test transactions. Training-derived frequency mappings are applied to the test set. `valeur_fonciere` is excluded from the predictors; `surface_reelle_bati` is retained because it is available from a property listing.

DVF is produced by the French tax authority from notarial deeds and cadastral information. Its geolocalized schema describes `id_mutation` as the identifier used to group the rows of a mutation. See the [official DVF dataset](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres) and its [geolocalized field definitions](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees).

## Held-out test results

| Metric | Median Dummy | Linear Regression | Random Forest | Gradient Boosting |
|---|---:|---:|---:|---:|
| Train MAE (€/m²) | 1,080.30 | 923.93 | 707.77 | 693.32 |
| Train RMSE (€/m²) | 1,476.94 | 1,291.82 | 1,003.89 | 935.61 |
| Train R² | -0.0110 | 0.2265 | 0.5329 | 0.5943 |
| Test MAE (€/m²) | 1,013.60 | 878.50 | **847.00** | 874.54 |
| Test RMSE (€/m²) | 1,358.95 | 1,216.16 | **1,198.03** | 1,235.38 |
| Test R² | -0.0060 | 0.1943 | **0.2181** | 0.1686 |

The median dummy predicts the training-set median for every test transaction. It provides a simple reference with no location or property information. All three trained models improve on this reference; the Random Forest performs best on this particular held-out split.

## Five-fold Random Forest validation

The Random Forest settings were kept fixed. For every fold, frequency encodings were learned from that fold's training data and applied to its validation data.

| Metric | Random Forest mean ± std | Median Dummy mean ± std |
|---|---:|---:|
| MAE (€/m²) | 891.14 ± 26.74 | 1,067.14 ± 38.42 |
| RMSE (€/m²) | 1,250.10 ± 43.66 | 1,453.84 ± 72.53 |
| R² | 0.25 ± 0.04 | -0.01 ± 0.01 |

Per-fold scores are in `rf_cross_validation.csv`; see `rf_cross_validation.md` for the short report. The folds contain different transactions from the same 2024 Bordeaux dataset, so they assess stability within this dataset, not performance in another year or city.

## Interpretation and limits

- The Random Forest is the strongest of the tested models, but its held-out R² of 0.2181 is modest. Treat this as a portfolio experiment, not a production-ready price estimator.
- The split is random, so nearby transactions can appear in both training and test sets. A spatial holdout would provide a stricter test for predictions in unfamiliar neighborhoods.
- The dataset covers 2024 only. It does not show whether the model generalizes to later years.
- The target is aggregated at mutation level. A mutation can contain multiple housing components; the model does not estimate a separate price for each dwelling in such a sale.
- `surface_reelle_bati` is a legitimate predictor for a listing-time estimate if its value is available before sale and matches the listing's surface definition. Its importance score alone is not evidence that a model is free of leakage.

## Next work

1. Compare against other held-out years and spatially held-out areas when suitable data is available.
2. Consider whether the intended product should predict transaction-level average price or the price per individual dwelling; DVF's mutation total cannot by itself allocate a package price among separate dwellings.
3. Keep the exact DVF source snapshot used for each published result because source files can be revised.
