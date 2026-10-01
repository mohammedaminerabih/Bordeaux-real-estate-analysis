# 5-fold Random Forest validation

The model settings were fixed before validation, Feature encodings are fitted separately inside each training fold.

| Metric | Random Forest mean ± std | Median dummy mean ± std |
|---|---:|---:|
| MAE (€/m²) | 888.49 ± 27.82 | 1067.14 ± 38.42 |
| RMSE (€/m²) | 1250.37 ± 43.79 | 1453.84 ± 72.53 |
| R² | 0.25 ± 0.04 | -0.01 ± 0.01 |

Each fold holds out different transactions from the same 2024 Bordeaux dataset, this measures stability within that dataset; it doesn't test performance in another city or year.

Per-fold scores are saved in `results/rf_cross_validation.csv`.
