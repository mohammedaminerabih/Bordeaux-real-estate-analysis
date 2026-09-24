"""Five-fold validation for the Random Forest using train-only encodings."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from model_rf import (
    engineer_features_test,
    engineer_features_train,
    load_cleaned_data,
    prepare_modeling_data,
    train_rf_model,
)


def main():
    """Evaluate the fixed Random Forest settings across five held-out folds."""
    base_dir = Path(__file__).resolve().parent.parent
    df = load_cleaned_data()
    y = df['prix_m2'].reset_index(drop=True)
    X = df.drop(columns=['prix_m2']).reset_index(drop=True)
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold_number, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
        X_train_raw, X_valid_raw = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        train_features, transforms = engineer_features_train(X_train_raw)
        valid_features = engineer_features_test(X_valid_raw, transforms)
        X_train, feature_names = prepare_modeling_data(train_features)
        X_valid, valid_feature_names = prepare_modeling_data(valid_features)

        if feature_names != valid_feature_names:
            raise ValueError(f"Feature columns differ in fold {fold_number}")

        forest = train_rf_model(X_train, y_train)
        dummy = DummyRegressor(strategy='median').fit(X_train, y_train)
        predictions = forest.predict(X_valid)
        dummy_predictions = dummy.predict(X_valid)

        fold_results.append({
            'fold': fold_number,
            'samples': len(valid_idx),
            'rf_mae': mean_absolute_error(y_valid, predictions),
            'rf_rmse': np.sqrt(mean_squared_error(y_valid, predictions)),
            'rf_r2': r2_score(y_valid, predictions),
            'dummy_mae': mean_absolute_error(y_valid, dummy_predictions),
            'dummy_rmse': np.sqrt(mean_squared_error(y_valid, dummy_predictions)),
            'dummy_r2': r2_score(y_valid, dummy_predictions),
        })
        print(f"Fold {fold_number}/5 complete")

    results = pd.DataFrame(fold_results)
    output_path = base_dir / 'results' / 'rf_cross_validation.csv'
    results.to_csv(output_path, index=False)

    summary_lines = [
        '# Five-fold Random Forest validation',
        '',
        'The model settings were fixed before validation. Feature encodings are fitted separately inside each training fold.',
        '',
        '| Metric | Random Forest mean ± std | Median dummy mean ± std |',
        '|---|---:|---:|',
    ]
    for metric, label in [('mae', 'MAE (€/m²)'), ('rmse', 'RMSE (€/m²)'), ('r2', 'R²')]:
        rf_values = results[f'rf_{metric}']
        dummy_values = results[f'dummy_{metric}']
        summary_lines.append(
            f"| {label} | {rf_values.mean():.2f} ± {rf_values.std(ddof=1):.2f} "
            f"| {dummy_values.mean():.2f} ± {dummy_values.std(ddof=1):.2f} |"
        )
    summary_lines.extend([
        '',
        'Each fold holds out different transactions from the same 2024 Bordeaux dataset. This measures stability within that dataset; it does not test performance in another city or year.',
        '',
        f"Per-fold scores are saved in `{output_path.relative_to(base_dir).as_posix()}`.",
    ])
    summary_path = base_dir / 'results' / 'rf_cross_validation.md'
    summary_path.write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    print(f"Saved fold scores to: {output_path}")
    print(f"Saved summary to: {summary_path}")
    print('\n'.join(summary_lines))


if __name__ == '__main__':
    main()
