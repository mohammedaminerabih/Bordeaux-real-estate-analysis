import unittest

import pandas as pd

from src.feature_utils import add_postal_area_features, postal_area_labels
from src.model_rf import (
    engineer_features_test,
    engineer_features_train,
    prepare_modeling_data,
)


class PostalAreaFeatureTests(unittest.TestCase):
    def test_postal_codes_use_their_actual_100_code_ranges(self):
        codes = pd.Series([32999, 33000, 33099, 33100, 33999, 34000, None])

        self.assertEqual(
            postal_area_labels(codes).tolist(),
            [
                "other",
                "33000-33099",
                "33000-33099",
                "33100-33199",
                "33900-33999",
                "other",
                "other",
            ],
        )

    def test_one_hot_schema_is_stable_and_unknown_codes_are_explicit(self):
        frame = pd.DataFrame({"code_postal": [33000, 33100, 99999, None]})

        encoded = add_postal_area_features(frame)

        # 33000-33099 is the reference area, so it is represented by all zeros.
        self.assertEqual(encoded.loc[0, "code_postal_area_33100_33199"], 0)
        self.assertEqual(encoded.loc[1, "code_postal_area_33100_33199"], 1)
        self.assertEqual(encoded.loc[2, "code_postal_area_other"], 1)
        self.assertEqual(encoded.loc[3, "code_postal_area_other"], 1)
        self.assertFalse(any(column.endswith("_code") for column in encoded.columns))

    def test_model_features_exclude_sale_date_and_keep_a_shared_schema(self):
        sample = pd.DataFrame(
            {
                "id_mutation": ["a", "b", "c", "d"],
                "date_mutation": ["2024-01-05", "2024-02-06", "2024-03-07", "2024-04-08"],
                "valeur_fonciere": [100000, 120000, 140000, 160000],
                "type_local": ["Appartement", "Maison", "Mixte", "Appartement"],
                "code_postal": [33000, 33100, 33200, 33800],
                "nom_commune": ["BORDEAUX"] * 4,
                "surface_reelle_bati": [25, 60, 80, 30],
                "nombre_pieces_principales": [1, 3, 4, 2],
                "latitude": [44.83, 44.84, 44.85, 44.86],
                "longitude": [-0.58, -0.57, -0.56, -0.55],
            }
        )

        train_features, transforms = engineer_features_train(sample)
        test_features = engineer_features_test(sample.iloc[:2].copy(), transforms)
        train_x, feature_names = prepare_modeling_data(train_features)
        test_x, test_feature_names = prepare_modeling_data(test_features)

        self.assertEqual(feature_names, test_feature_names)
        self.assertEqual(list(train_x.columns), list(test_x.columns))
        self.assertFalse(any(name.startswith("date_") for name in feature_names))
        self.assertIn("code_postal_area_33100_33199", feature_names)
        self.assertNotIn("valeur_fonciere", feature_names)


if __name__ == "__main__":
    unittest.main()