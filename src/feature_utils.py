"""Shared feature helpers for the Bordeaux real-estate models."""

import numpy as np
import pandas as pd


POSTAL_AREA_CATEGORIES = tuple(
    [f"{start}-{start + 99}" for start in range(33000, 34000, 100)] + ["other"]
)


def postal_area_labels(postal_codes):
    """Map postal codes to their exact 100-code interval, or ``other``."""
    numeric_codes = pd.to_numeric(postal_codes, errors="coerce")
    labels = pd.Series("other", index=postal_codes.index, dtype="object")
    in_bordeaux_range = numeric_codes.between(33000, 33999, inclusive="both")

    area_starts = (numeric_codes.loc[in_bordeaux_range] // 100 * 100).astype(int)
    labels.loc[in_bordeaux_range] = area_starts.map(
        lambda start: f"{start}-{start + 99}"
    )
    return labels


def add_postal_area_features(frame):
    """Add stable one-hot postal-area features without ordinal category codes.

    The first area (33000-33099) is the reference category. Missing and
    out-of-range postal codes are represented by an explicit ``other`` flag.
    The fixed output schema is identical for train, validation, and test data.
    """
    labels = postal_area_labels(frame["code_postal"])
    for category in POSTAL_AREA_CATEGORIES[1:]:
        feature_name = f"code_postal_area_{category.replace('-', '_')}"
        frame[feature_name] = labels.eq(category).astype(np.int8)
    return frame
