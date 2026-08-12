"""
test_feature_engineering.py — Tests for build_features().
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from app.core.feature_engineering import (
    CONTRACT_TYPES, PAYMENT_METHODS, TENURE_LABELS, build_features,
)


# ── Sample input row ──────────────────────────────────────────────────────────
SAMPLE_ROW = {
    "tenure_months": 18,
    "monthly_charges": 65.0,
    "total_charges": 1170.0,
    "contract_type": "month-to-month",
    "payment_method": "Credit Card",
    "avg_usage_hours": 55.0,
    "avg_support_tickets": 2.0,
    "avg_late_payments": 1.0,
    "avg_feature_adoption": 0.5,
    "cluster_id": 2,
}


@pytest.fixture
def sample_df():
    return pd.DataFrame([SAMPLE_ROW] * 10)


@pytest.fixture
def batch_df():
    """Varied batch including edge cases."""
    rows = [
        {**SAMPLE_ROW, "tenure_months": 0, "contract_type": "2yr", "cluster_id": -1},
        {**SAMPLE_ROW, "tenure_months": 72, "contract_type": "1yr"},
        {**SAMPLE_ROW, "payment_method": "Mailed Check", "cluster_id": 4},
    ]
    return pd.DataFrame(rows)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_returns_tuple(sample_df):
    result = build_features(sample_df)
    assert isinstance(result, tuple) and len(result) == 2


def test_output_is_dataframe(sample_df):
    feat, names = build_features(sample_df)
    assert isinstance(feat, pd.DataFrame)


def test_feature_names_match_columns(sample_df):
    feat, names = build_features(sample_df)
    assert list(feat.columns) == names


def test_no_nan(sample_df):
    feat, _ = build_features(sample_df)
    assert feat.isnull().sum().sum() == 0, "Feature matrix contains NaN"


def test_no_inf(sample_df):
    feat, _ = build_features(sample_df)
    import numpy as np
    assert not np.isinf(feat.values).any(), "Feature matrix contains Inf"


def test_tenure_bucket_columns_present(sample_df):
    feat, names = build_features(sample_df)
    for label in TENURE_LABELS:
        safe = label.replace("-", "_").replace("+", "plus")
        expected = f"tenure_{safe}"
        assert expected in names, f"Missing tenure bucket: {expected}"


def test_contract_onehot_present(sample_df):
    feat, names = build_features(sample_df)
    for ct in CONTRACT_TYPES:
        key = f"contract_{ct.replace('-', '_').replace(' ', '_')}"
        assert key in names, f"Missing contract column: {key}"


def test_payment_onehot_present(sample_df):
    feat, names = build_features(sample_df)
    for pm in PAYMENT_METHODS:
        key = f"payment_{pm.lower().replace(' ', '_')}"
        assert key in names, f"Missing payment column: {key}"


def test_cluster_onehot_present(sample_df):
    feat, names = build_features(sample_df)
    for cid in range(5):
        assert f"cluster_{cid}" in names


def test_charges_per_tenure_computed(sample_df):
    feat, _ = build_features(sample_df)
    assert "charges_per_tenure" in feat.columns
    expected = 65.0 / 18
    assert abs(feat["charges_per_tenure"].iloc[0] - expected) < 0.01


def test_zero_tenure_no_division_error(batch_df):
    feat, _ = build_features(batch_df)
    assert feat.isnull().sum().sum() == 0


def test_missing_optional_columns():
    """build_features should work even if optional columns are absent."""
    df = pd.DataFrame([{
        "tenure_months": 12,
        "monthly_charges": 50.0,
        "total_charges": 600.0,
        "contract_type": "1yr",
        "payment_method": "Bank Transfer",
    }])
    feat, names = build_features(df)
    assert len(feat) == 1
    assert feat.isnull().sum().sum() == 0


def test_shape_consistent(sample_df):
    feat, names = build_features(sample_df)
    assert feat.shape[0] == len(sample_df)
    assert feat.shape[1] == len(names)
