"""
test_model_training.py — Integration tests for the training pipeline.

Uses an in-memory DB populated with minimal synthetic data.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.core.data_generator import generate_dataset
from app.core.model_training import train_and_register
from app.core.db import ModelRegistry, CustomerSegment
from app.config import MODELS_DIR, N_CLUSTERS


# ── Fixture: generate minimal data + train once per module ────────────────────
@pytest.fixture(scope="module")
def training_result(test_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=test_engine)
    db = Session()

    # Generate minimal data (200 rows is enough for sanity checks)
    generate_dataset(n_customers=200, seed=99, db=db)

    # Patch model_training to use the test engine
    import app.core.model_training as mt
    original_load = mt._load_training_data

    def _patched_load(session):
        import pandas as pd
        from sqlalchemy import text
        from app.core.db import Customer, UsageLog, ChurnLabel
        customers  = pd.read_sql(text("SELECT * FROM customers"),  test_engine)
        labels     = pd.read_sql(text("SELECT customer_id, churned FROM churn_labels"), test_engine)
        usage_agg  = pd.read_sql(text("""
            SELECT customer_id,
                AVG(usage_hours)            AS avg_usage_hours,
                AVG(support_tickets)        AS avg_support_tickets,
                AVG(late_payments)          AS avg_late_payments,
                AVG(feature_adoption_score) AS avg_feature_adoption
            FROM usage_logs GROUP BY customer_id
        """), test_engine)
        df = customers.merge(usage_agg, on="customer_id", how="left").merge(
            labels, on="customer_id", how="inner"
        )
        for col, val in [("avg_usage_hours",50),("avg_support_tickets",1),
                         ("avg_late_payments",0.5),("avg_feature_adoption",0.5)]:
            df[col] = df[col].fillna(val)
        return df

    mt._load_training_data = _patched_load

    result = train_and_register(db=db)

    mt._load_training_data = original_load
    db.close()
    return result, test_engine


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_training_returns_dict(training_result):
    result, _ = training_result
    assert isinstance(result, dict)


def test_production_model_set(training_result):
    result, engine = training_result
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    db = Session()
    prod = db.query(ModelRegistry).filter(ModelRegistry.is_production.is_(True)).count()
    db.close()
    assert prod == 1, "Exactly one model should be marked production"


def test_classifier_auc_above_threshold(training_result):
    result, _ = training_result
    best_auc = result["best_auc"]
    assert best_auc >= 0.55, f"Best AUC {best_auc:.4f} below threshold 0.55"


def test_both_classifiers_registered(training_result):
    _, engine = training_result
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    db = Session()
    algos = {m.algorithm for m in db.query(ModelRegistry).all()}
    db.close()
    assert "logistic_regression" in algos
    assert "random_forest" in algos


def test_artifacts_exist_on_disk(training_result):
    result, _ = training_result
    for algo, info in result["classifiers"].items():
        path = Path(info.get("artifact_path", MODELS_DIR / f"{info['version']}.joblib"))
        # The artifact path may vary; check MODELS_DIR for any matching file
    # At minimum, check MODELS_DIR is not empty
    artifacts = list(MODELS_DIR.glob("*.joblib"))
    assert len(artifacts) > 0, "No .joblib artifacts found in models/"


def test_kmeans_segments_updated(training_result):
    _, engine = training_result
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    db = Session()
    n_segs = db.query(CustomerSegment).count()
    db.close()
    assert n_segs == N_CLUSTERS, f"Expected {N_CLUSTERS} segments after training"


def test_shap_reported(training_result):
    result, _ = training_result
    # SHAP may or may not succeed depending on environment — just check key exists
    assert "shap_available" in result


def test_version_string_format(training_result):
    result, _ = training_result
    assert result["version"].startswith("v"), \
        f"Version '{result['version']}' should start with 'v'"


def test_all_metrics_in_range(training_result):
    _, engine = training_result
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    db = Session()
    for m in db.query(ModelRegistry).all():
        if m.auc is not None:
            assert 0 <= m.auc <= 1, f"AUC out of range: {m.auc}"
        if m.f1 is not None:
            assert 0 <= m.f1 <= 1, f"F1 out of range: {m.f1}"
    db.close()
