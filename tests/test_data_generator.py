"""
test_data_generator.py — Tests for the synthetic data generator.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.core.data_generator import generate_dataset, ARCHETYPES
from app.core.db import (
    ChurnLabel, Customer, CustomerSegment,
    Subscription, UsageLog,
)
from app.config import N_CLUSTERS


# ── Fixture: generate a small dataset once per module ────────────────────────
@pytest.fixture(scope="module")
def generated(test_engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=test_engine)
    db = Session()
    counts = generate_dataset(n_customers=300, seed=0, db=db)
    db.close()
    return counts, test_engine


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_row_counts_customers(generated):
    counts, engine = generated
    assert counts["customers"] == 300, "Expected 300 customers"


def test_row_counts_subscriptions(generated):
    counts, _ = generated
    assert counts["subscriptions"] == 300, "One subscription per customer"


def test_row_counts_churn_labels(generated):
    counts, _ = generated
    assert counts["churn_labels"] == 300, "One churn label per customer"


def test_usage_logs_generated(generated):
    counts, _ = generated
    assert counts["usage_logs"] > 0, "Usage logs should be generated"


def test_churn_rate_realistic(generated):
    counts, _ = generated
    rate = counts["churn_rate"]
    assert 0.10 <= rate <= 0.70, f"Churn rate {rate:.1%} outside realistic 10–70% range"


def test_segments_created(generated):
    counts, engine = generated
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    db = Session()
    n_segs = db.query(CustomerSegment).count()
    db.close()
    assert n_segs == N_CLUSTERS, f"Expected {N_CLUSTERS} segments, got {n_segs}"


def test_cluster_ids_in_range(generated):
    _, engine = generated
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    db = Session()
    cids = [c.cluster_id for c in db.query(Customer.cluster_id).all()]
    db.close()
    unique_cids = set(cids)
    assert unique_cids.issubset(set(range(N_CLUSTERS))), \
        f"Unexpected cluster IDs: {unique_cids}"


def test_deterministic_with_seed(test_engine):
    """Same seed → same churn rate."""
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=test_engine)

    db1 = Session()
    c1 = generate_dataset(n_customers=100, seed=77, db=db1)
    db1.close()

    db2 = Session()
    c2 = generate_dataset(n_customers=100, seed=77, db=db2)
    db2.close()

    assert abs(c1["churn_rate"] - c2["churn_rate"]) < 1e-9, \
        "Dataset generation is not deterministic with same seed"


def test_no_null_customer_ids(generated):
    _, engine = generated
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    db = Session()
    null_ids = db.query(Customer).filter(Customer.customer_id.is_(None)).count()
    db.close()
    assert null_ids == 0


def test_archetype_weights_sum_to_one():
    total = sum(a["weight"] for a in ARCHETYPES)
    assert abs(total - 1.0) < 1e-9, f"Archetype weights sum to {total}, expected 1.0"
