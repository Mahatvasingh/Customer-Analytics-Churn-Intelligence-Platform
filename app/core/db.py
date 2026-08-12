"""
db.py — SQLAlchemy ORM models + engine/session factory.

Swap SQLite → PostgreSQL by changing DATABASE_URL in .env only.
No other code changes required (all models are DB-agnostic).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, JSON, String, Text, create_engine, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from app.config import DATABASE_URL


# ── Base class ────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── ORM Models ────────────────────────────────────────────────────────────────

class Customer(Base):
    """Core customer demographic + contract data."""
    __tablename__ = "customers"

    customer_id     = Column(Integer, primary_key=True, index=True, autoincrement=True)
    signup_date     = Column(DateTime, default=datetime.utcnow, nullable=False)
    age             = Column(Integer, nullable=False)
    gender          = Column(String(20), nullable=False)
    region          = Column(String(50), nullable=False)
    contract_type   = Column(String(20), nullable=False)   # month-to-month | 1yr | 2yr
    payment_method  = Column(String(40), nullable=False)
    monthly_charges = Column(Float, nullable=False)
    total_charges   = Column(Float, nullable=False)
    tenure_months   = Column(Integer, nullable=False)
    is_active       = Column(Boolean, default=True, nullable=False)
    cluster_id      = Column(Integer, nullable=True)       # assigned by K-Means

    # Relationships
    subscriptions = relationship("Subscription",   back_populates="customer", cascade="all, delete-orphan")
    usage_logs    = relationship("UsageLog",       back_populates="customer", cascade="all, delete-orphan")
    churn_label   = relationship("ChurnLabel",     back_populates="customer", uselist=False, cascade="all, delete-orphan")
    predictions   = relationship("ChurnPrediction", back_populates="customer", cascade="all, delete-orphan")


class Subscription(Base):
    """Customer subscription / plan history."""
    __tablename__ = "subscriptions"

    subscription_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    customer_id     = Column(Integer, ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False)
    plan_name       = Column(String(60), nullable=False)
    start_date      = Column(DateTime, nullable=False)
    end_date        = Column(DateTime, nullable=True)   # NULL = still active
    status          = Column(String(20), nullable=False)  # active | cancelled | paused

    customer = relationship("Customer", back_populates="subscriptions")


class UsageLog(Base):
    """Monthly usage telemetry per customer."""
    __tablename__ = "usage_logs"

    log_id                 = Column(Integer, primary_key=True, index=True, autoincrement=True)
    customer_id            = Column(Integer, ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False)
    month                  = Column(String(7), nullable=False)   # YYYY-MM
    usage_hours            = Column(Float, nullable=False)
    support_tickets        = Column(Integer, nullable=False, default=0)
    late_payments          = Column(Integer, nullable=False, default=0)
    feature_adoption_score = Column(Float, nullable=False)       # 0.0–1.0

    customer = relationship("Customer", back_populates="usage_logs")


class ChurnLabel(Base):
    """Ground-truth churn label (used for ML training)."""
    __tablename__ = "churn_labels"

    customer_id = Column(Integer, ForeignKey("customers.customer_id", ondelete="CASCADE"), primary_key=True)
    churned     = Column(Boolean, nullable=False)
    labeled_at  = Column(DateTime, default=datetime.utcnow, nullable=False)

    customer = relationship("Customer", back_populates="churn_label")


class ChurnPrediction(Base):
    """Logged churn predictions (single + batch) with SHAP top factors."""
    __tablename__ = "churn_predictions"

    prediction_id      = Column(Integer, primary_key=True, index=True, autoincrement=True)
    customer_id        = Column(Integer, ForeignKey("customers.customer_id", ondelete="CASCADE"), nullable=False)
    model_version      = Column(String(40), nullable=False)
    churn_probability  = Column(Float, nullable=False)
    predicted_label    = Column(Boolean, nullable=False)
    predicted_at       = Column(DateTime, default=datetime.utcnow, nullable=False)
    top_factors        = Column(JSON, nullable=True)   # {"feature": shap_value, ...}

    customer = relationship("Customer", back_populates="predictions")


class ModelRegistry(Base):
    """Versioned model registry — one row per training run per algorithm."""
    __tablename__ = "model_registry"

    model_id      = Column(Integer, primary_key=True, index=True, autoincrement=True)
    version       = Column(String(40), unique=True, nullable=False)
    algorithm     = Column(String(60), nullable=False)
    trained_at    = Column(DateTime, default=datetime.utcnow, nullable=False)
    auc           = Column(Float, nullable=True)
    f1            = Column(Float, nullable=True)
    precision     = Column(Float, nullable=True)
    recall        = Column(Float, nullable=True)
    is_production = Column(Boolean, default=False, nullable=False)
    artifact_path = Column(String(512), nullable=False)
    n_features    = Column(Integer, nullable=True)
    feature_names = Column(JSON, nullable=True)
    confusion_matrix = Column(JSON, nullable=True)


class CustomerSegment(Base):
    """K-Means cluster profiles, updated each training run."""
    __tablename__ = "customer_segments"

    segment_id    = Column(Integer, primary_key=True, index=True, autoincrement=True)
    cluster_id    = Column(Integer, unique=True, nullable=False)
    label         = Column(String(120), nullable=False)
    avg_churn_rate = Column(Float, nullable=False)
    size          = Column(Integer, nullable=False)
    profile_json  = Column(JSON, nullable=True)    # avg feature values for this cluster
    updated_at    = Column(DateTime, default=datetime.utcnow, nullable=False)


# ── Engine & Session factory ──────────────────────────────────────────────────
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI / plain dependency-injection helper."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables (idempotent)."""
    Base.metadata.create_all(bind=engine)
