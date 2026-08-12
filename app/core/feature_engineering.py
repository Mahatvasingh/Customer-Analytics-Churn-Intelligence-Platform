"""
feature_engineering.py — Shared feature pipeline.

``build_features(df)`` is called identically at training time and inference time
to eliminate train/serve skew. Any change to features must be made here only.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
TENURE_BINS   = [0, 12, 24, 48, float("inf")]
TENURE_LABELS = ["0-12mo", "12-24mo", "24-48mo", "48+mo"]

CONTRACT_TYPES  = ["month-to-month", "1yr", "2yr"]
PAYMENT_METHODS = ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"]
N_CLUSTERS      = 5   # must match config.N_CLUSTERS


# ── Main function ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the engineered feature matrix from a raw customer DataFrame.

    Required input columns
    ----------------------
    tenure_months, monthly_charges, total_charges,
    contract_type, payment_method,
    avg_usage_hours, avg_support_tickets, avg_late_payments, avg_feature_adoption,
    cluster_id  (optional — filled with -1 if absent)

    Returns
    -------
    (feature_df, feature_names)
    """
    feat = pd.DataFrame(index=df.index)

    # ── 1. Raw numeric ────────────────────────────────────────────────────────
    feat["tenure_months"]    = df["tenure_months"].fillna(0).astype(float)
    feat["monthly_charges"]  = df["monthly_charges"].fillna(df["monthly_charges"].median()
                                                             if "monthly_charges" in df.columns else 65.0)
    feat["total_charges"]    = df["total_charges"].fillna(0).astype(float)

    # ── 2. Usage aggregates ───────────────────────────────────────────────────
    feat["avg_usage_hours"]       = _safe_col(df, "avg_usage_hours", 50.0)
    feat["avg_support_tickets"]   = _safe_col(df, "avg_support_tickets", 1.0)
    feat["avg_late_payments"]     = _safe_col(df, "avg_late_payments", 0.5)
    feat["avg_feature_adoption"]  = _safe_col(df, "avg_feature_adoption", 0.5)

    # ── 3. Derived ratios ─────────────────────────────────────────────────────
    feat["charges_per_tenure"] = np.where(
        feat["tenure_months"] > 0,
        feat["monthly_charges"] / feat["tenure_months"],
        feat["monthly_charges"],
    )
    feat["support_ticket_rate"] = np.where(
        feat["tenure_months"] > 0,
        feat["avg_support_tickets"] / feat["tenure_months"],
        feat["avg_support_tickets"],
    )
    feat["late_payment_ratio"] = np.where(
        feat["tenure_months"] > 0,
        feat["avg_late_payments"] / feat["tenure_months"],
        feat["avg_late_payments"],
    )

    # ── 4. Tenure buckets (one-hot) ───────────────────────────────────────────
    bucket = pd.cut(
        feat["tenure_months"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        right=True,
        include_lowest=True,
    )
    for label in TENURE_LABELS:
        safe = label.replace("-", "_").replace("+", "plus").replace("mo", "mo")
        feat[f"tenure_{safe}"] = (bucket == label).astype(int)

    # ── 5. Contract type one-hot ──────────────────────────────────────────────
    contract = _safe_col_str(df, "contract_type", "month-to-month")
    for ct in CONTRACT_TYPES:
        key = f"contract_{ct.replace('-', '_').replace(' ', '_')}"
        feat[key] = (contract == ct).astype(int)

    # ── 6. Payment method one-hot ─────────────────────────────────────────────
    payment = _safe_col_str(df, "payment_method", "Credit Card")
    for pm in PAYMENT_METHODS:
        key = f"payment_{pm.lower().replace(' ', '_')}"
        feat[key] = (payment == pm).astype(int)

    # ── 7. K-Means cluster one-hot ────────────────────────────────────────────
    cluster = _safe_col(df, "cluster_id", -1).astype(int)
    for cid in range(N_CLUSTERS):
        feat[f"cluster_{cid}"] = (cluster == cid).astype(int)

    # ── Sanitise ──────────────────────────────────────────────────────────────
    feat = feat.replace([np.inf, -np.inf], 0).fillna(0)

    feature_names = list(feat.columns)
    return feat, feature_names


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_col(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default).astype(float)
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def _safe_col_str(df: pd.DataFrame, col: str, default: str) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default).astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype=str)
