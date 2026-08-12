"""
data_generator.py — K-Means seeded synthetic customer data generator.

Architecture
────────────
1. Five customer archetype clusters are defined with realistic feature distributions.
2. Customers are sampled from those distributions (multivariate-Gaussian per cluster).
3. A churn label is injected via a weighted logistic function so the ML model has
   real, learnable signal — not pure noise.
4. Actual K-Means (k=5) is then fit on the generated feature matrix so that cluster
   IDs reflect real data geometry, not just archetype membership.
5. Cluster profiles are stored in the `customer_segments` table.

Entry point: generate_dataset(n_customers, seed, db)
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from app.config import N_CLUSTERS, RANDOM_STATE
from app.core.db import (
    ChurnLabel, ChurnPrediction, Customer, CustomerSegment,
    SessionLocal, Subscription, UsageLog,
)

fake = Faker()

# ── Archetype definitions ─────────────────────────────────────────────────────
ARCHETYPES = [
    {   # 0 — High-Risk Churners
        "name": "High-Risk Churners",
        "weight": 0.20,
        "age_range": (20, 35),
        "tenure_range": (1, 12),
        "monthly_charges_range": (80, 125),
        "contract_probs": {"month-to-month": 0.85, "1yr": 0.12, "2yr": 0.03},
        "base_churn_rate": 0.65,
        "usage_mean": 42,  "usage_std": 14,
        "tickets_mean": 3.8,
        "late_mean": 2.2,
        "adoption_alpha": 1.5, "adoption_beta": 3.5,  # beta params → mean ~0.30
    },
    {   # 1 — Stable Mid-Tenure
        "name": "Stable Mid-Tenure",
        "weight": 0.25,
        "age_range": (30, 50),
        "tenure_range": (12, 36),
        "monthly_charges_range": (50, 85),
        "contract_probs": {"month-to-month": 0.30, "1yr": 0.55, "2yr": 0.15},
        "base_churn_rate": 0.20,
        "usage_mean": 62, "usage_std": 18,
        "tickets_mean": 1.5,
        "late_mean": 0.8,
        "adoption_alpha": 2.8, "adoption_beta": 2.2,  # mean ~0.56
    },
    {   # 2 — Loyal Long-Term
        "name": "Loyal Long-Term",
        "weight": 0.20,
        "age_range": (35, 65),
        "tenure_range": (36, 72),
        "monthly_charges_range": (40, 70),
        "contract_probs": {"month-to-month": 0.05, "1yr": 0.30, "2yr": 0.65},
        "base_churn_rate": 0.08,
        "usage_mean": 78, "usage_std": 16,
        "tickets_mean": 0.7,
        "late_mean": 0.2,
        "adoption_alpha": 4.0, "adoption_beta": 1.5,  # mean ~0.73
    },
    {   # 3 — Support-Heavy
        "name": "Support-Heavy",
        "weight": 0.15,
        "age_range": (25, 55),
        "tenure_range": (3, 24),
        "monthly_charges_range": (60, 105),
        "contract_probs": {"month-to-month": 0.60, "1yr": 0.30, "2yr": 0.10},
        "base_churn_rate": 0.55,
        "usage_mean": 38, "usage_std": 18,
        "tickets_mean": 5.5,
        "late_mean": 3.8,
        "adoption_alpha": 1.5, "adoption_beta": 3.0,  # mean ~0.33
    },
    {   # 4 — Price-Sensitive / Senior Stable
        "name": "Price-Sensitive",
        "weight": 0.20,
        "age_range": (50, 75),
        "tenure_range": (6, 48),
        "monthly_charges_range": (28, 58),
        "contract_probs": {"month-to-month": 0.20, "1yr": 0.50, "2yr": 0.30},
        "base_churn_rate": 0.12,
        "usage_mean": 33, "usage_std": 11,
        "tickets_mean": 1.1,
        "late_mean": 0.4,
        "adoption_alpha": 2.2, "adoption_beta": 2.8,  # mean ~0.44
    },
]

REGIONS         = ["North", "South", "East", "West", "Central"]
PAYMENT_METHODS = ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"]
PLAN_NAMES      = {
    "month-to-month": ["Basic Monthly", "Pro Monthly", "Starter Monthly"],
    "1yr":            ["Standard Annual", "Pro Annual"],
    "2yr":            ["Premium 2-Year", "Business 2-Year"],
}


# ── Churn probability function ────────────────────────────────────────────────

def _churn_probability(row: dict, base_rate: float) -> float:
    """
    Weighted logistic sigmoid that produces a realistic churn probability.
    Uses causal signals: contract type, tenure, charges, tickets, late payments,
    feature adoption — matching real-world churn drivers.
    """
    contract_risk = {"month-to-month": 1.5, "1yr": 0.4, "2yr": -0.9}[row["contract_type"]]
    tenure_risk   = -0.035 * row["tenure_months"]
    charge_risk   = 0.014  * (row["monthly_charges"] - 68)
    ticket_risk   = 0.28   * row["avg_support_tickets"]
    late_risk     = 0.38   * row["avg_late_payments"]
    adoption_risk = -1.3   * row["avg_feature_adoption"]

    logit = contract_risk + tenure_risk + charge_risk + ticket_risk + late_risk + adoption_risk
    computed = 1.0 / (1.0 + np.exp(-logit))

    # 30 % archetype weight, 70 % computed signal
    return float(0.30 * base_rate + 0.70 * computed)


# ── Main generator ────────────────────────────────────────────────────────────

def generate_dataset(
    n_customers: int = 8000,
    seed: int = 42,
    db: Optional[Session] = None,
) -> dict:
    """
    Generate a full synthetic dataset and populate all DB tables.

    Steps
    -----
    1. Sample customers from archetype Gaussian distributions.
    2. Inject churn labels via ``_churn_probability``.
    3. Fit K-Means (k=5) on the feature matrix → assign real cluster IDs.
    4. Store cluster profiles in ``customer_segments``.
    5. Insert all rows in a single transaction.

    Returns
    -------
    dict with row counts per table and overall churn rate.
    """
    np.random.seed(seed)
    random.seed(seed)
    Faker.seed(seed)

    _close = db is None
    if _close:
        db = SessionLocal()

    try:
        # ── Clear existing data (cascade order) ──────────────────────────────
        db.query(ChurnPrediction).delete()
        db.query(ChurnLabel).delete()
        db.query(UsageLog).delete()
        db.query(Subscription).delete()
        db.query(Customer).delete()
        db.query(CustomerSegment).delete()
        db.commit()

        # ── Step 1: Sample archetypes ────────────────────────────────────────
        arch_indices = np.random.choice(
            len(ARCHETYPES),
            size=n_customers,
            p=[a["weight"] for a in ARCHETYPES],
        )

        records = []
        for arch_idx in arch_indices:
            a = ARCHETYPES[arch_idx]

            tenure = int(np.clip(
                np.random.normal(np.mean(a["tenure_range"]),
                                 (a["tenure_range"][1] - a["tenure_range"][0]) / 4),
                a["tenure_range"][0], a["tenure_range"][1],
            ))
            monthly_charges = float(np.clip(
                np.random.normal(np.mean(a["monthly_charges_range"]),
                                 (a["monthly_charges_range"][1] - a["monthly_charges_range"][0]) / 4),
                a["monthly_charges_range"][0], a["monthly_charges_range"][1],
            ))
            age            = random.randint(*a["age_range"])
            contract_type  = np.random.choice(list(a["contract_probs"]), p=list(a["contract_probs"].values()))
            payment_method = random.choice(PAYMENT_METHODS)
            region         = random.choice(REGIONS)
            gender         = random.choice(["Male", "Female", "Non-binary"])
            signup_date    = datetime.now() - timedelta(days=tenure * 30 + random.randint(0, 29))

            # Monthly usage logs (last min(tenure, 12) months)
            n_months = min(tenure, 12)
            usage_h  = [max(0.0, np.random.normal(a["usage_mean"],   a["usage_std"]))  for _ in range(n_months)]
            tickets  = [max(0,   int(np.random.poisson(a["tickets_mean"])))             for _ in range(n_months)]
            late_pay = [max(0,   int(np.random.poisson(a["late_mean"])))                for _ in range(n_months)]
            adoption = [float(np.clip(np.random.beta(a["adoption_alpha"], a["adoption_beta"]), 0, 1))
                        for _ in range(n_months)]

            records.append({
                "arch_idx":            arch_idx,
                "age":                 age,
                "gender":              gender,
                "region":              region,
                "contract_type":       contract_type,
                "payment_method":      payment_method,
                "monthly_charges":     round(monthly_charges, 2),
                "total_charges":       round(monthly_charges * tenure, 2),
                "tenure_months":       tenure,
                "signup_date":         signup_date,
                "n_months":            n_months,
                "usage_h":             usage_h,
                "tickets":             tickets,
                "late_pay":            late_pay,
                "adoption":            adoption,
                "avg_usage_hours":     float(np.mean(usage_h))  if usage_h else a["usage_mean"],
                "avg_support_tickets": float(np.mean(tickets))  if tickets else a["tickets_mean"],
                "avg_late_payments":   float(np.mean(late_pay)) if late_pay else a["late_mean"],
                "avg_feature_adoption":float(np.mean(adoption)) if adoption else 0.5,
                "base_churn_rate":     a["base_churn_rate"],
            })

        df = pd.DataFrame(records)

        # ── Step 2: Churn labels ─────────────────────────────────────────────
        churn_probs = df.apply(lambda r: _churn_probability(r, r["base_churn_rate"]), axis=1)
        df["churned"] = (np.random.rand(len(df)) < churn_probs).astype(bool)

        # ── Step 3: K-Means clustering ───────────────────────────────────────
        feat_matrix = np.column_stack([
            df["monthly_charges"].values,
            df["tenure_months"].values,
            df["avg_usage_hours"].values,
            df["avg_support_tickets"].values,
            df["avg_late_payments"].values,
            df["avg_feature_adoption"].values,
        ])
        scaler     = StandardScaler()
        feat_scaled = scaler.fit_transform(feat_matrix)
        kmeans     = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        df["cluster_id"] = kmeans.fit_predict(feat_scaled)

        # ── Step 4: Build cluster profiles ───────────────────────────────────
        cluster_profiles = []
        for cid in range(N_CLUSTERS):
            mask     = df["cluster_id"] == cid
            seg      = df[mask]
            churn_rt = float(seg["churned"].mean()) if len(seg) > 0 else 0.0
            dom_contract = seg["contract_type"].mode().iloc[0] if len(seg) > 0 else "month-to-month"

            if churn_rt > 0.50:
                label = f"⚠️ High-Risk Segment {cid}"
            elif churn_rt > 0.25:
                label = f"⚡ Moderate-Risk Segment {cid}"
            else:
                label = f"✅ Low-Risk Segment {cid}"

            cluster_profiles.append({
                "cluster_id":    cid,
                "label":         label,
                "avg_churn_rate": round(churn_rt, 4),
                "size":          int(mask.sum()),
                "profile_json":  {
                    "avg_monthly_charges":  round(float(seg["monthly_charges"].mean()), 2),
                    "avg_tenure_months":    round(float(seg["tenure_months"].mean()), 1),
                    "avg_usage_hours":      round(float(seg["avg_usage_hours"].mean()), 1),
                    "avg_support_tickets":  round(float(seg["avg_support_tickets"].mean()), 2),
                    "avg_late_payments":    round(float(seg["avg_late_payments"].mean()), 2),
                    "avg_feature_adoption": round(float(seg["avg_feature_adoption"].mean()), 3),
                    "dominant_contract":    dom_contract,
                    "churn_rate":           round(churn_rt, 3),
                },
            })

        # ── Step 5: Insert into DB ────────────────────────────────────────────
        customer_objs = []
        for _, row in df.iterrows():
            customer_objs.append(Customer(
                age=int(row["age"]),
                gender=row["gender"],
                region=row["region"],
                contract_type=row["contract_type"],
                payment_method=row["payment_method"],
                monthly_charges=float(row["monthly_charges"]),
                total_charges=float(row["total_charges"]),
                tenure_months=int(row["tenure_months"]),
                signup_date=row["signup_date"],
                is_active=not bool(row["churned"]),
                cluster_id=int(row["cluster_id"]),
            ))

        db.add_all(customer_objs)
        db.flush()   # populate customer_ids

        sub_objs    = []
        usage_objs  = []
        label_objs  = []

        for idx, (_, row) in enumerate(df.iterrows()):
            cobj = customer_objs[idx]
            cid  = cobj.customer_id

            # Subscription
            plans     = PLAN_NAMES.get(row["contract_type"], ["Basic"])
            sub_start = row["signup_date"]
            sub_end   = (sub_start + timedelta(days=int(row["tenure_months"]) * 30)
                         if row["churned"] else None)
            sub_objs.append(Subscription(
                customer_id=cid,
                plan_name=random.choice(plans),
                start_date=sub_start,
                end_date=sub_end,
                status="cancelled" if row["churned"] else "active",
            ))

            # Usage logs
            n_months = int(row["n_months"])
            for m in range(n_months):
                month_str = (datetime.now() - timedelta(days=(n_months - m) * 30)).strftime("%Y-%m")
                usage_objs.append(UsageLog(
                    customer_id=cid,
                    month=month_str,
                    usage_hours=round(float(row["usage_h"][m]), 2),
                    support_tickets=int(row["tickets"][m]),
                    late_payments=int(row["late_pay"][m]),
                    feature_adoption_score=round(float(row["adoption"][m]), 4),
                ))

            # Churn label
            label_objs.append(ChurnLabel(
                customer_id=cid,
                churned=bool(row["churned"]),
            ))

        db.add_all(sub_objs)
        db.add_all(usage_objs)
        db.add_all(label_objs)

        for cp in cluster_profiles:
            db.add(CustomerSegment(
                cluster_id=cp["cluster_id"],
                label=cp["label"],
                avg_churn_rate=cp["avg_churn_rate"],
                size=cp["size"],
                profile_json=cp["profile_json"],
                updated_at=datetime.utcnow(),
            ))

        db.commit()

        return {
            "customers":     len(customer_objs),
            "subscriptions": len(sub_objs),
            "usage_logs":    len(usage_objs),
            "churn_labels":  len(label_objs),
            "segments":      len(cluster_profiles),
            "churn_rate":    round(float(df["churned"].mean()), 4),
        }

    except Exception:
        db.rollback()
        raise
    finally:
        if _close:
            db.close()
