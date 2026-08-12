"""
model_training.py — Train Logistic Regression + Random Forest classifiers
                    and a K-Means segmentation model.

Flow
────
1. Pull joined customer + usage + churn data from the DB.
2. Call ``build_features()`` (same pipeline as inference).
3. Stratified 80/20 split + 5-fold CV on both classifiers.
4. Compute AUC, F1, precision, recall, confusion matrix.
5. Select best classifier by AUC → promote as production model.
6. Compute SHAP values (TreeExplainer for RF, LinearExplainer for LR).
7. Fit K-Means → update ``customer_segments`` table.
8. Register every artifact in ``model_registry``.

Each artifact saved as models/{algo}_{version}.joblib.
Artifact is a dict: {"model", "background_X", "feature_names"}
so downstream code can always recreate SHAP explainers.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from sqlalchemy.orm import Session

import shap

from app.config import CV_FOLDS, MODELS_DIR, N_CLUSTERS, RANDOM_STATE, TEST_SIZE
from app.core.db import CustomerSegment, ModelRegistry, SessionLocal
from app.core.feature_engineering import build_features

warnings.filterwarnings("ignore", category=UserWarning)

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_training_data(db: Session) -> pd.DataFrame:
    """Join customers + usage aggregates + churn labels into a flat DataFrame."""
    from app.core.db import engine

    customers = pd.read_sql(
        text("SELECT * FROM customers"), engine
    )
    labels = pd.read_sql(
        text("SELECT customer_id, churned FROM churn_labels"), engine
    )
    usage_agg = pd.read_sql(
        text("""
            SELECT
                customer_id,
                AVG(usage_hours)            AS avg_usage_hours,
                AVG(support_tickets)        AS avg_support_tickets,
                AVG(late_payments)          AS avg_late_payments,
                AVG(feature_adoption_score) AS avg_feature_adoption
            FROM usage_logs
            GROUP BY customer_id
        """), engine
    )

    df = (
        customers
        .merge(usage_agg, on="customer_id", how="left")
        .merge(labels,    on="customer_id", how="inner")
    )

    # Fill missing usage stats
    for col, val in [
        ("avg_usage_hours", df["avg_usage_hours"].median()),
        ("avg_support_tickets", 1.0),
        ("avg_late_payments", 0.5),
        ("avg_feature_adoption", 0.5),
    ]:
        df[col] = df[col].fillna(val)

    return df


def _next_version(db: Session) -> str:
    count = db.query(ModelRegistry).count()
    return f"v{count + 1}.0"


# ── SHAP helpers ──────────────────────────────────────────────────────────────

def _compute_shap(model, X_train_bg: pd.DataFrame, X_test: pd.DataFrame,
                  feature_names: list) -> Optional[dict]:
    """Compute mean |SHAP| importances. Returns feature→value dict or None."""
    try:
        algo = type(model).__name__
        if "Forest" in algo or "Tree" in algo:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_train_bg)

        sample = X_test.iloc[:min(500, len(X_test))]
        sv = explainer.shap_values(sample)
        if isinstance(sv, list):
            sv = sv[1]  # class 1
        mean_abs = np.abs(sv).mean(axis=0)
        return dict(zip(feature_names, [float(v) for v in mean_abs]))
    except Exception as e:
        print(f"   ⚠️  SHAP failed ({e}); skipping importance export.")
        return None


def _shap_for_row(model, background_X: pd.DataFrame, X_row: pd.DataFrame,
                  feature_names: list) -> dict:
    """Compute SHAP values for a single inference row."""
    try:
        algo = type(model).__name__
        if "Forest" in algo or "Tree" in algo:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, background_X)
        sv = explainer.shap_values(X_row)
        if isinstance(sv, list):
            sv = sv[1]
        values = sv[0] if sv.ndim > 1 else sv
        return dict(zip(feature_names, [float(v) for v in values]))
    except Exception:
        return {}


# ── Main training entry point ─────────────────────────────────────────────────

def train_and_register(db: Optional[Session] = None) -> dict:
    """
    Train LR + RF classifiers and K-Means segmentation.
    Registers all artifacts in the DB. Returns a summary dict.
    """
    _close = db is None
    if _close:
        db = SessionLocal()

    try:
        print("📊 Loading training data …")
        df = _load_training_data(db)

        if len(df) < 50:
            raise RuntimeError(
                f"Only {len(df)} labelled rows found. Run seed.py first."
            )

        X, feature_names = build_features(df)
        y = df["churned"].astype(int)

        print(f"   Rows: {len(df):,}  |  Features: {len(feature_names)}  |  Churn rate: {y.mean():.1%}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        # Background sample for SHAP (kept small for memory)
        bg_idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X_train), size=min(100, len(X_train)), replace=False
        )
        background_X = X_train.iloc[bg_idx].reset_index(drop=True)

        version = _next_version(db)
        results = {}
        best_auc = -1.0
        best_version = None
        best_shap = None

        # ── Train classifiers ─────────────────────────────────────────────────
        classifiers = [
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=1000, random_state=RANDOM_STATE,
                    class_weight="balanced", solver="lbfgs",
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=200, random_state=RANDOM_STATE,
                    class_weight="balanced", n_jobs=-1,
                    max_depth=12, min_samples_leaf=5,
                ),
            ),
        ]

        for algo_name, clf in classifiers:
            print(f"\n🔧 Training {algo_name} …")

            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            cv_aucs = cross_val_score(clf, X_train, y_train, cv=cv, scoring="roc_auc")
            print(f"   CV AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

            clf.fit(X_train, y_train)
            y_proba = clf.predict_proba(X_test)[:, 1]
            y_pred  = (y_proba >= 0.5).astype(int)

            auc  = roc_auc_score(y_test, y_proba)
            f1   = f1_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            cm   = confusion_matrix(y_test, y_pred).tolist()

            print(f"   AUC={auc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

            # SHAP importances
            shap_dict = _compute_shap(clf, background_X, X_test, feature_names)
            if shap_dict and algo_name == "random_forest":
                shap_path = MODELS_DIR / f"shap_importances_{version}.json"
                shap_path.write_text(json.dumps(shap_dict, indent=2))
                print(f"   SHAP importances → {shap_path.name}")

            # Save artifact (dict so downstream can recreate SHAP explainers)
            algo_version  = f"{algo_name}_{version}"
            artifact_path = MODELS_DIR / f"{algo_version}.joblib"
            joblib.dump(
                {
                    "model":        clf,
                    "background_X": background_X,
                    "feature_names": feature_names,
                },
                artifact_path,
            )

            # Metadata JSON
            meta = {
                "version": algo_version, "algorithm": algo_name,
                "trained_at": datetime.utcnow().isoformat(),
                "auc": auc, "f1": f1, "precision": prec, "recall": rec,
                "confusion_matrix": cm,
                "cv_auc_mean": float(cv_aucs.mean()), "cv_auc_std": float(cv_aucs.std()),
                "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                "n_features": len(feature_names), "feature_names": feature_names,
                "shap_importances": shap_dict,
            }
            (MODELS_DIR / f"{algo_version}_metadata.json").write_text(
                json.dumps(meta, indent=2)
            )

            # Registry row
            entry = ModelRegistry(
                version=algo_version,
                algorithm=algo_name,
                auc=auc, f1=f1, precision=prec, recall=rec,
                is_production=False,
                artifact_path=str(artifact_path),
                n_features=len(feature_names),
                feature_names=feature_names,
                confusion_matrix=cm,
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)

            results[algo_name] = {
                "version": algo_version, "model_id": entry.model_id,
                "auc": auc, "f1": f1,
            }

            if auc > best_auc:
                best_auc, best_version = auc, algo_version
                if algo_name == "random_forest":
                    best_shap = shap_dict

        # ── Promote best classifier ───────────────────────────────────────────
        db.query(ModelRegistry).update({"is_production": False})
        db.query(ModelRegistry).filter(
            ModelRegistry.version == best_version
        ).update({"is_production": True})
        db.commit()
        print(f"\n🏆 Production model → {best_version}  (AUC={best_auc:.4f})")

        # ── K-Means segmentation ──────────────────────────────────────────────
        print("\n🔮 Fitting K-Means segmentation …")
        km_features = ["monthly_charges", "tenure_months", "avg_usage_hours",
                       "avg_support_tickets", "avg_late_payments", "avg_feature_adoption"]
        km_matrix = df[km_features].fillna(0).values
        scaler     = StandardScaler()
        km_scaled  = scaler.fit_transform(km_matrix)
        kmeans     = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        km_labels  = kmeans.fit_predict(km_scaled)

        # Save K-Means artifact
        km_path = MODELS_DIR / f"kmeans_{version}.joblib"
        joblib.dump({"model": kmeans, "scaler": scaler, "feature_names": km_features}, km_path)

        # Update customer_segments
        df_km = df.copy()
        df_km["km_cluster"] = km_labels

        db.query(CustomerSegment).delete()
        for cid in range(N_CLUSTERS):
            mask    = df_km["km_cluster"] == cid
            seg     = df_km[mask]
            churn_r = float(seg["churned"].mean()) if len(seg) else 0.0
            dom_c   = seg["contract_type"].mode().iloc[0] if len(seg) else "unknown"

            if churn_r > 0.50:
                lbl = f"⚠️ High-Risk Segment {cid}"
            elif churn_r > 0.25:
                lbl = f"⚡ Moderate-Risk Segment {cid}"
            else:
                lbl = f"✅ Low-Risk Segment {cid}"

            db.add(CustomerSegment(
                cluster_id=cid, label=lbl,
                avg_churn_rate=round(churn_r, 4),
                size=int(mask.sum()),
                profile_json={
                    "avg_monthly_charges":  round(float(seg["monthly_charges"].mean()), 2),
                    "avg_tenure_months":    round(float(seg["tenure_months"].mean()), 1),
                    "avg_usage_hours":      round(float(seg["avg_usage_hours"].mean()), 1),
                    "avg_support_tickets":  round(float(seg["avg_support_tickets"].mean()), 2),
                    "avg_feature_adoption": round(float(seg["avg_feature_adoption"].mean()), 3),
                    "dominant_contract":    dom_c,
                    "churn_rate":           round(churn_r, 3),
                },
                updated_at=datetime.utcnow(),
            ))
        db.commit()
        print("   Customer segments updated in DB ✅")

        return {
            "version":        version,
            "classifiers":    results,
            "production":     best_version,
            "best_auc":       best_auc,
            "shap_available": best_shap is not None,
            "n_samples":      len(df),
        }

    except Exception:
        db.rollback()
        raise
    finally:
        if _close:
            db.close()


# ── Inference SHAP helper (used by Predict page) ──────────────────────────────

def compute_shap_for_row(artifact: dict, X_row: pd.DataFrame) -> dict:
    """
    Given a loaded model artifact dict and a 1-row feature DataFrame,
    return {feature: shap_value} for top-factor display.
    """
    return _shap_for_row(
        artifact["model"],
        artifact["background_X"],
        X_row,
        artifact["feature_names"],
    )
