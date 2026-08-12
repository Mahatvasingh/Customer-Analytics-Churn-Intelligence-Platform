"""
model_registry.py — CRUD operations for the model_registry table.

Functions
─────────
list_versions()         → list of ModelRegistry rows (newest first)
get_production_model()  → (artifact_dict, ModelRegistry_entry)
promote(version)        → atomically set is_production=True on one version
rollback_to(version)    → alias for promote
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
from sqlalchemy.orm import Session

from app.core.db import ModelRegistry, SessionLocal


def list_versions(db: Optional[Session] = None):
    """Return all model versions ordered newest-first."""
    _close = db is None
    if _close:
        db = SessionLocal()
    try:
        return db.query(ModelRegistry).order_by(ModelRegistry.trained_at.desc()).all()
    finally:
        if _close:
            db.close()


def get_production_model(db: Optional[Session] = None) -> Tuple[dict, ModelRegistry]:
    """
    Load the currently promoted production model artifact from disk.

    Returns
    -------
    (artifact_dict, registry_entry)
    artifact_dict has keys: "model", "background_X", "feature_names"
    """
    _close = db is None
    if _close:
        db = SessionLocal()
    try:
        entry = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.is_production.is_(True))
            .first()
        )
        if entry is None:
            raise RuntimeError(
                "No production model found. Run seed.py to train a model first."
            )

        path = Path(entry.artifact_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Artifact not found: {path}\n"
                "Re-run training or check your MODELS_DIR."
            )

        artifact = joblib.load(path)
        # Backward-compat: wrap bare sklearn models
        if not isinstance(artifact, dict):
            artifact = {"model": artifact, "background_X": None, "feature_names": []}

        return artifact, entry
    finally:
        if _close:
            db.close()


def promote(version: str, db: Optional[Session] = None) -> bool:
    """
    Atomically promote a model version to production.
    Clears is_production on all other rows first.
    """
    _close = db is None
    if _close:
        db = SessionLocal()
    try:
        db.query(ModelRegistry).update({"is_production": False})
        n = db.query(ModelRegistry).filter(
            ModelRegistry.version == version
        ).update({"is_production": True})
        if n == 0:
            raise ValueError(f"Version '{version}' not found in model registry.")
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        if _close:
            db.close()


def rollback_to(version: str, db: Optional[Session] = None) -> bool:
    """Alias for promote — semantically used for rollback operations."""
    return promote(version, db)
