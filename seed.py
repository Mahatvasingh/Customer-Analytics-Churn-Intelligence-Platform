"""
seed.py — One-time setup script.

Run once from the project root:
    python seed.py

What it does
────────────
1. Creates all database tables.
2. Generates synthetic dataset (8 000 customers by default).
3. Trains the initial ML models and registers the production model.
4. Prints a summary of row counts and model metrics.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from app.core.db import init_db, SessionLocal
from app.core.data_generator import generate_dataset
from app.core.model_training import train_and_register


def banner(msg: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print(f"{'─'*60}")


def main() -> None:
    t0 = time.time()

    banner("🔮 Customer Churn Intelligence Platform — Seed")

    # ── 1. Init DB ────────────────────────────────────────────────────────────
    print("\n[1/3] Initialising database tables …")
    init_db()
    print("      ✅ Tables created (or already exist)")

    # ── 2. Generate data ──────────────────────────────────────────────────────
    print("\n[2/3] Generating synthetic dataset …")
    print("      This may take 20–60 seconds for 8 000 customers.")
    t1 = time.time()
    db = SessionLocal()
    try:
        counts = generate_dataset(n_customers=8000, seed=42, db=db)
    finally:
        db.close()

    print(f"      ✅ Done in {time.time()-t1:.1f}s")
    print(f"      Customers   : {counts['customers']:,}")
    print(f"      Subscriptions: {counts['subscriptions']:,}")
    print(f"      Usage logs  : {counts['usage_logs']:,}")
    print(f"      Churn rate  : {counts['churn_rate']:.1%}")
    print(f"      K-Means segments: {counts['segments']}")

    # ── 3. Train models ───────────────────────────────────────────────────────
    print("\n[3/3] Training ML models …")
    print("      Logistic Regression + Random Forest + K-Means segmentation")
    t2 = time.time()
    summary = train_and_register()
    elapsed = time.time() - t2
    print(f"      ✅ Training complete in {elapsed:.1f}s")
    print(f"      Production model : {summary['production']}")
    print(f"      Best AUC         : {summary['best_auc']:.4f}")
    print(f"      SHAP available   : {summary['shap_available']}")

    # ── Final summary ─────────────────────────────────────────────────────────
    banner(f"✅ Seed complete in {time.time()-t0:.1f}s total")
    print(
        "\n  Launch the app with:\n"
        "      streamlit run app/main.py\n"
    )


if __name__ == "__main__":
    main()
