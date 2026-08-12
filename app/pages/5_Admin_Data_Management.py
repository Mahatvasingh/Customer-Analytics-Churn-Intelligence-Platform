"""
5_Admin_Data_Management.py — Dataset regeneration and DB statistics.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st
from sqlalchemy import text

from app.core.data_generator import generate_dataset
from app.core.db import SessionLocal, engine
from app.core.utils import GLOBAL_CSS

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
st.title("⚙️ Data Management")
st.caption("Regenerate the synthetic dataset and inspect database statistics")

# ── DB Stats ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def _db_stats():
    tables = [
        "customers", "subscriptions", "usage_logs",
        "churn_labels", "churn_predictions",
        "model_registry", "customer_segments",
    ]
    stats = []
    with engine.connect() as conn:
        for t in tables:
            try:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar() or 0
            except Exception:
                count = "—"
            stats.append({"Table": t, "Rows": count})
    return pd.DataFrame(stats)


@st.cache_data(ttl=30)
def _last_refresh():
    try:
        with engine.connect() as conn:
            ts = conn.execute(
                text("SELECT MAX(labeled_at) FROM churn_labels")
            ).scalar()
        return str(ts)[:16] if ts else "Never"
    except Exception:
        return "Unknown"


# ── Stats panel ───────────────────────────────────────────────────────────────
st.markdown('<span class="section-title">📋 Database Statistics</span>', unsafe_allow_html=True)

col_stats, col_info = st.columns([1, 1])

with col_stats:
    df_stats = _db_stats()
    st.dataframe(df_stats, use_container_width=True, hide_index=True,
                 column_config={
                     "Rows": st.column_config.NumberColumn("Rows", format="%d"),
                 })

with col_info:
    last_refresh = _last_refresh()
    st.markdown(f"""
    <div class="info-card">
        <h4>📅 Last Data Refresh</h4>
        <p><strong style="color:#f1f5f9">{last_refresh}</strong></p>
    </div>
    <br>
    <div class="info-card">
        <h4>💾 Database</h4>
        <p>SQLite (local dev).<br>
           Swap to PostgreSQL by setting<br>
           <code>DATABASE_URL</code> in <code>.env</code>.</p>
    </div>
    <br>
    <div class="info-card">
        <h4>📁 Storage</h4>
        <p>Database: <code>data/churn.db</code><br>
           Models: <code>models/*.joblib</code></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Dataset Regeneration ──────────────────────────────────────────────────────
st.markdown('<span class="section-title">🔄 Regenerate Synthetic Dataset</span>', unsafe_allow_html=True)

st.warning(
    "⚠️ **This will delete ALL existing customer data, predictions, and churn labels.** "
    "Model versions and artifacts are preserved. You will need to retrain models after regeneration.",
    icon="⚠️",
)

with st.form("regen_form"):
    col_n, col_s = st.columns(2)
    with col_n:
        n_customers = st.slider(
            "Number of customers to generate",
            min_value=1000, max_value=20000, value=8000, step=500,
        )
    with col_s:
        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

    st.markdown("**What will be generated:**")
    st.markdown(f"""
    - 👥 **{n_customers:,}** synthetic customers across 5 K-Means archetypes
    - 📅 Subscriptions, usage logs (up to 12 months per customer)
    - 🏷️ Churn labels via weighted logistic function
    - 🔮 K-Means clustering → customer segments table
    """)

    col_go, col_cancel = st.columns([1, 3])
    with col_go:
        submitted = st.form_submit_button(
            "🚀 Regenerate Dataset", use_container_width=True, type="primary"
        )

if submitted:
    progress = st.progress(0, "Clearing existing data …")
    try:
        with st.spinner(f"Generating {n_customers:,} customers (seed={seed}) …"):
            progress.progress(20, "Sampling archetypes …")
            db = SessionLocal()
            try:
                counts = generate_dataset(n_customers=n_customers, seed=int(seed), db=db)
            finally:
                db.close()
            progress.progress(100, "Done!")

        st.success(
            f"✅ Dataset regenerated successfully!\n\n"
            f"- **{counts['customers']:,}** customers\n"
            f"- **{counts['subscriptions']:,}** subscriptions\n"
            f"- **{counts['usage_logs']:,}** usage log entries\n"
            f"- **{counts['churn_rate']:.1%}** overall churn rate\n"
            f"- **{counts['segments']}** customer segments"
        )
        st.cache_data.clear()
        st.info(
            "💡 **Next step:** Go to **Model Registry** and click "
            "**'Train New Model Version'** to retrain on the new data."
        )

    except Exception as e:
        st.error(f"Regeneration failed: {e}")
        raise

st.markdown("---")

# ── Model artifact cleanup ────────────────────────────────────────────────────
st.markdown('<span class="section-title">🗑️ Model Artifact Info</span>', unsafe_allow_html=True)

from app.config import MODELS_DIR

artifacts = list(MODELS_DIR.glob("*.joblib"))
meta_files = list(MODELS_DIR.glob("*_metadata.json"))

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.metric("Model artifacts (.joblib)", len(artifacts))
with col_m2:
    st.metric("Metadata files (.json)", len(meta_files))

if artifacts:
    with st.expander("📁 Show artifact files"):
        for f in sorted(artifacts):
            size_kb = f.stat().st_size / 1024
            st.markdown(f"- `{f.name}` — **{size_kb:.1f} KB**")
