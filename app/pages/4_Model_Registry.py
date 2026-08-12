"""
4_Model_Registry.py — View, compare, promote, and rollback trained model versions.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.db import SessionLocal
from app.core.model_registry import list_versions, promote, rollback_to
from app.core.model_training import train_and_register
from app.core.utils import GLOBAL_CSS

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
st.title("🗂️ Model Registry")
st.caption("Compare model versions, promote to production, and retrain on current data")

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Inter"),
    margin=dict(l=20, r=20, t=40, b=20),
)


# ── Load registry ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def _load_registry():
    models = list_versions()
    if not models:
        return pd.DataFrame()
    rows = []
    for m in models:
        rows.append({
            "version":       m.version,
            "algorithm":     m.algorithm.replace("_", " ").title(),
            "trained_at":    str(m.trained_at)[:16] if m.trained_at else "—",
            "AUC":           round(m.auc, 4) if m.auc else None,
            "F1":            round(m.f1, 4) if m.f1 else None,
            "Precision":     round(m.precision, 4) if m.precision else None,
            "Recall":        round(m.recall, 4) if m.recall else None,
            "is_production": m.is_production,
            "model_id":      m.model_id,
        })
    return pd.DataFrame(rows)


def _refresh():
    st.cache_data.clear()
    st.rerun()


df_registry = _load_registry()

if df_registry.empty:
    st.warning("No models trained yet. Run `python seed.py` or use the button below.")
else:
    # ── Current production badge ──────────────────────────────────────────────
    prod_rows = df_registry[df_registry["is_production"]]
    if not prod_rows.empty:
        prod = prod_rows.iloc[0]
        st.markdown(f"""
        <div class="info-card">
            <h4>🏆 Production Model</h4>
            <p>
                <strong style="color:#f1f5f9">{prod['version']}</strong>
                &nbsp;·&nbsp; {prod['algorithm']}
                &nbsp;·&nbsp; AUC: <strong style="color:#6366f1">{prod['AUC']}</strong>
                &nbsp;·&nbsp; F1: <strong style="color:#14b8a6">{prod['F1']}</strong>
                &nbsp;·&nbsp; Trained: {prod['trained_at']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<span class="section-title">All Model Versions</span>', unsafe_allow_html=True)

    # ── Registry table ────────────────────────────────────────────────────────
    display_df = df_registry.copy()
    display_df["Status"] = display_df["is_production"].apply(
        lambda x: "🏆 Production" if x else "—"
    )
    show_cols = ["version", "algorithm", "AUC", "F1", "Precision", "Recall", "trained_at", "Status"]
    st.dataframe(
        display_df[show_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "AUC":       st.column_config.ProgressColumn("AUC", min_value=0, max_value=1, format="%.4f"),
            "F1":        st.column_config.ProgressColumn("F1",  min_value=0, max_value=1, format="%.4f"),
            "Precision": st.column_config.ProgressColumn("Precision", min_value=0, max_value=1, format="%.4f"),
            "Recall":    st.column_config.ProgressColumn("Recall",    min_value=0, max_value=1, format="%.4f"),
        },
    )

    # ── Metrics comparison chart ──────────────────────────────────────────────
    st.markdown('<span class="section-title">Metrics Comparison</span>', unsafe_allow_html=True)
    fig = go.Figure()
    metrics = ["AUC", "F1", "Precision", "Recall"]
    colors  = ["#6366f1", "#14b8a6", "#f59e0b", "#ef4444"]
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric,
            x=display_df["version"],
            y=display_df[metric],
            marker_color=color,
        ))
    fig.update_layout(
        **PLOT_LAYOUT, barmode="group", height=360,
        xaxis_title="", yaxis_title="Score",
        legend=dict(orientation="h", y=1.1),
    )
    fig.update_yaxes(range=[0, 1], gridcolor="#1e293b", color="#94a3b8")
    fig.update_xaxes(color="#94a3b8")
    st.plotly_chart(fig, use_container_width=True)

    # ── Promote / Rollback controls ───────────────────────────────────────────
    st.markdown('<span class="section-title">Promote / Rollback</span>', unsafe_allow_html=True)
    non_prod = df_registry[~df_registry["is_production"]]
    if non_prod.empty:
        st.info("Only one model version exists — nothing to promote/rollback to.")
    else:
        col_sel, col_btn = st.columns([2, 1])
        with col_sel:
            selected = st.selectbox(
                "Select a version to promote as production",
                non_prod["version"].tolist(),
                key="promote_select",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Promote to Production", use_container_width=True):
                try:
                    promote(selected)
                    st.success(f"✅ `{selected}` is now the production model.")
                    _refresh()
                except Exception as e:
                    st.error(f"Promotion failed: {e}")

        if len(df_registry) >= 2:
            with col_btn:
                if st.button("⏪ Rollback to Selected", use_container_width=True, type="secondary"):
                    try:
                        rollback_to(selected)
                        st.success(f"✅ Rolled back to `{selected}`.")
                        _refresh()
                    except Exception as e:
                        st.error(f"Rollback failed: {e}")

# ── Train new version ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<span class="section-title">Train New Version</span>', unsafe_allow_html=True)
st.markdown("Retrain both classifiers on the current database. "
            "The best model (by AUC) will be promoted automatically.")

with st.expander("⚙️ Training Notes"):
    st.markdown("""
    - Trains **Logistic Regression** (baseline) + **Random Forest** (primary)
    - Uses **stratified 80/20 split** + **5-fold cross-validation**
    - Selects best model by **AUC-ROC**
    - Computes **SHAP** global importances (Random Forest)
    - Updates **K-Means** customer segmentation
    - Saves versioned `.joblib` artifact to `models/`
    """)

if st.button("🔧 Train New Model Version", use_container_width=True):
    progress = st.progress(0, "Starting training …")
    try:
        with st.spinner("Training in progress (this may take 1–3 minutes) …"):
            progress.progress(10, "Loading data …")
            result = train_and_register()
            progress.progress(100, "Complete!")

        st.success(
            f"✅ Training complete!  \n"
            f"New version: **{result['production']}**  |  "
            f"Best AUC: **{result['best_auc']:.4f}**  |  "
            f"Samples: **{result['n_samples']:,}**"
        )
        _refresh()
    except Exception as e:
        st.error(f"Training failed: {e}")
