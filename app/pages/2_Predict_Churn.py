"""
2_Predict_Churn.py — Single and batch churn prediction UI.
"""
from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.db import ChurnPrediction, Customer, SessionLocal, engine
from app.core.feature_engineering import build_features
from app.core.model_registry import get_production_model
from app.core.model_training import compute_shap_for_row
from app.core.utils import (
    GLOBAL_CSS, fmt_currency, fmt_pct, format_shap_factors, get_risk_tier,
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
st.title("🎯 Predict Churn")
st.caption("Run churn predictions powered by your production ML model")


# ── Load production model (cached for session) ────────────────────────────────
@st.cache_resource
def _load_model():
    return get_production_model()


try:
    artifact, registry_entry = _load_model()
except Exception as e:
    st.error(f"⚠️ Could not load production model: {e}\n\nRun `python seed.py` first.")
    st.stop()

model          = artifact["model"]
feature_names  = artifact.get("feature_names", [])
background_X   = artifact.get("background_X")

st.sidebar.success(
    f"**Active Model**  \n"
    f"`{registry_entry.version}`  \n"
    f"AUC: **{registry_entry.auc:.4f}**  \n"
    f"Algo: {registry_entry.algorithm.replace('_', ' ').title()}"
)


# ── Gauge chart ───────────────────────────────────────────────────────────────
def _gauge(prob: float) -> go.Figure:
    tier, color, _ = get_risk_tier(prob)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 36, "color": "#f1f5f9"}},
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#94a3b8"),
            bar=dict(color=color),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#334155",
            steps=[
                dict(range=[0, 30],  color="rgba(34,197,94,0.15)"),
                dict(range=[30, 60], color="rgba(245,158,11,0.15)"),
                dict(range=[60, 100],color="rgba(239,68,68,0.15)"),
            ],
            threshold=dict(line=dict(color=color, width=4), thickness=0.8, value=prob*100),
        ),
        title={"text": f"Churn Probability — {tier} Risk", "font": {"color": "#94a3b8", "size": 14}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9"),
        height=280,
        margin=dict(l=20, r=20, t=20, b=10),
    )
    return fig


# ── SHAP bar chart ────────────────────────────────────────────────────────────
def _shap_bar(shap_dict: dict) -> go.Figure:
    if not shap_dict:
        return None
    top = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    names  = [k.replace("_", " ").title() for k, _ in top]
    values = [v for _, v in top]
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter"),
        height=300, margin=dict(l=10, r=40, t=10, b=10),
        xaxis_title="SHAP Value (red = pushes churn, green = reduces churn)",
        xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
        yaxis=dict(color="#f1f5f9"),
    )
    return fig


# ── Single prediction ─────────────────────────────────────────────────────────
def _predict_single(row_dict: dict, customer_id: int | None = None) -> dict:
    """Run prediction on a dict of raw feature inputs."""
    df_in = pd.DataFrame([row_dict])
    X, _ = build_features(df_in)

    # Align columns to training feature set
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    prob   = float(model.predict_proba(X)[0, 1])
    label  = prob >= 0.5
    tier, color, emoji = get_risk_tier(prob)

    # SHAP
    shap_dict = {}
    if background_X is not None:
        try:
            shap_dict = compute_shap_for_row(artifact, X)
        except Exception:
            pass

    top_factors = format_shap_factors(shap_dict, top_n=3)

    # Persist prediction
    db = SessionLocal()
    try:
        pred_row = ChurnPrediction(
            customer_id=customer_id or 0,
            model_version=registry_entry.version,
            churn_probability=prob,
            predicted_label=label,
            predicted_at=datetime.utcnow(),
            top_factors=dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]),
        )
        db.add(pred_row)
        db.commit()
    finally:
        db.close()

    return {
        "probability": prob, "label": label,
        "tier": tier, "color": color, "emoji": emoji,
        "shap_dict": shap_dict, "top_factors": top_factors,
    }


# ── TABS ──────────────────────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["🧑 Single Customer", "📂 Batch Upload"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ═══════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.markdown("### Enter Customer Details")

    with st.form("single_pred_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📋 Demographics**")
            age            = st.slider("Age", 18, 80, 35)
            gender         = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
            region         = st.selectbox("Region", ["North", "South", "East", "West", "Central"])

        with col2:
            st.markdown("**📄 Contract & Billing**")
            contract_type  = st.selectbox("Contract Type", ["month-to-month", "1yr", "2yr"])
            payment_method = st.selectbox("Payment Method",
                                          ["Credit Card", "Bank Transfer",
                                           "Electronic Check", "Mailed Check"])
            monthly_charges= st.slider("Monthly Charges ($)", 20.0, 130.0, 65.0, 0.5)
            tenure_months  = st.slider("Tenure (months)", 1, 72, 12)

        with col3:
            st.markdown("**📊 Usage & Behaviour**")
            avg_usage        = st.slider("Avg Usage Hours / month", 0.0, 150.0, 50.0, 1.0)
            avg_tickets      = st.slider("Avg Support Tickets / month", 0.0, 10.0, 1.0, 0.5)
            avg_late         = st.slider("Avg Late Payments / month", 0.0, 8.0, 0.5, 0.5)
            avg_adoption     = st.slider("Feature Adoption Score", 0.0, 1.0, 0.5, 0.05)
            cluster_id       = st.selectbox("Customer Cluster (if known)", [-1, 0, 1, 2, 3, 4],
                                            format_func=lambda x: "Auto-detect" if x == -1 else f"Cluster {x}")

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        row = {
            "age": age, "gender": gender, "region": region,
            "contract_type": contract_type, "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": monthly_charges * tenure_months,
            "tenure_months": tenure_months,
            "avg_usage_hours": avg_usage,
            "avg_support_tickets": avg_tickets,
            "avg_late_payments": avg_late,
            "avg_feature_adoption": avg_adoption,
            "cluster_id": cluster_id if cluster_id >= 0 else -1,
        }

        with st.spinner("Running prediction …"):
            result = _predict_single(row)

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        r1, r2, r3 = st.columns([1.2, 1, 1])

        with r1:
            st.plotly_chart(_gauge(result["probability"]), use_container_width=True)

        with r2:
            tier_class = f"risk-{result['tier'].lower()}"
            st.markdown(f"""
            <div class="info-card" style="height:180px;">
                <h4>Risk Assessment</h4>
                <p>Churn probability: <strong style="color:#f1f5f9">{result['probability']:.1%}</strong></p>
                <p>Verdict: <span class="{tier_class}">{result['emoji']} {result['tier']} Risk</span></p>
                <p style="margin-top:12px; color:#64748b; font-size:.8rem;">
                    Threshold: Low &lt;30% · Medium 30–60% · High ≥60%
                </p>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown('<div class="info-card" style="height:180px;">', unsafe_allow_html=True)
            st.markdown('<h4 style="color:#6366f1; margin:0 0 8px;">Top Churn Drivers</h4>',
                        unsafe_allow_html=True)
            for f in result["top_factors"]:
                arrow = "🔴" if f["value"] > 0 else "🟢"
                st.markdown(
                    f"<p style='margin:4px 0; color:#cbd5e1; font-size:.85rem;'>"
                    f"{arrow} {f['description']}</p>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # SHAP bar chart
        if result["shap_dict"]:
            st.markdown("#### Feature Contribution (SHAP Values)")
            fig_shap = _shap_bar(result["shap_dict"])
            if fig_shap:
                st.plotly_chart(fig_shap, use_container_width=True)
                st.caption("🔴 Red bars push toward churn · 🟢 Green bars reduce churn risk")

        st.success("✅ Prediction logged to database.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ═══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### Upload a CSV for Batch Scoring")
    st.markdown("""
    **Required columns:** `tenure_months`, `monthly_charges`, `contract_type`,
    `payment_method`, `avg_usage_hours`, `avg_support_tickets`,
    `avg_late_payments`, `avg_feature_adoption`
    """)

    with st.expander("📥 Download sample CSV template"):
        sample = pd.DataFrame([{
            "tenure_months": 12, "monthly_charges": 70.5,
            "total_charges": 846.0, "contract_type": "month-to-month",
            "payment_method": "Credit Card", "age": 34,
            "gender": "Female", "region": "North",
            "avg_usage_hours": 45.0, "avg_support_tickets": 2.5,
            "avg_late_payments": 1.0, "avg_feature_adoption": 0.4,
            "cluster_id": -1,
        }])
        csv_sample = sample.to_csv(index=False)
        st.download_button("⬇️ Download template", csv_sample, "churn_template.csv", "text/csv")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            df_batch = pd.read_csv(uploaded)
            st.info(f"📄 Loaded **{len(df_batch):,}** rows")

            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                with st.spinner(f"Scoring {len(df_batch):,} customers …"):
                    # Fill defaults for optional columns
                    for col, val in [
                        ("total_charges", None), ("age", 35), ("gender", "Unknown"),
                        ("region", "Unknown"), ("cluster_id", -1),
                    ]:
                        if col not in df_batch.columns:
                            df_batch[col] = val
                    if df_batch["total_charges"].isnull().any():
                        df_batch["total_charges"] = (
                            df_batch["monthly_charges"] * df_batch["tenure_months"]
                        )

                    X_batch, _ = build_features(df_batch)
                    for col in feature_names:
                        if col not in X_batch.columns:
                            X_batch[col] = 0
                    X_batch = X_batch[feature_names]

                    probs  = model.predict_proba(X_batch)[:, 1]
                    labels = (probs >= 0.5).astype(bool)
                    tiers  = [get_risk_tier(p)[0] for p in probs]

                    df_batch["churn_probability"] = probs.round(4)
                    df_batch["predicted_label"]   = labels
                    df_batch["risk_tier"]         = tiers

                    # Persist predictions
                    db = SessionLocal()
                    try:
                        pred_rows = [
                            ChurnPrediction(
                                customer_id=int(row.get("customer_id", 0)) if "customer_id" in df_batch.columns else 0,
                                model_version=registry_entry.version,
                                churn_probability=float(probs[i]),
                                predicted_label=bool(labels[i]),
                                predicted_at=datetime.utcnow(),
                            )
                            for i, (_, row) in enumerate(df_batch.iterrows())
                        ]
                        db.add_all(pred_rows)
                        db.commit()
                    finally:
                        db.close()

                st.success(f"✅ Scored {len(df_batch):,} customers. All predictions logged.")

                # Results table
                st.markdown("#### Scored Results")
                show_cols = (
                    ["customer_id"] if "customer_id" in df_batch.columns else []
                ) + ["churn_probability", "predicted_label", "risk_tier"]
                st.dataframe(
                    df_batch[show_cols].style.background_gradient(
                        subset=["churn_probability"],
                        cmap="RdYlGn_r",
                    ),
                    use_container_width=True, height=400,
                )

                # Summary
                high_risk = (df_batch["risk_tier"] == "High").sum()
                med_risk  = (df_batch["risk_tier"] == "Medium").sum()
                m1, m2, m3 = st.columns(3)
                m1.metric("🔴 High Risk",   high_risk)
                m2.metric("🟡 Medium Risk", med_risk)
                m3.metric("🟢 Low Risk",    len(df_batch) - high_risk - med_risk)

                # Download
                csv_out = df_batch.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Results CSV",
                    csv_out, "churn_predictions.csv", "text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")
