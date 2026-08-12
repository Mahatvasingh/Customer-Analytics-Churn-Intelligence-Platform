"""
1_Dashboard.py — Executive Dashboard with KPIs, charts, and cluster analysis.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

from app.config import APP_TITLE, MODELS_DIR
from app.core.db import engine
from app.core.utils import GLOBAL_CSS, fmt_currency, fmt_number, fmt_pct

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
st.title("📊 Dashboard")
st.caption("Real-time customer churn analytics powered by your live database")

# ── Plotly dark theme ─────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Inter"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)
COLORS = ["#6366f1", "#14b8a6", "#f59e0b", "#ef4444", "#22c55e"]


# ── Data loaders (cached) ─────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_kpis():
    with engine.connect() as conn:
        total     = conn.execute(text("SELECT COUNT(*) FROM customers")).scalar() or 0
        churn_r   = conn.execute(text("SELECT AVG(CAST(churned AS REAL)) FROM churn_labels")).scalar() or 0
        avg_ch    = conn.execute(text("SELECT AVG(monthly_charges) FROM customers")).scalar() or 0
        avg_ten   = conn.execute(text("SELECT AVG(tenure_months) FROM customers")).scalar() or 0
        active    = conn.execute(text("SELECT COUNT(*) FROM customers WHERE is_active=1")).scalar() or 0
    return dict(total=int(total), churn_rate=float(churn_r),
                avg_charges=float(avg_ch), avg_tenure=float(avg_ten),
                active=int(active))


@st.cache_data(ttl=300)
def load_churn_by_contract():
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT c.contract_type,
                   COUNT(*) AS total,
                   SUM(CAST(cl.churned AS INTEGER)) AS churned
            FROM customers c
            JOIN churn_labels cl ON c.customer_id=cl.customer_id
            GROUP BY c.contract_type
        """)).fetchall()
    return pd.DataFrame(rows, columns=["contract_type", "total", "churned"])


@st.cache_data(ttl=300)
def load_churn_by_tenure():
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                CASE
                    WHEN tenure_months <= 12  THEN '0–12 mo'
                    WHEN tenure_months <= 24  THEN '13–24 mo'
                    WHEN tenure_months <= 48  THEN '25–48 mo'
                    ELSE '48+ mo'
                END AS bucket,
                COUNT(*) AS total,
                SUM(CAST(cl.churned AS INTEGER)) AS churned
            FROM customers c
            JOIN churn_labels cl ON c.customer_id=cl.customer_id
            GROUP BY bucket
        """)).fetchall()
    df = pd.DataFrame(rows, columns=["bucket", "total", "churned"])
    order = ["0–12 mo", "13–24 mo", "25–48 mo", "48+ mo"]
    df["bucket"] = pd.Categorical(df["bucket"], categories=order, ordered=True)
    return df.sort_values("bucket")


@st.cache_data(ttl=300)
def load_monthly_usage():
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT month, AVG(usage_hours) AS avg_hours
            FROM usage_logs
            GROUP BY month
            ORDER BY month DESC
            LIMIT 12
        """)).fetchall()
    return pd.DataFrame(rows, columns=["month", "avg_hours"]).sort_values("month")


@st.cache_data(ttl=300)
def load_segments():
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT cluster_id, label, avg_churn_rate, size, profile_json FROM customer_segments"
        )).fetchall()
    if not rows:
        return pd.DataFrame()
    records = []
    for r in rows:
        profile = json.loads(r[4]) if isinstance(r[4], str) else (r[4] or {})
        records.append({
            "Cluster": r[0], "Label": r[1],
            "Churn Rate": r[2], "Size": r[3],
            **{k.replace("_", " ").title(): v for k, v in profile.items()},
        })
    return pd.DataFrame(records)


@st.cache_data(ttl=300)
def load_shap_importances():
    files = sorted(MODELS_DIR.glob("shap_importances_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return {}
    return json.loads(files[0].read_text())


@st.cache_data(ttl=300)
def load_region_churn():
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT c.region,
                   COUNT(*) AS total,
                   SUM(CAST(cl.churned AS INTEGER)) AS churned
            FROM customers c
            JOIN churn_labels cl ON c.customer_id=cl.customer_id
            GROUP BY c.region
        """)).fetchall()
    return pd.DataFrame(rows, columns=["region", "total", "churned"])


@st.cache_data(ttl=300)
def load_cluster_scatter():
    """Pull a sample for PCA scatter plot."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT c.monthly_charges, c.tenure_months,
                   u.avg_usage, u.avg_tickets, c.cluster_id,
                   cl.churned
            FROM customers c
            JOIN churn_labels cl ON c.customer_id=cl.customer_id
            LEFT JOIN (
                SELECT customer_id,
                       AVG(usage_hours) AS avg_usage,
                       AVG(support_tickets) AS avg_tickets
                FROM usage_logs GROUP BY customer_id
            ) u ON c.customer_id=u.customer_id
            ORDER BY RANDOM()
            LIMIT 2000
        """)).fetchall()
    return pd.DataFrame(rows, columns=["monthly_charges", "tenure_months",
                                        "avg_usage", "avg_tickets",
                                        "cluster_id", "churned"])


# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    try:
        regions_all = ["All"] + sorted([
            r[0] for r in engine.connect().execute(text("SELECT DISTINCT region FROM customers")).fetchall()
        ])
    except Exception:
        regions_all = ["All"]
    sel_region = st.selectbox("Region", regions_all)

    contracts_all = ["All", "month-to-month", "1yr", "2yr"]
    sel_contract  = st.selectbox("Contract Type", contracts_all)

    st.caption("Filters affect charts below (not KPI cards).")


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if sel_region != "All" and "region" in df.columns:
        df = df[df["region"] == sel_region]
    if sel_contract != "All" and "contract_type" in df.columns:
        df = df[df["contract_type"] == sel_contract]
    return df


# ── Check data ────────────────────────────────────────────────────────────────
try:
    kpis = load_kpis()
    if kpis["total"] == 0:
        st.warning("⚠️ No data found. Run `python seed.py` first, then refresh.")
        st.stop()
except Exception as e:
    st.error(f"Database error: {e}\n\nMake sure you've run `python seed.py`.")
    st.stop()

# ── KPI Row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
kpi_items = [
    (c1, "👥", "Total Customers",  fmt_number(kpis["total"]),       f"{kpis['active']:,} active"),
    (c2, "📉", "Churn Rate",       fmt_pct(kpis["churn_rate"]),     "across all customers"),
    (c3, "💰", "Avg Monthly Charge", fmt_currency(kpis["avg_charges"]), "per customer"),
    (c4, "📅", "Avg Tenure",       f"{kpis['avg_tenure']:.1f} mo",  "customer lifetime"),
]
for col, icon, title, value, delta in kpi_items:
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 1: Churn by Contract + Churn by Tenure ────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<span class="section-title">Churn Rate by Contract Type</span>', unsafe_allow_html=True)
    df_contract = load_churn_by_contract()
    if not df_contract.empty:
        df_contract["churn_rate"] = df_contract["churned"] / df_contract["total"]
        fig = go.Figure(go.Bar(
            x=df_contract["contract_type"],
            y=df_contract["churn_rate"] * 100,
            marker_color=COLORS[:3],
            text=[f"{r:.1%}" for r in df_contract["churn_rate"]],
            textposition="outside",
        ))
        fig.update_layout(
            **PLOT_LAYOUT,
            yaxis_title="Churn Rate (%)",
            xaxis_title="",
            height=300,
        )
        fig.update_yaxes(gridcolor="#1e293b", color="#94a3b8")
        fig.update_xaxes(color="#94a3b8")
        st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown('<span class="section-title">Churn Rate by Tenure</span>', unsafe_allow_html=True)
    df_tenure = load_churn_by_tenure()
    if not df_tenure.empty:
        df_tenure["churn_rate"] = df_tenure["churned"] / df_tenure["total"]
        fig2 = go.Figure(go.Bar(
            x=df_tenure["bucket"].astype(str),
            y=df_tenure["churn_rate"] * 100,
            marker=dict(
                color=df_tenure["churn_rate"] * 100,
                colorscale=[[0, "#22c55e"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                showscale=True,
                colorbar=dict(title="Churn %", tickfont=dict(color="#94a3b8")),
            ),
            text=[f"{r:.1%}" for r in df_tenure["churn_rate"]],
            textposition="outside",
        ))
        fig2.update_layout(
            **PLOT_LAYOUT, yaxis_title="Churn Rate (%)", xaxis_title="", height=300
        )
        fig2.update_yaxes(gridcolor="#1e293b", color="#94a3b8")
        fig2.update_xaxes(color="#94a3b8")
        st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Monthly Usage Trend + Region Churn ─────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.markdown('<span class="section-title">Monthly Usage Trend</span>', unsafe_allow_html=True)
    df_usage = load_monthly_usage()
    if not df_usage.empty:
        fig3 = go.Figure(go.Scatter(
            x=df_usage["month"], y=df_usage["avg_hours"],
            mode="lines+markers",
            line=dict(color="#6366f1", width=2.5),
            marker=dict(color="#14b8a6", size=7),
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.1)",
        ))
        fig3.update_layout(
            **PLOT_LAYOUT, yaxis_title="Avg Usage Hours", xaxis_title="", height=300
        )
        fig3.update_yaxes(gridcolor="#1e293b", color="#94a3b8")
        fig3.update_xaxes(color="#94a3b8", tickangle=-30)
        st.plotly_chart(fig3, use_container_width=True)

with col_d:
    st.markdown('<span class="section-title">Churn Rate by Region</span>', unsafe_allow_html=True)
    df_region = load_region_churn()
    if not df_region.empty:
        df_region["churn_rate"] = df_region["churned"] / df_region["total"]
        fig4 = px.bar(
            df_region, x="region", y="churn_rate",
            color="churn_rate",
            color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
            labels={"churn_rate": "Churn Rate", "region": ""},
            height=300,
        )
        fig4.update_traces(texttemplate="%{y:.1%}", textposition="outside")
        fig4.update_layout(**PLOT_LAYOUT)
        fig4.update_yaxes(tickformat=".0%", gridcolor="#1e293b", color="#94a3b8")
        fig4.update_xaxes(color="#94a3b8")
        st.plotly_chart(fig4, use_container_width=True)

# ── Row 3: SHAP Importances ───────────────────────────────────────────────────
st.markdown('<span class="section-title">🧠 Global Feature Importances (SHAP)</span>', unsafe_allow_html=True)
shap_data = load_shap_importances()
if shap_data:
    shap_df = (
        pd.DataFrame(list(shap_data.items()), columns=["feature", "importance"])
        .sort_values("importance", ascending=False)
        .head(15)
    )
    shap_df["feature"] = shap_df["feature"].str.replace("_", " ").str.title()
    fig_shap = go.Figure(go.Bar(
        x=shap_df["importance"],
        y=shap_df["feature"],
        orientation="h",
        marker=dict(
            color=shap_df["importance"],
            colorscale=[[0, "#14b8a6"], [1, "#6366f1"]],
        ),
        text=[f"{v:.4f}" for v in shap_df["importance"]],
        textposition="outside",
    ))
    fig_shap.update_layout(
        **PLOT_LAYOUT, height=420,
        xaxis_title="Mean |SHAP| Value", yaxis_title="",
    )
    fig_shap.update_xaxes(gridcolor="#1e293b", color="#94a3b8")
    fig_shap.update_yaxes(color="#f1f5f9")
    st.plotly_chart(fig_shap, use_container_width=True)
else:
    st.info("SHAP importances will appear here after training a Random Forest model.")

# ── Row 4: Customer Segment Analysis ─────────────────────────────────────────
st.markdown('<span class="section-title">🔮 Customer Segments (K-Means)</span>', unsafe_allow_html=True)

seg_df = load_segments()
if not seg_df.empty:
    col_e, col_f = st.columns([1.2, 1])

    with col_e:
        # Bubble chart: size=cluster size, color=churn rate
        fig_seg = px.scatter(
            seg_df,
            x="Avg Monthly Charges" if "Avg Monthly Charges" in seg_df.columns else seg_df.columns[4],
            y="Avg Tenure Months"   if "Avg Tenure Months"   in seg_df.columns else seg_df.columns[5],
            size="Size",
            color="Churn Rate",
            color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
            hover_name="Label",
            text="Cluster",
            size_max=60,
            labels={"Avg Monthly Charges": "Avg Monthly Charges ($)",
                    "Avg Tenure Months": "Avg Tenure (months)"},
            height=380,
        )
        fig_seg.update_traces(textfont=dict(color="white", size=12))
        fig_seg.update_layout(**PLOT_LAYOUT)
        fig_seg.update_xaxes(gridcolor="#1e293b", color="#94a3b8")
        fig_seg.update_yaxes(gridcolor="#1e293b", color="#94a3b8")
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_f:
        st.markdown("**Cluster Profiles**")
        display_cols = ["Cluster", "Label", "Size", "Churn Rate"]
        existing = [c for c in display_cols if c in seg_df.columns]
        disp = seg_df[existing].copy()
        if "Churn Rate" in disp.columns:
            disp["Churn Rate"] = disp["Churn Rate"].apply(lambda x: f"{x:.1%}")
        st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
            height=380,
        )
else:
    st.info("Segment data will appear here after running seed.py.")

# ── Refresh hint ──────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("💡 Charts are cached for 5 minutes. Click **Rerun** (⋮ menu) to force a refresh.")
