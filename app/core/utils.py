"""
utils.py — Shared helper functions used across pages and core modules.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from app.config import RISK_LOW_THRESHOLD, RISK_HIGH_THRESHOLD

# ── Risk tier helpers ─────────────────────────────────────────────────────────

def get_risk_tier(probability: float) -> Tuple[str, str, str]:
    """
    Map a churn probability to a risk tier.
    Returns (tier_name, hex_color, emoji).
    """
    if probability < RISK_LOW_THRESHOLD:
        return "Low", "#22c55e", "🟢"
    elif probability < RISK_HIGH_THRESHOLD:
        return "Medium", "#f59e0b", "🟡"
    else:
        return "High", "#ef4444", "🔴"


def risk_badge_html(probability: float) -> str:
    """Return an HTML badge string for the risk tier."""
    tier, color, emoji = get_risk_tier(probability)
    css_class = f"risk-{tier.lower()}"
    return f'<span class="{css_class}">{emoji} {tier} Risk</span>'


# ── SHAP formatting ───────────────────────────────────────────────────────────

_READABLE_NAMES: Dict[str, str] = {
    "tenure_months":            "Customer Tenure",
    "monthly_charges":          "Monthly Charges",
    "total_charges":            "Total Charges",
    "charges_per_tenure":       "Charges-per-Tenure Ratio",
    "avg_usage_hours":          "Avg Monthly Usage (hrs)",
    "avg_support_tickets":      "Avg Support Tickets",
    "avg_late_payments":        "Avg Late Payments",
    "avg_feature_adoption":     "Feature Adoption Score",
    "support_ticket_rate":      "Support Ticket Rate",
    "late_payment_ratio":       "Late Payment Ratio",
    "contract_month_to_month":  "Month-to-Month Contract",
    "contract_1yr":             "1-Year Contract",
    "contract_2yr":             "2-Year Contract",
    "payment_credit_card":      "Payment: Credit Card",
    "payment_bank_transfer":    "Payment: Bank Transfer",
    "payment_electronic_check": "Payment: Electronic Check",
    "payment_mailed_check":     "Payment: Mailed Check",
}


def readable_feature_name(feature: str) -> str:
    """Convert raw feature name to human-readable label."""
    return _READABLE_NAMES.get(feature, feature.replace("_", " ").title())


def format_shap_factors(
    shap_dict: Dict[str, float],
    top_n: int = 3,
) -> List[Dict]:
    """
    Format top-N SHAP factors into plain-English dicts.
    Returns list of {"feature", "value", "readable", "direction", "description"}.
    """
    if not shap_dict:
        return []

    sorted_factors = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

    result = []
    for feature, value in sorted_factors:
        direction = "increases" if value > 0 else "reduces"
        magnitude = "strongly" if abs(value) > 0.05 else "slightly"
        readable = readable_feature_name(feature)
        result.append({
            "feature":     feature,
            "readable":    readable,
            "value":       value,
            "direction":   direction,
            "description": f"{readable} {magnitude} {direction} churn risk",
        })
    return result


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_currency(value: float) -> str:
    return f"${value:,.2f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}%}"


def fmt_number(value: float) -> str:
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:,.0f}"


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den else default


# ── Shared Streamlit CSS ──────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── KPI Cards ─────────────────────────────────────────────── */
.kpi-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    transition: transform .2s ease, box-shadow .2s ease;
    height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 36px rgba(99,102,241,.25);
    border-color: #6366f1;
}
.kpi-icon  { font-size: 1.8rem; margin-bottom: 6px; }
.kpi-title { color: #94a3b8; font-size: .75rem; font-weight: 600;
             text-transform: uppercase; letter-spacing: .12em; }
.kpi-value { color: #f1f5f9; font-size: 2rem; font-weight: 800; margin: 6px 0 4px; }
.kpi-delta { color: #22c55e; font-size: .8rem; font-weight: 500; }
.kpi-delta.negative { color: #ef4444; }

/* ── Section headers ───────────────────────────────────────── */
.section-title {
    font-size: 1.25rem; font-weight: 700;
    background: linear-gradient(90deg, #6366f1, #14b8a6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 28px 0 12px;
    display: block;
}

/* ── Risk badges ───────────────────────────────────────────── */
.risk-high   { background:rgba(239,68,68,.15);  color:#ef4444;
               border:1px solid rgba(239,68,68,.35);
               border-radius:8px; padding:4px 14px; font-weight:600; display:inline-block; }
.risk-medium { background:rgba(245,158,11,.15); color:#f59e0b;
               border:1px solid rgba(245,158,11,.35);
               border-radius:8px; padding:4px 14px; font-weight:600; display:inline-block; }
.risk-low    { background:rgba(34,197,94,.15);  color:#22c55e;
               border:1px solid rgba(34,197,94,.35);
               border-radius:8px; padding:4px 14px; font-weight:600; display:inline-block; }

/* ── Info cards ────────────────────────────────────────────── */
.info-card {
    background: #1e293b; border: 1px solid #334155; border-radius: 12px;
    padding: 16px 20px; margin: 8px 0;
}
.info-card h4 { color: #6366f1; margin: 0 0 8px; font-size: .95rem; }
.info-card p  { color: #cbd5e1; margin: 0; font-size: .9rem; }

/* ── Factor bar ────────────────────────────────────────────── */
.factor-row {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 0; border-bottom: 1px solid #1e293b;
}
.factor-name  { color: #cbd5e1; font-size: .9rem; min-width: 220px; }
.factor-bar   { flex: 1; height: 8px; border-radius: 4px;
                background: rgba(99,102,241,.3); }
.factor-fill  { height: 100%; border-radius: 4px;
                background: linear-gradient(90deg, #6366f1, #14b8a6); }
.factor-value { color: #94a3b8; font-size: .8rem; min-width: 60px; text-align: right; }

/* ── Cluster badge ─────────────────────────────────────────── */
.cluster-badge {
    display: inline-block;
    padding: 3px 12px; border-radius: 20px;
    font-size: .8rem; font-weight: 600;
    background: rgba(99,102,241,.2); color: #a5b4fc;
    border: 1px solid rgba(99,102,241,.35);
}

/* ── Probability gauge text ────────────────────────────────── */
.prob-number {
    font-size: 3.5rem; font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #14b8a6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; line-height: 1.1;
}
.prob-label { color: #94a3b8; font-size: .9rem; text-align: center; margin-top: 4px; }

/* ── Chat bubbles ──────────────────────────────────────────── */
.sql-expander { background: #0f172a; border-radius: 8px; padding: 12px;
                font-family: 'Courier New', monospace; font-size: .85rem;
                color: #a5b4fc; }
</style>
"""
