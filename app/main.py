"""
main.py — Streamlit navigation shell.
Entry point: streamlit run app/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────────
# This makes `from app.core.xxx import ...` work in all page scripts.
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.config import APP_ICON, APP_TITLE

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar branding ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 12px 0 24px;">
            <div style="font-size:2.4rem;">{APP_ICON}</div>
            <div style="font-size:1rem; font-weight:700; color:#f1f5f9; margin-top:6px;">
                Churn Intelligence
            </div>
            <div style="font-size:.75rem; color:#64748b; margin-top:2px;">
                AI-Powered Customer Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

# ── Page definitions ──────────────────────────────────────────────────────────
pages = [
    st.Page("pages/1_Dashboard.py",            title="Dashboard",       icon="📊"),
    st.Page("pages/2_Predict_Churn.py",        title="Predict Churn",   icon="🎯"),
    st.Page("pages/3_AI_Insights_Chatbot.py",  title="AI Chatbot",      icon="🤖"),
    st.Page("pages/4_Model_Registry.py",       title="Model Registry",  icon="🗂️"),
    st.Page("pages/5_Admin_Data_Management.py", title="Data Management", icon="⚙️"),
]

pg = st.navigation(pages)
pg.run()
