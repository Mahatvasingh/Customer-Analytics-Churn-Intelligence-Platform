"""
3_AI_Insights_Chatbot.py — Natural-language chatbot over the churn database.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from app.core.chatbot import ChurnChatbot, STARTER_QUESTIONS
from app.core.llm_client import get_llm_client
from app.core.utils import GLOBAL_CSS

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Extra chat-specific CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
.stChatMessage { background: #1e293b !important; border-radius: 12px !important; }
.stChatMessage p { color: #f1f5f9 !important; }
.starter-btn {
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    color: #a5b4fc;
    padding: 6px 14px;
    font-size: .85rem;
    cursor: pointer;
    margin: 4px;
    display: inline-block;
    transition: background .2s;
}
.starter-btn:hover { background: rgba(99,102,241,0.25); }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Insights Chatbot")
st.caption("Ask natural-language questions about your customer data, churn trends, and predictions")

# ── LLM status ────────────────────────────────────────────────────────────────
llm = get_llm_client()
if not llm.is_available:
    st.warning(
        "⚠️ **LLM not configured.** Set `ANTHROPIC_API_KEY` in your `.env` file to enable the chatbot.\n\n"
        "All other platform pages work without an API key."
    )
    st.stop()

st.sidebar.success("✅ LLM Connected")
st.sidebar.markdown("**Capabilities**")
st.sidebar.markdown("""
- 📊 Data queries (NL → SQL)
- 🔍 Prediction explanations (SHAP)
- 🔮 Customer segment analysis
- 💡 General churn insights
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Security**")
st.sidebar.markdown("All SQL is validated as SELECT-only before execution.")

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []    # list of {"role", "content", "sql"}
if "chatbot" not in st.session_state:
    st.session_state.chatbot = ChurnChatbot()

chatbot = st.session_state.chatbot

# ── Starter questions ─────────────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("#### 💬 Try asking…")
    cols = st.columns(2)
    for i, q in enumerate(STARTER_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"starter_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sql"):
            with st.expander("🔍 How I got this (SQL)", expanded=False):
                st.markdown(
                    f'<div class="sql-expander"><pre>{msg["sql"]}</pre></div>',
                    unsafe_allow_html=True,
                )

# ── Handle pending starter question ──────────────────────────────────────────
user_input = st.chat_input("Ask about your churn data…")
if "pending_question" in st.session_state:
    user_input = st.session_state.pop("pending_question")

# ── Process new message ───────────────────────────────────────────────────────
if user_input:
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": user_input, "sql": None})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build history for LLM (last 6 turns only to stay within context)
    llm_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_history[-12:]
        if m["role"] in ("user", "assistant")
    ]

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            try:
                answer, sql = chatbot.chat(user_input, history=llm_history[:-1])
            except Exception as exc:
                answer = f"⚠️ An error occurred: `{exc}`"
                sql    = None

        st.markdown(answer)
        if sql:
            with st.expander("🔍 How I got this (SQL)", expanded=False):
                st.markdown(
                    f'<div class="sql-expander"><pre>{sql}</pre></div>',
                    unsafe_allow_html=True,
                )

    st.session_state.chat_history.append({"role": "assistant", "content": answer, "sql": sql})

# ── Clear chat ────────────────────────────────────────────────────────────────
if st.session_state.chat_history:
    if st.button("🗑️ Clear conversation", use_container_width=False):
        st.session_state.chat_history = []
        st.rerun()
