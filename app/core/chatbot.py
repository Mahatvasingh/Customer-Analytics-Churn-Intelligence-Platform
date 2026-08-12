"""
chatbot.py — NL-to-SQL + SHAP explanation chatbot.

Architecture
────────────
User message → intent classifier → one of four handlers:
  A. data_query       → LLM generates SQL → validate (SELECT-only) → execute → LLM summarises
  B. pred_explain     → fetch SHAP top_factors for customer_id → LLM phrases in plain English
  C. segment_question → query customer_segments → LLM describes cluster profiles
  D. general          → LLM answers using dashboard summary stats as context

SQL SECURITY
────────────
All generated SQL is validated with ``is_safe_sql()`` before execution.
Anything that is not a plain SELECT (INSERT, UPDATE, DROP, etc.) is rejected
immediately — this is a hard security boundary, not a suggestion.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.db import ChurnPrediction, Customer, CustomerSegment, SessionLocal
from app.core.llm_client import get_llm_client

# ── DB schema string (used as LLM context) ────────────────────────────────────
_SCHEMA = """
Database tables (SQLite):

customers      (customer_id INTEGER PK, signup_date TEXT, age INTEGER, gender TEXT,
                region TEXT, contract_type TEXT, payment_method TEXT,
                monthly_charges REAL, total_charges REAL, tenure_months INTEGER,
                is_active INTEGER 0/1, cluster_id INTEGER)

subscriptions  (subscription_id INTEGER PK, customer_id INTEGER FK,
                plan_name TEXT, start_date TEXT, end_date TEXT, status TEXT)

usage_logs     (log_id INTEGER PK, customer_id INTEGER FK, month TEXT YYYY-MM,
                usage_hours REAL, support_tickets INTEGER,
                late_payments INTEGER, feature_adoption_score REAL 0-1)

churn_labels   (customer_id INTEGER PK FK, churned INTEGER 0=stayed 1=churned,
                labeled_at TEXT)

churn_predictions (prediction_id INTEGER PK, customer_id INTEGER FK,
                   model_version TEXT, churn_probability REAL,
                   predicted_label INTEGER, predicted_at TEXT, top_factors TEXT JSON)

model_registry (model_id INTEGER PK, version TEXT, algorithm TEXT, trained_at TEXT,
                auc REAL, f1 REAL, precision REAL, recall REAL,
                is_production INTEGER 0/1, artifact_path TEXT)

customer_segments (segment_id INTEGER PK, cluster_id INTEGER, label TEXT,
                   avg_churn_rate REAL, size INTEGER, profile_json TEXT)
"""

# ── SQL Safety ────────────────────────────────────────────────────────────────
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE"
    r"|EXEC|EXECUTE|GRANT|REVOKE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)


def is_safe_sql(sql: str) -> bool:
    """
    Return True only if the SQL is a plain SELECT with no dangerous keywords.
    This is a hard security gate — never weaken without explicit review.
    """
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        return False
    if _FORBIDDEN.search(stripped):
        return False
    # Reject UNION-based injections
    if re.search(r"\bUNION\b", stripped, re.IGNORECASE) and "--" in stripped:
        return False
    return True


# ── Intent classifier ─────────────────────────────────────────────────────────

def _classify(msg: str) -> str:
    m = msg.lower()
    if any(k in m for k in ["why is customer", "explain customer", "customer id",
                             "customer #", "why did customer", "prediction for customer"]):
        return "pred_explain"
    if any(k in m for k in ["segment", "cluster", "group", "profile", "archetype"]):
        return "segment_question"
    if any(k in m for k in ["how many", "which", "what is", "what are", "show me",
                             "list", "count", "average", "avg", "top", "highest",
                             "lowest", "trend", "compare", "breakdown", "rate"]):
        return "data_query"
    return "general"


def _extract_customer_id(msg: str) -> Optional[int]:
    for pat in [
        r"customer\s+(?:id\s+)?#?(\d+)",
        r"customer\s+(\d+)",
        r"#(\d+)",
        r"\bid\s*[:=]?\s*(\d+)\b",
    ]:
        m = re.search(pat, msg, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _dashboard_summary(db: Session) -> str:
    try:
        total      = db.execute(text("SELECT COUNT(*) FROM customers")).scalar() or 0
        churn_rate = db.execute(text("SELECT AVG(CAST(churned AS REAL)) FROM churn_labels")).scalar() or 0
        avg_ch     = db.execute(text("SELECT AVG(monthly_charges) FROM customers")).scalar() or 0
        avg_ten    = db.execute(text("SELECT AVG(tenure_months) FROM customers")).scalar() or 0
        return (
            f"{int(total):,} customers | {churn_rate:.1%} churn rate | "
            f"${avg_ch:.2f} avg monthly charges | {avg_ten:.1f} months avg tenure"
        )
    except Exception:
        return "Platform summary unavailable."


# ── Main chatbot class ────────────────────────────────────────────────────────

class ChurnChatbot:
    """Stateless chatbot — pass history per call."""

    def __init__(self) -> None:
        self.llm = get_llm_client()

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        db: Optional[Session] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Process a user message.

        Returns
        -------
        (answer_text, sql_or_None)
        ``sql_or_None`` is the SQL that was executed (for the transparency panel).
        """
        _close = db is None
        if _close:
            db = SessionLocal()
        history = history or []
        try:
            intent = _classify(user_message)
            if intent == "pred_explain":
                return self._pred_explain(user_message, db)
            if intent == "segment_question":
                return self._segment_question(user_message, db)
            if intent == "data_query":
                return self._data_query(user_message, history, db)
            return self._general(user_message, history, db)
        finally:
            if _close:
                db.close()

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _data_query(self, msg: str, history: list, db: Session) -> Tuple[str, str]:
        summary = _dashboard_summary(db)
        sql_system = f"""You are a data analyst generating SQLite SELECT queries.

Schema:
{_SCHEMA}

Platform state: {summary}

Rules:
- Output ONLY the raw SQL statement — no markdown, no explanation, no fences.
- Only SELECT statements. No subquery tricks. LIMIT 20.
- If the question cannot be answered with SQL, output exactly: CANNOT_ANSWER
"""
        sql_raw = self.llm.chat(
            [{"role": "user", "content": f"Write SQL to answer: {msg}"}],
            system=sql_system, max_tokens=400,
        )
        sql = sql_raw.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip()

        if sql.upper() == "CANNOT_ANSWER":
            return self._general(msg, history, db), None

        if not is_safe_sql(sql):
            return (
                "⚠️ The generated query did not pass the read-only safety check "
                "and was not executed.", sql
            )

        try:
            result  = db.execute(text(sql))
            rows    = result.fetchall()
            cols    = list(result.keys())
            if rows:
                df_r = pd.DataFrame(rows, columns=cols)
                data_ctx = df_r.to_string(index=False, max_rows=20)
            else:
                data_ctx = "Query returned no rows."

            sum_system = f"""You are a friendly data analyst. 
Summarise these SQL results in clear, concise natural language.
Use bullet points if there are multiple insights. 
Platform context: {summary}"""
            answer = self.llm.chat(
                [{"role": "user",
                  "content": f"Question: {msg}\n\nResults:\n{data_ctx}\n\nSummarise."}],
                system=sum_system, max_tokens=600,
            )
            return answer, sql

        except Exception as exc:
            return f"⚠️ Query execution failed: `{str(exc)[:300]}`", sql

    def _pred_explain(self, msg: str, db: Session) -> Tuple[str, None]:
        cid = _extract_customer_id(msg)
        if not cid:
            return (
                "Please include the customer ID, e.g. "
                "*'Why is customer 1042 predicted to churn?'*",
                None,
            )

        customer = db.query(Customer).filter(Customer.customer_id == cid).first()
        if not customer:
            return f"Customer **{cid}** was not found in the database.", None

        pred = (
            db.query(ChurnPrediction)
            .filter(ChurnPrediction.customer_id == cid)
            .order_by(ChurnPrediction.predicted_at.desc())
            .first()
        )
        if not pred:
            return (
                f"No predictions exist for customer **{cid}** yet. "
                "Run a prediction from the **Predict Churn** page first.", None
            )

        factors_str = json.dumps(pred.top_factors or {}, indent=2)
        context = f"""Customer {cid}:
- Age: {customer.age} | Region: {customer.region} | Gender: {customer.gender}
- Contract: {customer.contract_type} | Tenure: {customer.tenure_months} months
- Monthly charges: ${customer.monthly_charges:.2f}
- Cluster: {customer.cluster_id}
- Predicted churn probability: {pred.churn_probability:.1%}
- SHAP top factors (feature → value, positive = pushes toward churn): {factors_str}"""

        system = """You are a customer retention specialist.
Explain this churn prediction in plain English for a business audience.
1. State the overall risk level clearly.
2. Explain the top 3 SHAP factors in business terms (what they mean, not raw numbers).
3. Suggest 2–3 concrete retention actions tailored to this customer.
Be concise and empathetic."""
        answer = self.llm.chat(
            [{"role": "user", "content": f"Explain the churn prediction for customer {cid}:\n{context}"}],
            system=system, max_tokens=700,
        )
        return answer, None

    def _segment_question(self, msg: str, db: Session) -> Tuple[str, None]:
        segs = db.query(CustomerSegment).order_by(CustomerSegment.cluster_id).all()
        if not segs:
            return "No customer segments found. Run **seed.py** to generate data.", None

        ctx = "\n".join(
            f"Segment {s.cluster_id} — {s.label}: "
            f"{s.size} customers, {s.avg_churn_rate:.1%} churn rate. "
            f"Profile: {json.dumps(s.profile_json)}"
            for s in segs
        )
        system = """You are a customer analytics expert.
Answer the question using the segment data below.
Focus on actionable business insights and keep it concise."""
        answer = self.llm.chat(
            [{"role": "user", "content": f"Segments:\n{ctx}\n\nQuestion: {msg}"}],
            system=system, max_tokens=700,
        )
        return answer, None

    def _general(self, msg: str, history: list, db: Session) -> Tuple[str, None]:
        summary = _dashboard_summary(db)
        system = f"""You are an AI assistant for a Customer Churn Intelligence Platform.
Current platform data: {summary}
{_SCHEMA}
Answer questions about customer churn, retention strategies, and data insights.
Be concise, helpful, and data-driven. Say so if you don't have enough information."""
        messages = history + [{"role": "user", "content": msg}]
        answer = self.llm.chat(messages, system=system, max_tokens=600)
        return answer, None


# ── Suggested starter questions ───────────────────────────────────────────────
STARTER_QUESTIONS = [
    "Which customer segment has the highest churn rate?",
    "What is the average monthly charge for churned customers?",
    "Which region has the most at-risk customers?",
    "How many customers are on month-to-month contracts?",
    "Show me the top 5 customers by churn probability.",
    "What is the churn rate by contract type?",
]
