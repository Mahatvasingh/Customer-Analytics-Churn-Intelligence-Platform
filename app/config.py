"""
config.py — Central configuration loaded from .env
All other modules import constants from here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# ── Project directories ───────────────────────────────────────────────────────
# BASE_DIR is the project root (one level up from app/)
BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist on import
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Database ──────────────────────────────────────────────────────────────────
# Default: SQLite stored in data/churn.db
# To use PostgreSQL: set DATABASE_URL=postgresql://user:pass@host/db in .env
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{DATA_DIR / 'churn.db'}"
)

# ── LLM Provider ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
AI_MODEL_NAME: str = os.getenv("AI_MODEL_NAME", "claude-sonnet-4-6")

# ── App metadata ─────────────────────────────────────────────────────────────
APP_TITLE = "Customer Churn Intelligence Platform"
APP_ICON = "🔮"
APP_VERSION = "1.0.0"

# ── ML hyperparameters ────────────────────────────────────────────────────────
N_CLUSTERS: int = 5          # K-Means clusters
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

# ── Risk tier thresholds ─────────────────────────────────────────────────────
RISK_LOW_THRESHOLD: float = 0.30
RISK_HIGH_THRESHOLD: float = 0.60
