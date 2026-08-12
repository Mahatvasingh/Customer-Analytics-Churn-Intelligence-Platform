# 🔮 Customer Churn Intelligence Platform

A production-style, full-stack machine learning application that predicts customer churn, clusters customers into actionable segments, and provides natural-language AI insights over your data.

Built entirely with Python — no external data dependencies required to start.

## 🌟 Features

- **Synthetic Data Generation**: K-Means seeded generator creates realistic customer profiles with causal churn signals.
- **ML Pipeline**: Trains Logistic Regression and Random Forest classifiers alongside a K-Means customer segmentation model. Includes a versioned model registry.
- **AI Chatbot**: A natural-language interface (powered by Anthropic Claude) that translates questions into safe SQL queries, explains model predictions using SHAP values, and summarizes customer segments.
- **Streamlit Dashboard**: 5-page interactive UI featuring KPI cards, Plotly charts, single/batch predictions, and an admin data management interface.
- **Database Agnostic**: Uses SQLAlchemy ORM. Defaults to local SQLite, but can be swapped to PostgreSQL with a single `.env` change.

## 📁 Project Structure

```text
churn-platform/
├── app/
│   ├── main.py                     # Entry point: Streamlit navigation shell
│   ├── pages/
│   │   ├── 1_Dashboard.py          # EDA + KPIs + charts + cluster map
│   │   ├── 2_Predict_Churn.py      # Single/batch prediction UI
│   │   ├── 3_AI_Insights_Chatbot.py# NL chatbot over data + model
│   │   ├── 4_Model_Registry.py     # Compare/promote/rollback model versions
│   │   └── 5_Admin_Data_Management.py # Regenerate data, manage DB stats
│   ├── core/
│   │   ├── db.py                   # SQLAlchemy engine, session, models
│   │   ├── data_generator.py       # K-Means seeded synthetic data generator
│   │   ├── feature_engineering.py  # Feature pipeline (shared by train + inference)
│   │   ├── model_training.py       # Train LR + RF + K-Means segmentation
│   │   ├── model_registry.py       # Version tracking, metrics, promote/rollback
│   │   ├── llm_client.py           # Thin wrapper over Anthropic Claude
│   │   ├── chatbot.py              # NL-to-SQL + insight generation + SHAP explain
│   │   └── utils.py                # Shared formatting and styling
│   └── config.py                   # Loads .env, central constants
├── models/                         # Saved model artifacts (.joblib) + metadata.json
├── data/                           # churn.db (SQLite) lives here
├── tests/                          # pytest suite
├── .env.example
├── requirements.txt
├── README.md
└── seed.py                         # CLI script: init DB, generate data, train first model
```

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.11+
- Virtual environment (recommended)

### 2. Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configuration

Copy the example `.env` file:

```bash
cp .env.example .env
```

Open `.env` and add your **Anthropic API Key** to enable the AI Chatbot. 
*(Note: If you don't have a key, the rest of the platform will still work fully, and the chatbot page will display a friendly "not configured" message).*

### 4. Seed the Database and Train Models

Run the setup script. This will create the database schema, generate 8,000 synthetic customers, train the initial Machine Learning models, and compute customer segments:

```bash
python seed.py
```
*This step may take 1–2 minutes to complete.*

### 5. Launch the Application

Start the Streamlit server:

```bash
streamlit run app/main.py
```

The application will open in your default web browser (typically at `http://localhost:8501`).

## 🧪 Running Tests

A `pytest` suite is included to verify the data generator, feature engineering pipeline, and model training logic. Tests use an in-memory SQLite database to avoid modifying your local `churn.db`.

```bash
pytest tests/ -v
```

## 🔌 Swapping to PostgreSQL

To switch from the default local SQLite database to PostgreSQL:

1. Ensure your PostgreSQL server is running.
2. Edit your `.env` file and set the `DATABASE_URL`:
   ```env
   DATABASE_URL=postgresql://username:password@localhost:5432/your_database_name
   ```
3. Rerun `python seed.py` to create the schema and seed data in the new database.
