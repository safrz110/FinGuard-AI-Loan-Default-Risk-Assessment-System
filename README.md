# 🏦 FinGuard AI — Loan Default Risk System

> **Instant, AI-powered loan default risk assessment for modern lending workflows.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-GBC-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-90.6%25-D4AF37?style=flat-square)]()

---

##  Overview

**FinGuard AI** is a production-ready credit risk intelligence application built with Streamlit. It leverages a **Gradient Boosting Classifier** trained on 6,000 synthetic applicant records to predict loan default probability in real time — delivering actionable lending recommendations in under a second.

The app is designed for financial analysts, credit risk teams, and fintech developers who need a transparent, explainable risk-scoring interface without the overhead of a full enterprise stack.

---

##  Features

- **Real-time risk scoring** — Predicts default probability from 6 key applicant inputs
- **Three-tier risk classification** — Low / Moderate / High with colour-coded result cards
- **Actionable lending recommendations** — Approve, review, or decline guidance per assessment
- **Visual risk gauge** — Intuitive colour-banded bar with a probability marker
- **Engineered feature pipeline** — Automatic derivation of 10+ financial ratios and log transforms
- **Persistent model caching** — `@st.cache_resource` prevents redundant model re-training
- **Premium dark UI** — Custom CSS with JetBrains Mono and Plus Jakarta Sans typefaces

---

##  Model Details

| Property | Value |
|---|---|
| Algorithm | Gradient Boosting Classifier (GBC) |
| Training samples | 6,000 synthetic applicant records |
| Test split | 15% (stratified) |
| Accuracy | **90.6%** |
| Risk classes | 2 — `No Default (0)` / `Default (1)` |
| Preprocessor | `ColumnTransformer` (loaded from `best_model.pkl`) |

### Input Features

| Feature | Type | Range / Options |
|---|---|---|
| Age | Integer | 21 – 70 years |
| Gender | Categorical | Male, Female |
| Marital Status | Categorical | Married, Single |
| Annual Income (₹) | Integer | ₹15,000 – ₹2,00,000 |
| Loan Amount (₹) | Integer | ₹2,000 – ₹80,000 |
| Credit Score | Integer | 300 – 900 |
| Employment Status | Categorical | Employed, Self-employed, Unemployed |

### Engineered Features (auto-derived)

`income_loan_ratio` · `loan_burden` · `risk_score` · `income_x_age` · `loan_x_income` · `loan_per_age` · `loan_amount_log` · `income_log`

---

##  Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/your-username/finguard-ai.git
cd finguard-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Ensure model artifacts are present

The following files must be in the **project root** alongside `app.py`:

```
finguard-ai/
├── app.py
├── best_model.pkl          # Trained ColumnTransformer preprocessor
├── feature_columns.pkl     # Ordered list of feature column names
└── requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

##  Requirements

Create a `requirements.txt` with the following (versions tested):

```
streamlit>=1.32.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

---


##  Live Demo 
https://finguard-ai-loan-default-risk-assessment-0.streamlit.app


##  App Walkthrough

1. **Applicant Profile** — Enter age, gender, and marital status
2. **Financial Profile** — Set annual income, loan amount, credit score, and employment status
3. **Assess Loan Risk** — Click the button to run the model
4. **Result Card** — Displays default probability, risk tier, and a lending recommendation
5. **Risk Gauge** — Visual indicator showing where the applicant falls on the Low → High spectrum
6. **Summary Chips** — At-a-glance breakdown of the four key risk drivers

---

##  Risk Tier Logic

| Default Probability | Tier | Recommendation |
|---|---|---|
| < 25% |  Low Risk | Approve — standard terms |
| 25% – 54% |  Moderate Risk | Review — consider collateral or co-signer |
| ≥ 55% |  High Risk | Decline or restructure with higher rate |

---

##  Project Structure

```
finguard-ai/
├── app.py                  # Main Streamlit application
├── best_model.pkl          # Serialised ColumnTransformer preprocessor
├── feature_columns.pkl     # Feature column order for inference
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

##  Disclaimer

This application is built for **demonstration and educational purposes**. The training data is synthetically generated and does not represent real financial records. It is **not intended for use in actual lending or credit decisions**. Always consult qualified financial and compliance professionals before deploying any credit-scoring system in production.

---

##  Contributing

Contributions are welcome. Please open an issue first to discuss any significant changes, then submit a pull request against the `main` branch.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

##  License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Decision Tree · Credit Risk Engine · FinGuard AI v1.0</sub>
</div>
