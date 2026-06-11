import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import math

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="FinGuard AI – Loan Default Risk System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# MODEL METADATA (UPDATE WITH YOUR REAL DECISION TREE METRICS)
# ============================================================
MODEL_NAME = "Decision Tree Classifier"
ACCURACY = 0.862      # 86.2%
F1_SCORE = 0.847      # 84.7%
TRAINING_SAMPLES = 12500

# ============================================================
# CUSTOM CSS – SLIGHTLY DARK BACKGROUND, LIGHT CARDS, GOOD CONTRAST
# ============================================================
st.markdown("""
<style>
    /* Global – slightly dark background */
    .stApp {
        background: linear-gradient(135deg, #1a2a3a 0%, #0f1a24 100%);
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #5bc0be 0%, #3a9b9b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center;
        color: #a8c9c9;
        margin-bottom: 2rem;
        font-size: 1rem;
        font-weight: 500;
    }
    /* Metric cards (top row) – light, elevated */
    .metric-card {
        background: #ffffff;
        border-radius: 1.5rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        border: none;
        transition: 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1f6d6d;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #2c5a5a;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
        font-weight: 600;
    }
    /* Input sections – light cards */
    .input-section {
        background: #ffffff;
        border-radius: 1.5rem;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        border: none;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1f6d6d;
        margin-bottom: 1rem;
        border-left: 4px solid #1f6d6d;
        padding-left: 0.8rem;
    }
    /* Risk result card */
    .risk-card {
        background: #ffffff;
        border-radius: 1.5rem;
        padding: 1.5rem;
        margin-top: 1rem;
        text-align: center;
        box-shadow: 0 12px 28px rgba(0,0,0,0.2);
    }
    .risk-low {
        background: linear-gradient(135deg, #d0f0e8 0%, #b2dfdb 100%);
        border-left: 8px solid #2e7d32;
        color: #004d40;
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffe0b2 0%, #ffcc80 100%);
        border-left: 8px solid #e65100;
        color: #8b4513;
    }
    .risk-high {
        background: linear-gradient(135deg, #ffcdd2 0%, #ef9a9a 100%);
        border-left: 8px solid #c62828;
        color: #b71c1c;
    }
    .risk-value {
        font-size: 2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    /* Buttons – vibrant to stand out */
    .stButton > button {
        background: linear-gradient(135deg, #2c7a7a 0%, #1f6d6d 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 700;
        border-radius: 2.5rem;
        width: 100%;
        transition: 0.2s;
        font-size: 1rem;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #3a9b9b 0%, #2c7a7a 100%);
    }
    /* Form inputs */
    .stSelectbox label, .stNumberInput label {
        font-weight: 600;
        color: #1f6d6d;
    }
    hr {
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        color: #8fbbbb;
        font-size: 0.7rem;
        margin-top: 2rem;
        border-top: 1px solid #2c5a5a;
        padding-top: 1rem;
    }
    /* Info & warning boxes – light background, dark text */
    .stAlert {
        border-radius: 0.8rem;
        background-color: #f8f9fa;
        color: #1a2a3a;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# RULE‑BASED RISK CALCULATOR (Decision Tree Logic)
# ============================================================
def calculate_risk(age, income, loan_amount, credit_score, employment_status, marital_status):
    """
    Transparent rule-based risk score (0-100).
    Mimics a decision tree using financial & demographic rules.
    """
    risk = 0

    # 1. Credit score branch
    if credit_score >= 750:
        risk -= 25
    elif credit_score >= 650:
        risk -= 5
    elif credit_score >= 550:
        risk += 15
    else:
        risk += 35

    # 2. Debt-to-income ratio (loan/income)
    dti = loan_amount / income if income > 0 else 999
    if dti > 0.5:
        risk += 30
    elif dti > 0.3:
        risk += 12
    else:
        risk -= 10

    # 3. Employment status
    if employment_status == "Unemployed":
        risk += 25
    elif employment_status == "Self-employed":
        risk += 8
    else:  # Employed
        risk -= 5

    # 4. Age factor
    if age < 25:
        risk += 12
    elif age > 60:
        risk += 8
    elif 35 <= age <= 50:
        risk -= 5

    # 5. Marital status (small adjustment)
    if marital_status == "Single":
        risk += 3

    # 6. Loan amount absolute (extra safety)
    if loan_amount > 50000:
        risk += 10
    elif loan_amount < 10000:
        risk -= 5

    # Clamp between 0 and 100
    return np.clip(risk, 0, 100)

def get_risk_category(risk_score):
    if risk_score >= 70:
        return "High Risk", "risk-high", f"{risk_score:.1f} / 100", "#c62828"
    elif risk_score >= 30:
        return "Medium Risk", "risk-medium", f"{risk_score:.1f} / 100", "#e65100"
    else:
        return "Low Risk", "risk-low", f"{risk_score:.1f} / 100", "#2e7d32"

# ============================================================
# GAUGE CHART (Plotly) – Enhanced Colors
# ============================================================
def create_gauge(risk_score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        title = {'text': "Risk Meter", 'font': {'size': 16, 'color': '#1f6d6d', 'weight': 'bold'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1f6d6d"},
            'bar': {'color': "#2c7a7a"},
            'bgcolor': "#fafcfd",
            'borderwidth': 2,
            'bordercolor': "#cce6e6",
            'steps': [
                {'range': [0, 30], 'color': '#b2dfdb'},
                {'range': [30, 70], 'color': '#ffe0b2'},
                {'range': [70, 100], 'color': '#ffcdd2'}
            ],
            'threshold': {
                'line': {'color': "#c62828", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ============================================================
# MAIN APP
# ============================================================
def main():
    # Header
    st.markdown('<div class="main-header">🛡️ FinGuard AI – Loan Default Risk Assessment System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intelligent credit risk analytics powered by a Decision Tree classifier</div>', unsafe_allow_html=True)

    # Top metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{ACCURACY:.1%}</div><div class="metric-label">MODEL ACCURACY</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{F1_SCORE:.3f}</div><div class="metric-label">F1 SCORE</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{MODEL_NAME}</div><div class="metric-label">ALGORITHM</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{TRAINING_SAMPLES:,}</div><div class="metric-label">TRAINING SAMPLES</div></div>', unsafe_allow_html=True)

    # Two-column layout for inputs
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 DEMOGRAPHIC & EMPLOYMENT</div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Female", "Male"], help="Legal gender")
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=30, step=1)
        employment_status = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed"])
        marital_status = st.selectbox("Marital Status", ["Married", "Single"])
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💰 FINANCIAL METRICS</div>', unsafe_allow_html=True)
        income = st.number_input("Annual Income ($)", min_value=0, max_value=500_000, value=50_000, step=1_000)
        loan_amount = st.number_input("Loan Amount Requested ($)", min_value=0, max_value=200_000, value=20_000, step=1_000)
        credit_score = st.number_input("Credit Score (300–850)", min_value=300, max_value=850, value=650, step=10, help="Higher is better")
        st.markdown('</div>', unsafe_allow_html=True)

    # Analyze button
    analyze = st.button("🔍 ANALYZE RISK WITH FINGUARD AI", use_container_width=True)

    # Result area
    st.markdown('<div class="risk-card">', unsafe_allow_html=True)
    if analyze:
        if income <= 0 or loan_amount <= 0:
            st.error("❌ Income and loan amount must be positive values.")
        else:
            # Compute risk using the rule-based decision tree
            risk_score = calculate_risk(age, income, loan_amount, credit_score, employment_status, marital_status)
            risk_label, risk_class, risk_display, color = get_risk_category(risk_score)
            
            # Show result
            st.markdown(f'<div class="{risk_class}" style="border-radius:1.2rem; padding:1.2rem;">'
                        f'<h3 style="margin:0; font-weight:700;">{risk_label}</h3>'
                        f'<p class="risk-value">{risk_display}</p>'
                        f'<p style="margin:0; font-weight:500;">{"High probability of default – caution required" if risk_score >= 70 else "Moderate risk – manual review recommended" if risk_score >= 30 else "Low probability of default – favorable for approval"}</p>'
                        f'</div>', unsafe_allow_html=True)
            
            # Risk gauge
            gauge = create_gauge(risk_score)
            st.plotly_chart(gauge, use_container_width=True)
            
            # Additional derived metrics expander
            with st.expander("📊 View detailed financial ratios"):
                dti = loan_amount / income if income > 0 else 0
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Debt-to-Income Ratio", f"{dti:.2f}")
                with col_b:
                    st.metric("Loan Burden (%)", f"{(dti*100):.1f}%")
                with col_c:
                    st.metric("Credit Score Health", f"{credit_score} / 850")
                st.info(f"🧬 Age group: {'<25' if age<25 else '26-35' if age<=35 else '36-50' if age<=50 else '50+'} | "
                        f"Income level: {'<30k' if income<30000 else '30k-80k' if income<=80000 else '>80k'} | "
                        f"Loan level: {'<10k' if loan_amount<10000 else '10k-50k' if loan_amount<=50000 else '>50k'}")
    else:
        st.info("👈 Fill in the applicant details and click **Analyze Risk** to get a loan default prediction.")
        # Show a placeholder gauge
        placeholder_gauge = create_gauge(0)
        st.plotly_chart(placeholder_gauge, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown('<div class="footer">⚠️ FinGuard AI is an AI‑powered risk assessment tool. Final lending decisions should include human underwriting and additional verification.<br>© 2026 FinGuard AI – Decision Tree Risk Engine</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()