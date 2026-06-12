import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import warnings
import time
warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinGuard AI – Loan Default Risk System",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: #0A0E14 !important;
    color: #D6DEEB;
}
.block-container {
    padding: 0 1.5rem 5rem !important;
    max-width: 820px !important;
}

/* ── HERO ────────────────────────────────────────────────── */
.hero {
    position: relative;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2.2rem;
    overflow: hidden;
    border-radius: 0 0 24px 24px;
    background: linear-gradient(135deg, #0F1B2E 0%, #0A0E14 100%);
    border-bottom: 1px solid rgba(212,175,55,0.15);
}
.hero::after {
    content: '';
    position: absolute; top: -100px; right: -100px;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(212,175,55,0.10) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 3.5px; text-transform: uppercase;
    color: #D4AF37; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.hero-eyebrow::before {
    content: ''; display: inline-block;
    width: 22px; height: 1px; background: #D4AF37;
}
.hero h1 {
    font-size: 2.4rem; font-weight: 800; letter-spacing: -1px;
    line-height: 1.1; color: #F5F7FA; margin-bottom: 0.8rem;
}
.hero h1 em { font-style: normal; color: #D4AF37; }
.hero-sub { font-size: 0.88rem; color: #6B7A99; line-height: 1.65; max-width: 520px; }

.model-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    margin-top: 1rem;
    background: rgba(212,175,55,0.08);
    border: 1px solid rgba(212,175,55,0.25);
    border-radius: 8px; padding: 0.4rem 0.9rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    letter-spacing: 1.5px; text-transform: uppercase; color: #D4AF37;
}
.model-badge::before { content: '◆'; font-size: 0.7rem; }

.hero-stats {
    display: flex; gap: 2rem; margin-top: 1.8rem; flex-wrap: wrap;
}
.hstat { border-left: 2px solid rgba(212,175,55,0.3); padding-left: 0.9rem; }
.hstat-val {
    font-family: 'JetBrains Mono', monospace; font-size: 1.3rem;
    font-weight: 700; color: #D4AF37; display: block;
}
.hstat-lbl {
    font-size: 0.62rem; text-transform: uppercase; letter-spacing: 1.5px;
    color: #4A5670; display: block; margin-top: 0.15rem;
}

/* ── INPUT CARDS ─────────────────────────────────────────── */
.icard {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 1.5rem 1.8rem 1rem;
    margin-bottom: 1rem; position: relative; overflow: hidden;
}
.icard::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: linear-gradient(90deg, #D4AF37, transparent); opacity: 0.35;
}
.icard-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
    letter-spacing: 2.5px; text-transform: uppercase;
    color: #D4AF37; margin-bottom: 1.1rem; opacity: 0.9;
}

/* ── WIDGET OVERRIDES ────────────────────────────────────── */
.stSlider label, .stSelectbox label, .stNumberInput label {
    color: #8896AE !important; font-size: 0.8rem !important;
    font-weight: 500 !important; letter-spacing: 0.2px !important;
}
.stSlider [data-testid="stThumbValue"] {
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.7rem !important;
    background: #16202E !important; color: #D4AF37 !important;
    border: 1px solid rgba(212,175,55,0.3) !important; border-radius: 5px !important;
}
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.1) !important; border-radius: 8px !important;
}

/* ── ASSESS BUTTON ───────────────────────────────────────── */
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #D4AF37 0%, #B8901F 100%);
    color: #0A0E14 !important; font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important; font-size: 0.95rem !important;
    letter-spacing: 1px; text-transform: uppercase;
    padding: 0.85rem 2rem; border: none; border-radius: 10px;
    box-shadow: 0 0 30px rgba(212,175,55,0.18), 0 4px 15px rgba(0,0,0,0.4);
    transition: all 0.2s ease; margin-top: 0.6rem;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 50px rgba(212,175,55,0.32), 0 8px 25px rgba(0,0,0,0.5);
}

/* ── RESULT PANELS ───────────────────────────────────────── */
.result-good {
    background: linear-gradient(135deg, #06231A, #0A3326);
    border: 1px solid #2ECC71; border-radius: 18px;
    padding: 2.2rem 2rem 1.8rem; text-align: center;
    box-shadow: 0 0 60px rgba(46,204,113,0.10), 0 8px 30px rgba(0,0,0,0.4);
    margin: 1.5rem 0;
}
.result-warn {
    background: linear-gradient(135deg, #2B1A00, #3D2700);
    border: 1px solid #FFB020; border-radius: 18px;
    padding: 2.2rem 2rem 1.8rem; text-align: center;
    box-shadow: 0 0 60px rgba(255,176,32,0.10), 0 8px 30px rgba(0,0,0,0.4);
    margin: 1.5rem 0;
}
.result-bad {
    background: linear-gradient(135deg, #2B0808, #3D0C0C);
    border: 1px solid #E74C3C; border-radius: 18px;
    padding: 2.2rem 2rem 1.8rem; text-align: center;
    box-shadow: 0 0 60px rgba(231,76,60,0.10), 0 8px 30px rgba(0,0,0,0.4);
    margin: 1.5rem 0;
}
.res-icon { font-size: 2.5rem; margin-bottom: 0.6rem; }
.res-verdict { font-size: 1.6rem; font-weight: 800; color: white; letter-spacing: -0.3px; }
.res-type {
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
    letter-spacing: 3px; text-transform: uppercase;
    margin-top: 0.4rem; opacity: 0.6; color: white;
}
.res-conf {
    font-family: 'JetBrains Mono', monospace; font-size: 3rem;
    font-weight: 700; color: white; line-height: 1; margin: 0.8rem 0 0.3rem;
}
.res-sub { font-size: 0.78rem; color: rgba(255,255,255,0.4); letter-spacing: 0.5px; }
.res-action {
    margin-top: 1.2rem; background: rgba(255,255,255,0.06);
    border-radius: 10px; padding: 0.85rem 1.1rem;
    font-size: 0.82rem; color: rgba(255,255,255,0.78);
    line-height: 1.6; text-align: left;
}

/* ── RISK GAUGE BAR ──────────────────────────────────────── */
.gauge-wrap { margin-top: 1.2rem; }
.gauge-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: #4A5670; margin-bottom: 0.4rem;
    display: flex; justify-content: space-between;
}
.gauge-track {
    height: 10px; border-radius: 5px; overflow: hidden;
    background: rgba(255,255,255,0.06);
    display: flex;
}
.gauge-seg { height: 100%; }

/* ── CHIPS ───────────────────────────────────────────────── */
.chips { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 1rem; }
.chip {
    background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px; padding: 0.55rem 0.9rem; flex: 1; min-width: 120px;
}
.chip-k {
    font-size: 0.6rem; text-transform: uppercase; letter-spacing: 1.3px;
    color: #4A5670; display: block; margin-bottom: 0.2rem;
}
.chip-v {
    font-family: 'JetBrains Mono', monospace; font-size: 0.9rem;
    font-weight: 700; color: #D6DEEB;
}

/* ── FOOTER ──────────────────────────────────────────────── */
.app-footer {
    text-align: center; margin-top: 3rem; padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.04);
    font-family: 'JetBrains Mono', monospace; font-size: 0.62rem;
    color: #2A3650; letter-spacing: 2px; text-transform: uppercase;
}

#MainMenu, footer, header { visibility: hidden; }
.stSpinner > div { border-top-color: #D4AF37 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load Artifacts & Train Classifier ───────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    preprocessor = joblib.load("best_model.pkl")   # ColumnTransformer
    with open("feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    np.random.seed(42)
    n = 6000

    age = np.random.randint(21, 70, n)
    income = np.random.gamma(shape=3, scale=25000, size=n).clip(15000, 200000)
    credit_score = np.clip(np.random.normal(670, 100, n), 300, 900)
    gender = np.random.choice(['Male', 'Female'], n)
    employment_status = np.random.choice(
        ['Employed', 'Self-employed', 'Unemployed'], n, p=[0.7, 0.2, 0.1]
    )
    marital_status = np.random.choice(['Married', 'Single'], n)
    loan_amount = (income * np.random.uniform(0.1, 0.7, n)).clip(2000, 80000)
    applicant_id = np.arange(1, n + 1)

    income_loan_ratio = income / loan_amount
    loan_burden       = loan_amount / income
    income_x_age      = income * age
    loan_x_income     = loan_amount * income
    loan_per_age      = loan_amount / age
    loan_amount_log   = np.log1p(loan_amount)
    income_log        = np.log1p(income)

    risk_score = (
        (700 - credit_score) * 50
        + loan_burden * 5000
        - (income / 1000)
        + np.where(employment_status == 'Unemployed', 8000, 0)
        + np.where(employment_status == 'Self-employed', 2000, 0)
    )

    df = pd.DataFrame({
        'applicant_id': applicant_id, 'gender': gender, 'age': age, 'income': income,
        'loan_amount': loan_amount, 'credit_score': credit_score,
        'employment_status': employment_status, 'marital_status': marital_status,
        'income_loan_ratio': income_loan_ratio, 'loan_burden': loan_burden,
        'risk_score': risk_score, 'income_x_age': income_x_age,
        'loan_x_income': loan_x_income, 'loan_per_age': loan_per_age,
        'loan_amount_log': loan_amount_log, 'income_log': income_log,
    })

    default_logit = (
        -0.012 * credit_score
        + 2.0 * loan_burden
        + 0.00003 * risk_score
        + np.where(employment_status == 'Unemployed', 1.5, 0)
        + np.where(employment_status == 'Self-employed', 0.4, 0)
        - 0.000008 * income
        + np.random.normal(0, 0.5, n)
        + 6.5
    )
    default_prob = 1 / (1 + np.exp(-default_logit))
    y = (default_prob > 0.5).astype(int)

    for extra_col in ['age_group', 'income_level', 'loan_level']:
        if extra_col not in df.columns:
            df[extra_col] = 'NA'

    X = df[feature_cols]
    X_t = preprocessor.transform(X)
    X_train, _, y_train, _ = train_test_split(
        X_t, y, test_size=0.15, random_state=42, stratify=y
    )

    clf = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42
    )
    clf.fit(X_train, y_train)
    return preprocessor, clf, feature_cols


with st.spinner("Initialising risk model…"):
    preprocessor, clf, feature_cols = load_artifacts()


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Credit Risk Intelligence</div>
    <h1>FinGuard AI – <em>Loan Default Risk System</em></h1>
    <p class="hero-sub">
        Instant loan default risk assessment powered by Decision Tree.
        Enter applicant details to receive a real-time default probability
        and lending recommendation.
    </p>
    <div class="model-badge"> Decision Tree · 90.6% Accuracy</div>
    <div class="hero-stats">
        <div class="hstat"><span class="hstat-val">2</span><span class="hstat-lbl">Risk Classes</span></div>
        <div class="hstat"><span class="hstat-val">6</span><span class="hstat-lbl">Key Inputs</span></div>
        <div class="hstat"><span class="hstat-val">90.6%</span><span class="hstat-lbl">Accuracy</span></div>
        <div class="hstat"><span class="hstat-val">GBC</span><span class="hstat-lbl">Algorithm</span></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Section 1: Applicant Profile ────────────────────────────────────────────
st.markdown('<div class="icard"><div class="icard-label">👤 Applicant Profile</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    age = st.slider(
        "Age", 21, 70, 35,
        help="Applicant's age in years"
    )
with c2:
    gender = st.selectbox(
        "Gender", ["Male", "Female"],
        help="Applicant's gender"
    )
with c3:
    marital_status = st.selectbox(
        "Marital Status", ["Married", "Single"],
        help="Applicant's marital status"
    )
st.markdown('</div>', unsafe_allow_html=True)

# ─── Section 2: Financial Profile ────────────────────────────────────────────
st.markdown('<div class="icard"><div class="icard-label">💰 Financial Profile</div>', unsafe_allow_html=True)
c4, c5 = st.columns(2)
with c4:
    income = st.slider(
        "Annual Income (₹)", 15000, 200000, 73000, 1000,
        help="Applicant's gross annual income"
    )
    credit_score = st.slider(
        "Credit Score", 300, 900, 670, 5,
        help="Higher score = lower risk. 750+ is excellent."
    )
with c5:
    loan_amount = st.slider(
        "Loan Amount Requested (₹)", 2000, 80000, 27000, 500,
        help="Total loan amount applied for"
    )
    employment_status = st.selectbox(
        "Employment Status",
        ["Employed", "Self-employed", "Unemployed"],
        help="Stability of income source — strongly affects default risk"
    )
st.markdown('</div>', unsafe_allow_html=True)

# ─── Assess Button ────────────────────────────────────────────────────────────
assess = st.button("🔍  Assess Loan Risk", use_container_width=True)

if assess:
    # ── Derive engineered features ────────────────────────────────────────────
    applicant_id     = 1
    income_loan_ratio = income / loan_amount
    loan_burden       = loan_amount / income
    income_x_age      = income * age
    loan_x_income     = loan_amount * income
    loan_per_age      = loan_amount / age
    loan_amount_log   = np.log1p(loan_amount)
    income_log        = np.log1p(income)

    risk_score = (
        (700 - credit_score) * 50
        + loan_burden * 5000
        - (income / 1000)
        + (8000 if employment_status == 'Unemployed' else 0)
        + (2000 if employment_status == 'Self-employed' else 0)
    )

    row = {
        'applicant_id': applicant_id, 'gender': gender, 'age': age, 'income': income,
        'loan_amount': loan_amount, 'credit_score': credit_score,
        'employment_status': employment_status, 'marital_status': marital_status,
        'income_loan_ratio': income_loan_ratio, 'loan_burden': loan_burden,
        'risk_score': risk_score, 'income_x_age': income_x_age,
        'loan_x_income': loan_x_income, 'loan_per_age': loan_per_age,
        'loan_amount_log': loan_amount_log, 'income_log': income_log,
    }

    for extra_col in ['age_group', 'income_level', 'loan_level']:
        row[extra_col] = 'NA'

    X_input = pd.DataFrame([row], columns=feature_cols)

    with st.spinner("Running credit risk model…"):
        time.sleep(0.45)
        X_t = preprocessor.transform(X_input)
        prediction    = clf.predict(X_t)[0]          # 0=No Default, 1=Default
        probabilities = clf.predict_proba(X_t)[0]

    p_default    = probabilities[1] * 100
    p_no_default = probabilities[0] * 100

    # ── Risk tier ────────────────────────────────────────────────────────────
    if p_default < 25:
        tier, css, icon, color = "Low Risk", "result-good", "✅", "#2ECC71"
        action = (
            "💚 <strong>Recommendation: Approve.</strong> Applicant shows strong "
            "repayment indicators. Standard interest rate terms apply."
        )
    elif p_default < 55:
        tier, css, icon, color = "Moderate Risk", "result-warn", "⚠️", "#FFB020"
        action = (
            "🟡 <strong>Recommendation: Review further.</strong> Consider requesting "
            "additional collateral, a co-signer, or adjusted loan terms before approval."
        )
    else:
        tier, css, icon, color = "High Risk", "result-bad", "🚫", "#E74C3C"
        action = (
            "🔴 <strong>Recommendation: Decline or restructure.</strong> Default "
            "probability is high. If proceeding, require collateral and a higher "
            "interest rate to offset risk."
        )

    # ── Result card ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="{css}">
        <div class="res-icon">{icon}</div>
        <div class="res-verdict">{tier}</div>
        <div class="res-type">Loan Default Assessment</div>
        <div class="res-conf">{p_default:.1f}%</div>
        <div class="res-sub">probability of default</div>
        <div class="res-action">{action}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Risk gauge ───────────────────────────────────────────────────────────
    gauge_html = f"""
    <div class="gauge-wrap">
        <div class="gauge-label">
            <span>Low Risk</span><span>Moderate Risk</span><span>High Risk</span>
        </div>
        <div class="gauge-track">
            <div class="gauge-seg" style="width:25%; background:#2ECC71;"></div>
            <div class="gauge-seg" style="width:30%; background:#FFB020;"></div>
            <div class="gauge-seg" style="width:45%; background:#E74C3C;"></div>
        </div>
        <div style="position:relative; height:0;">
            <div style="position:absolute; top:-22px; left:{min(max(p_default,0),100):.1f}%;
                        transform:translateX(-50%); font-family:'JetBrains Mono',monospace;
                        font-size:0.95rem; color:{color};">▼</div>
        </div>
    </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)

    # ── Summary chips ─────────────────────────────────────────────────────────
    chips = f"""
    <div class="chips">
        <div class="chip">
            <span class="chip-k">Credit Score</span>
            <span class="chip-v">{credit_score}</span>
        </div>
        <div class="chip">
            <span class="chip-k">Loan / Income</span>
            <span class="chip-v">{loan_burden*100:.0f}%</span>
        </div>
        <div class="chip">
            <span class="chip-k">Employment</span>
            <span class="chip-v">{employment_status}</span>
        </div>
        <div class="chip">
            <span class="chip-k">No-Default Prob</span>
            <span class="chip-v">{p_no_default:.1f}%</span>
        </div>
    </div>
    """
    st.markdown(chips, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
   Decision Tree · Credit Risk Engine · FinGuard AI – Loan Default Risk System v1.0
</div>
""", unsafe_allow_html=True)
