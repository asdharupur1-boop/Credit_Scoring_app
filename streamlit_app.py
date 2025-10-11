import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(
    page_title="Credit Scoring AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------ LOGOS FUNCTION ------------------------
def get_logo_svg(platform):
    logos = {
        'linkedin': '''
        <svg width="20" height="20" viewBox="0 0 24 24" fill="#0077B5">
            <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
        </svg>
        ''',
        'github': '''
        <svg width="20" height="20" viewBox="0 0 24 24" fill="#333">
            <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
        </svg>
        ''',
        'email': '''
        <svg width="20" height="20" viewBox="0 0 24 24" fill="#D44638">
            <path d="M12 12.713l-11.985-9.713h23.97l-11.985 9.713zm0 2.574l-12-9.725v15.438h24v-15.438l-12 9.725z"/>
        </svg>
        ''',
    }
    return logos.get(platform.lower(), '')

# ------------------------ SIDEBAR ------------------------
with st.sidebar:
    st.title("🏦 Credit Scoring AI")
    st.markdown("---")
    st.subheader("👨‍💻 Developed by Ayush Shukla")
    st.markdown("**Data Scientist & ML Engineer**")
    st.markdown("---")
    st.subheader("🌐 Connect With Me")

    contact_links = [
        {"platform": "LinkedIn", "url": "https://www.linkedin.com/in/ayush-shukla-890072337/", "logo": get_logo_svg('linkedin'), "color": "#0077B5"},
        {"platform": "GitHub", "url": "https://github.com/asdharupur1-boop/Credit_Scoring_app", "logo": get_logo_svg('github'), "color": "#333333"},
        {"platform": "Email", "url": "Asdharupur1@gmail.com.com", "logo": get_logo_svg('email'), "color": "#D44638"},
    ]

    for c in contact_links:
        st.markdown(f"""
        <div style="display:flex;align-items:center;margin:10px 0;padding:8px;border-radius:5px;background-color:{c['color']}10;">
            <div style="margin-right:10px;">{c['logo']}</div>
            <a href="{c['url']}" target="_blank" style="color:{c['color']};text-decoration:none;font-weight:500;">
                {c['platform']}
            </a>
        </div>
        """, unsafe_allow_html=True)

# ------------------------ CSS ------------------------
st.markdown("""
<style>
.main-header { font-size:2.5rem; color:#1f77b4; text-align:center; margin-bottom:1rem; }
.prediction-card { background:linear-gradient(45deg,#667eea 0%,#764ba2 100%); color:white;
    padding:20px; border-radius:15px; margin:10px 0; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ------------------------ HEADER ------------------------
st.markdown('<h1 class="main-header">🏦 AI-Powered Credit Scoring System</h1>', unsafe_allow_html=True)

# ------------------------ MODEL LOADING ------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model/best_model.pkl')
    except Exception as e:
        st.error(f"Model load error: {e}")
        model = None

    try:
        scaler = joblib.load('model/feature_scaler.pkl')
    except Exception as e:
        st.warning(f"Scaler load warning: {e}")
        scaler = None

    return model, scaler

# ------------------------ ANALYTICS FUNCTIONS ------------------------
def analyze_risk_factors(data):
    risk_factors = []
    if data.get('credit_score', 0) < 580: risk_factors.append("Low credit score (<580)")
    if data.get('debt_to_income_ratio', 0) > 0.5: risk_factors.append("High DTI (>50%)")
    if data.get('number_of_late_payments', 0) > 5: risk_factors.append("Many late payments (>5)")
    if data.get('credit_utilization', 0) > 0.7: risk_factors.append("High utilization (>70%)")
    if data.get('has_previous_default', 0) == 1: risk_factors.append("Previous default")
    if data.get('employment_length', 0) < 2: risk_factors.append("Short job history (<2 yrs)")
    return risk_factors

def get_feature_importance(data):
    return {
        'Credit Score': data.get('credit_score', 0) / 850,
        'Debt-to-Income Ratio': data.get('debt_to_income_ratio', 0),
        'Late Payments': data.get('number_of_late_payments', 0) / 10,
        'Income Level': 1 - (data.get('income', 0) / 200000),
        'Credit Utilization': data.get('credit_utilization', 0),
    }

# ------------------------ PREDICTION ------------------------
def predict_credit(data, model, scaler):
    df = pd.DataFrame([data])
    numerical_features = ['age', 'income', 'employment_length', 'loan_amount',
                          'credit_score', 'debt_to_income_ratio', 'years_at_residence',
                          'number_of_credit_lines', 'number_of_late_payments', 'credit_utilization']
    existing_features = [f for f in numerical_features if f in df.columns]
    df_scaled = df.copy()

    if scaler is not None and len(existing_features) > 0:
        try:
            df_scaled[existing_features] = scaler.transform(df[existing_features])
        except Exception as e:
            st.warning(f"Scaler transform skipped: {e}")

    try:
        if hasattr(model, "predict_proba"):
            default_prob = model.predict_proba(df_scaled)[0, 1]
        elif hasattr(model, "decision_function"):
            default_prob = 1 / (1 + np.exp(-model.decision_function(df_scaled)[0]))
        else:
            default_prob = float(model.predict(df_scaled)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

    default_prob = max(0, min(1, float(default_prob)))
    credit_score = int(300 + (1 - default_prob) * 500)
    credit_score = max(300, min(850, credit_score))

    if credit_score >= 800:
        rating = "Excellent"; color = "#27ae60"
    elif credit_score >= 740:
        rating = "Very Good"; color = "#2ecc71"
    elif credit_score >= 670:
        rating = "Good"; color = "#f39c12"
    elif credit_score >= 580:
        rating = "Fair"; color = "#e67e22"
    else:
        rating = "Poor"; color = "#e74c3c"

    decision = "APPROVED" if default_prob < 0.5 else "REJECTED"
    risk = "Low" if default_prob < 0.3 else "Medium" if default_prob < 0.7 else "High"

    return {
        'default_probability': round(default_prob, 3),
        'credit_score': credit_score,
        'rating': rating,
        'rating_color': color,
        'decision': decision,
        'risk_level': risk,
        'risk_factors': analyze_risk_factors(data),
        'feature_importance': get_feature_importance(data)
    }

# ------------------------ MAIN APP ------------------------
model, scaler = load_model()

if model:
    tab1, tab2 = st.tabs(["🎯 Credit Assessment", "📊 Prediction Details"])

    with tab1:
        st.subheader("Enter Applicant Information")

        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 70, 35)
            income = st.number_input("Annual Income ($)", 10000, 500000, 65000, 5000)
            employment = st.slider("Employment Years", 0, 40, 5)
            residence = st.slider("Years at Residence", 0, 20, 3)
        with col2:
            credit_score = st.slider("Credit Score", 300, 850, 720)
            loan_amount = st.number_input("Loan Amount ($)", 1000, 100000, 15000, 1000)
            dti = st.slider("Debt-to-Income Ratio", 0.1, 0.8, 0.35, 0.05)
            credit_util = st.slider("Credit Utilization", 0.1, 0.9, 0.4, 0.05)

        credit_lines = st.slider("Number of Credit Lines", 1, 20, 8)
        late_payments = st.slider("Late Payments (last 2 years)", 0, 10, 2)
        previous_default = st.selectbox("Previous Default", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        if st.button("🚀 Analyze Credit Application"):
            with st.spinner("Analyzing your application..."):
                data = {
                    'age': age, 'income': income, 'employment_length': employment,
                    'loan_amount': loan_amount, 'credit_score': credit_score,
                    'debt_to_income_ratio': dti, 'years_at_residence': residence,
                    'number_of_credit_lines': credit_lines, 'number_of_late_payments': late_payments,
                    'credit_utilization': credit_util, 'has_previous_default': previous_default
                }

                result = predict_credit(data, model, scaler)
                if result:
                    st.success("✅ Analysis Complete!")
                    st.markdown("---")

                    colA, colB, colC = st.columns(3)
                    with colA:
                        st.markdown(f'<div class="prediction-card"><h2>📊 {result["credit_score"]}</h2><p>Credit Score</p></div>', unsafe_allow_html=True)
                    with colB:
                        st.markdown(f'<div class="prediction-card"><h2>{result["rating"]}</h2><p>Rating</p></div>', unsafe_allow_html=True)
                    with colC:
                        st.markdown(f'<div class="prediction-card"><h2>{result["decision"]}</h2><p>Decision</p></div>', unsafe_allow_html=True)

    with tab2:
        st.subheader("📊 Prediction Breakdown")
        st.write("### Risk Factors:")
        if model:
            if result:
                st.write(result["risk_factors"])
else:
    st.error("❌ Model not loaded. Please ensure model files exist in the `model/` folder.")

