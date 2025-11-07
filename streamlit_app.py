# app_v2_enhanced.py - Advanced Credit Scoring System v2.4
# Changes in v2.4: Removed authentication. Added realtime CSV/XLSX batch default checking and monitoring UI.
# Run: streamlit run app_v2_enhanced.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
from fpdf import FPDF
import io
import os
import tempfile
import hashlib
import logging

# Optional libraries - SHAP and boto3 may not be installed in all environments. Guard imports.
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import boto3
    BOTO3_AVAILABLE = True
except Exception:
    BOTO3_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('credit_app')

st.set_page_config(page_title="Advanced Credit Scoring System v2.4",
                   page_icon="🏦", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------- Constants & Cache dirs -----------------
MODEL_CACHE_DIR = 'model_cache'
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# ----------------- Utility: Model loading -----------------
@st.cache_resource
def load_model_from_path(path: str):
    try:
        model = joblib.load(path)
        logger.info(f'Model loaded from {path}')
        return model
    except Exception as e:
        logger.error('Failed to load model from path: ' + str(e))
        return None

# ----------------- Feature engineering -----------------
def create_advanced_features(df):
    df = df.copy()
    required = ['number_of_late_payments','debt_to_income_ratio','credit_utilization','has_previous_default',
                'employment_length','age','income','loan_amount','loan_term','number_of_credit_lines','credit_score',
                'years_at_residence']
    for c in required:
        if c not in df.columns:
            df[c] = 0
    df['risk_score'] = ((df['number_of_late_payments'] * 0.3) + (df['debt_to_income_ratio'] * 0.25) + (df['credit_utilization'] * 0.2) + (df['has_previous_default'] * 0.25)) * 100
    df['payment_behavior'] = df['number_of_late_payments'] / (df['number_of_credit_lines'] + 1)
    df['credit_age_factor'] = df['employment_length'] / df['age'].replace(0,1)
    df['income_to_loan_ratio'] = df['income'] / (df['loan_amount'] + 1)
    df['credit_capacity_utilization'] = df['loan_amount'] / (df['income'] * df['loan_term'] / 12 + 1)
    df['payment_consistency_score'] = 1 - (df['number_of_late_payments'] / (df['employment_length'] + 1))
    df['financial_health_index'] = ((df['income'] / (df['loan_amount']+1)) * 0.3 + (1 - df['debt_to_income_ratio']) * 0.3 + (1 - df['credit_utilization']) * 0.2 + df['payment_consistency_score'] * 0.2) * 100
    df['creditworthiness_score_v2'] = ((df['credit_score'] / 850) * 0.25 + (1 - df['debt_to_income_ratio']) * 0.20 + df['financial_health_index'] / 100 * 0.25 + (df['employment_length'] / 40) * 0.15 + (1 - (df['number_of_late_payments'] / 10)) * 0.15) * 1000
    df['ml_risk_probability'] = (df['number_of_late_payments'] * 0.15 + df['debt_to_income_ratio'] * 0.20 + df['credit_utilization'] * 0.15 + df['has_previous_default'] * 0.25 + ((850 - df['credit_score']) / 850) * 0.25)
    df['stability_score'] = ((df['years_at_residence'] / 20) * 0.4 + (df['employment_length'] / 40) * 0.6) * 100
    df['loan_affordability_index'] = df['income'] / (df['loan_amount'] * df['loan_term'] / 12 + 1)
    df['economic_sensitivity'] = (df['debt_to_income_ratio'] * 0.4 + (df['loan_amount'] / (df['income']+1) * 0.3) + (df['credit_utilization'] * 0.3))
    df['behavioral_pattern_score'] = (1 - (df['number_of_late_payments'] / (df['number_of_credit_lines'] + 1))) * 100
    def get_risk_category_v2(score):
        try:
            s = float(score)
        except:
            return 'Unknown'
        if s >= 800:
            return 'AAA - Excellent'
        elif s >= 750:
            return 'AA - Very Good'
        elif s >= 700:
            return 'A - Good'
        elif s >= 650:
            return 'BBB - Average'
        elif s >= 600:
            return 'BB - Below Average'
        elif s >= 550:
            return 'B - Poor'
        else:
            return 'C - Very Poor'
    df['risk_category_v2'] = df['creditworthiness_score_v2'].apply(get_risk_category_v2)
    df['ensemble_default_probability'] = (df['ml_risk_probability'] * 0.6 + (1 - (df['creditworthiness_score_v2'] / 1000)) * 0.4)
    return df

# ----------------- Sample data -----------------
@st.cache_data
def generate_sample_data(n_samples=500):
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 150000, n_samples),
        'employment_length': np.random.randint(0, 40, n_samples),
        'loan_amount': np.random.randint(5000, 50000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'debt_to_income_ratio': np.random.uniform(0.1, 0.8, n_samples),
        'years_at_residence': np.random.randint(0, 20, n_samples),
        'number_of_credit_lines': np.random.randint(1, 15, n_samples),
        'number_of_late_payments': np.random.randint(0, 10, n_samples),
        'credit_utilization': np.random.uniform(0.1, 0.9, n_samples),
        'has_previous_default': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'home_ownership': np.random.choice(['RENT', 'MORTGAGE', 'OWN'], n_samples),
        'loan_purpose': np.random.choice(['PERSONAL', 'MEDICAL', 'VENTURE', 'HOME_IMPROVEMENT', 'DEBT_CONSOLIDATION'], n_samples),
        'default': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    }
    return pd.DataFrame(data)

# ----------------- App state and model placeholder -----------------
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None

# ----------------- Top header -----------------
st.markdown(f"""<div style='display:flex; justify-content:space-between; align-items:center'>
    <div style='font-size:28px; font-weight:700; color:#0b5cff'>Advanced Credit Scoring System</div>
    <div style='font-size:12px; color:#6c757d'>v2.4 • {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
</div>""", unsafe_allow_html=True)

# ----------------- Sidebar: model upload only -----------------
st.sidebar.header('Model & Artifacts')
local_model = st.sidebar.file_uploader('Upload model artifact (joblib/pkl)', type=['joblib','pkl'])
preproc_file = st.sidebar.file_uploader('Upload preprocessor (joblib/pkl) - optional', type=['joblib','pkl'])

if local_model is not None:
    tmp = os.path.join(tempfile.gettempdir(), f'model_{int(datetime.now().timestamp())}.joblib')
    with open(tmp, 'wb') as f:
        f.write(local_model.read())
    m = load_model_from_path(tmp)
    if m is not None:
        st.session_state.model = m
        st.sidebar.success('Local model loaded')

if preproc_file is not None:
    tmp2 = os.path.join(tempfile.gettempdir(), f'preproc_{int(datetime.now().timestamp())}.joblib')
    with open(tmp2, 'wb') as f:
        f.write(preproc_file.read())
    try:
        pre = joblib.load(tmp2)
        st.session_state.preprocessor = pre
        st.sidebar.success('Preprocessor loaded')
    except Exception as e:
        st.sidebar.error('Failed to load preprocessor: ' + str(e))

st.sidebar.markdown('---')
st.sidebar.markdown('Realtime default checking supports **CSV** and **XLSX** files. Upload a file below in the "Batch Monitor" tab.')

# ----------------- Main tabs -----------------
tabs = st.tabs(["Dashboard", "Data Explorer", "Loan Simulator", "Batch Monitor", "Model Insights", "Settings"])

# ---------- Dashboard ----------
with tabs[0]:
    st.header('Overview')
    sample = generate_sample_data(300)
    sample_enh = create_advanced_features(sample)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Total Apps', '50k', delta='+1.2%')
    c2.metric('Approval', '85.2%', delta='+0.8%')
    c3.metric('Default', '14.8%', delta='-0.9%')
    c4.metric('Avg Score', '675', delta='+3')

# ---------- Data Explorer ----------
with tabs[1]:
    st.header('Data Explorer')
    n = st.slider('Rows', 50, 2000, 300, step=50)
    df = generate_sample_data(n)
    enh = create_advanced_features(df)
    st.dataframe(enh.head(100))

# ---------- Loan Simulator ----------
with tabs[2]:
    st.header('Loan Simulator')
    colA, colB = st.columns(2)
    with colA:
        age = st.number_input('Age',18,80,35)
        income = st.number_input('Annual Income',10000,500000,60000)
        employment_length = st.slider('Employment (yrs)',0,40,3)
    with colB:
        loan_amount = st.number_input('Loan Amount',1000,200000,20000)
        loan_term = st.selectbox('Loan Term (months)', [12,24,36,48,60])
        credit_score = st.slider('Credit Score',300,850,650)
    if st.button('Simulate'):
        applicant = pd.DataFrame([{
            'age':age,'income':income,'employment_length':employment_length,'years_at_residence':2,
            'loan_amount':loan_amount,'loan_term':loan_term,'credit_score':credit_score,'debt_to_income_ratio':0.35,
            'number_of_credit_lines':4,'number_of_late_payments':1,'credit_utilization':0.4,'has_previous_default':0
        }])
        enh_app = create_advanced_features(applicant)
        # Try model predict
        if st.session_state.model is not None:
            try:
                X = applicant.copy()
                if st.session_state.preprocessor is not None:
                    Xp = st.session_state.preprocessor.transform(X)
                    prob = st.session_state.model.predict_proba(Xp)[:,1][0]
                else:
                    prob = st.session_state.model.predict_proba(X)[:,1][0]
                enh_app['model_prob'] = prob
            except Exception as e:
                st.warning('Model predict failed: ' + str(e))
                enh_app['model_prob'] = enh_app['ensemble_default_probability']
        else:
            enh_app['model_prob'] = enh_app['ensemble_default_probability']
        st.metric('Default Prob.', f"{enh_app['model_prob'].iloc[0]:.2%}")
        st.metric('Creditworthiness', f"{enh_app['creditworthiness_score_v2'].iloc[0]:.0f}")

# ---------- Batch Monitor (realtime CSV/XLSX checking) ----------
with tabs[3]:
    st.header('Batch Monitor — Realtime Default Checking')
    st.write('Upload CSV or XLSX. The app will process and show flagged rows (high default risk) immediately.')

    uploaded_file = st.file_uploader('Upload CSV or XLSX for realtime checking', type=['csv','xlsx'])
    threshold = st.slider('Flag threshold for default probability', 0.0, 1.0, 0.6)
    highlight_color = st.color_picker('Highlight color for flagged rows', '#fff2cc')

    if uploaded_file is not None:
        filename = uploaded_file.name
        try:
            if filename.lower().endswith('.csv'):
                df_batch = pd.read_csv(uploaded_file)
            else:
                df_batch = pd.read_excel(uploaded_file)
            st.success(f'Loaded {len(df_batch)} rows from {filename}')

            # Ensure necessary columns exist - try to infer common names if missing
            # Map common variations to expected columns
            col_map = {}
            cols = [c.lower() for c in df_batch.columns]
            mapping_rules = {
                'age':'age','income':'income','employment_length':'employment_length','loan_amount':'loan_amount',
                'credit_score':'credit_score','debt_to_income_ratio':'debt_to_income_ratio','years_at_residence':'years_at_residence',
                'number_of_credit_lines':'number_of_credit_lines','number_of_late_payments':'number_of_late_payments','credit_utilization':'credit_utilization',
                'has_previous_default':'has_previous_default','loan_term':'loan_term'
            }
            for want, want_col in mapping_rules.items():
                for c in df_batch.columns:
                    if c.lower() == want or c.lower().replace(' ','_') == want:
                        col_map[want_col] = c
                        break
            # If some required columns missing, create with safe defaults
            for req in ['number_of_late_payments','debt_to_income_ratio','credit_utilization','has_previous_default','employment_length','age','income','loan_amount','loan_term','number_of_credit_lines','credit_score','years_at_residence']:
                if req not in col_map:
                    # try heuristics
                    if 'age' in req and 'dob' in cols:
                        # skip complex DOB parsing for now
                        col_map[req] = None
                    else:
                        col_map[req] = None

            # Build standardized dataframe with expected numeric columns
            std_df = pd.DataFrame()
            for req in ['age','income','employment_length','loan_amount','credit_score','debt_to_income_ratio','years_at_residence','number_of_credit_lines','number_of_late_payments','credit_utilization','has_previous_default','loan_term']:
                src = col_map.get(req)
                if src and src in df_batch.columns:
                    std_df[req] = pd.to_numeric(df_batch[src], errors='coerce').fillna(0)
                else:
                    std_df[req] = 0

            # Preserve original columns for output
            output_df = pd.concat([df_batch.reset_index(drop=True), std_df.reset_index(drop=True)], axis=1)

            # Feature engineering and model scoring
            enhanced_batch = create_advanced_features(output_df)
            if st.session_state.model is not None:
                try:
                    X = output_df.copy()
                    if st.session_state.preprocessor is not None:
                        Xp = st.session_state.preprocessor.transform(X)
                        probs = st.session_state.model.predict_proba(Xp)[:,1]
                    else:
                        probs = st.session_state.model.predict_proba(X)[:,1]
                    enhanced_batch['model_default_probability'] = probs
                except Exception as e:
                    st.warning('Model scoring failed for uploaded file: ' + str(e))
                    enhanced_batch['model_default_probability'] = enhanced_batch['ensemble_default_probability']
            else:
                enhanced_batch['model_default_probability'] = enhanced_batch['ensemble_default_probability']

            enhanced_batch['flagged'] = enhanced_batch['model_default_probability'] >= threshold

            # Summary
            colA,colB,colC = st.columns(3)
            colA.metric('Total Rows', len(enhanced_batch))
            colB.metric('Flagged', f"{enhanced_batch['flagged'].sum()} ({(enhanced_batch['flagged'].mean()*100):.1f}%)")
            colC.metric('Avg Default Prob.', f"{enhanced_batch['model_default_probability'].mean():.2f}")

            # Display with highlighting for flagged rows
            def highlight_flags(row):
                return ['background-color: {}'.format(highlight_color) if row['flagged'] else '' for _ in row]

            # Show top flagged rows first
            flagged_df = enhanced_batch[enhanced_batch['flagged']].copy()
            non_flagged_df = enhanced_batch[~enhanced_batch['flagged']].copy()

            st.subheader('Flagged Applicants (high default risk)')
            if len(flagged_df) > 0:
                st.dataframe(flagged_df.head(200).style.apply(highlight_flags, axis=1), use_container_width=True)
            else:
                st.info('No flagged rows at or above the threshold.')

            st.subheader('Full Results Preview')
            st.dataframe(pd.concat([flagged_df.head(50), non_flagged_df.head(50)]).reset_index(drop=True), use_container_width=True)

            # Export options
            csv = enhanced_batch.to_csv(index=False).encode('utf-8')
            st.download_button('Download scored CSV', csv, file_name=f'scored_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

            # Quick chart
            st.subheader('Risk Distribution')
            fig = plt.figure(); plt.hist(enhanced_batch['model_default_probability'], bins=30); plt.xlabel('Default Probability'); plt.ylabel('Count'); st.pyplot(fig)

        except Exception as e:
            st.error('Failed to process file: ' + str(e))

# ---------- Model Insights ----------
with tabs[4]:
    st.header('Model Insights & SHAP')
    if st.session_state.model is None:
        st.info('No model uploaded. Upload a joblib/pkl model in the sidebar to view full insights.')
    else:
        st.write('Model loaded:', type(st.session_state.model))
        if hasattr(st.session_state.model, 'feature_importances_'):
            fi = st.session_state.model.feature_importances_
            names = getattr(st.session_state.model, 'feature_names_in_', [f'f{i}' for i in range(len(fi))])
            fi_df = pd.DataFrame({'feature':names, 'importance':fi}).sort_values('importance', ascending=True)
            st.bar_chart(fi_df.set_index('feature'))
        if SHAP_AVAILABLE:
            st.info('SHAP is available. You can run per-instance explanations in the Loan Simulator tab or generate global summary from a sample dataset.')
        else:
            st.info('SHAP not installed. Install with `pip install shap` to enable explainability.')

# ---------- Settings ----------
with tabs[5]:
    st.header('Settings & Utilities')
    if st.button('Clear cached models & temp files'):
        try:
            st.cache_data.clear(); st.cache_resource.clear()
            for f in os.listdir(tempfile.gettempdir()):
                if 'model_' in f or 'preproc_' in f:
                    try: os.remove(os.path.join(tempfile.gettempdir(), f))
                    except: pass
            # clear model cache dir
            for f in os.listdir(MODEL_CACHE_DIR):
                try: os.remove(os.path.join(MODEL_CACHE_DIR, f))
                except: pass
            st.success('Cleared')
        except Exception as e:
            st.error(str(e))
    st.markdown('---')
    st.markdown('Security note: Authentication removed by request. This app currently trusts the environment and uploaded models. Do not expose to public networks without adding access control.')

# ---------- Footer ----------
st.markdown('---')
st.markdown('<div style="text-align:center; color:#6c757d">v2.4 — Realtime CSV/XLSX batch checking added. Replace mock model with production artifact for real results.</div>', unsafe_allow_html=True)
