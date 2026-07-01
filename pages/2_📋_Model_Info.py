"""
Model Information Page — Performance Metrics, Feature Importance, Model Details
Updated with Boruta-selected features and new model coefficients, medical-grade design
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.predict import (
    BETAS, SCALE_SD, SCALE_MEAN, TIME_POINTS,
    CONT_VARS, BINARY_VARS, CAT_VARS_WITH_LEVELS,
    BASELINE_HAZARD, DEFAULT_INPUT, MODEL_COLS
)

st.set_page_config(page_title="Model Info — CKM Predictor", page_icon="", layout="wide")

st.markdown("""
<style>
    :root {
        --navy: #1B2A4A;
        --navy-light: #2C3E6B;
        --steel: #4A6FA5;
        --slate: #6B7F9E;
        --gray-50: #F7F8FA;
        --gray-100: #EDEFF2;
        --gray-200: #DDE0E6;
        --gray-300: #B8BFC8;
        --white: #FFFFFF;
        --red: #C44536;
        --green: #3A7D5C;
        --amber: #D4893B;
    }

    .stApp { background-color: var(--gray-50); }

    .page-header {
        background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
        margin: -1rem -1.5rem 1.5rem;
        padding: 1.75rem 2rem;
        color: var(--white);
    }
    .page-header h1 {
        font-size: 1.4rem; font-weight: 600; margin: 0;
    }
    .page-header .sub {
        font-size: 0.82rem; color: rgba(255,255,255,0.65); margin-top: 0.2rem;
    }

    .section-title {
        font-size: 0.85rem; font-weight: 600; color: var(--navy);
        text-transform: uppercase; letter-spacing: 0.06em;
        margin: 1.75rem 0 1rem; padding-bottom: 0.5rem;
        border-bottom: 1.5px solid var(--gray-200);
    }

    .metric-card {
        background: var(--white);
        border-radius: 8px;
        border: 1px solid var(--gray-200);
        padding: 1.25rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.6rem; font-weight: 700; color: var(--navy);
    }
    .metric-card .label {
        font-size: 0.72rem; color: var(--slate);
        text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500;
        margin-top: 0.2rem;
    }
    .metric-card .tag {
        display: inline-block;
        font-size: 0.62rem; padding: 0.1rem 0.4rem;
        border-radius: 3px; margin-top: 0.3rem;
    }
    .tag-green { background: #F2F9F5; color: var(--green); }

    .data-frame {
        background: var(--white);
        border-radius: 8px;
        border: 1px solid var(--gray-200);
        overflow: hidden;
    }

    .info-block {
        background: var(--white);
        border-radius: 8px;
        border: 1px solid var(--gray-200);
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        line-height: 1.6;
        color: #2C3E6B;
    }

    .app-footer {
        text-align: center; padding: 1.5rem 0 0.5rem;
        font-size: 0.7rem; color: var(--gray-300);
        border-top: 1px solid var(--gray-200); margin-top: 1.5rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-header">
    <h1>Model Documentation</h1>
    <div class="sub">Cox Proportional Hazards · Boruta Feature Selection · CHARLS Cohort</div>
</div>
""", unsafe_allow_html=True)

# ── Overview Metrics ───────────────────────────────────────
st.markdown('<div class="section-title">Performance Summary</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="value">Cox PH</div>
        <div class="label">Model</div>
        <span class="tag tag-green">Selected</span>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="value">0.809</div>
        <div class="label">C-index (Test Set)</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="value">11</div>
        <div class="label">Input Features</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="value">6,953</div>
        <div class="label">Total Samples</div>
    </div>
    """, unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("""
    <div class="metric-card">
        <div class="value">0.805 ± 0.019</div>
        <div class="label">CV C-index (5-fold)</div>
    </div>
    """, unsafe_allow_html=True)
with col_b:
    st.markdown("""
    <div class="metric-card">
        <div class="value">0.808</div>
        <div class="label">Train C-index</div>
    </div>
    """, unsafe_allow_html=True)
with col_c:
    st.markdown("""
    <div class="metric-card">
        <div class="value">3, 5, 8</div>
        <div class="label">Prediction Time Points (years)</div>
    </div>
    """, unsafe_allow_html=True)

# ── Cross-model Comparison ──
st.markdown('<div class="section-title">Model Comparison (Test Set)</div>', unsafe_allow_html=True)

comp_data = {
    "Model": ["Cox PH (selected)", "CV-CoxBoost", "CV-GlmNet", "GBM", "Ranger (RF)"],
    "C-index": [0.809, 0.789, 0.798, 0.812, 0.974],
    "Graf Score": [0.052, 0.055, 0.054, 0.071, 0.026],
    "LogLoss": [9.17, 4.31, 9.19, 12.71, 1.32],
}
comp_df = pd.DataFrame(comp_data)
st.dataframe(
    comp_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Model": st.column_config.TextColumn("Model"),
        "C-index": st.column_config.NumberColumn("C-index", format="%.3f"),
        "Graf Score": st.column_config.NumberColumn("Graf Score", format="%.3f"),
        "LogLoss": st.column_config.NumberColumn("LogLoss", format="%.2f"),
    }
)

st.caption("Cox PH selected as final model for interpretability and clinical parsimony. Ranger (RF) shows higher C-index but lacks coefficient transparency.")

# ── Brier Scores ──
st.markdown('<div class="section-title">Calibration: Brier Scores</div>', unsafe_allow_html=True)
brier_data = {
    "Time Point": ["3-Year", "5-Year", "8-Year"],
    "Train Brier": [0.042, 0.076, 0.094],
    "Train CI": ["(0.037–0.047)", "(0.070–0.081)", "(0.088–0.100)"],
    "Test Brier": [0.041, 0.076, 0.098],
    "Test CI": ["(0.034–0.049)", "(0.068–0.085)", "(0.089–0.107)"],
}
st.dataframe(
    pd.DataFrame(brier_data),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Time Point": st.column_config.TextColumn("Time Point"),
        "Train Brier": st.column_config.NumberColumn("Train Brier", format="%.3f"),
        "Test Brier": st.column_config.NumberColumn("Test Brier", format="%.3f"),
    }
)

# ── Model Coefficients ──
st.markdown('<div class="section-title">Model Coefficients</div>', unsafe_allow_html=True)

feature_labels = {
    "CCR": "Cre/CysC Ratio", "age": "Age", "bl_crp": "C-Reactive Protein",
    "bl_cysc": "Cystatin C", "bmi": "BMI",
    "gender1": "Sex (Male)", "hypertension1": "Hypertension",
    "lunge1": "Chronic Lung Disease", "marry1": "Married",
    "ckm_stage1": "CKM Stage 1", "ckm_stage2": "CKM Stage 2", "ckm_stage3": "CKM Stage 3",
    "edu2": "Education: Primary", "edu3": "Education: Middle School", "edu4": "Education: High School+",
}

beta_df = pd.DataFrame({
    "Feature": [feature_labels.get(k, k) for k in BETAS.keys()],
    "β Coefficient": [round(v, 4) for v in BETAS.values()],
    "HR (exp β)": [round(np.exp(v), 3) for v in BETAS.values()],
    "95% CI": ["—"] * len(BETAS),
    "p-value": ["—"] * len(BETAS),
})

st.dataframe(
    beta_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "β Coefficient": st.column_config.NumberColumn("β", format="%.4f"),
        "HR (exp β)": st.column_config.NumberColumn("HR", format="%.3f"),
    }
)

# ── Feature Importance via |β| ──
st.markdown('<div class="section-title">Feature Contribution (Cox |β|)</div>', unsafe_allow_html=True)

fi = {}
for k, v in BETAS.items():
    if k.startswith("ckm_stage"):
        group = "CKM Stage"
    elif k.startswith("edu"):
        group = "Education Level"
    else:
        group = feature_labels.get(k, k)
    fi[group] = max(fi.get(group, 0), abs(v))

fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
fi_df = pd.DataFrame(fi_sorted, columns=["Feature", "|β|"])

import plotly.express as px
fig = px.bar(
    fi_df, x="|β|", y="Feature", orientation="h",
    color="|β|", color_continuous_scale=["#E8EEF6", "#4A6FA5"],
    title="", text="|β|",
)
fig.update_layout(
    yaxis={"categoryorder": "total ascending"},
    xaxis_title="Absolute Coefficient |β|",
    height=400, plot_bgcolor="white",
    margin=dict(l=0, r=0, t=10, b=0),
    font=dict(family="system-ui"),
)
fig.update_traces(texttemplate="%{x:.3f}", textposition="outside")
st.plotly_chart(fig, use_container_width=True)

# ── Boruta Feature Selection ──
st.markdown('<div class="section-title">Boruta Feature Selection</div>', unsafe_allow_html=True)

boruta_data = [
    ("age", 1.00, 23.03, "Confirmed"), ("bl_crp", 1.00, 6.07, "Confirmed"),
    ("bl_cysc", 1.00, 9.18, "Confirmed"), ("ckm_stage", 1.00, 9.11, "Confirmed"),
    ("gender", 1.00, 7.68, "Confirmed"), ("lunge", 1.00, 7.31, "Confirmed"),
    ("marry", 1.00, 8.36, "Confirmed"), ("edu", 0.99, 5.04, "Confirmed"),
    ("CCR", 0.94, 3.83, "Confirmed"), ("hypertension", 0.93, 3.43, "Confirmed"),
    ("bmi", 0.85, 3.26, "Confirmed"),
    ("diabetes", 0.35, 1.72, "Tentative"), ("drinkl", 0.37, 1.50, "Tentative"),
    ("smoken", 0.36, 1.75, "Tentative"),
    ("bl_hbalc", 0.03, 0.81, "Rejected"), ("bl_hgb", 0.19, 1.29, "Rejected"),
    ("bl_ua", 0.01, 0.78, "Rejected"), ("bl_wbc", 0.01, 0.18, "Rejected"),
    ("hhcperc", 0.01, 0.21, "Rejected"), ("mwaist", 0.00, 0.26, "Rejected"),
    ("rural", 0.01, 0.09, "Rejected"), ("sleep", 0.02, 0.12, "Rejected"),
    ("tyg", 0.01, 1.03, "Rejected"),
]

boruta_df = pd.DataFrame(boruta_data, columns=["Variable", "Norm Hits", "Median Imp.", "Decision"])
st.dataframe(boruta_df, use_container_width=True, hide_index=True)

st.markdown("""
<div class="info-block">
<strong>Boruta Algorithm:</strong> 23 candidate variables tested via 100 shadow permutations.
<strong>11 confirmed</strong> (normHits ≥ 0.80), <strong>3 tentative</strong> (0.20–0.80), <strong>9 rejected</strong> (≤ 0.20).
Sex, Cystatin C, CKM stage, and marital status show the highest median importance.
</div>
""", unsafe_allow_html=True)

# ── Preprocessing ──
st.markdown('<div class="section-title">Data Preprocessing</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-block">
<strong>Pipeline</strong><br>
① Missing values imputed: continuous → median, categorical → mode<br>
② Continuous variables standardized: (x − μ) / σ<br>
③ Categorical variables: dummy coding (reference level omitted)<br>
④ Cox PH: ties = "efron"<br>
⑤ Feature selection: Boruta (11 confirmed from 23 candidates)
</div>
""", unsafe_allow_html=True)

st.markdown("**Scaling Parameters**")
sd_df = pd.DataFrame({
    "Variable": list(SCALE_SD.keys()),
    "Std Dev": [round(v, 4) for v in SCALE_SD.values()],
    "Mean": [round(SCALE_MEAN.get(v, 0), 4) for v in SCALE_SD.keys()],
})
st.dataframe(sd_df, use_container_width=True, hide_index=True)

st.markdown("**Baseline Hazard & Survival**")
h0_display = pd.DataFrame({
    "Time (yr)": TIME_POINTS,
    "H₀(t)": [f"{BASELINE_HAZARD[t]:.6f}" for t in TIME_POINTS],
    "S₀(t)": [f"{np.exp(-BASELINE_HAZARD[t]):.4f}" for t in TIME_POINTS],
})
st.dataframe(h0_display, use_container_width=True, hide_index=True)

# ── Variable Descriptions ──
st.markdown('<div class="section-title">Variable Dictionary</div>', unsafe_allow_html=True)
var_desc = [
    ["Age", "Continuous", "years", "45–101", "Age at baseline examination"],
    ["BMI", "Continuous", "kg/m²", "10–60", "Body mass index"],
    ["CRP", "Continuous", "mg/L", "0.01–130", "C-reactive protein, inflammation"],
    ["Cystatin C", "Continuous", "mg/L", "0.30–9.0", "Kidney function marker"],
    ["CCR", "Continuous", "ratio", "0.20–2.5", "Creatinine/cystatin C, sarcopenia proxy"],
    ["Sex", "Binary", "—", "0/1", "0 = Female, 1 = Male"],
    ["Education", "Ordinal", "—", "1–4", "1=None, 2=Primary, 3=Middle, 4=High School+"],
    ["Marital Status", "Binary", "—", "0/1", "0 = Other, 1 = Married"],
    ["Hypertension", "Binary", "—", "0/1", "Self-reported diagnosis"],
    ["Lung Disease", "Binary", "—", "0/1", "Self-reported chronic lung disease"],
    ["CKM Stage", "Ordinal", "—", "0–3", "AHA CKM syndrome stage (0=No risk to 3=Subclinical CVD)"],
]
st.dataframe(
    pd.DataFrame(var_desc, columns=["Variable", "Type", "Unit", "Range", "Description"]),
    use_container_width=True, hide_index=True,
)

# ── Footer ──
st.markdown("""
<div class="app-footer">
    <strong>CKM Mortality Risk Predictor v2.0</strong>
    &nbsp;·&nbsp; Cox PH (Boruta-selected)
    &nbsp;·&nbsp; CHARLS Cohort N = 6,953
</div>
""", unsafe_allow_html=True)
