"""
Model Information Page — Performance Metrics, Feature Importance, Model Details
Updated with Boruta-selected features and new model coefficients
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

st.set_page_config(page_title="Model Info", page_icon="📋", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #7f8c8d;
        margin-top: 0.25rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ecf0f1;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📋 Model Information")
st.markdown("---")

# ── Model Overview ──────────────────────────────────────────
st.markdown('<div class="section-header">📌 Model Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="value">Cox PH</div>
        <div class="label">Model Type</div>
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
        <div class="label">Raw Features</div>
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
        <div class="value">0.7905</div>
        <div class="label">CV C-index (5-fold)</div>
    </div>
    """, unsafe_allow_html=True)
with col_b:
    st.markdown("""
    <div class="metric-card">
        <div class="value">0.8046</div>
        <div class="label">Train C-index</div>
    </div>
    """, unsafe_allow_html=True)
with col_c:
    st.markdown("""
    <div class="metric-card">
        <div class="value">0.8085</div>
        <div class="label">Test C-index</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Model Coefficients ──────────────────────────────────────
st.markdown('<div class="section-header">📊 Model Coefficients (β / log(HR))</div>', unsafe_allow_html=True)

feature_labels = {
    "CCR": "Cre/CysC Ratio",
    "age": "Age", "bl_crp": "C-Reactive Protein", "bl_cysc": "Cystatin C",
    "bmi": "BMI",
    "gender1": "Gender (Male)", "hypertension1": "Hypertension",
    "lunge1": "Lung Disease", "marry1": "Married",
    "ckm_stage1": "CKM Stage 1", "ckm_stage2": "CKM Stage 2", "ckm_stage3": "CKM Stage 3",
    "edu2": "Education: Primary", "edu3": "Education: Middle School", "edu4": "Education: High School+",
}

beta_df = pd.DataFrame({
    "Variable": list(BETAS.keys()),
    "Label": [feature_labels.get(k, k) for k in BETAS.keys()],
    "Beta (β)": [round(v, 6) for v in BETAS.values()],
    "HR (exp β)": [round(np.exp(v), 4) for v in BETAS.values()],
})

beta_df["Type"] = ""
for i, k in enumerate(beta_df["Variable"]):
    if k in CONT_VARS:
        beta_df.loc[i, "Type"] = "Continuous (scaled)"
    elif k.startswith("ckm_stage"):
        beta_df.loc[i, "Type"] = "CKM Stage (dummy, ref=0)"
    elif k.startswith("edu"):
        beta_df.loc[i, "Type"] = "Education (dummy, ref=None)"
    else:
        beta_df.loc[i, "Type"] = "Binary"

st.dataframe(
    beta_df[["Label", "Beta (β)", "HR (exp β)", "Type"]],
    use_container_width=True,
    hide_index=True,
)

# ── Feature Importance ──────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">🔍 Feature Importance Ranking</div>', unsafe_allow_html=True)

feature_importance = {}
for k, v in BETAS.items():
    if k.startswith("ckm_stage"):
        group = "CKM Stage"
    elif k.startswith("edu"):
        group = "Education"
    else:
        group = feature_labels.get(k, k)
    if group in feature_importance:
        feature_importance[group] = max(feature_importance[group], abs(v))
    else:
        feature_importance[group] = abs(v)

fi_sorted = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
fi_df = pd.DataFrame(fi_sorted, columns=["Feature", "|Beta|"])
st.dataframe(fi_df, use_container_width=True, hide_index=True)

import plotly.express as px
fig = px.bar(
    fi_df, x="|Beta|", y="Feature", orientation="h",
    color="|Beta|", color_continuous_scale="Reds",
    title="Feature Importance (Cox Coefficient Absolute Value)",
    text="|Beta|",
)
fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500, plot_bgcolor="white")
fig.update_traces(texttemplate="%{x:.4f}")
st.plotly_chart(fig, use_container_width=True)

# ── Boruta Feature Selection ────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">🎯 Boruta Feature Selection Results</div>', unsafe_allow_html=True)

boruta_data = [
    ("age", 1.00, 23.03, "Confirmed"),
    ("bl_crp", 1.00, 6.07, "Confirmed"),
    ("bl_cysc", 1.00, 9.18, "Confirmed"),
    ("ckm_stage", 1.00, 9.11, "Confirmed"),
    ("gender", 1.00, 7.68, "Confirmed"),
    ("lunge", 1.00, 7.31, "Confirmed"),
    ("marry", 1.00, 8.36, "Confirmed"),
    ("edu", 0.99, 5.04, "Confirmed"),
    ("CCR", 0.94, 3.83, "Confirmed"),
    ("hypertension", 0.93, 3.43, "Confirmed"),
    ("bmi", 0.85, 3.26, "Confirmed"),
    ("diabetes", 0.35, 1.72, "Tentative"),
    ("drinkl", 0.37, 1.50, "Tentative"),
    ("smoken", 0.36, 1.75, "Tentative"),
    ("bl_hbalc", 0.03, 0.81, "Rejected"),
    ("bl_hgb", 0.19, 1.29, "Rejected"),
    ("bl_ua", 0.01, 0.78, "Rejected"),
    ("bl_wbc", 0.01, 0.18, "Rejected"),
    ("hhcperc", 0.01, 0.21, "Rejected"),
    ("mwaist", 0.00, 0.26, "Rejected"),
    ("rural", 0.01, 0.09, "Rejected"),
    ("sleep", 0.02, 0.12, "Rejected"),
    ("tyg", 0.01, 1.03, "Rejected"),
]

boruta_df = pd.DataFrame(boruta_data, columns=["Variable", "Norm Hits", "Median Imp", "Decision"])
st.dataframe(boruta_df[["Variable", "Median Imp", "Norm Hits", "Decision"]], use_container_width=True, hide_index=True)

# ── Preprocessing ───────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">⚙️ Data Preprocessing</div>', unsafe_allow_html=True)

st.markdown("""
**Preprocessing steps applied in the original model:**

1. **Missing Value Imputation**: Continuous variables → median; Categorical → mode
2. **Continuous Scaling**: `scale(center=TRUE, scale=TRUE)` — each continuous variable standardized (mean=0, sd=1)
3. **Categorical Encoding**: Dummy coding for categorical variables (reference level dropped)
4. **Cox PH Model**: `ties="efron"` for handling tied event times
5. **Feature Selection**: Boruta algorithm — 11 confirmed features selected from 23 candidates
""")

st.markdown("**Continuous Variable Scaling Parameters:**")
sd_df = pd.DataFrame({
    "Variable": list(SCALE_SD.keys()),
    "Std Dev (SD)": [round(v, 6) for v in SCALE_SD.values()],
    "Mean": [round(SCALE_MEAN.get(v, 0), 6) for v in SCALE_SD.keys()],
})
st.dataframe(sd_df, use_container_width=True, hide_index=True)

st.markdown("**Baseline Survival Probabilities S₀(t):**")
h0_df = pd.DataFrame({
    "Time Point": TIME_POINTS,
    "Baseline Cumulative Risk H₀(t)": [BASELINE_HAZARD[t] for t in TIME_POINTS],
    "Baseline Survival S₀(t)": [f"{np.exp(-BASELINE_HAZARD[t]):.4f}" for t in TIME_POINTS],
})
st.dataframe(h0_df, use_container_width=True, hide_index=True)

# ── Variable Descriptions ───────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">📖 Variable Descriptions</div>', unsafe_allow_html=True)

var_desc_df = pd.DataFrame([
    ["age", "Age", "Continuous", "45-101 years"],
    ["bmi", "Body Mass Index", "Continuous", "10-60 kg/m²"],
    ["bl_crp", "C-Reactive Protein", "Continuous", "0.01-130 mg/L"],
    ["bl_cysc", "Cystatin C", "Continuous", "0.30-9.0 mg/L"],
    ["CCR", "Creatinine/Cystatin C Ratio", "Continuous", "0.20-2.5"],
    ["gender", "Gender", "Categorical", "0=Female, 1=Male"],
    ["edu", "Education", "Categorical", "1=None, 2=Primary, 3=Middle, 4=High School+"],
    ["marry", "Marital Status", "Categorical", "0=Other, 1=Married"],
    ["hypertension", "Hypertension", "Categorical", "0=No, 1=Yes"],
    ["lunge", "Lung Disease", "Categorical", "0=No, 1=Yes"],
    ["ckm_stage", "CKM Stage", "Categorical", "0-3"],
], columns=["Variable", "Description", "Type", "Range"])

st.dataframe(var_desc_df, use_container_width=True, hide_index=True)

# ── Research Info ───────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">📚 Research Information</div>', unsafe_allow_html=True)

st.markdown("""
**Data Source**: CHARLS (China Health and Retirement Longitudinal Study) cohort

**Population**: Chinese adults aged 45 and older

**Sample Size**: 6,953 participants (Training: 4,658 + Test: 2,295)

**Outcome**: All-cause mortality

**Prediction Time Points**: 3-year, 5-year, and 8-year survival probabilities

**Model Selection**: Cox Proportional Hazards model after Boruta feature selection (11 confirmed features from 23 candidates)

**Model Performance**:
- Train C-index: 0.8046 ± 0.0194
- 5-Fold CV C-index: 0.7905 ± 0.0236
- Test C-index: 0.8085
- Test Brier Score: 0.041

**Citation**: Please cite the relevant research paper when using this tool.
""")
