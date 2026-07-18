"""
Model Information Page — Performance Metrics, Feature Importance, Model Details
Final 15-variable Cox PH model with raw continuous variables
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.predict import (
    BETAS, SCALE_SD, SCALE_MEAN, TIME_POINTS,
    CONT_VARS, BINARY_VARS, CAT_VARS_WITH_LEVELS,
    BASELINE_HAZARD, DEFAULT_INPUT
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
        <div class="value">0.785</div>
        <div class="label">C-index (Test Set)</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="value">15</div>
        <div class="label">Coefficients</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="value">4,658</div>
        <div class="label">Training Samples</div>
    </div>
    """, unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
    <div class="metric-card">
        <div class="value">0.805</div>
        <div class="label">CV C-index (5-fold)</div>
    </div>
    """, unsafe_allow_html=True)
with col_b:
    st.markdown("""
    <div class="metric-card">
        <div class="value">0.808</div>
        <div class="label">Train Set C-index</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Model Coefficients ──────────────────────────────────────
st.markdown('<div class="section-header">📊 Model Coefficients (β / log(HR))</div>', unsafe_allow_html=True)

feature_labels = {
    "age": "Age", "bl_crp": "C-Reactive Protein", "bl_cysc": "Cystatin C",
    "bmi": "BMI", "CCR": "Cre/CysC Ratio",
    "gender1": "Gender (Male)", "hypertension1": "Hypertension",
    "lunge1": "Lung Disease", "marry1": "Married",
    "ckm_stage1": "CKM Stage 1", "ckm_stage2": "CKM Stage 2",
    "ckm_stage3": "CKM Stage 3",
    "edu2": "Education: Junior high", "edu3": "Education: Senior high",
    "edu4": "Education: College or above",
}

feature_types = {
    "age": "Continuous (raw)", "bl_crp": "Continuous (raw)",
    "bl_cysc": "Continuous (raw)", "bmi": "Continuous (raw)",
    "CCR": "Continuous (raw)",
    "gender1": "Binary (ref: Female)", "hypertension1": "Binary (ref: No)",
    "lunge1": "Binary (ref: No)", "marry1": "Binary (ref: Other)",
    "ckm_stage1": "Categorical (ref: Stage 0)", "ckm_stage2": "Categorical (ref: Stage 0)",
    "ckm_stage3": "Categorical (ref: Stage 0)",
    "edu2": "Categorical (ref: Elementary or below)",
    "edu3": "Categorical (ref: Elementary or below)",
    "edu4": "Categorical (ref: Elementary or below)",
}

beta_df = pd.DataFrame({
    "Variable": list(BETAS.keys()),
    "Label": [feature_labels.get(k, k) for k in BETAS.keys()],
    "Beta": [round(v, 6) for v in BETAS.values()],
    "HR": [f"{np.exp(v):.4f}" for v in BETAS.values()],
    "Type": [feature_types.get(k, "") for k in BETAS.keys()],
})

st.dataframe(
    beta_df[["Label", "Beta", "HR", "Type"]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Beta": st.column_config.NumberColumn(format="%.6f"),
        "HR": st.column_config.TextColumn(),
        "Type": st.column_config.TextColumn(),
    }
)

st.markdown("""
**Note**: Continuous predictors use raw (unscaled) values. HR for continuous variables 
represents the hazard ratio per unit increase. For example, HR for Age = exp(0.0815) = 1.085 
per additional year.
""")

# ── Feature Importance ──────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">🔍 Feature Importance Ranking</div>', unsafe_allow_html=True)

# For raw variables, importance = |β × SD| (standardized coefficient)
feature_importance = {}
for k, v in BETAS.items():
    if k in CONT_VARS:
        # For continuous: importance = |β × SD| (effect of 1 SD change)
        group = feature_labels.get(k, k)
        importance = abs(v * SCALE_SD.get(k, 1))
    elif k in ["gender1", "hypertension1", "lunge1", "marry1"]:
        group = feature_labels.get(k, k)
        importance = abs(v)
    elif k.startswith("ckm_stage"):
        group = "CKM Stage"
        importance = max(importance, abs(v)) if group in feature_importance else abs(v)
    elif k.startswith("edu"):
        group = "Education"
        importance = max(importance, abs(v)) if group in feature_importance else abs(v)
    else:
        group = feature_labels.get(k, k)
        importance = abs(v)
    
    if group in feature_importance:
        feature_importance[group] = max(feature_importance[group], importance)
    else:
        feature_importance[group] = importance

fi_sorted = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
fi_df = pd.DataFrame(fi_sorted, columns=["Feature", "Importance (|β×SD|)"])

st.dataframe(fi_df, use_container_width=True, hide_index=True)

# Bar chart
import plotly.express as px
fig = px.bar(
    fi_df,
    x="Importance (|β×SD|)",
    y="Feature",
    orientation="h",
    color="Importance (|β×SD|)",
    color_continuous_scale="Reds",
    title="Feature Importance (Standardized |β×SD|)",
    text="Importance (|β×SD|)",
)
fig.update_layout(
    yaxis={"categoryorder": "total ascending"},
    xaxis_title="|β × SD|",
    height=500,
    plot_bgcolor="white",
)
fig.update_traces(texttemplate="%{x:.4f}")
st.plotly_chart(fig, use_container_width=True)


# ── Preprocessing ───────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">⚙️ Data Preprocessing</div>', unsafe_allow_html=True)

st.markdown("""
**Preprocessing steps applied in the final model:**

1. **Missing Value Imputation**: Continuous variables → median; Categorical → mode
2. **Continuous Variables**: Used as raw (unscaled) values directly in the model
3. **Binary Variables**: 0/1 coding (0 = reference category)
4. **Categorical Encoding**: Dummy coding with reference levels
   - Education: reference = Elementary or below (level 1)
   - CKM Stage: reference = Stage 0
5. **Cox PH Model**: `ties="efron"` for handling tied event times

**Note**: The R model pipeline uses `scale(center=FALSE, scale=TRUE)` internally, 
which divides continuous variables by their training-set SD. The coefficients 
extracted in this webapp have been pre-divided by the SD, so they operate on 
**raw** variable values directly.
""")

st.markdown("**Training-Set Descriptive Statistics (for reference):**")
stats_df = pd.DataFrame({
    "Variable": list(SCALE_SD.keys()),
    "Mean (μ)": [round(SCALE_MEAN[v], 4) for v in SCALE_SD.keys()],
    "Std Dev (σ)": [round(v, 4) for v in SCALE_SD.values()],
})
st.dataframe(stats_df, use_container_width=True, hide_index=True)

st.markdown("**Baseline Cumulative Hazard H₀(t) and Survival S₀(t):**")
h0_df = pd.DataFrame({
    "Time Point": TIME_POINTS,
    "Cumulative Baseline Hazard H₀(t)": [f"{BASELINE_HAZARD[t]:.15f}" for t in TIME_POINTS],
    "Baseline Survival S₀(t)": [f"{np.exp(-BASELINE_HAZARD[t]):.6f}" for t in TIME_POINTS],
})
st.dataframe(h0_df, use_container_width=True, hide_index=True)

st.markdown("""
**Survival function**: S(t|X) = exp[−H₀(t) × exp(βᵀX)]

**Linear predictor**: LP = β₁×Age + β₂×Gender + β₃×CystatinC + β₄×Edu2 + β₅×Edu3 
+ β₆×Edu4 + β₇×CRP + β₈×Married + β₉×Lung + β₁₀×Hypertension 
+ β₁₁×CCR + β₁₂×CKM1 + β₁₃×CKM2 + β₁₄×CKM3 + β₁₅×BMI
""")


# ── Time-dependent Discrimination ───────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">📈 Time-dependent Discrimination (Test Set)</div>', unsafe_allow_html=True)

auc_df = pd.DataFrame({
    "Time Point": ["3 years", "5 years", "8 years"],
    "AUROC": [0.782, 0.789, 0.803],
    "95% CI": ["(0.730–0.833)", "(0.755–0.819)", "(0.775–0.829)"],
    "AUPR": [0.247, 0.358, 0.456],
    "AUPR 95% CI": ["(0.193–0.321)", "(0.291–0.422)", "(0.391–0.519)"],
})
st.dataframe(auc_df, use_container_width=True, hide_index=True)


# ── Variable Descriptions ───────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">📖 Variable Descriptions</div>', unsafe_allow_html=True)

var_desc_df = pd.DataFrame([
    ["age", "Age", "Continuous", "45–101 years"],
    ["bmi", "Body Mass Index", "Continuous", "10–60 kg/m²"],
    ["bl_crp", "C-Reactive Protein", "Continuous", "0.01–130 mg/L"],
    ["bl_cysc", "Cystatin C", "Continuous", "0.30–9.0 mg/L"],
    ["CCR", "Creatinine/Cystatin C Ratio", "Continuous", "0.20–2.5"],
    ["gender", "Gender", "Categorical", "0=Female, 1=Male"],
    ["edu", "Education", "Categorical", "1=Elementary or below, 2=Junior high, 3=Senior high, 4=College or above"],
    ["marry", "Marital Status", "Categorical", "0=Other, 1=Married"],
    ["hypertension", "Hypertension", "Categorical", "0=No, 1=Yes"],
    ["lunge", "Lung Disease", "Categorical", "0=No, 1=Yes"],
    ["ckm_stage", "CKM Stage", "Categorical", "0–3"],
], columns=["Variable", "Description", "Type", "Range"])

st.dataframe(var_desc_df, use_container_width=True, hide_index=True)


# ── Research Info ───────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">📚 Research Information</div>', unsafe_allow_html=True)

st.markdown("""
**Data Source**: CHARLS (China Health and Retirement Longitudinal Study) cohort

**Population**: Chinese adults aged 45 and older

**Sample Size**: 6,953 participants (Training: 4,658 + Test: 2,295)

**Outcome**: All-cause mortality (14% event rate over mean 7.7-year follow-up)

**Prediction Time Points**: 3-year, 5-year, and 8-year survival probabilities

**Model Selection**: Cox Proportional Hazards model after Boruta feature selection (11 raw features → 15 model terms)

**Model Performance**:
- 5-Fold CV C-index: 0.805
- Train C-index: 0.808
- Test C-index: 0.785
- Test Brier Score: 0.057
- Test Calibration α: 1.066, β: 0.952
- Test D-calibration: 2.446

**Citation**: Please cite the relevant research paper when using this tool.
""")
