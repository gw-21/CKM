"""
CKM Mortality Risk Prediction System — Main Entry
Interactive Survival Prediction Web App Based on Cox Proportional Hazards Model
Updated: Boruta-selected features (11 inputs → 15 model columns)
"""

import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.predict import (
    predict_survival, get_survival_curve,
    get_baseline_survival, get_baseline_curve,
    TIME_POINTS, SCALE_SD, SCALE_MEAN, DEFAULT_INPUT
)
from utils.visualization import create_survival_plot, create_gauge

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="CKM Mortality Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS Styles ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #7f8c8d;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #34495e;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.25rem;
        border-bottom: 2px solid #ecf0f1;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .result-card h3 {
        font-size: 1rem;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    .result-card .value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .result-card .sub {
        font-size: 0.85rem;
        opacity: 0.75;
    }
    .risk-low {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        box-shadow: 0 4px 15px rgba(39, 174, 96, 0.4);
    }
    .risk-medium {
        background: linear-gradient(135deg, #e67e22 0%, #f39c12 100%);
        box-shadow: 0 4px 15px rgba(230, 126, 34, 0.4);
    }
    .risk-high {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        box-shadow: 0 4px 15px rgba(192, 57, 43, 0.4);
    }
    .info-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
    }
    .input-label {
        font-weight: 500;
        color: #2c3e50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        font-size: 1rem;
    }
    @media (max-width: 768px) {
        .result-card { padding: 1rem; }
        .result-card .value { font-size: 1.8rem; }
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Variable Info ───────────────────────────────────────────
VAR_INFO = {
    "age":              {"label": "Age",             "unit": "years",  "type": "continuous", "min": 45,  "max": 101, "step": 1,   "default": 60,  "help": "Patient age (45-101 years)"},
    "bmi":              {"label": "BMI",             "unit": "kg/m²",  "type": "continuous", "min": 10.0,  "max": 60.0, "step": 0.1, "default": 24.0,  "help": "Body Mass Index"},
    "bl_crp":           {"label": "C-Reactive Protein", "unit": "mg/L", "type": "continuous", "min": 0.01,"max": 130.0, "step": 0.1, "default": 2.8, "help": "C-reactive protein level"},
    "bl_cysc":          {"label": "Cystatin C",      "unit": "mg/L",   "type": "continuous", "min": 0.30,"max": 9.0, "step": 0.01,"default": 1.0, "help": "Serum cystatin C"},
    "cre_cyc_ratio":    {"label": "Cre/CysC Ratio (CCR)",  "unit": "",        "type": "continuous", "min": 0.20,"max": 2.5, "step": 0.01,"default": 0.79,"help": "Creatinine to cystatin C ratio"},
    "gender":           {"label": "Gender",           "unit": "",        "type": "categorical",
                         "options": {"0": "Female", "1": "Male"}, "default": "0"},
    "edu":              {"label": "Education",        "unit": "",        "type": "categorical",
                         "options": {"1": "None", "2": "Primary", "3": "Middle School", "4": "High School+"}, "default": "1"},
    "marry":            {"label": "Marital Status",   "unit": "",        "type": "categorical",
                         "options": {"0": "Other", "1": "Married"}, "default": "1"},
    "hypertension":     {"label": "Hypertension",     "unit": "",        "type": "categorical",
                         "options": {"0": "No", "1": "Yes"}, "default": "0"},
    "lunge":            {"label": "Lung Disease",     "unit": "",        "type": "categorical",
                         "options": {"0": "No", "1": "Yes"}, "default": "0"},
    "ckm_stage":        {"label": "CKM Stage",       "unit": "",        "type": "categorical",
                         "options": {"0": "Stage 0", "1": "Stage 1", "2": "Stage 2", "3": "Stage 3"}, "default": "2"},
}


def render_result_card(time_point, surv_prob, baseline_prob):
    """Render a single timepoint result card"""
    risk = 1 - surv_prob
    card_class = "result-card"
    if surv_prob >= 0.85:
        card_class += " risk-low"
    elif surv_prob >= 0.60:
        card_class += " risk-medium"
    else:
        card_class += " risk-high"

    baseline_text = f"Baseline S({time_point}yr) = {baseline_prob:.1%}"

    st.markdown(f"""
    <div class="{card_class}">
        <h3>📊 {time_point}-Year Survival Probability</h3>
        <div class="value">{surv_prob:.1%}</div>
        <div class="sub">{baseline_text}</div>
        <div class="sub">Mortality Risk: {risk:.1%}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Session State Init ──────────────────────────────────────
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "results" not in st.session_state:
    st.session_state.results = None
if "lp" not in st.session_state:
    st.session_state.lp = None
if "input_values" not in st.session_state:
    st.session_state.input_values = {}


# ── Page Layout ─────────────────────────────────────────────
col_logo, col_title = st.columns([0.6, 3.8])
with col_logo:
    st.markdown("""
    <div style="font-size: 3rem;">🔬</div>
    """, unsafe_allow_html=True)
with col_title:
    st.markdown('<div class="main-header">CKM Mortality Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Individualized Survival Prediction Based on Cox PH Model — Validated in CHARLS Cohort (N=6,953)</div>', unsafe_allow_html=True)

st.markdown("---")

left_col, right_col = st.columns([1, 1.6])

# ═══════════════════════════════════════════════════════════════
# Left Column: Input Panel
# ═══════════════════════════════════════════════════════════════
with left_col:
    st.markdown('<div class="section-title">📋 Patient Information</div>', unsafe_allow_html=True)

    with st.container():
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            age = st.number_input(
                f"**{VAR_INFO['age']['label']}** ({VAR_INFO['age']['unit']})",
                min_value=VAR_INFO['age']['min'],
                max_value=VAR_INFO['age']['max'],
                value=VAR_INFO['age']['default'],
                step=VAR_INFO['age']['step'],
                help=VAR_INFO['age']['help'],
            )
        with r1c2:
            bmi = st.number_input(
                f"**{VAR_INFO['bmi']['label']}** ({VAR_INFO['bmi']['unit']})",
                min_value=VAR_INFO['bmi']['min'],
                max_value=VAR_INFO['bmi']['max'],
                value=VAR_INFO['bmi']['default'],
                step=VAR_INFO['bmi']['step'],
                help=VAR_INFO['bmi']['help'],
            )

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            bl_cysc = st.number_input(
                f"**{VAR_INFO['bl_cysc']['label']}** ({VAR_INFO['bl_cysc']['unit']})",
                min_value=VAR_INFO['bl_cysc']['min'],
                max_value=VAR_INFO['bl_cysc']['max'],
                value=VAR_INFO['bl_cysc']['default'],
                step=VAR_INFO['bl_cysc']['step'],
                help=VAR_INFO['bl_cysc']['help'],
            )
        with r2c2:
            bl_crp = st.number_input(
                f"**{VAR_INFO['bl_crp']['label']}** ({VAR_INFO['bl_crp']['unit']})",
                min_value=VAR_INFO['bl_crp']['min'],
                max_value=VAR_INFO['bl_crp']['max'],
                value=VAR_INFO['bl_crp']['default'],
                step=VAR_INFO['bl_crp']['step'],
                help=VAR_INFO['bl_crp']['help'],
            )

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            cre_cyc_ratio = st.number_input(
                f"**{VAR_INFO['cre_cyc_ratio']['label']}** ({VAR_INFO['cre_cyc_ratio']['unit']})",
                min_value=VAR_INFO['cre_cyc_ratio']['min'],
                max_value=VAR_INFO['cre_cyc_ratio']['max'],
                value=VAR_INFO['cre_cyc_ratio']['default'],
                step=VAR_INFO['cre_cyc_ratio']['step'],
                help=VAR_INFO['cre_cyc_ratio']['help'],
            )

        with r3c2:
            gender = st.selectbox(
                f"**{VAR_INFO['gender']['label']}**",
                options=list(VAR_INFO['gender']['options'].keys()),
                format_func=lambda x: VAR_INFO['gender']['options'][x],
                index=0,
            )

        r4c1, r4c2 = st.columns(2)
        with r4c1:
            edu = st.selectbox(
                f"**{VAR_INFO['edu']['label']}**",
                options=list(VAR_INFO['edu']['options'].keys()),
                format_func=lambda x: VAR_INFO['edu']['options'][x],
                index=0,
            )
        with r4c2:
            marry = st.selectbox(
                f"**{VAR_INFO['marry']['label']}**",
                options=list(VAR_INFO['marry']['options'].keys()),
                format_func=lambda x: VAR_INFO['marry']['options'][x],
                index=1,
            )

        r5c1, r5c2 = st.columns(2)
        with r5c1:
            hypertension = st.selectbox(
                f"**{VAR_INFO['hypertension']['label']}**",
                options=list(VAR_INFO['hypertension']['options'].keys()),
                format_func=lambda x: VAR_INFO['hypertension']['options'][x],
                index=0,
            )
        with r5c2:
            lunge = st.selectbox(
                f"**{VAR_INFO['lunge']['label']}**",
                options=list(VAR_INFO['lunge']['options'].keys()),
                format_func=lambda x: VAR_INFO['lunge']['options'][x],
                index=0,
            )

        ckm_stage = st.selectbox(
            f"**{VAR_INFO['ckm_stage']['label']}**",
            options=list(VAR_INFO['ckm_stage']['options'].keys()),
            format_func=lambda x: VAR_INFO['ckm_stage']['options'][x],
            index=0,
            help="CKM Cardiovascular-Kidney-Metabolic Syndrome Staging (0-3)",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_col1, predict_col2 = st.columns([1, 1])
    with predict_col1:
        predict_btn = st.button("🚀 Predict", type="primary", use_container_width=True)
    with predict_col2:
        reset_btn = st.button("🔄 Reset", use_container_width=True)

    if reset_btn:
        st.session_state.predicted = False
        st.session_state.results = None
        st.rerun()


# ═══════════════════════════════════════════════════════════════
# Right Column: Results Panel
# ═══════════════════════════════════════════════════════════════
with right_col:
    st.markdown('<div class="section-title">📊 Prediction Results</div>', unsafe_allow_html=True)

    if predict_btn:
        input_dict = {
            "age": age,
            "bmi": bmi,
            "bl_cysc": bl_cysc,
            "bl_crp": bl_crp,
            "cre_cyc_ratio": cre_cyc_ratio,
            "gender": int(gender),
            "edu": int(edu),
            "marry": int(marry),
            "hypertension": int(hypertension),
            "lunge": int(lunge),
            "ckm_stage": int(ckm_stage),
        }

        with st.spinner("⏳ Calculating..."):
            results, lp = predict_survival(input_dict)
            st.session_state.results = results
            st.session_state.lp = lp
            st.session_state.predicted = True
            st.session_state.input_values = input_dict

    if st.session_state.predicted:
        results = st.session_state.results
        lp = st.session_state.lp
        baseline = get_baseline_survival()

        risk_3 = 1 - results[3]
        if risk_3 < 0.15:
            risk_badge = "🟢 Low Risk"
        elif risk_3 < 0.40:
            risk_badge = "🟡 Moderate Risk"
        else:
            risk_badge = "🔴 High Risk"

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="info-box">
                <strong>Risk Level</strong><br>
                <span style="font-size: 1.5rem;">{risk_badge}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="info-box">
                <strong>Linear Predictor (LP)</strong><br>
                <span style="font-size: 1.2rem;">{lp:+.4f}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        card1, card2, card3 = st.columns(3)
        with card1:
            render_result_card(3, results[3], baseline[3])
        with card2:
            render_result_card(5, results[5], baseline[5])
        with card3:
            render_result_card(8, results[8], baseline[8])

        st.markdown("---")
        st.markdown('<div class="section-title">📈 Survival Curve</div>', unsafe_allow_html=True)

        patient_curve = get_survival_curve(st.session_state.input_values)
        baseline_curve = get_baseline_curve()
        fig = create_survival_plot(patient_curve, baseline_curve)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-title">📝 Interpretation</div>', unsafe_allow_html=True)

        comparison_text = ""
        for t in TIME_POINTS:
            diff = results[t] - baseline[t]
            if diff > 0.05:
                comparison_text += f"- **{t}yr**: Survival probability {results[t]:.1%}, higher than baseline ({baseline[t]:.1%}) by {abs(diff):.1%} — favorable prognosis ✅\n"
            elif diff < -0.05:
                comparison_text += f"- **{t}yr**: Survival probability {results[t]:.1%}, lower than baseline ({baseline[t]:.1%}) by {abs(diff):.1%} — elevated risk ⚠️\n"
            else:
                comparison_text += f"- **{t}yr**: Survival probability {results[t]:.1%}, comparable to baseline ({baseline[t]:.1%})\n"

        st.markdown(comparison_text)

        st.markdown("<br>**🔍 Identified Risk Factors**", unsafe_allow_html=True)
        input_v = st.session_state.input_values
        risk_factors = []
        if input_v["age"] > 70:
            risk_factors.append("Advanced age (>70)")
        if input_v["bl_cysc"] > 1.2:
            risk_factors.append("Elevated Cystatin C")
        if input_v["bl_crp"] > 10:
            risk_factors.append("Markedly elevated CRP")
        if input_v["bmi"] > 30:
            risk_factors.append("Obesity (BMI>30)")
        if input_v["bmi"] < 18.5:
            risk_factors.append("Underweight (BMI<18.5)")
        if input_v["hypertension"] == 1:
            risk_factors.append("Hypertension")
        if input_v["lunge"] == 1:
            risk_factors.append("Lung disease")
        if input_v["ckm_stage"] >= 3:
            risk_factors.append(f"CKM Stage {input_v['ckm_stage']}")

        if risk_factors:
            st.markdown("Detected risk factors: " + ", ".join(risk_factors))
        else:
            st.markdown("No significant high-risk factors detected.")
    else:
        st.markdown("""
        <div style="text-align:center; padding:4rem 0; color:#95a5a6;">
            <div style="font-size:4rem; margin-bottom:1rem;">👈</div>
            <h3>Enter Patient Information to Predict</h3>
            <p>Fill in 11 clinical indicators in the left panel.<br>The system will calculate 3/5/8-year survival probabilities.</p>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#95a5a6; font-size:0.8rem;">
    <strong>CKM Mortality Risk Prediction System v2.0</strong> |
    Model: Cox PH (Boruta-selected) |
    Data Source: CHARLS Cohort (N=6,953) |
    C-index: 0.809 (Test Set) |
    Features: 11 (5 continuous + 6 categorical)
</div>
""", unsafe_allow_html=True)
