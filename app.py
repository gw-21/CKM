"""
CKM Mortality Risk Prediction System — Main Entry
Medical-grade survival prediction interface — AHA CKM Framework aligned
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.predict import (
    predict_survival, get_survival_curve,
    get_baseline_survival, get_baseline_curve,
    TIME_POINTS, SCALE_SD, SCALE_MEAN, DEFAULT_INPUT
)
from utils.visualization import create_survival_plot, create_gauge

st.set_page_config(
    page_title="CKM Risk Predictor — CHARLS",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Medical Style System ─────────────────────────────────────
st.markdown("""
<style>
    /* ── Design Tokens ── */
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
        --red-bg: #FFF5F3;
        --amber: #D4893B;
        --amber-bg: #FFFAF0;
        --green: #3A7D5C;
        --green-bg: #F2F9F5;
        --blue-accent: #3B7DD8;
    }

    /* ── Global ── */
    .stApp {
        background-color: var(--gray-50);
    }
    .main > div {
        padding: 0 1.5rem 2rem;
    }

    /* ── Header ── */
    .app-header {
        background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
        margin: -1rem -1.5rem 1.5rem;
        padding: 1.75rem 2rem;
        color: var(--white);
    }
    .app-header h1 {
        font-size: 1.6rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        margin: 0;
        line-height: 1.3;
    }
    .app-header .sub {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.7);
        margin-top: 0.2rem;
        font-weight: 400;
    }
    .app-header .badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 4px;
        padding: 0.15rem 0.6rem;
        font-size: 0.7rem;
        color: rgba(255,255,255,0.75);
        margin-left: 0.5rem;
        letter-spacing: 0.02em;
    }
    .app-header .logo {
        font-size: 1.4rem;
        font-weight: 600;
        color: rgba(255,255,255,0.3);
        letter-spacing: 0.05em;
    }

    /* ── Section Titles ── */
    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--navy);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1.5px solid var(--gray-200);
    }

    /* ── Form Cards ── */
    .form-card {
        background: var(--white);
        border-radius: 8px;
        border: 1px solid var(--gray-200);
        padding: 0.1rem 1.25rem 1rem;
        margin-bottom: 0.75rem;
    }
    .form-card .field-row {
        display: flex; gap: 1rem;
    }
    .form-card .field-row > div {
        flex: 1;
    }

    /* ── Clinical Result Cards ── */
    .result-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0.75rem;
        margin: 0.5rem 0;
    }
    .clinical-card {
        background: var(--white);
        border-radius: 8px;
        border: 1px solid var(--gray-200);
        padding: 1rem 1.1rem;
        text-align: center;
        transition: box-shadow 0.15s;
    }
    .clinical-card .label {
        font-size: 0.72rem;
        color: var(--slate);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    .clinical-card .value {
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0.25rem 0 0.15rem;
        line-height: 1.1;
    }
    .clinical-card .sub {
        font-size: 0.72rem;
        color: var(--slate);
    }
    .clinical-card .value.low { color: var(--green); }
    .clinical-card .value.medium { color: var(--amber); }
    .clinical-card .value.high { color: var(--red); }

    /* ── Info Badges ── */
    .info-panel {
        background: var(--white);
        border-radius: 8px;
        border: 1px solid var(--gray-200);
        padding: 1rem 1.25rem;
        margin: 0.4rem 0;
    }
    .info-panel .label {
        font-size: 0.7rem;
        color: var(--slate);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    .info-panel .main-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--navy);
        margin-top: 0.15rem;
    }

    /* ── Risk Level Badge ── */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.8rem;
        border-radius: 5px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .risk-badge.low { background: var(--green-bg); color: var(--green); }
    .risk-badge.medium { background: var(--amber-bg); color: var(--amber); }
    .risk-badge.high { background: var(--red-bg); color: var(--red); }

    /* ── Comparison Text ── */
    .comparison-item {
        padding: 0.35rem 0;
        font-size: 0.88rem;
        border-bottom: 1px solid var(--gray-100);
    }
    .comparison-item:last-child { border: none; }

    /* ── Empty State ── */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: var(--slate);
    }
    .empty-state .icon { font-size: 2.5rem; margin-bottom: 0.5rem; opacity: 0.3; }
    .empty-state h3 {
        font-size: 1.05rem; font-weight: 500; color: var(--gray-300);
        margin: 0.5rem 0;
    }
    .empty-state p {
        font-size: 0.85rem; color: var(--gray-300);
        max-width: 300px; margin: 0 auto;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        font-size: 0.7rem;
        color: var(--gray-300);
        border-top: 1px solid var(--gray-200);
        margin-top: 1.5rem;
    }
    .app-footer strong { color: var(--slate); }

    /* ── Streamlit Overrides ── */
    .stButton button {
        font-weight: 500;
        font-size: 0.85rem;
        border-radius: 6px;
    }
    .stButton button[kind="primary"] {
        background: var(--navy);
        border: 1px solid var(--navy);
        color: white;
    }
    .stButton button[kind="primary"]:hover {
        background: var(--navy-light);
        border-color: var(--navy-light);
    }
    .stNumberInput label, .stSelectbox label {
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: var(--navy) !important;
    }
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        border-radius: 6px !important;
        border: 1px solid var(--gray-200) !important;
        font-size: 0.85rem !important;
    }
    .stSelectbox div[data-baseweb="select"]:hover {
        border-color: var(--steel) !important;
    }
    div[data-testid="stForm"] { border: none; padding: 0; }
    .row-widget.stButton { margin-top: 0.5rem; }
    .stSpinner > div { border-color: var(--steel) !important; }

    /* Plotly chart container */
    .stPlotlyChart {
        background: var(--white);
        border-radius: 8px;
        border: 1px solid var(--gray-200);
        padding: 0.75rem;
    }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .app-header { padding: 1.25rem; }
        .app-header h1 { font-size: 1.2rem; }
        .result-grid { grid-template-columns: 1fr; }
    }

    /* Hide defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ── Variable Info ───────────────────────────────────────────
VAR_INFO = {
    "age":              {"label": "Age",             "unit": "years",  "type": "continuous",
                         "min": 45,  "max": 101, "step": 1,  "default": 60,
                         "help": "Patient age at baseline (45–101 years)"},
    "bmi":              {"label": "Body Mass Index", "unit": "kg/m²",  "type": "continuous",
                         "min": 10.0,  "max": 60.0, "step": 0.1, "default": 24.0,
                         "help": "BMI = weight(kg) / height(m²)"},
    "bl_crp":           {"label": "C-Reactive Protein", "unit": "mg/L", "type": "continuous",
                         "min": 0.01, "max": 130.0, "step": 0.1, "default": 2.8,
                         "help": "Serum CRP, inflammation marker"},
    "bl_cysc":          {"label": "Cystatin C",      "unit": "mg/L",   "type": "continuous",
                         "min": 0.30, "max": 9.0, "step": 0.01, "default": 1.0,
                         "help": "Serum cystatin C, kidney function marker"},
    "cre_cyc_ratio":    {"label": "Cre/CysC Ratio (CCR)",  "unit": "",
                         "type": "continuous", "min": 0.20, "max": 2.5, "step": 0.01,
                         "default": 0.79, "help": "Creatinine-to-cystatin C ratio, sarcopenia marker"},
    "gender":           {"label": "Sex", "unit": "", "type": "categorical",
                         "options": {"0": "Female", "1": "Male"}, "default": "0"},
    "edu":              {"label": "Education", "unit": "", "type": "categorical",
                         "options": {"1": "None", "2": "Primary", "3": "Middle School", "4": "High School+"},
                         "default": "1"},
    "marry":            {"label": "Marital Status", "unit": "", "type": "categorical",
                         "options": {"0": "Other", "1": "Married"}, "default": "1"},
    "hypertension":     {"label": "Hypertension", "unit": "", "type": "categorical",
                         "options": {"0": "No", "1": "Yes"}, "default": "0"},
    "lunge":            {"label": "Chronic Lung Disease", "unit": "", "type": "categorical",
                         "options": {"0": "No", "1": "Yes"}, "default": "0",
                         "help": "Self-reported lung disease (COPD, asthma, etc.)"},
    "ckm_stage":        {"label": "CKM Stage", "unit": "", "type": "categorical",
                         "options": {"0": "Stage 0 — No Risk", "1": "Stage 1 — Excess Risk",
                                     "2": "Stage 2 — Metabolic Risk", "3": "Stage 3 — Subclinical CVD"},
                         "default": "2",
                         "help": "AHA Cardiovascular-Kidney-Metabolic Syndrome Staging"},
}

CKM_STAGE_COLORS = {
    "0": ("Stage 0", "#3A7D5C"),
    "1": ("Stage 1", "#D4893B"),
    "2": ("Stage 2", "#C44536"),
    "3": ("Stage 3", "#8B2233"),
}


def render_clinical_card(time_point, surv_prob, baseline_prob):
    risk = 1 - surv_prob
    if surv_prob >= 0.90:
        level = "low"
        label = "Low Risk"
    elif surv_prob >= 0.70:
        level = "medium"
        label = "Moderate Risk"
    else:
        level = "high"
        label = "Elevated Risk"

    st.markdown(f"""
    <div class="clinical-card">
        <div class="label">{time_point}-Year Survival</div>
        <div class="value {level}">{surv_prob:.1%}</div>
        <div class="sub">Baseline S₀({time_point}yr) = {baseline_prob:.1%}</div>
        <div class="sub" style="margin-top:0.2rem; font-weight:500;">
            Mortality Risk: {risk:.1%} · <span style="color:{'#3A7D5C' if level=='low' else '#D4893B' if level=='medium' else '#C44536'}">{label}</span>
        </div>
    </div>
    """ , unsafe_allow_html=True)


# ── Session State ───────────────────────────────────────────
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "results" not in st.session_state:
    st.session_state.results = None
if "lp" not in st.session_state:
    st.session_state.lp = None
if "input_values" not in st.session_state:
    st.session_state.input_values = {}


# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown(f"""
<div class="app-header">
    <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.15rem;">
        <span class="logo">CKM</span>
        <span style="font-size:0.75rem; color:rgba(255,255,255,0.3);">|</span>
        <h1>Mortality Risk Predictor</h1>
    </div>
    <div class="sub">
        Cox Proportional Hazards Model · CHARLS Cohort (N = 6,953)
        <span class="badge">v2.0</span>
        <span class="badge">C-index 0.809</span>
        <span class="badge">AHA CKM Framework</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════
left_col, spacer_col, right_col = st.columns([1, 0.04, 1.4])

# ═══════════════════════════════════════════════════════════
# LEFT: Clinical Data Entry
# ═══════════════════════════════════════════════════════════
with left_col:
    st.markdown('<div class="section-title">Patient Assessment</div>', unsafe_allow_html=True)

    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        age = st.number_input("Age (years)", min_value=45, max_value=101,
                              value=60, step=1, help="45–101 years at baseline")
    with r1c2:
        gender = st.selectbox("Sex", options=["0", "1"],
                              format_func=lambda x: "Female" if x == "0" else "Male",
                              index=0)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0,
                              value=24.0, step=0.1, format="%.1f")
    with r2c2:
        ckm_stage = st.selectbox("CKM Stage", options=["0","1","2","3"],
            format_func=lambda x: f"Stage {x}",
            index=2,
            help="AHA Cardiovascular-Kidney-Metabolic Syndrome Staging")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Laboratory Values</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        bl_cysc = st.number_input("Cystatin C (mg/L)", min_value=0.30, max_value=9.0,
                                  value=1.0, step=0.01, format="%.2f")
    with r3c2:
        bl_crp = st.number_input("C-Reactive Protein (mg/L)", min_value=0.01, max_value=130.0,
                                 value=2.8, step=0.1, format="%.1f")

    r4c1, r4c2 = st.columns(2)
    with r4c1:
        cre_cyc_ratio = st.number_input("CCR Ratio", min_value=0.20, max_value=2.5,
                                        value=0.79, step=0.01, format="%.2f",
                                        help="Creatinine / Cystatin C ratio")
    with r4c2:
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Comorbidities & Social</div>', unsafe_allow_html=True)
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    r5c1, r5c2 = st.columns(2)
    with r5c1:
        hypertension = st.selectbox("Hypertension", options=["0","1"],
                                    format_func=lambda x: "Yes" if x=="1" else "No", index=0)
    with r5c2:
        lunge = st.selectbox("Chronic Lung Disease", options=["0","1"],
                             format_func=lambda x: "Yes" if x=="1" else "No", index=0)
    r6c1, r6c2 = st.columns(2)
    with r6c1:
        edu = st.selectbox("Education", options=["1","2","3","4"],
            format_func=lambda x: {"1":"None","2":"Primary","3":"Middle","4":"High School+"}[x],
            index=0)
    with r6c2:
        marry = st.selectbox("Married", options=["0","1"],
                             format_func=lambda x: "Yes" if x=="1" else "No", index=1)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Action Buttons ──
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        predict_btn = st.button("Calculate Risk", type="primary", use_container_width=True)
    with btn_col2:
        reset_btn = st.button("Clear", use_container_width=True)

    if reset_btn:
        st.session_state.predicted = False
        st.session_state.results = None
        st.rerun()


# ═══════════════════════════════════════════════════════════
# RIGHT: Results Panel
# ═══════════════════════════════════════════════════════════
with right_col:
    st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)

    if predict_btn:
        input_dict = {
            "age": age, "bmi": bmi, "bl_cysc": bl_cysc, "bl_crp": bl_crp,
            "cre_cyc_ratio": cre_cyc_ratio, "gender": int(gender), "edu": int(edu),
            "marry": int(marry), "hypertension": int(hypertension),
            "lunge": int(lunge), "ckm_stage": int(ckm_stage),
        }
        with st.spinner("Calculating survival estimates..."):
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
        if risk_3 < 0.10:
            risk_class = "low"
            risk_level = "Low"
        elif risk_3 < 0.25:
            risk_class = "medium"
            risk_level = "Moderate"
        else:
            risk_class = "high"
            risk_level = "Elevated"

        # ── Summary Line ──
        sum_col1, sum_col2 = st.columns([1, 1])
        with sum_col1:
            st.markdown(f"""
            <div class="info-panel">
                <div class="label">3-Year Mortality Risk Classification</div>
                <div class="main-value">
                    <span class="risk-badge {risk_class}">● {risk_level} Risk</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with sum_col2:
            ckm_stage_value = int(st.session_state.input_values.get("ckm_stage", 0))
            stage_name, stage_color = CKM_STAGE_COLORS[str(ckm_stage_value)]
            st.markdown(f"""
            <div class="info-panel">
                <div class="label">AHA CKM Syndrome Stage</div>
                <div class="main-value" style="color:{stage_color};">{stage_name}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Three Clinical Cards ──
        st.markdown('<div class="section-title" style="margin-top:0.75rem;">Survival Estimates</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-grid">', unsafe_allow_html=True)
        # Use columns for actual layout
        card_c1, card_c2, card_c3 = st.columns(3)
        with card_c1:
            render_clinical_card(3, results[3], baseline[3])
        with card_c2:
            render_clinical_card(5, results[5], baseline[5])
        with card_c3:
            render_clinical_card(8, results[8], baseline[8])

        # ── Survival Curve ──
        st.markdown('<div class="section-title" style="margin-top:1rem;">Survival Trajectory</div>', unsafe_allow_html=True)
        patient_curve = get_survival_curve(st.session_state.input_values)
        baseline_curve = get_baseline_curve()
        fig = create_survival_plot(patient_curve, baseline_curve)
        st.plotly_chart(fig, use_container_width=True)

        # ── Clinical Interpretation ──
        st.markdown('<div class="section-title">Clinical Interpretation</div>', unsafe_allow_html=True)

        comparison_lines = []
        for t in TIME_POINTS:
            diff = results[t] - baseline[t]
            if diff > 0.03:
                label = "Favorable prognosis compared to cohort baseline"
            elif diff < -0.03:
                label = "Elevated risk compared to cohort baseline"
            else:
                label = "Comparable to cohort baseline"
            comparison_lines.append((t, results[t], baseline[t], diff, label))

        for t, res, base, diff, label in comparison_lines:
            dir_symbol = "↑" if diff > 0.03 else ("↓" if diff < -0.03 else "≈")
            dir_color = "#3A7D5C" if diff > 0.03 else "#C44536" if diff < -0.03 else "#6B7F9E"
            st.markdown(f"""
            <div class="comparison-item">
                <strong>{t}-Year:</strong>&ensp;
                Patient <strong>{res:.1%}</strong>
                vs Baseline <strong>{base:.1%}</strong>
                <span style="color:{dir_color}; font-weight:600; margin-left:0.3rem;">{dir_symbol} {abs(diff):.1%}</span>
                <span style="color:var(--slate); font-size:0.82rem; margin-left:0.5rem;">({label})</span>
            </div>
            """, unsafe_allow_html=True)

        # ── Risk Factors ──
        input_v = st.session_state.input_values
        risk_factors = []
        if input_v["age"] > 70:
            risk_factors.append("Age > 70 years")
        if input_v["bl_cysc"] > 1.2:
            risk_factors.append("Cystatin C > 1.2 mg/L")
        if input_v["bl_crp"] > 10:
            risk_factors.append("CRP > 10 mg/L")
        if input_v["bmi"] > 30:
            risk_factors.append("BMI > 30 (obesity)")
        if input_v["bmi"] < 18.5:
            risk_factors.append("BMI < 18.5 (underweight)")
        if input_v["hypertension"] == 1:
            risk_factors.append("Hypertension")
        if input_v["lunge"] == 1:
            risk_factors.append("Chronic lung disease")
        if input_v["ckm_stage"] >= 3:
            risk_factors.append(f"CKM Stage {input_v['ckm_stage']} (subclinical CVD)")

        if risk_factors:
            st.markdown("**Identified Risk Factors:**  " + " · ".join(
                [f'<span style="background:var(--red-bg);color:var(--red);padding:0.1rem 0.4rem;border-radius:3px;font-size:0.78rem;">{rf}</span>' for rf in risk_factors]
            ), unsafe_allow_html=True)
        else:
            st.markdown("*No significant risk factors identified.*")

    else:
        # ── Empty State ──
        st.markdown("""
        <div class="empty-state">
            <div class="icon">⚕</div>
            <h3>Enter patient data to begin</h3>
            <p>Complete the clinical assessment form on the left, then calculate individualized 3/5/8-year survival probabilities.</p>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <strong>CKM Mortality Risk Predictor v2.0</strong>
    &nbsp;·&nbsp; Cox PH (Boruta-selected, 15 features)
    &nbsp;·&nbsp; CHARLS Cohort N = 6,953
    &nbsp;·&nbsp; C-index 0.809
</div>
""", unsafe_allow_html=True)
