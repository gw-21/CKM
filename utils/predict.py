"""
CKM Mortality Risk Prediction — Core Calculation Logic
Implements Cox PH prediction using coefficients from the final R model.

Model: 15-variable Cox PH with raw (unscaled) continuous variables,
binary 0/1 coding, and dummy-coded categoricals.
- CKM stage: reference = Stage 0
- Education: reference = Level 1 (Elementary or below)
- Continuous: raw values (age, CCR, bl_crp, bl_cysc, bmi)
"""

import numpy as np
import pandas as pd
import os

# ── File Paths ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Handle being called from pages/ directory
if BASE_DIR.endswith("pages"):
    BASE_DIR = os.path.dirname(BASE_DIR)
elif BASE_DIR.endswith("utils"):
    BASE_DIR = os.path.dirname(BASE_DIR)
elif BASE_DIR.endswith("streamlit_app"):
    pass
else:
    _p = BASE_DIR
    for _ in range(3):
        if _p.endswith("streamlit_app"):
            break
        _p = os.path.dirname(_p)
    else:
        _p = BASE_DIR
    BASE_DIR = _p

DATA_DIR = BASE_DIR

# ── Load Model Parameters ───────────────────────────────────
_coef = pd.read_csv(os.path.join(DATA_DIR, "model_coefficients.csv"))
BETAS = dict(zip(_coef["variable"], _coef["coefficient"]))
BETAS = {k: v for k, v in BETAS.items() if not np.isnan(v)}

_h0 = pd.read_csv(os.path.join(DATA_DIR, "baseline_hazard.csv"))
BASELINE_HAZARD = dict(zip(_h0["time"].astype(int), _h0["hazard"]))

TIME_POINTS = [3, 5, 8]

# ── Variable Lists ──────────────────────────────────────────
# Continuous variables (used as RAW values, no transformation)
CONT_VARS = ["age", "bl_crp", "bl_cysc", "bmi", "CCR"]

# Binary variables (0/1 coding, 0 = reference)
BINARY_VARS = ["gender", "hypertension", "lunge", "marry"]

# Categorical variables with dummy coding
# CKM stage: reference = 0, dummies = 1, 2, 3
# Education: reference = 1, dummies = 2, 3, 4
CAT_VARS_WITH_LEVELS = {
    "gender":         ["0", "1"],
    "hypertension":   ["0", "1"],
    "lunge":          ["0", "1"],
    "marry":          ["0", "1"],
    "ckm_stage":      ["0", "1", "2", "3"],
    "edu":            ["1", "2", "3", "4"],
}

# Scale SD (kept for backward compatibility / Model Info display)
# NOTE: The model uses RAW variable values, not z-scored.
#       SD values are from the training set for reference only.
SCALE_SD = {
    "age": 10.100403,
    "bl_crp": 7.502316,
    "bl_cysc": 0.280433,
    "bmi": 15.176226,
    "CCR": 0.201022,
}
SCALE_MEAN = {
    "age": 60.425290,
    "bl_crp": 2.803987,
    "bl_cysc": 1.019416,
    "bmi": 23.726265,
    "CCR": 0.793090,
}


# ── Preprocessing ───────────────────────────────────────────
def preprocess(input_dict):
    """
    Convert 11 raw clinical features into the model's linear predictor terms.

    Pipeline:
    1. Continuous variables: used as-is (RAW values, no scaling)
    2. Binary variables: 0/1 coding
    3. Education: dummy coding (reference = level 1)
    4. CKM stage: dummy coding (reference = stage 0)

    Returns: dict of {variable_name: value} aligned with BETAS keys
    """
    features = {}

    # 1. Continuous variables: RAW values (no transformation)
    for v in CONT_VARS:
        features[v] = float(input_dict[v])

    # 2. Binary variables (suffix "1" in model)
    features["gender1"] = float(input_dict["gender"])
    features["hypertension1"] = float(input_dict["hypertension"])
    features["lunge1"] = float(input_dict["lunge"])
    features["marry1"] = float(input_dict["marry"])

    # 3. Education: dummy coding (reference = edu level 1)
    edu_val = int(input_dict["edu"])
    features["edu2"] = 1.0 if edu_val == 2 else 0.0
    features["edu3"] = 1.0 if edu_val == 3 else 0.0
    features["edu4"] = 1.0 if edu_val == 4 else 0.0

    # 4. CKM stage: dummy coding (reference = stage 0)
    ckm_val = int(input_dict["ckm_stage"])
    features["ckm_stage1"] = 1.0 if ckm_val == 1 else 0.0
    features["ckm_stage2"] = 1.0 if ckm_val == 2 else 0.0
    features["ckm_stage3"] = 1.0 if ckm_val == 3 else 0.0

    return features


def predict_survival(input_dict):
    """
    Input: dict of 11 features (variable_name → value)
    Returns: (results_dict {3: survival_prob, 5: survival_prob, 8: survival_prob}, lp)
    """
    features = preprocess(input_dict)

    # Compute linear predictor: lp = Σ β_i × X_i
    lp = 0.0
    for var_name, beta in BETAS.items():
        if var_name in features:
            lp += beta * features[var_name]

    # Compute survival probabilities at each timepoint
    # S(t|X) = exp(-H₀(t) × exp(lp))
    results = {}
    for t in TIME_POINTS:
        H0 = BASELINE_HAZARD[t]
        surv = np.exp(-H0 * np.exp(lp))
        results[t] = surv

    return results, lp


def get_survival_curve(input_dict):
    """
    Generate full survival curve for Plotly chart
    Returns: (times_array, surv_probs_array)
    """
    results, lp = predict_survival(input_dict)

    # Create step function
    fine_times = [0]
    fine_surv = [1.0]
    for t in TIME_POINTS:
        H0 = BASELINE_HAZARD[t]
        surv = np.exp(-H0 * np.exp(lp))
        fine_times.append(t)
        fine_surv.append(surv)

    return np.array(fine_times), np.array(fine_surv)


# ── Baseline Population (Typical Patient) ───────────────────
# A typical patient: 60-year-old, mean biomarker values, married, CKM stage 2
DEFAULT_INPUT = {
    "age": 60,
    "bl_crp": 2.0,
    "bl_cysc": 1.0,
    "bmi": 24,
    "CCR": 0.79,
    "gender": 0,
    "hypertension": 0,
    "lunge": 0,
    "marry": 1,
    "ckm_stage": 2,
    "edu": 1,
}

# Precompute baseline survival curve
_baseline_results, _ = predict_survival(DEFAULT_INPUT)


def get_baseline_survival():
    """Return baseline population survival probabilities at 3/5/8 years"""
    return {t: _baseline_results[t] for t in TIME_POINTS}


def get_baseline_curve():
    """Return baseline survival curve"""
    times = [0]
    survs = [1.0]
    for t in TIME_POINTS:
        s = _baseline_results[t]
        times.append(t)
        survs.append(s)
    return np.array(times), np.array(survs)
