"""
CKM Mortality Risk Prediction — Core Calculation Logic
Implements Cox PH prediction in Python using coefficients extracted from the R model
Updated with Boruta-selected features and new preprocessing pipeline
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

_sd = pd.read_csv(os.path.join(DATA_DIR, "preprocessing_params.csv"))
SCALE_SD = dict(zip(_sd["parameter"], _sd["scale_sd"]))
SCALE_MEAN = dict(zip(_sd["parameter"], _sd["scale_mean"]))

TIME_POINTS = [3, 5, 8]

# ── Variable Lists ──────────────────────────────────────────
CONT_VARS = ["age", "bl_crp", "bl_cysc", "bmi", "CCR"]
BINARY_VARS = ["gender", "hypertension", "lunge", "marry"]
CAT_VARS_WITH_LEVELS = {
    "gender":         ["0", "1"],
    "hypertension":   ["0", "1"],
    "lunge":          ["0", "1"],
    "marry":          ["0", "1"],
    "ckm_stage":      ["0", "1", "2", "3"],
    "edu":            ["1", "2", "3", "4"],
}

MODEL_COLS = [
    "CCR", "age", "bl_crp", "bl_cysc", "bmi",
    "ckm_stage1", "ckm_stage2", "ckm_stage3",
    "edu2", "edu3", "edu4",
    "gender1", "hypertension1", "lunge1", "marry1",
]


# ── Preprocessing ───────────────────────────────────────────
def preprocess(input_dict):
    """
    Convert 11 raw clinical features into the 15-dimensional model input vector
    Pipeline:
      1. Continuous: (val - mean) / sd  (standard scaling)
      2. Binary: keep as-is (0/1), renamed to gender1/hypertension1/lunge1/marry1
      3. ckm_stage: dummy-coded (reference=stage0, so columns 1/2/3)
      4. edu: dummy-coded (reference=none/edu1, so columns 2/3/4)
    """
    features = {}

    # 1. Continuous variables: standard scaling (center + scale)
    cont_val = float(input_dict["cre_cyc_ratio"])
    features["CCR"] = (cont_val - SCALE_MEAN["CCR"]) / SCALE_SD["CCR"]

    for v in ["age", "bl_crp", "bl_cysc", "bmi"]:
        val = float(input_dict[v])
        features[v] = (val - SCALE_MEAN[v]) / SCALE_SD[v]

    # 2. Binary variables: store as gender1, hypertension1, etc.
    bin_map = {"gender": "gender1", "hypertension": "hypertension1",
               "lunge": "lunge1", "marry": "marry1"}
    for raw, model_name in bin_map.items():
        features[model_name] = float(input_dict[raw])

    # 3. ckm_stage: dummy (reference = stage 0)
    stage = int(input_dict["ckm_stage"])
    for s in [1, 2, 3]:
        features[f"ckm_stage{s}"] = 1.0 if stage == s else 0.0

    # 4. edu: dummy (reference = level 1 = None)
    edu = int(input_dict["edu"])
    for e in [2, 3, 4]:
        features[f"edu{e}"] = 1.0 if edu == e else 0.0

    # Return in MODEL_COLS order
    return np.array([features[col] for col in MODEL_COLS])


def predict_survival(input_dict):
    """
    Input: dict of 11 features (variable_name → value)
    Returns: (results_dict {3: survival_prob, 5: survival_prob, 8: survival_prob}, lp)
    """
    X = preprocess(input_dict)

    beta_values = np.array([BETAS[col] for col in MODEL_COLS])
    lp = np.dot(X, beta_values)

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

    fine_times = [0]
    fine_surv = [1.0]
    for t in TIME_POINTS:
        H0 = BASELINE_HAZARD[t]
        surv = np.exp(-H0 * np.exp(lp))
        fine_times.append(t)
        fine_surv.append(surv)

    return np.array(fine_times), np.array(fine_surv)


# ── Baseline Population (Median Patient) ───────────────────
DEFAULT_INPUT = {
    "age": 60.43,
    "bl_crp": 2.80,
    "bl_cysc": 1.02,
    "bmi": 23.73,
    "cre_cyc_ratio": 0.79,
    "gender": 0,
    "hypertension": 0,
    "lunge": 0,
    "marry": 1,
    "ckm_stage": 2,
    "edu": 1,
}

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
