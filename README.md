# CKM Mortality Risk Prediction System

Individualized survival prediction based on Cox Proportional Hazards model, validated in the CHARLS cohort.

## Features

- Input 11 clinical indicators to predict 3/5/8-year survival probabilities
- Generate personalized survival curves via Plotly interactive visualization
- Compare with baseline population
- Identify key risk factors
- Model information page with Boruta feature selection results

## Model Performance

- Data Source: CHARLS Cohort (N=6,953)
- Model: Cox Proportional Hazards (Boruta-selected)
- C-index: 0.809 (Test Set)
- Features: 11 confirmed variables (5 continuous + 6 categorical)

## Technical Stack

- Streamlit + Python
- Plotly interactive visualization
- Cox PH regression model
- Boruta feature selection

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Online Access

Deployed on Streamlit Community Cloud
