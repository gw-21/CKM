"""
Visualization functions — generates survival curve Plotly charts
"""

import plotly.graph_objects as go
import numpy as np


def create_survival_plot(patient_curve, baseline_curve):
    """
    Generate interactive survival curve
    patient_curve: (times, surv_probs) — patient survival curve
    baseline_curve: (times, surv_probs) — baseline population curve
    """
    p_times, p_surv = patient_curve
    b_times, b_surv = baseline_curve

    fig = go.Figure()

    # Patient survival curve
    fig.add_trace(go.Scatter(
        x=p_times,
        y=p_surv,
        mode="lines+markers",
        name="Patient",
        line=dict(color="#e74c3c", width=3, shape="hv"),
        marker=dict(size=8, symbol="circle"),
    ))

    # Baseline population curve
    fig.add_trace(go.Scatter(
        x=b_times,
        y=b_surv,
        mode="lines+markers",
        name="Baseline Population",
        line=dict(color="#3498db", width=2, dash="dash", shape="hv"),
        marker=dict(size=8, symbol="diamond"),
    ))

    fig.update_layout(
        title=dict(
            text="Kaplan-Meier Survival Curve",
            font=dict(size=18, color="#2c3e50"),
        ),
        xaxis=dict(
            title="Follow-up Time (Years)",
            range=[0, 10],
            dtick=1,
            gridcolor="#ecf0f1",
        ),
        yaxis=dict(
            title="Survival Probability",
            range=[0, 1],
            gridcolor="#ecf0f1",
            tickformat=".0%",
        ),
        legend=dict(
            x=0.75,
            y=0.25,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        plot_bgcolor="white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def create_gauge(surv_prob, title=""):
    """
    Generate a gauge indicator for survival probability at a single timepoint
    """
    if isinstance(surv_prob, np.floating):
        surv_prob = float(surv_prob)

    # Color based on risk level
    if surv_prob >= 0.9:
        color = "#27ae60"  # Green — low risk
    elif surv_prob >= 0.7:
        color = "#f39c12"  # Orange — moderate risk
    else:
        color = "#e74c3c"  # Red — high risk

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=surv_prob * 100,
        number=dict(
            suffix="%",
            font=dict(size=28, color=color),
        ),
        title=dict(
            text=title,
            font=dict(size=14, color="#7f8c8d"),
        ),
        delta=dict(
            reference=100,
            decreasing=dict(color="#e74c3c"),
        ),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="darkgray"),
            bar=dict(color=color, thickness=0.3),
            bgcolor="white",
            borderwidth=1,
            bordercolor="lightgray",
            steps=[
                dict(range=[0, 30], color="#fadbd8"),
                dict(range=[30, 70], color="#fdebd0"),
                dict(range=[70, 100], color="#d5f5e3"),
            ],
            threshold=dict(
                line=dict(color="red", width=3),
                thickness=0.75,
                value=50,
            ),
        ),
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font=dict(color="#2c3e50"),
    )

    return fig
