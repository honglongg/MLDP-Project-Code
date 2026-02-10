"""
BuyerIQ - Intelligent Purchase Prediction for E-Commerce
Streamlit Application with 2026 Design Trends
Enterprise-Grade UI/UX for Business Professionals
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# -----------------------------
# Page config (must be first Streamlit call)
# -----------------------------
st.set_page_config(
    page_title="BuyerIQ - AI Purchase Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="auto",
)

# -----------------------------
# Constants
# -----------------------------
ARTIFACT_DIR = Path("model_artifacts")
MODEL_PATH = ARTIFACT_DIR / "final_model_pipeline.pkl"
METRICS_PATH = ARTIFACT_DIR / "final_metrics.pkl"

MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITORS = ["Returning_Visitor", "New_Visitor", "Other"]

REQUIRED_COLUMNS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues",
    "SpecialDay", "Month",
    "OperatingSystems", "Browser", "Region", "TrafficType",
    "VisitorType", "Weekend",
]

# -----------------------------
# Custom CSS
# (unchanged visually, just injected once)
# -----------------------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        --warning-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.1);
        --glass-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.7);
        --surface-1: rgba(255, 255, 255, 0.05);
        --surface-2: rgba(255, 255, 255, 0.08);
    }

    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1b3d 25%, #2d1b4e 50%, #1a1b3d 75%, #0a0e27 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Sora', sans-serif;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .gradient-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 30%, #f093fb 60%, #4facfe 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 4rem;
        letter-spacing: -0.02em;
        line-height: 1.1;
        animation: gradientFlow 8s ease infinite;
        filter: drop-shadow(0 0 30px rgba(102, 126, 234, 0.5));
    }

    @keyframes gradientFlow {
        0%, 100% { background-position: 0% center; }
        50% { background-position: 100% center; }
    }

    .glass-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow:
            0 8px 32px 0 rgba(31, 38, 135, 0.37),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.1),
            0 0 0 1px rgba(0, 0, 0, 0.1);
        padding: 2rem;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }

    .glass-card:hover::before { left: 100%; }

    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow:
            0 16px 48px 0 rgba(31, 38, 135, 0.5),
            inset 0 1px 0 0 rgba(255, 255, 255, 0.15),
            0 0 0 1px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.3);
    }

    .metric-card {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 100%);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 2rem 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }

    .metric-card:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow:
            0 12px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 14, 39, 0.95) 0%, rgba(29, 27, 61, 0.95) 100%);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    [data-testid="stSidebar"] .element-container {
        animation: fadeInUp 0.6s ease-out backwards;
    }

    [data-testid="stSidebar"] .element-container:nth-child(1) { animation-delay: 0.1s; }
    [data-testid="stSidebar"] .element-container:nth-child(2) { animation-delay: 0.2s; }
    [data-testid="stSidebar"] .element-container:nth-child(3) { animation-delay: 0.3s; }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 0.875rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.02em;
        box-shadow:
            0 8px 24px rgba(102, 126, 234, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 12px 32px rgba(102, 126, 234, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
    }

    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div,
    .stTextInput>div>div>input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
        transition: all 0.3s ease;
    }

    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.25rem;
        border-radius: 100px;
        font-weight: 600;
        font-size: 0.875rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        backdrop-filter: blur(10px);
        border: 1px solid;
        animation: pulse 2s ease-in-out infinite;
    }

    .status-high {
        background: linear-gradient(135deg, rgba(17, 153, 142, 0.2) 0%, rgba(56, 239, 125, 0.2) 100%);
        border-color: rgba(56, 239, 125, 0.4);
        color: #38ef7d;
    }

    .status-moderate {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(251, 191, 36, 0.2) 100%);
        border-color: rgba(251, 191, 36, 0.4);
        color: #fbbf24;
    }

    .status-low {
        background: linear-gradient(135deg, rgba(245, 87, 108, 0.2) 0%, rgba(239, 68, 68, 0.2) 100%);
        border-color: rgba(239, 68, 68, 0.4);
        color: #f56c6c;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }

    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        margin: 2rem 0;
    }

    /* Hide Streamlit Branding (SAFE: keeps sidebar toggle) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Do NOT hide the header; just make it invisible-looking */
    [data-testid="stHeader"] {
    background: transparent;
    box-shadow: none;
    }

    /* Optional: reduce extra padding at the top */
    .block-container {
    padding-top: 1rem;
    }

    .feature-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    color: rgba(255,255,255,0.85);
    font-weight: 700;
    font-size: 0.85rem;
    margin-right: 0.75rem;
    }
    .feature-item {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.1rem;
    }
    .feature-text {
    color: rgba(255,255,255,0.6);
    font-size: 0.9rem;
    line-height: 1.35;
    }

    /* Streamlit labels / help / small captions */
    label, .stMarkdown, .stCaption {
    color: rgba(255,255,255,0.82) !important;
    }

    /* Small helper text under widgets (Streamlit uses data-testid in many versions) */
    [data-testid="stWidgetLabel"] p {
    color: rgba(255,255,255,0.78) !important;
    }

    /* Input placeholder text */
    .stTextInput input::placeholder,
    .stNumberInput input::placeholder {
    color: rgba(255,255,255,0.35) !important;
    }

    /* Input focus states */
    .stNumberInput input:focus,
    .stTextInput input:focus {
    border-color: rgba(102,126,234,0.6) !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.18) !important;
    outline: none !important;
    }

    /* Selectbox focus (baseweb select) */
    .stSelectbox [data-baseweb="select"] > div:focus-within {
    border-color: rgba(102,126,234,0.6) !important;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.18) !important;
    }

    /* Sidebar radio buttons */
    [data-testid="stSidebar"] .stRadio > div {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 0.75rem;
    }

    .glass-card.tight { padding: 1.5rem; }

    .pill {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    color: rgba(255,255,255,0.75);
    font-size: 0.8rem;
    margin: 0.25rem 0.35rem 0 0;
    }

</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Loading artifacts (safer + clearer errors)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    try:
        return joblib.load(METRICS_PATH)
    except Exception:
        return None


model = load_model()
metrics = load_metrics()

# -----------------------------
# Helpers
# -----------------------------
def safe_metric_value(m: dict, key: str, default: float = 0.0) -> float:
    try:
        v = m.get(key, default)
        return float(v)
    except Exception:
        return float(default)


def get_recommendation(prob: float) -> dict:
    if prob > 0.7:
        return {
            "cat": "HIGH",
            "color": "#38ef7d",
            "icon": "🟢",
            "action": "Premium targeting",
            "class": "status-high",
            "strategy": "Deploy personalized offers, expedited checkout, live chat support",
            "expected_roi": "3-5x average",
        }
    if prob > 0.4:
        return {
            "cat": "MODERATE",
            "color": "#fbbf24",
            "icon": "🟡",
            "action": "Standard marketing",
            "class": "status-moderate",
            "strategy": "Email nurture campaigns, retargeting ads, product recommendations",
            "expected_roi": "1.5-2.5x average",
        }
    return {
        "cat": "LOW",
        "color": "#f56c6c",
        "icon": "🔴",
        "action": "Passive monitoring",
        "class": "status-low",
        "strategy": "Minimal spend, basic retargeting, content marketing focus",
        "expected_roi": "0.5-1x average",
    }


def gauge_chart(prob: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(prob) * 100,
            number={"suffix": "%", "font": {"size": 48, "color": "#ffffff", "family": "Sora"}},
            title={"text": "Purchase Probability", "font": {"size": 16, "color": "rgba(255,255,255,0.7)", "family": "Sora"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "rgba(255,255,255,0.2)"},
                "bar": {"color": "#667eea", "thickness": 0.75},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 2,
                "bordercolor": "rgba(255,255,255,0.1)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(245, 108, 108, 0.2)"},
                    {"range": [40, 70], "color": "rgba(251, 191, 36, 0.2)"},
                    {"range": [70, 100], "color": "rgba(56, 239, 125, 0.2)"},
                ],
                "threshold": {"line": {"color": "#ffffff", "width": 4}, "thickness": 0.75, "value": float(prob) * 100},
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "family": "Sora"},
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def ensure_model_ready(m) -> tuple[bool, str]:
    if m is None:
        return False, "Model not loaded. Ensure model_artifacts/final_model_pipeline.pkl exists."
    if not hasattr(m, "predict"):
        return False, "Loaded object has no .predict(). Please re-export your pipeline."
    if not hasattr(m, "predict_proba"):
        return False, "Loaded object has no .predict_proba(). Your model must support probability outputs."
    return True, ""


def reorder_to_model_features(df: pd.DataFrame, m) -> pd.DataFrame:
    """
    If the pipeline/model has feature_names_in_, align columns in that order.
    Otherwise keep df as-is.
    """
    cols = getattr(m, "feature_names_in_", None)
    if cols is None:
        return df
    cols = list(cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        # keep original; missing columns will fail later with a clearer error
        return df
    return df[cols]


def validate_and_coerce_batch(df: pd.DataFrame) -> tuple[bool, str, pd.DataFrame]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}", df

    out = df.copy()

    # Coerce types (best-effort)
    numeric_cols = [
        "Administrative", "Administrative_Duration",
        "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates", "PageValues",
        "SpecialDay",
        "OperatingSystems", "Browser", "Region", "TrafficType",
    ]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Weekend: allow True/False, 1/0, "true"/"false"
    if out["Weekend"].dtype != bool:
        out["Weekend"] = (
            out["Weekend"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        )
    # If still NaN, fill False (conservative)
    out["Weekend"] = out["Weekend"].fillna(False).astype(bool)

    # Quick domain sanity (don’t hard-fail; just warn later if lots of NaNs)
    return True, "", out


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown(
        '<h2 style="color: #667eea; font-size: 1.75rem; margin-bottom: 0.5rem;">BuyerIQ</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color: rgba(255,255,255,0.6); font-size: 0.875rem; margin-bottom: 2rem;">Predictive Intelligence Platform</p>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Home", "Prediction", "Performance", "ROI", "Batch"],
        label_visibility="collapsed",
    )

    if isinstance(metrics, dict):
        st.markdown(
            '<p style="color: rgba(255,255,255,0.5); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem;">Model Performance</p>',
            unsafe_allow_html=True,
        )
        st.metric("F1-Score", f"{safe_metric_value(metrics, 'f1_score'):.1%}")
        st.metric("Precision", f"{safe_metric_value(metrics, 'precision'):.1%}")
        st.metric("Recall", f"{safe_metric_value(metrics, 'recall'):.1%}")

# -----------------------------
# PAGE 1: HOME
# -----------------------------
if page == "Home":
    st.markdown('<h1 class="gradient-text">BuyerIQ</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size: 1.25rem; color: rgba(255,255,255,0.7); margin-bottom: 1rem;">AI-Powered Purchase Prediction Platform</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size: 1rem; color: rgba(255,255,255,0.5); margin-bottom: 3rem;">Identify high-intent buyers in real-time • Optimize marketing spend • Maximize conversion efficiency</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="glass-card tight">', unsafe_allow_html=True)
    st.markdown("### Executive Summary")
    st.markdown(
        """
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-top: 1.5rem;">
        <div style="text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">84.5%</div>
            <div style="color: rgba(255,255,255,0.6); margin-top: 0.5rem;">Baseline Cart Abandonment</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">59%</div>
            <div style="color: rgba(255,255,255,0.6); margin-top: 0.5rem;">Buyer Capture Rate</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">72%</div>
            <div style="color: rgba(255,255,255,0.6); margin-top: 0.5rem;">Marketing Efficiency Gain</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">&lt;10ms</div>
            <div style="color: rgba(255,255,255,0.6); margin-top: 0.5rem;">Prediction Latency</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="glass-card tight">', unsafe_allow_html=True)
        st.markdown("### Business Challenge")
        st.markdown('<div style="margin-top: 1.5rem;">', unsafe_allow_html=True)
        st.markdown(
            """
        <div class="feature-item">
            <span class="feature-icon">B</span>
            <div>
                <div style="font-weight: 600; color: #ffffff; margin-bottom: 0.25rem;">Wasted Marketing Budget</div>
                <div class="feature-text">84.5% of visitors abandon without purchasing, yet marketing treats all equally</div>
            </div>
        </div>
        <div class="feature-item">
            <span class="feature-icon">R</span>
            <div>
                <div style="font-weight: 600; color: #ffffff; margin-bottom: 0.25rem;">Lost Revenue Opportunities</div>
                <div class="feature-text">High-intent buyers leave due to lack of timely, personalized engagement</div>
            </div>
        </div>
        <div class="feature-item">
            <span class="feature-icon">T</span>
            <div>
                <div style="font-weight: 600; color: #ffffff; margin-bottom: 0.25rem;">Inefficient Targeting</div>
                <div class="feature-text">Traditional analytics provide hindsight, not real-time predictive insights</div>
            </div>
        </div>
        <div class="feature-item">
            <span class="feature-icon">S</span>
            <div>
                <div style="font-weight: 600; color: #ffffff; margin-bottom: 0.25rem;">Delayed Action</div>
                <div class="feature-text">Manual segmentation and campaign setup miss critical conversion windows</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown("</div></div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card tight">', unsafe_allow_html=True)
        st.markdown("### BuyerIQ Solution")
        st.markdown(
            """
        <div style="margin-top: 1.5rem;">
            <div class="feature-item">
                <span class="feature-icon">AI</span>
                <div>
                    <div style="font-weight: 600; color: #ffffff; margin-bottom: 0.25rem;">Real-Time AI Predictions</div>
                    <div class="feature-text">Instant purchase probability scoring for every active session (&lt;10ms latency)</div>
                </div>
            </div>
            <div class="feature-item">
                <span class="feature-icon">P</span>
                <div>
                    <div style="font-weight: 600; color: #ffffff; margin-bottom: 0.25rem;">Precision Targeting</div>
                    <div class="feature-text">59% buyer capture rate with 72% improvement in marketing efficiency</div>
                </div>
            </div>
            <div class="feature-item">
                <span class="feature-icon">A</span>
                <div>
                    <div style="font-weight: 600; color: #ffffff; margin-bottom: 0.25rem;">Automated Activation</div>
                    <div class="feature-text">Trigger personalized interventions automatically based on intent signals</div>
                </div>
            </div>
            <div class="feature-item">
                <span class="feature-icon">I</span>
                <div>
                    <div style="font-weight: 600; color: #ffffff; margin-bottom: 0.25rem;">Actionable Intelligence</div>
                    <div class="feature-text">Clear recommendations: Premium, Standard, or Passive engagement strategies</div>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if isinstance(metrics, dict):
        st.markdown("---")
        st.markdown("### Model Performance Indicators")
        st.markdown(
            '<p style="color: rgba(255,255,255,0.6); margin-bottom: 2rem;">Validated on 12,330 real e-commerce sessions with rigorous cross-validation</p>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value">{safe_metric_value(metrics, 'f1_score'):.1%}</div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.5rem;">Balanced Accuracy</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{safe_metric_value(metrics, 'precision'):.1%}</div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.5rem;">Prediction Accuracy</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{safe_metric_value(metrics, 'recall'):.1%}</div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.5rem;">Buyer Capture Rate</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{safe_metric_value(metrics, 'accuracy'):.1%}</div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.5rem;">Overall Correctness</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

# -----------------------------
# PAGE 2: PREDICTION
# -----------------------------
elif page == "Prediction":
    st.markdown('<h1 class="gradient-text">Live Prediction</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size: 1.125rem; color: rgba(255,255,255,0.7); margin-bottom: 2rem;">Real-time purchase probability analysis</p>',
        unsafe_allow_html=True,
    )

    ok, msg = ensure_model_ready(model)
    if not ok:
        st.error(f"⚠️ {msg}")
        st.stop()

    st.markdown("### Sample Scenarios")
    st.markdown(
        '<p style="color: rgba(255,255,255,0.6); margin-bottom: 1rem;">Click to load pre-configured examples and see how the model performs</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 0.9])
    high = c1.button("High-Intent Sample", use_container_width=True)
    mod = c2.button("Moderate-Intent Sample", use_container_width=True)
    low = c3.button("Low-Intent Sample", use_container_width=True)

    reset = c4.button("Reset", use_container_width=True)
    if reset:
        st.session_state.vals = {}

    if "vals" not in st.session_state:
        st.session_state.vals = {}

    if high:
        st.session_state.vals = {
            "Admin": 5, "AdminDur": 300, "Info": 2, "InfoDur": 50,
            "Prod": 30, "ProdDur": 1200, "Bounce": 0.001, "Exit": 0.01,
            "PageVal": 150, "Special": 0, "Month": "Nov", "OS": 2,
            "Browser": 2, "Region": 1, "Traffic": 2, "Visitor": "Returning_Visitor", "Weekend": False,
        }
    elif mod:
        st.session_state.vals = {
            "Admin": 2, "AdminDur": 80, "Info": 1, "InfoDur": 20,
            "Prod": 15, "ProdDur": 400, "Bounce": 0.01, "Exit": 0.03,
            "PageVal": 25, "Special": 0, "Month": "May", "OS": 2,
            "Browser": 2, "Region": 3, "Traffic": 3, "Visitor": "Returning_Visitor", "Weekend": False,
        }
    elif low:
        st.session_state.vals = {
            "Admin": 1, "AdminDur": 4, "Info": 0, "InfoDur": 0,
            "Prod": 13, "ProdDur": 161, "Bounce": 0.025, "Exit": 0.062,
            "PageVal": 0, "Special": 0.6, "Month": "May", "OS": 2,
            "Browser": 5, "Region": 9, "Traffic": 5, "Visitor": "Returning_Visitor", "Weekend": False,
        }

    st.markdown("---")
    st.markdown("### Session Parameters")

    with st.form("pred_form"):
        a, b, c = st.columns(3)

        with a:
            with st.expander("Page Metrics", expanded=True):
                admin = st.number_input("Admin Pages", 0, 100, int(st.session_state.vals.get("Admin", 1)))
                admin_dur = st.number_input("Admin Duration (s)", 0.0, 10000.0, float(st.session_state.vals.get("AdminDur", 4.0)))
                info = st.number_input("Info Pages", 0, 100, int(st.session_state.vals.get("Info", 0)))
                info_dur = st.number_input("Info Duration (s)", 0.0, 10000.0, float(st.session_state.vals.get("InfoDur", 0.0)))
                prod = st.number_input("Product Pages", 0, 200, int(st.session_state.vals.get("Prod", 13)))
                prod_dur = st.number_input("Product Duration (s)", 0.0, 20000.0, float(st.session_state.vals.get("ProdDur", 161.0)))

        with b:
            with st.expander("Engagement Metrics", expanded=True):
                bounce = st.number_input("Bounce Rate", 0.0, 1.0, float(st.session_state.vals.get("Bounce", 0.025)), format="%.6f")
                exit_r = st.number_input("Exit Rate", 0.0, 1.0, float(st.session_state.vals.get("Exit", 0.062)), format="%.6f")
                page_val = st.number_input("Page Values ($)", 0.0, 500.0, float(st.session_state.vals.get("PageVal", 0.0)))
                special = st.number_input("Special Day", 0.0, 1.0, float(st.session_state.vals.get("Special", 0.6)))

        with c:
            with st.expander("Session Context", expanded=True):
                preset_month = st.session_state.vals.get("Month", MONTHS[4])
                month_index = MONTHS.index(preset_month) if preset_month in MONTHS else 4
                month = st.selectbox("Month", MONTHS, index=month_index)

                os_ = st.number_input("Operating System", 1, 8, int(st.session_state.vals.get("OS", 2)))
                browser = st.number_input("Browser", 1, 13, int(st.session_state.vals.get("Browser", 5)))
                region = st.number_input("Region", 1, 9, int(st.session_state.vals.get("Region", 9)))
                traffic = st.number_input("Traffic Type", 1, 20, int(st.session_state.vals.get("Traffic", 5)))

                preset_visitor = st.session_state.vals.get("Visitor", VISITORS[0])
                visitor_index = VISITORS.index(preset_visitor) if preset_visitor in VISITORS else 0
                visitor = st.selectbox("Visitor Type", VISITORS, index=visitor_index)

                weekend = st.checkbox("Weekend Visit", bool(st.session_state.vals.get("Weekend", False)))


        submit = st.form_submit_button("Generate Prediction", use_container_width=True)

    if submit:
        input_df = pd.DataFrame(
            {
                "Administrative": [admin],
                "Administrative_Duration": [admin_dur],
                "Informational": [info],
                "Informational_Duration": [info_dur],
                "ProductRelated": [prod],
                "ProductRelated_Duration": [prod_dur],
                "BounceRates": [bounce],
                "ExitRates": [exit_r],
                "PageValues": [page_val],
                "SpecialDay": [special],
                "Month": [month],
                "OperatingSystems": [os_],
                "Browser": [browser],
                "Region": [region],
                "TrafficType": [traffic],
                "VisitorType": [visitor],
                "Weekend": [weekend],
            }
        )

        input_df = reorder_to_model_features(input_df, model)

        try:
            prob = float(model.predict_proba(input_df)[0][1])
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.info("💡 This usually means your input columns/types do not match what the model pipeline was trained on.")
            st.stop()

        st.markdown("---")
        st.markdown("### Prediction Results")

        r1, r2 = st.columns([1.2, 1])
        with r1:
            st.plotly_chart(gauge_chart(prob), use_container_width=True)

        with r2:
            rec = get_recommendation(prob)
            st.markdown('<div class="glass-card tight" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown(
                f'<div style="text-align: center; margin-top: 1.5rem;"><span class="status-badge {rec["class"]}">{rec["icon"]} {rec["cat"]} INTENT</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p style="text-align: center; font-size: 3rem; font-weight: 700; margin: 1.5rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{prob:.1%}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p style="text-align: center; color: rgba(255,255,255,0.5); font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1.5rem;">Purchase Probability</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 1.5rem; margin-top: 1.5rem;">',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p style="color: rgba(255,255,255,0.7); font-size: 0.875rem; margin-bottom: 0.75rem;"><strong>Recommended Action:</strong></p>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<p style="color: rgba(255,255,255,0.9); font-size: 1rem; margin-bottom: 1rem;">{rec["action"]}</p>', unsafe_allow_html=True)
            st.markdown(
                '<p style="color: rgba(255,255,255,0.7); font-size: 0.875rem; margin-bottom: 0.75rem;"><strong>Engagement Strategy:</strong></p>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<p style="color: rgba(255,255,255,0.9); font-size: 0.875rem; margin-bottom: 1rem;">{rec["strategy"]}</p>', unsafe_allow_html=True)
            st.markdown(
                '<p style="color: rgba(255,255,255,0.7); font-size: 0.875rem; margin-bottom: 0.5rem;"><strong>Expected ROI:</strong></p>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<p style="color: {rec["color"]}; font-size: 1rem; font-weight: 600;">{rec["expected_roi"]}</p>', unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

# -----------------------------
# PAGE 3: PERFORMANCE
# -----------------------------
elif page == "Performance":
    st.markdown('<h1 class="gradient-text">Model Performance</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size: 1.125rem; color: rgba(255,255,255,0.7); margin-bottom: 2rem;">Comprehensive evaluation metrics and analysis</p>',
        unsafe_allow_html=True,
    )

    if not isinstance(metrics, dict):
        st.warning("Performance metrics not available. Please ensure model_artifacts/final_metrics.pkl exists.")
        st.stop()

    st.markdown("### Core Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">F1-Score</div><div class="metric-value">{safe_metric_value(metrics,"f1_score"):.1%}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Precision</div><div class="metric-value">{safe_metric_value(metrics,"precision"):.1%}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Recall</div><div class="metric-value">{safe_metric_value(metrics,"recall"):.1%}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Accuracy</div><div class="metric-value">{safe_metric_value(metrics,"accuracy"):.1%}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Confusion Matrix Analysis")
    st.markdown(
        '<p style="color: rgba(255,255,255,0.6); margin-bottom: 2rem;">Detailed breakdown of model predictions vs. actual outcomes</p>',
        unsafe_allow_html=True,
    )

    cm = metrics.get("confusion_matrix", {})
    tn = int(cm.get("true_negatives", 0))
    fp = int(cm.get("false_positives", 0))
    fn = int(cm.get("false_negatives", 0))
    tp = int(cm.get("true_positives", 0))

    cm_fig = go.Figure(
        go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Predicted: No Purchase", "Predicted: Purchase"],
            y=["Actual: No Purchase", "Actual: Purchase"],
            colorscale=[
                [0, "rgba(102, 126, 234, 0.2)"],
                [0.5, "rgba(102, 126, 234, 0.5)"],
                [1, "rgba(102, 126, 234, 0.9)"],
            ],
            text=[[f"TN\n{tn:,}", f"FP\n{fp:,}"], [f"FN\n{fn:,}", f"TP\n{tp:,}"]],
            texttemplate="%{text}",
            textfont={"size": 20, "color": "white", "family": "Sora"},
            hovertemplate="%{y}<br>%{x}<br>Count: %{z:,}<extra></extra>",
            showscale=False,
        )
    )
    cm_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white", "family": "Sora"},
        height=500,
        xaxis={"side": "bottom"},
        yaxis={"side": "left"},
        margin=dict(l=100, r=50, t=50, b=100),
    )
    st.plotly_chart(cm_fig, use_container_width=True)

# -----------------------------
# PAGE 4: ROI
# -----------------------------
elif page == "ROI":
    st.markdown('<h1 class="gradient-text">Business Impact Calculator</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size: 1.125rem; color: rgba(255,255,255,0.7); margin-bottom: 1rem;">Project revenue impact and marketing ROI with AI-powered predictions</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size: 0.875rem; color: rgba(255,255,255,0.5); margin-bottom: 2rem;">This calculator uses your actual model performance metrics to estimate real-world business outcomes</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="glass-card tight">', unsafe_allow_html=True)
    st.markdown("### Your Business Parameters")
    st.markdown(
        '<p style="color: rgba(255,255,255,0.6); margin-bottom: 1.5rem;">Enter your current metrics to calculate potential impact</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        sessions = st.number_input("Monthly Website Sessions", min_value=1000, max_value=10000000, value=100000, step=10000)
    with col2:
        avg_order = st.number_input("Average Order Value ($)", min_value=1.0, max_value=10000.0, value=50.0, step=5.0)
    with col3:
        mkt_cost = st.number_input("Marketing Cost per Contact ($)", min_value=0.1, max_value=100.0, value=2.0, step=0.5)
    st.markdown("</div>", unsafe_allow_html=True)

    if not isinstance(metrics, dict):
        st.warning("Metrics not available for ROI calculation. Please ensure model_artifacts/final_metrics.pkl exists.")
        st.stop()

    precision = safe_metric_value(metrics, "precision", 0.0)
    recall = safe_metric_value(metrics, "recall", 0.0)

    if precision <= 0 or recall <= 0:
        st.warning("Precision/Recall in metrics are missing or zero, ROI results may be invalid.")
        precision = max(precision, 1e-9)
        recall = max(recall, 1e-9)

    st.markdown("---")
    st.markdown("### Projected Financial Impact")

    baseline_conversion = 0.015
    buyers_captured = sessions * baseline_conversion * recall
    revenue = buyers_captured * avg_order
    contacts_needed = buyers_captured / precision
    cost = contacts_needed * mkt_cost
    net = revenue - cost
    roi = ((net / cost) * 100) if cost > 0 else 0

    traditional_contacts = sessions * 0.1
    traditional_cost = traditional_contacts * mkt_cost
    traditional_buyers = traditional_contacts * baseline_conversion
    traditional_revenue = traditional_buyers * avg_order
    traditional_net = traditional_revenue - traditional_cost

    cost_savings = traditional_cost - cost
    efficiency_gain = ((cost_savings / traditional_cost) * 100) if traditional_cost > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Projected Revenue</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">${revenue:,.0f}</div>
            <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.5rem;">From {int(buyers_captured):,} captured buyers</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Marketing Investment</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">${cost:,.0f}</div>
            <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.5rem;">Targeting {int(contacts_needed):,} high-intent users</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Net Profit</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">${net:,.0f}</div>
            <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.5rem;">After marketing costs</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-label">Return on Investment</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{roi:.0f}%</div>
            <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.5rem;">Marketing ROI</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="glass-card tight">', unsafe_allow_html=True)
    st.markdown("#### Understanding Your Results")
    st.markdown(
    """
    <div style="margin: 0.5rem 0 1.0rem 0;">
      <span class="pill">Baseline conversion: 1.5%</span>
      <span class="pill">Traditional targeting: 10% sessions</span>
      <span class="pill">BuyerIQ targeting: model-based</span>
    </div>
    """,
    unsafe_allow_html=True,
    )
    st.markdown(
        f"""
**How BuyerIQ Improves ROI:**
- **Precision Targeting:** Only market to users our AI identifies as high-intent, reducing wasted spend by **{efficiency_gain:.0f}%**
- **Higher Capture Rate:** Identify and convert **{recall:.1%}** of potential buyers vs industry average
- **Cost Efficiency:** Spend **${(cost / buyers_captured):.2f}** to acquire each customer vs **${(traditional_cost / traditional_buyers):.2f}** with traditional approaches

**Model Assumptions:**
- Baseline conversion rate: 1.5% (industry standard for e-commerce)
- Traditional approach: Broad targeting of 10% of all sessions
- BuyerIQ approach: Precision targeting based on AI predictions
        """.strip()
    )
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# PAGE 5: BATCH
# -----------------------------
else:
    st.markdown('<h1 class="gradient-text">Batch Predictions</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size: 1.125rem; color: rgba(255,255,255,0.7); margin-bottom: 1rem;">Process multiple sessions simultaneously for bulk analysis</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="font-size: 0.875rem; color: rgba(255,255,255,0.5); margin-bottom: 2rem;">Upload your session data as CSV to get purchase predictions for thousands of users at once</p>',
        unsafe_allow_html=True,
    )

    ok, msg = ensure_model_ready(model)
    if not ok:
        st.error(f"⚠️ {msg}")
        st.stop()

    with st.expander("CSV Format Requirements", expanded=False):
        st.markdown(
            """
**Required Columns:**
- `Administrative`, `Administrative_Duration`
- `Informational`, `Informational_Duration`
- `ProductRelated`, `ProductRelated_Duration`
- `BounceRates`, `ExitRates`, `PageValues`
- `SpecialDay`, `Month`
- `OperatingSystems`, `Browser`, `Region`, `TrafficType`
- `VisitorType`, `Weekend`

**Data Types:**
- Numeric fields: integers or decimals
- Month: 'Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
- VisitorType: 'Returning_Visitor', 'New_Visitor', 'Other'
- Weekend: True/False or 1/0
"""
        )

    st.markdown("### Upload Session Data")
    uploaded = st.file_uploader("Upload CSV file with session data", type=["csv"])

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            st.success(f"Successfully loaded **{len(df_raw):,} sessions** from your file")

            st.markdown("### Data Preview")
            st.markdown('<div class="glass-card tight">', unsafe_allow_html=True)
            st.dataframe(df_raw.head(10), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            valid, err, df = validate_and_coerce_batch(df_raw)
            if not valid:
                st.error(f"Error validating batch data: {err}")
                st.info("Please re-export your CSV with the required columns.")
                st.stop()

            # Optional: warn if many NaNs after coercion
            nan_rate = df[REQUIRED_COLUMNS].isna().mean().mean()
            if nan_rate > 0.05:
                st.warning("Some columns contain many invalid/missing values after type conversion. Predictions may be unreliable.")

            if st.button("Process Batch Predictions", use_container_width=True):
                with st.spinner("AI is analyzing your sessions... This may take a moment for large datasets."):
                    df_in = reorder_to_model_features(df, model)

                    try:
                        preds = model.predict(df_in)
                        probs = model.predict_proba(df_in)[:, 1]
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        st.info("This usually means your CSV columns/types do not match what the model pipeline expects.")
                        st.stop()

                    out = df.copy()
                    out["Prediction"] = ["Purchase" if bool(p) else "No Purchase" for p in preds]
                    out["Probability"] = probs.astype(float)
                    out["Intent_Category"] = np.where(
                        out["Probability"] > 0.7,
                        "High",
                        np.where(out["Probability"] > 0.4, "Moderate", "Low"),
                    )

                    st.success("Batch processing complete!")

                    st.markdown("---")
                    st.markdown("### Results Summary")

                    high_intent = int((out["Probability"] > 0.7).sum())
                    mod_intent = int(((out["Probability"] > 0.4) & (out["Probability"] <= 0.7)).sum())
                    low_intent = int((out["Probability"] <= 0.4).sum())
                    avg_prob = float(out["Probability"].mean())

                    summary_cols = st.columns(4)
                    with summary_cols[0]:
                        st.markdown(
                            f"""
                        <div class="metric-card">
                            <div class="metric-label">🟢 High Intent</div>
                            <div class="metric-value" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{high_intent:,}</div>
                            <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.5rem;">{(high_intent/len(out)*100):.1f}% of sessions</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    with summary_cols[1]:
                        st.markdown(
                            f"""
                        <div class="metric-card">
                            <div class="metric-label">🟡 Moderate Intent</div>
                            <div class="metric-value" style="background: linear-gradient(135deg, #f093fb 0%, #fbbf24 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{mod_intent:,}</div>
                            <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.5rem;">{(mod_intent/len(out)*100):.1f}% of sessions</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    with summary_cols[2]:
                        st.markdown(
                            f"""
                        <div class="metric-card">
                            <div class="metric-label">🔴 Low Intent</div>
                            <div class="metric-value" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{low_intent:,}</div>
                            <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.5rem;">{(low_intent/len(out)*100):.1f}% of sessions</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                    with summary_cols[3]:
                        st.markdown(
                            f"""
                        <div class="metric-card">
                            <div class="metric-label">Average Probability</div>
                            <div class="metric-value">{avg_prob:.1%}</div>
                            <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-top: 0.5rem;">Mean purchase intent</div>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("---")
                    st.markdown("### Full Results Table")
                    st.markdown('<div class="glass-card tight">', unsafe_allow_html=True)
                    st.dataframe(out, use_container_width=True, height=400)
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("---")
                    csv_bytes = out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇Download Complete Results (CSV)",
                        data=csv_bytes,
                        file_name=f"buyeriq_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.info("Please ensure your file is a valid CSV format.")
