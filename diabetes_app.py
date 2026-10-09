import streamlit as st
import numpy as np
import joblib
import warnings
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

# ─── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor 🩺",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

/* ── Global ── */
.stApp {
    background: linear-gradient(160deg, #F0F7FF 0%, #EFF6FF 25%, #F5F0FF 50%, #FFF5F7 75%, #F0F7FF 100%);
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit elements */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 800px;}

/* ── Hero Section ── */
.hero-container {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    animation: fadeInDown 0.8s ease-out;
}
@keyframes fadeInDown {
    from {opacity: 0; transform: translateY(-20px);}
    to {opacity: 1; transform: translateY(0);}
}
.hero-emoji {
    font-size: 3.5rem;
    margin-bottom: 0.5rem;
    display: inline-block;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% {transform: scale(1);}
    50% {transform: scale(1.1);}
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 50%, #A855F7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    line-height: 1.2;
}
.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 1.05rem;
    color: #8E8EA0;
    letter-spacing: 0.02em;
}

/* ── Glass Card ── */
.glass-card {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 24px;
    border: 1px solid rgba(255, 255, 255, 0.8);
    box-shadow:
        0 8px 32px rgba(14, 165, 233, 0.08),
        0 2px 8px rgba(99, 102, 241, 0.06);
    padding: 2rem 2.2rem;
    margin: 1rem 0;
    transition: all 0.3s ease;
}
.glass-card:hover {
    box-shadow:
        0 12px 40px rgba(14, 165, 233, 0.12),
        0 4px 12px rgba(99, 102, 241, 0.08);
    transform: translateY(-2px);
}

/* ── Section Labels ── */
.section-label {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #6366F1;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99, 102, 241, 0.3), transparent);
}

/* ── Input Styling ── */
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    border-radius: 16px !important;
    border: 2px solid rgba(99, 102, 241, 0.15) !important;
    padding: 0.7rem 1rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    background: rgba(255, 255, 255, 0.8) !important;
}
.stNumberInput > div > div > input:focus {
    border-color: rgba(99, 102, 241, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* ── Streamlit Labels ── */
.stNumberInput label, .stSelectbox label, .stSlider label {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: #4A4A5A !important;
    font-size: 0.92rem !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 50%, #A855F7 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.08rem !important;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3) !important;
    text-transform: none;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.4) !important;
}
.stButton > button:active {
    transform: translateY(0) scale(0.99) !important;
}

/* ── Result Card ── */
.result-card {
    backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 2rem;
    margin: 1.5rem 0;
    text-align: center;
    animation: scaleIn 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}
@keyframes scaleIn {
    from {opacity: 0; transform: scale(0.95);}
    to {opacity: 1; transform: scale(1);}
}
.result-low {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.08) 0%, rgba(16, 185, 129, 0.12) 100%);
    border: 1.5px solid rgba(34, 197, 94, 0.25);
}
.result-high {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(220, 38, 38, 0.12) 100%);
    border: 1.5px solid rgba(239, 68, 68, 0.25);
}
.result-value {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 3.2rem;
    line-height: 1.1;
}
.result-low .result-value {
    background: linear-gradient(135deg, #22C55E, #10B981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.result-high .result-value {
    background: linear-gradient(135deg, #EF4444, #DC2626);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.result-label {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 1.3rem;
    margin-top: 0.5rem;
}
.result-low .result-label { color: #16A34A; }
.result-high .result-label { color: #DC2626; }
.result-subtitle {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    font-size: 0.95rem;
    color: #8E8EA0;
    margin-top: 0.3rem;
}

/* ── Stat Pills ── */
.stat-row {
    display: flex;
    gap: 0.8rem;
    margin-top: 1rem;
    flex-wrap: wrap;
    justify-content: center;
}
.stat-pill {
    background: rgba(255, 255, 255, 0.7);
    border-radius: 14px;
    padding: 0.7rem 1.2rem;
    border: 1px solid rgba(99, 102, 241, 0.12);
    flex: 1;
    min-width: 140px;
    text-align: center;
    transition: all 0.3s ease;
}
.stat-pill:hover {
    background: rgba(255, 255, 255, 0.9);
    transform: translateY(-2px);
}
.stat-pill-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8E8EA0;
    margin-bottom: 0.2rem;
}
.stat-pill-value {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1.15rem;
    color: #4A4A5A;
}

/* ── Risk Badge ── */
.risk-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
    margin-top: 0.5rem;
}
.risk-low {background: rgba(74, 222, 128, 0.15); color: #16A34A; border: 1px solid rgba(74, 222, 128, 0.3);}
.risk-high {background: rgba(248, 113, 113, 0.15); color: #DC2626; border: 1px solid rgba(248, 113, 113, 0.3);}

/* ── Info Box ── */
.info-box {
    background: rgba(99, 102, 241, 0.06);
    border-radius: 14px;
    padding: 0.8rem 1.2rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #6366F1;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-top: 0.8rem;
    border: 1px solid rgba(99, 102, 241, 0.12);
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    color: #B0B0C0;
}

/* ── Plotly Chart Container ── */
.stPlotlyChart {
    border-radius: 20px;
    overflow: hidden;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
    background: linear-gradient(135deg, #0EA5E9, #6366F1) !important;
}

/* ── Divider ── */
.soft-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.2), transparent);
    margin: 1.5rem 0;
}

/* ── Disclaimer ── */
.disclaimer-box {
    background: rgba(251, 191, 36, 0.08);
    border: 1px solid rgba(251, 191, 36, 0.2);
    border-radius: 14px;
    padding: 1rem 1.5rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: #92400E;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model & Scaler ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("best_model_diabetes.pkl")


@st.cache_resource
def load_scaler():
    return joblib.load("scaler_diabetes.pkl")


model = load_model()
scaler = load_scaler()

FEATURE_NAMES = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']


# ─── Helper Functions ────────────────────────────────────────────────────
def predict_diabetes(age, hypertension, heart_disease, bmi, hba1c, glucose):
    """Predict diabetes risk using the loaded model with scaling."""
    features = np.array([[age, hypertension, heart_disease, bmi, hba1c, glucose]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    return prediction, probability


def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight", "#3B82F6"
    elif bmi < 25:
        return "Normal", "#22C55E"
    elif bmi < 30:
        return "Overweight", "#F59E0B"
    else:
        return "Obese", "#EF4444"


def get_hba1c_status(hba1c):
    if hba1c < 5.7:
        return "Normal", "#22C55E"
    elif hba1c < 6.5:
        return "Pre-diabetic", "#F59E0B"
    else:
        return "Diabetic Range", "#EF4444"


def get_glucose_status(glucose):
    if glucose < 100:
        return "Normal", "#22C55E"
    elif glucose < 126:
        return "Pre-diabetic", "#F59E0B"
    else:
        return "Diabetic Range", "#EF4444"


# ─── Hero Section ────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-emoji">🩺</div>
    <div class="hero-title">Diabetes Risk Predictor</div>
    <div class="hero-subtitle">Enter your health metrics and let our ML model assess your diabetes risk</div>
</div>
""", unsafe_allow_html=True)

# ─── Input Section ───────────────────────────────────────────────────────
st.markdown('<div class="section-label">📋 Your Health Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("🎂 Age (years)", min_value=1, max_value=120, value=35, step=1)
    bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1,
                          help="Body Mass Index. Normal range: 18.5 – 24.9")
    hypertension = st.selectbox("💓 Hypertension", ["No", "Yes"],
                                help="Do you have high blood pressure?")

with col2:
    blood_glucose = st.number_input("🩸 Blood Glucose Level (mg/dL)", min_value=50, max_value=400, value=120, step=1,
                                    help="Fasting blood glucose level")
    hba1c = st.number_input("🧪 HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.5, step=0.1,
                            help="Hemoglobin A1c (glycated hemoglobin). Normal: < 5.7%")
    heart_disease = st.selectbox("❤️ Heart Disease", ["No", "Yes"],
                                 help="Do you have heart disease?")

# Convert Yes/No to 1/0
hypertension_val = 1 if hypertension == "Yes" else 0
heart_disease_val = 1 if heart_disease == "Yes" else 0

# ── Quick Stats ──
bmi_cat, bmi_color = get_bmi_category(bmi)
hba1c_status, hba1c_color = get_hba1c_status(hba1c)
glucose_status, glucose_color = get_glucose_status(blood_glucose)

st.markdown(f"""
<div class="stat-row">
    <div class="stat-pill">
        <div class="stat-pill-label">BMI Status</div>
        <div class="stat-pill-value" style="color: {bmi_color};">{bmi_cat}</div>
    </div>
    <div class="stat-pill">
        <div class="stat-pill-label">HbA1c Status</div>
        <div class="stat-pill-value" style="color: {hba1c_color};">{hba1c_status}</div>
    </div>
    <div class="stat-pill">
        <div class="stat-pill-label">Glucose Status</div>
        <div class="stat-pill-value" style="color: {glucose_color};">{glucose_status}</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)

# ─── Predict Button ─────────────────────────────────────────────────────
predict_clicked = st.button("🔍  Assess My Diabetes Risk", use_container_width=True)

# ─── Results ─────────────────────────────────────────────────────────────
if predict_clicked:
    prediction, probability = predict_diabetes(
        age, hypertension_val, heart_disease_val, bmi, hba1c, blood_glucose
    )
    prob_diabetic = probability[1] * 100
    prob_healthy = probability[0] * 100
    is_diabetic = prediction == 1
    risk_class = "high" if is_diabetic else "low"

    # ── Result Card ──
    st.markdown(f"""
    <div class="result-card result-{risk_class}">
        <div class="result-value">{prob_diabetic:.1f}%</div>
        <div class="result-label">
            {"⚠️ High Diabetes Risk" if is_diabetic else "✅ Low Diabetes Risk"}
        </div>
        <div class="result-subtitle">
            Model Confidence: {max(prob_diabetic, prob_healthy):.1f}%
        </div>
        <div style="margin-top: 0.8rem;">
            <span class="risk-badge risk-{risk_class}">
                {"Diabetic" if is_diabetic else "Non-Diabetic"}
            </span>
        </div>
        <div class="stat-row">
            <div class="stat-pill">
                <div class="stat-pill-label">Risk Score</div>
                <div class="stat-pill-value">{prob_diabetic:.1f}%</div>
            </div>
            <div class="stat-pill">
                <div class="stat-pill-label">Age</div>
                <div class="stat-pill-value">{age} yrs</div>
            </div>
            <div class="stat-pill">
                <div class="stat-pill-label">BMI</div>
                <div class="stat-pill-value">{bmi:.1f}</div>
            </div>
            <div class="stat-pill">
                <div class="stat-pill-label">HbA1c</div>
                <div class="stat-pill-value">{hba1c:.1f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Risk Probability Gauge ──
    st.markdown('<div class="section-label">📊 Risk Probability</div>', unsafe_allow_html=True)

    gauge_color = "#EF4444" if is_diabetic else "#22C55E"
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_diabetic,
        number={'suffix': '%', 'font': {'size': 42, 'family': 'Outfit', 'color': '#4A4A5A'}},
        title={'text': 'Diabetes Risk Score', 'font': {'size': 16, 'family': 'Inter', 'color': '#8E8EA0'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#E5E7EB'},
            'bar': {'color': gauge_color, 'thickness': 0.3},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.1)'},
                {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.1)'},
            ],
            'threshold': {
                'line': {'color': '#6366F1', 'width': 3},
                'thickness': 0.8,
                'value': prob_diabetic,
            },
        },
    ))
    fig_gauge.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
    )
    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    # ── Radar Chart: Risk Factor Breakdown ──
    st.markdown('<div class="section-label">🕸️ Risk Factor Profile</div>', unsafe_allow_html=True)

    # Normalize each feature to a 0-100 scale for radar display
    radar_values = [
        min(age / 80 * 100, 100),                          # Age normalized to 80
        hypertension_val * 100,                              # Binary 0/100
        heart_disease_val * 100,                             # Binary 0/100
        min(bmi / 45 * 100, 100),                           # BMI normalized to 45
        min(hba1c / 10 * 100, 100),                         # HbA1c normalized to 10
        min(blood_glucose / 300 * 100, 100),                # Glucose normalized to 300
    ]
    radar_labels = ['Age', 'Hypertension', 'Heart Disease', 'BMI', 'HbA1c', 'Blood Glucose']

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_values + [radar_values[0]],  # close the polygon
        theta=radar_labels + [radar_labels[0]],
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.12)',
        line=dict(color='#6366F1', width=2.5),
        marker=dict(size=7, color='#6366F1'),
        name='Your Profile',
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.0f}/100<extra></extra>',
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor='rgba(99, 102, 241, 0.08)',
                tickfont=dict(size=10, color='#8E8EA0'),
            ),
            angularaxis=dict(
                gridcolor='rgba(99, 102, 241, 0.1)',
                tickfont=dict(size=12, family='Inter', color='#4A4A5A'),
            ),
            bgcolor='rgba(0,0,0,0)',
        ),
        showlegend=False,
        height=380,
        margin=dict(l=60, r=60, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

    # ── Feature Importance ──
    st.markdown('<div class="section-label">🏆 Feature Importance</div>', unsafe_allow_html=True)

    importances = model.feature_importances_
    feature_display = ['Age', 'Hypertension', 'Heart Disease', 'BMI', 'HbA1c Level', 'Blood Glucose']
    sorted_idx = np.argsort(importances)

    fig_imp = go.Figure()
    fig_imp.add_trace(go.Bar(
        x=importances[sorted_idx],
        y=[feature_display[i] for i in sorted_idx],
        orientation='h',
        marker=dict(
            color=importances[sorted_idx],
            colorscale=[[0, '#0EA5E9'], [0.5, '#6366F1'], [1, '#A855F7']],
            cornerradius=8,
        ),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
    ))
    fig_imp.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#4A4A5A'),
        xaxis=dict(
            title='Importance',
            gridcolor='rgba(99, 102, 241, 0.08)',
            zerolinecolor='rgba(99, 102, 241, 0.08)',
            tickfont=dict(size=11),
        ),
        yaxis=dict(tickfont=dict(size=12)),
    )
    st.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})

    # ── Reference Ranges Table ──
    st.markdown('<div class="section-label">📋 Your Values vs. Reference Ranges</div>', unsafe_allow_html=True)

    table_data = [
        ("Age", f"{age} years", "—", "Risk increases with age"),
        ("BMI", f"{bmi:.1f}", "18.5 – 24.9", bmi_cat),
        ("HbA1c", f"{hba1c:.1f}%", "< 5.7%", hba1c_status),
        ("Blood Glucose", f"{blood_glucose} mg/dL", "< 100 mg/dL", glucose_status),
        ("Hypertension", hypertension, "No", "⚠️ Risk factor" if hypertension_val else "✅ Normal"),
        ("Heart Disease", heart_disease, "No", "⚠️ Risk factor" if heart_disease_val else "✅ Normal"),
    ]

    table_html = """
    <table style="width: 100%; border-collapse: separate; border-spacing: 0 6px; font-family: 'Inter', sans-serif; font-size: 0.88rem;">
        <thead>
            <tr style="color: #8E8EA0; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.1em;">
                <th style="text-align: left; padding: 0.5rem 1rem; font-weight: 500;">Metric</th>
                <th style="text-align: center; padding: 0.5rem 1rem; font-weight: 500;">Your Value</th>
                <th style="text-align: center; padding: 0.5rem 1rem; font-weight: 500;">Normal Range</th>
                <th style="text-align: center; padding: 0.5rem 1rem; font-weight: 500;">Status</th>
            </tr>
        </thead>
        <tbody>
    """
    for metric, value, normal, status in table_data:
        is_warning = any(w in status for w in ["Overweight", "Obese", "Pre-diabetic", "Diabetic", "⚠️"])
        row_bg = "rgba(239, 68, 68, 0.06)" if is_warning else "rgba(255,255,255,0.5)"
        row_border = "2px solid rgba(239, 68, 68, 0.2)" if is_warning else "1px solid rgba(99, 102, 241, 0.08)"
        status_color = "#DC2626" if is_warning else "#16A34A"

        table_html += f"""
            <tr style="background: {row_bg};">
                <td style="padding: 0.6rem 1rem; border-left: {row_border}; border-top: {row_border}; border-bottom: {row_border}; border-radius: 12px 0 0 12px; font-weight: 500; color: #4A4A5A;">
                    {metric}
                </td>
                <td style="padding: 0.6rem 1rem; border-top: {row_border}; border-bottom: {row_border}; text-align: center; color: #6366F1; font-weight: 600;">
                    {value}
                </td>
                <td style="padding: 0.6rem 1rem; border-top: {row_border}; border-bottom: {row_border}; text-align: center; color: #8E8EA0;">
                    {normal}
                </td>
                <td style="padding: 0.6rem 1rem; border-right: {row_border}; border-top: {row_border}; border-bottom: {row_border}; border-radius: 0 12px 12px 0; text-align: center; font-weight: 600; color: {status_color};">
                    {status}
                </td>
            </tr>
        """
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    # ── Disclaimer ──
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational and informational purposes only.
        It is NOT a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified healthcare provider for medical decisions.
    </div>
    """, unsafe_allow_html=True)

# ─── Sidebar: About ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    ### 🩺 About This App

    This **Diabetes Risk Predictor** uses a **Random Forest Classifier** trained on health data to estimate diabetes risk.

    **Input Features:**
    - 🎂 Age
    - 💓 Hypertension
    - ❤️ Heart Disease
    - ⚖️ BMI
    - 🧪 HbA1c Level
    - 🩸 Blood Glucose Level

    **Model:** Random Forest (100 estimators)

    **Scaling:** StandardScaler

    ---
    *Built with Streamlit & Plotly*
    """)

# ─── Footer ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Powered by Machine Learning · Random Forest Classifier<br>
    <span style="font-size: 0.7rem;">Predictions are estimates based on training data. Consult a healthcare professional for medical advice.</span>
</div>
""", unsafe_allow_html=True)
