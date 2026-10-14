"""
📊 Sales Demand Forecast Dashboard
A minimalistic, aesthetic dashboard powered by a Lasso regression model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
import calendar

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Demand Forecast",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS — Soft, muted, Pinterest-like palette
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,700;1,9..40,400&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #faf9f6;
    --card:      #ffffff;
    --border:    #e8e4df;
    --text:      #2d2a26;
    --muted:     #8a857e;
    --accent:    #c4785b;
    --accent-lt: #f4ebe5;
    --sage:      #8fae8b;
    --sage-lt:   #e9f0e8;
    --warm:      #d4a574;
    --warm-lt:   #faf0e6;
    --slate:     #7b8fa1;
    --slate-lt:  #e8edf1;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text);
}

.stApp {
    background-color: var(--bg) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: var(--card) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    font-weight: 500;
    font-size: 0.82rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: var(--muted);
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 24px;
    transition: box-shadow 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 20px rgba(0,0,0,0.04);
}

div[data-testid="stMetric"] label {
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted) !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
}

/* ── Chart containers ── */
.chart-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 16px;
}

.chart-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 4px;
    font-weight: 500;
}

.chart-subtitle {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 18px;
}

/* ── Hero header ── */
.hero {
    padding: 12px 0 28px 0;
}

.hero h1 {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 2px;
    color: var(--text);
}

.hero p {
    color: var(--muted);
    font-size: 0.95rem;
    margin: 0;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    padding: 10px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.02em;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #b36a4e !important;
    box-shadow: 0 4px 16px rgba(196,120,91,0.25) !important;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    color: var(--muted) !important;
}

/* ── Tag pill ── */
.tag {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.tag-accent { background: var(--accent-lt); color: var(--accent); }
.tag-sage   { background: var(--sage-lt);   color: var(--sage);   }
.tag-warm   { background: var(--warm-lt);   color: var(--warm);   }
.tag-slate  { background: var(--slate-lt);  color: var(--slate);  }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\user\Downloads\best_demand_forecast_model.pkl")


data = load_model()
model = data["model"]
scaler = data["scaler"]
label_encoder = data["label_encoder"]
categorical_cols = data["categorical_cols"]
features = data["features"]

# ──────────────────────────────────────────────
# Reference Data (derived from scaler stats)
# ──────────────────────────────────────────────
REGIONS = [
    "Australia and Oceania", "Central America and the Caribbean",
    "Europe", "Asia", "Middle East and North Africa",
    "North America", "Sub-Saharan Africa",
]

COUNTRIES = {
    "Asia": ["China", "India", "Japan", "South Korea", "Indonesia", "Thailand", "Vietnam", "Myanmar", "Bangladesh"],
    "Europe": ["United Kingdom", "Germany", "France", "Italy", "Spain", "Poland", "Netherlands", "Sweden", "Norway"],
    "North America": ["United States", "Canada", "Mexico"],
    "Sub-Saharan Africa": ["Nigeria", "Kenya", "South Africa", "Ghana", "Ethiopia", "Tanzania", "Uganda", "Senegal"],
    "Middle East and North Africa": ["Saudi Arabia", "UAE", "Egypt", "Morocco", "Turkey", "Iran", "Iraq", "Algeria"],
    "Central America and the Caribbean": ["Guatemala", "Honduras", "Costa Rica", "Panama", "Cuba", "Jamaica", "Haiti"],
    "Australia and Oceania": ["Australia", "New Zealand", "Papua New Guinea", "Fiji"],
}

ITEM_TYPES = [
    "Baby Food", "Beverages", "Cereal", "Clothes",
    "Cosmetics", "Fruits", "Household", "Meat",
    "Office Supplies", "Personal Care", "Snacks", "Vegetables",
]

SALES_CHANNELS = ["Online", "Offline"]
ORDER_PRIORITIES = ["C", "H", "L", "M"]
PRIORITY_LABELS = {"C": "Critical", "H": "High", "L": "Low", "M": "Medium"}


def predict_demand(region, country, item_type, sales_channel,
                   order_priority, unit_price, unit_cost, month, year):
    """Run the Lasso model and return predicted units sold."""
    cat_map = {col: idx for idx, col in enumerate(categorical_cols)}
    encoded = {
        "Region": REGIONS.index(region) if region in REGIONS else 0,
        "Country": list(COUNTRIES.get(region, ["Unknown"])).index(country)
                   if country in COUNTRIES.get(region, []) else 0,
        "Item Type": ITEM_TYPES.index(item_type) if item_type in ITEM_TYPES else 0,
        "Sales Channel": SALES_CHANNELS.index(sales_channel),
        "Order Priority": ORDER_PRIORITIES.index(order_priority),
    }
    row = np.array([[
        encoded["Region"], encoded["Country"], encoded["Item Type"],
        encoded["Sales Channel"], encoded["Order Priority"],
        unit_price, unit_cost, month, year,
    ]])
    row_scaled = scaler.transform(row)
    pred = model.predict(row_scaled)[0]
    return max(0, pred)


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ✦ Forecast Settings")
    st.caption("Configure prediction parameters")
    st.markdown("---")

    sel_region = st.selectbox("Region", REGIONS, index=3)
    sel_country = st.selectbox(
        "Country", COUNTRIES.get(sel_region, ["Unknown"]), index=0
    )
    sel_item = st.selectbox("Item Type", ITEM_TYPES, index=1)
    sel_channel = st.selectbox("Sales Channel", SALES_CHANNELS, index=0)
    sel_priority = st.selectbox(
        "Order Priority",
        ORDER_PRIORITIES,
        format_func=lambda x: f"{PRIORITY_LABELS[x]}  ({x})",
        index=2,
    )

    st.markdown("---")
    sel_price = st.number_input("Unit Price (\u0024)", 1.0, 1000.0, 150.0, step=5.0)
    sel_cost = st.number_input("Unit Cost (\u0024)", 1.0, 1000.0, 100.0, step=5.0)

    st.markdown("---")
    sel_year = st.slider("Year", 2010, 2030, 2026)
    sel_month = st.slider("Month", 1, 12, 6)

    st.markdown("---")
    run = st.button("✦  Generate Forecast", use_container_width=True)


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>✦ Demand Forecast</h1>
    <p>Predict future demand with your trained Lasso regression model</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MAIN PREDICTION
# ──────────────────────────────────────────────
prediction = predict_demand(
    sel_region, sel_country, sel_item, sel_channel,
    sel_priority, sel_price, sel_cost, sel_month, sel_year,
)

margin = sel_price - sel_cost
revenue_est = prediction * sel_price
profit_est = prediction * margin

# ──────────────────────────────────────────────
# KPI ROW
# ──────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Predicted Units", f"{prediction:,.0f}")
k2.metric("Est. Revenue", f"${revenue_est:,.0f}")
k3.metric("Est. Profit", f"${profit_est:,.0f}")
k4.metric("Margin", f"{(margin / sel_price * 100):.1f}%")

st.markdown("---")

# ──────────────────────────────────────────────
# CHARTS — Row 1
# ──────────────────────────────────────────────
col_a, col_b = st.columns(2)

# — Monthly trend for selected config across the year —
with col_a:
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Demand Curve</div>
        <div class="chart-subtitle">Monthly Forecast · {year}</div>
    </div>
    """.replace("{year}", str(sel_year)), unsafe_allow_html=True)

    monthly = []
    for m in range(1, 13):
        p = predict_demand(
            sel_region, sel_country, sel_item, sel_channel,
            sel_priority, sel_price, sel_cost, m, sel_year,
        )
        monthly.append({"Month": calendar.month_abbr[m], "Units": p, "MonthNum": m})
    df_m = pd.DataFrame(monthly)

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=df_m["Month"], y=df_m["Units"],
        mode="lines+markers",
        line=dict(color="#c4785b", width=2.5, shape="spline"),
        marker=dict(size=7, color="#c4785b", line=dict(width=2, color="#fff")),
        fill="tozeroy",
        fillcolor="rgba(196,120,91,0.08)",
        hovertemplate="<b>%{x}</b><br>Units: %{y:,.0f}<extra></extra>",
    ))
    # Highlight selected month
    sel_row = df_m[df_m["MonthNum"] == sel_month].iloc[0]
    fig_line.add_trace(go.Scatter(
        x=[sel_row["Month"]], y=[sel_row["Units"]],
        mode="markers",
        marker=dict(size=13, color="#c4785b", line=dict(width=3, color="#fff"),
                    symbol="circle"),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig_line.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0), height=320,
        xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#8a857e")),
        yaxis=dict(showgrid=True, gridcolor="#f0ece7", zeroline=False,
                   tickfont=dict(size=11, color="#8a857e")),
        showlegend=False,
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})


# — Item type comparison —
with col_b:
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Product Mix</div>
        <div class="chart-subtitle">Demand by Item Type</div>
    </div>
    """, unsafe_allow_html=True)

    item_data = []
    for it in ITEM_TYPES:
        p = predict_demand(
            sel_region, sel_country, it, sel_channel,
            sel_priority, sel_price, sel_cost, sel_month, sel_year,
        )
        item_data.append({"Item": it, "Units": p})
    df_items = pd.DataFrame(item_data).sort_values("Units", ascending=True)

    colors = ["#c4785b" if it == sel_item else "#e8e4df" for it in df_items["Item"]]

    fig_bar = go.Figure(go.Bar(
        x=df_items["Units"], y=df_items["Item"],
        orientation="h",
        marker=dict(color=colors, cornerradius=6),
        hovertemplate="<b>%{y}</b><br>Units: %{x:,.0f}<extra></extra>",
    ))
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0), height=320,
        xaxis=dict(showgrid=True, gridcolor="#f0ece7", zeroline=False,
                   tickfont=dict(size=11, color="#8a857e")),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#8a857e")),
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})


# ──────────────────────────────────────────────
# CHARTS — Row 2
# ──────────────────────────────────────────────
col_c, col_d = st.columns(2)

# — Channel comparison —
with col_c:
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Channel Split</div>
        <div class="chart-subtitle">Online vs Offline</div>
    </div>
    """, unsafe_allow_html=True)

    channel_data = []
    for ch in SALES_CHANNELS:
        p = predict_demand(
            sel_region, sel_country, sel_item, ch,
            sel_priority, sel_price, sel_cost, sel_month, sel_year,
        )
        channel_data.append({"Channel": ch, "Units": p})
    df_ch = pd.DataFrame(channel_data)

    fig_donut = go.Figure(go.Pie(
        labels=df_ch["Channel"], values=df_ch["Units"],
        hole=0.65,
        marker=dict(colors=["#c4785b", "#8fae8b"], line=dict(color="#fff", width=3)),
        textinfo="label+percent",
        textfont=dict(size=12, color="#2d2a26", family="DM Sans"),
        hovertemplate="<b>%{label}</b><br>Units: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
    ))
    fig_donut.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=10, b=10), height=320,
        showlegend=False,
        font=dict(family="DM Sans"),
        annotations=[dict(
            text=f"<b>{sum(df_ch['Units']):,.0f}</b><br><span style='font-size:11px;color:#8a857e'>Total</span>",
            showarrow=False, font=dict(size=18, color="#2d2a26", family="DM Sans"),
        )],
    )
    st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})


# — Regional heatmap across months —
with col_d:
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Seasonality</div>
        <div class="chart-subtitle">Regional Demand Heatmap</div>
    </div>
    """, unsafe_allow_html=True)

    heat_data = []
    for reg in REGIONS:
        first_country = COUNTRIES.get(reg, ["Unknown"])[0]
        for m in range(1, 13):
            p = predict_demand(
                reg, first_country, sel_item, sel_channel,
                sel_priority, sel_price, sel_cost, m, sel_year,
            )
            heat_data.append({"Region": reg.split(" ")[0], "Month": calendar.month_abbr[m], "Units": p})
    df_heat = pd.DataFrame(heat_data)
    heat_pivot = df_heat.pivot(index="Region", columns="Month", values="Units")
    month_order = [calendar.month_abbr[i] for i in range(1, 13)]
    heat_pivot = heat_pivot[month_order]

    fig_heat = go.Figure(go.Heatmap(
        z=heat_pivot.values,
        x=heat_pivot.columns.tolist(),
        y=heat_pivot.index.tolist(),
        colorscale=[
            [0, "#faf9f6"],
            [0.3, "#f4ebe5"],
            [0.6, "#d4a574"],
            [1, "#c4785b"],
        ],
        showscale=False,
        hovertemplate="<b>%{y}</b> · %{x}<br>Units: %{z:,.0f}<extra></extra>",
    ))
    fig_heat.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0), height=320,
        xaxis=dict(tickfont=dict(size=10, color="#8a857e"), side="bottom"),
        yaxis=dict(tickfont=dict(size=10, color="#8a857e"), autorange="reversed"),
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})


# ──────────────────────────────────────────────
# CHARTS — Row 3: Multi-year & Priority
# ──────────────────────────────────────────────
col_e, col_f = st.columns(2)

# — Multi-year trend —
with col_e:
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Trend</div>
        <div class="chart-subtitle">Demand Over the Years</div>
    </div>
    """, unsafe_allow_html=True)

    year_data = []
    for y in range(2015, 2031):
        p = predict_demand(
            sel_region, sel_country, sel_item, sel_channel,
            sel_priority, sel_price, sel_cost, sel_month, y,
        )
        year_data.append({"Year": y, "Units": p})
    df_yr = pd.DataFrame(year_data)

    fig_area = go.Figure()
    fig_area.add_trace(go.Scatter(
        x=df_yr["Year"], y=df_yr["Units"],
        fill="tozeroy",
        fillcolor="rgba(143,174,139,0.12)",
        line=dict(color="#8fae8b", width=2.5, shape="spline"),
        mode="lines",
        hovertemplate="<b>%{x}</b><br>Units: %{y:,.0f}<extra></extra>",
    ))
    # Mark selected year
    sel_yr_row = df_yr[df_yr["Year"] == sel_year].iloc[0]
    fig_area.add_trace(go.Scatter(
        x=[sel_yr_row["Year"]], y=[sel_yr_row["Units"]],
        mode="markers",
        marker=dict(size=11, color="#8fae8b", line=dict(width=3, color="#fff")),
        showlegend=False, hoverinfo="skip",
    ))
    fig_area.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0), height=300,
        xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#8a857e"), dtick=2),
        yaxis=dict(showgrid=True, gridcolor="#f0ece7", zeroline=False,
                   tickfont=dict(size=11, color="#8a857e")),
        showlegend=False,
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig_area, use_container_width=True, config={"displayModeBar": False})


# — Priority breakdown —
with col_f:
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Priority</div>
        <div class="chart-subtitle">Demand by Order Priority</div>
    </div>
    """, unsafe_allow_html=True)

    pri_data = []
    pri_colors = {"C": "#c4785b", "H": "#d4a574", "L": "#8fae8b", "M": "#7b8fa1"}
    for pr in ORDER_PRIORITIES:
        p = predict_demand(
            sel_region, sel_country, sel_item, sel_channel,
            pr, sel_price, sel_cost, sel_month, sel_year,
        )
        pri_data.append({
            "Priority": PRIORITY_LABELS[pr],
            "Units": p,
            "Color": pri_colors[pr],
        })
    df_pri = pd.DataFrame(pri_data)

    fig_pri = go.Figure(go.Bar(
        x=df_pri["Priority"], y=df_pri["Units"],
        marker=dict(
            color=df_pri["Color"],
            cornerradius=8,
        ),
        width=0.55,
        hovertemplate="<b>%{x}</b><br>Units: %{y:,.0f}<extra></extra>",
    ))
    fig_pri.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0), height=300,
        xaxis=dict(showgrid=False, tickfont=dict(size=12, color="#8a857e")),
        yaxis=dict(showgrid=True, gridcolor="#f0ece7", zeroline=False,
                   tickfont=dict(size=11, color="#8a857e")),
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig_pri, use_container_width=True, config={"displayModeBar": False})


# ──────────────────────────────────────────────
# MODEL INFO (expandable)
# ──────────────────────────────────────────────
st.markdown("---")

with st.expander("✦  Model Details"):
    mi1, mi2, mi3 = st.columns(3)
    with mi1:
        st.markdown("**Algorithm**")
        st.code("Lasso Regression (L1)")
    with mi2:
        st.markdown("**Regularization (α)**")
        st.code(f"{model.alpha}")
    with mi3:
        st.markdown("**Features**")
        st.code(f"{model.n_features_in_}")

    st.markdown("**Feature Coefficients**")
    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_,
    }).sort_values("Coefficient", key=abs, ascending=False)

    fig_coef = go.Figure(go.Bar(
        x=coef_df["Coefficient"],
        y=coef_df["Feature"],
        orientation="h",
        marker=dict(
            color=["#c4785b" if c > 0 else "#7b8fa1" for c in coef_df["Coefficient"]],
            cornerradius=5,
        ),
        hovertemplate="<b>%{y}</b><br>Coeff: %{x:.2f}<extra></extra>",
    ))
    fig_coef.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0), height=280,
        xaxis=dict(showgrid=True, gridcolor="#f0ece7", zeroline=True,
                   zerolinecolor="#e8e4df", tickfont=dict(size=11, color="#8a857e")),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color="#8a857e")),
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig_coef, use_container_width=True, config={"displayModeBar": False})


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:36px 0 20px 0;">
    <span style="color:#c8c3bc; font-size:0.78rem; letter-spacing:0.06em;">
        BUILT WITH STREAMLIT · LASSO REGRESSION MODEL · ✦
    </span>
</div>
""", unsafe_allow_html=True)
