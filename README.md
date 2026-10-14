# ✦ Sales Demand Forecast Dashboard

A minimalistic, Pinterest-aesthetic Streamlit dashboard for sales demand forecasting, powered by a trained **Lasso Regression** model.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Lasso-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)

---

## ✦ Features

| Section | Description |
|---|---|
| **KPI Cards** | Predicted Units, Est. Revenue, Est. Profit, Margin % |
| **Demand Curve** | Monthly forecast across the selected year |
| **Product Mix** | Horizontal bar comparing all 12 item types |
| **Channel Split** | Donut chart — Online vs Offline share |
| **Seasonality Heatmap** | All regions × 12 months heat visualization |
| **Year-over-Year Trend** | 2015–2030 area chart |
| **Priority Breakdown** | Demand by Critical / High / Medium / Low |
| **Model Details** | Expandable section with coefficients & feature importance |

## ✦ Tech Stack

- **Frontend**: Streamlit with custom CSS (earth-tone Pinterest palette)
- **Charts**: Plotly (spline, bar, donut, heatmap, area)
- **Model**: Lasso Regression (scikit-learn) with StandardScaler
- **Typography**: DM Sans

## ✦ Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

### Model File

Place your trained model file at:

```
C:\Users\user\Downloads\best_demand_forecast_model.pkl
```

Or update the path in `app.py` → `load_model()` function.

## ✦ Model Info

- **Algorithm**: Lasso Regression (L1 Regularization)
- **Features**: Region, Country, Item Type, Sales Channel, Order Priority, Unit Price, Unit Cost, Order Month, Order Year
- **Target**: Units Sold
- **Preprocessing**: Label Encoding for categorical features + StandardScaler

## ✦ License

MIT
