# 🩺 Diabetes Prediction Dashboard

A beautiful, interactive **Streamlit Dashboard** that predicts diabetes risk using a **Random Forest Classifier** trained on health data.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)

## ✨ Features

- **🔍 Diabetes Risk Assessment** — Enter 6 health metrics and get an instant prediction
- **📊 Risk Probability Gauge** — Visual gauge chart showing diabetes probability (0–100%)
- **🕸️ Radar Chart** — Risk factor profile across all input features
- **🏆 Feature Importance** — See which features matter most to the model
- **📋 Reference Ranges** — Compare your values against normal medical ranges
- **💎 Glassmorphism UI** — Modern, Pinterest-inspired design with smooth animations

## 📋 Input Features

| Feature | Description | Normal Range |
|---------|-------------|--------------|
| 🎂 Age | Age in years | — |
| ⚖️ BMI | Body Mass Index | 18.5 – 24.9 |
| 🧪 HbA1c Level | Glycated hemoglobin (%) | < 5.7% |
| 🩸 Blood Glucose | Fasting blood glucose (mg/dL) | < 100 mg/dL |
| 💓 Hypertension | High blood pressure (Yes/No) | No |
| ❤️ Heart Disease | Heart disease history (Yes/No) | No |

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Astryn25/Diabetes-Prediction-Dashboard.git
   cd Diabetes-Prediction-Dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run diabetes_app.py
   ```

4. Open your browser at **http://localhost:8501**

## 🧠 Model Details

- **Algorithm:** Random Forest Classifier (100 estimators)
- **Preprocessing:** StandardScaler for feature normalization
- **Output:** Binary classification (Diabetic / Non-Diabetic) with probability scores

## 📁 Project Structure

```
Diabetes-Prediction-Dashboard/
├── diabetes_app.py              # Main Streamlit dashboard
├── best_model_diabetes.pkl      # Trained Random Forest model
├── scaler_diabetes.pkl          # StandardScaler for feature scaling
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## ⚠️ Disclaimer

This tool is for **educational and informational purposes only**. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using Streamlit & Plotly
</p>
