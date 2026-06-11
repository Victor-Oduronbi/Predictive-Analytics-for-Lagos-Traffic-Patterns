# 🚗 Lagos Traffic Congestion Engine & Predictive Analytics

An end-to-end Machine Learning web platform that quantifies urban transit gridlock severity and predicts economic time loss (lost man-hours) along major Lagos transit corridors. The engine is powered by a Scikit-Learn Multi-Model Stacking Ensemble and served via an interactive Streamlit UI featuring live 24-hour predictive trending charts.

🔗 [**Live Dashboard Link**:](https://predictive-analytics-for-lagos-traffic-patterns-jvczcesb3ozwgc.streamlit.app/#lagos-traffic-congestion-engine)
<p align="center">
  <img width="1366" height="766" alt="Google Search - Google Chrome 6_11_2026 2_11_00 AM" src="https://github.com/user-attachments/assets/cce38830-f8e4-4722-bdb0-128140d18d9f" />
  <br>
  <b>Lagos Traffic Engine Dashboard</b>
</p>

## 📈 Executive Summary
Urban gridlock costs major cities billions in lost productivity. This project bypasses commercial API paywalls by utilizing an empirical, domain-driven data simulation framework. By combining macro-level meteorological parameters with localized transit bottlenecks, it maps compound environmental and spatial friction rules onto a 1,000-observation matrix. An advanced ensemble pipeline reverse-engineers these non-linear transit states to predict trip delays down to the minute.

---

## 🛠️ The Architecture & Pipeline Logic

### 1. Domain-Driven Data Calibration & Acquisition
Instead of using generic open datasets, this engine features structured vectors built directly from authoritative local transport and climate data sources:
* **The Danne Institute "Connectivity & Productivity" Framework**: Grounded our temporal peak baseline modifiers (1.8x) using their baseline benchmark showing Lagosians lose a collective 14.12 million daily man-hours.
* **Ipaja-Ikotun Corridor Traffic Surveys**: Calibrated structural spatial penalties (1.3x) based on documented intersection bottlenecks and localized economic layout frictions.
* **NiMet Hydro-Meteorological Impact Indices**: Leveraged rainfall probability and flash-flood potential indices to scale a compounding transit delay modifier (1.5x).

### 2. Feature Engineering Suite
* **Cyclic Trigonometric Time Encoding**: Converted standard hours (0–23) into two-dimensional circular space coordinates (`Hour_Sin` and `Hour_Cos`). This forces the algorithm to recognize that Hour 23 seamlessly connects to Hour 0, matching real-world transit cycles.
* **Feature Standardization**: Implemented Scikit-Learn's `StandardScaler` fitted natively within the deployment pipeline to maintain feature scale equality across disparate inputs without data leakage.

### 3. Multi-Model Stacking Ensemble
To capture complex, multiplicative penalties that linear models fail to resolve, the engine deploys a heterogeneous machine learning stack:
* **Base Regressors**: `RandomForestRegressor` (Ensemble Trees) + `GradientBoostingRegressor` (Iterative Residual Correction).
* **Meta-Estimator**: A `Ridge` linear regression layer that learns how to optimally blend the predictions of both base models to minimize Mean Absolute Error (MAE).

### 4. Interactive Live 24-Hour Analytics Trend Overlay
The pipeline leverages point-inference iteration to generate full-day simulation data arrays dynamically based on active user runtime parameters, printing out real-time vector graphs inside the UI dashboard.
<p align="center">
<img width="1366" height="766" alt="Streamlit - Google Chrome 6_11_2026 2_40_29 AM" src="https://github.com/user-attachments/assets/40430618-ed92-4ae3-ab2d-aaf39f1632b3" />
  <br>
  <b>Lagos Traffic Engine Dashboard</b>
</p>

---

## 🚀 Installation & Usage

1. Clone this repository:
```bash
git clone https://github.com
cd lagos-traffic-engine
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn streamlit
```

3. Launch the Streamlit Web App Server:
```bash
streamlit run app.py
```
