# 🚗 Lagos Traffic Congestion Engine & Predictive Analytics

An end-to-end Machine Learning web platform that quantifies urban transit gridlock severity and predicts economic time loss (lost man-hours) along major Lagos transit corridors. The engine is powered by a Scikit-Learn Multi-Model Stacking Ensemble and served via an interactive Streamlit UI.
<p align="center">
  <img width="1366" height="766" alt="Lagos Traffic Predictor - Google Chrome 6_11_2026 12_45_02 AM" src="https://github.com/user-attachments/assets/421ca99d-40f0-4ad8-827c-2a8818889d24" />
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
* **Feature Standardization**: Implemented Scikit-Learn's `StandardScaler` fitted strictly on training slices to prevent data leakage while equalizing disparate feature scales.

### 3. Multi-Model Stacking Ensemble
To capture complex, multiplicative penalties that linear models fail to resolve, the engine deploys a heterogeneous machine learning stack:
* **Base Regressors**: `RandomForestRegressor` (Ensemble Trees) + `GradientBoostingRegressor` (Iterative Residual Correction).
* **Meta-Estimator**: A `Ridge` linear regression layer that learns how to optimally blend the predictions of both base models to minimize Mean Absolute Error (MAE).

---

## 🚀 Installation & Usage

1. Clone this repository:
```bash
git clone https://github.com
cd lagos-traffic-engine
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn joblib streamlit
```

3. Train the Stacked Ensemble and generate artifacts:
```bash
python train_stack.py
```

4. Launch the local Streamlit Web Server:
```bash
streamlit run app.py
```

---

## 📊 Key Insights & Explainability
Feature Importance Extraction (FIE) verifies that the Stacking Ensemble accurately identifies vehicular volume during peak commuting hours as the primary gridlock driver (~65% importance), while flagging rainfall indicators as the most volatile compound multiplier (~20% importance). 
