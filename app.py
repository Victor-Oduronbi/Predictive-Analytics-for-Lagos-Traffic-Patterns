import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor

# 1. Page Configuration Setup
st.set_page_config(page_title="Lagos Traffic Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Lagos Traffic Congestion Engine")
st.markdown("Predicting transit severity and lost man-hours along major economic corridors.")

# 2. Hybrid Model Loader / Auto-Trainer
@st.cache_resource
def get_model_and_scaler():
    try:
        df = pd.read_csv('lagos_traffic_proxies.csv')
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour_Of_Day'] / 24.0)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour_Of_Day'] / 24.0)

        X = df.drop(columns=['Hour_Of_Day', 'Target_Lost_Man_Hours'])
        y = df['Target_Lost_Man_Hours']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        base_models = [
            ('random_forest', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)),
            ('grad_boost', GradientBoostingRegressor(random_state=42))
        ]
        stacking_model = StackingRegressor(estimators=base_models, final_estimator=Ridge())
        stacking_model.fit(X_scaled, y)
        
        return stacking_model, scaler
    except Exception as e:
        st.error(f"Initialization Failed: {str(e)}")
        return None, None

model, scaler = get_model_and_scaler()

if model is None:
    st.warning("Please ensure 'lagos_traffic_proxies.csv' is uploaded to your GitHub repository root!")
    st.stop()

# 3. Sidebar Input Controls
st.sidebar.header("🕹️ Corridor Parameters")
corridor_name = st.sidebar.selectbox("Select Target Route Corridor", ["Ikotun-Ipaja Road", "Lekki-Epe Expressway"])
corridor_id = 0 if corridor_name == "Ikotun-Ipaja Road" else 1

hour = st.sidebar.slider("Hour of Day (24hr Clock)", 0, 23, 8)
is_peak = st.sidebar.checkbox("Is Peak Commuter Window?", value=(6 <= hour <= 10 or 16 <= hour <= 21))

st.sidebar.header("🌤️ Environmental Factors")
rain_prob = st.sidebar.slider("NiMet Precipitation Probability (%)", 0.0, 100.0, 75.0)
vehicle_volume = st.sidebar.number_input("Estimated Vehicle Density (Volume Counts)", min_value=100, max_value=5000, value=1800)

# Helper function to compute dynamic point inference
def run_inference(h, p_win, corr, r_prob, vol):
    h_sin = np.sin(2 * np.pi * h / 24.0)
    h_cos = np.cos(2 * np.pi * h / 24.0)
    input_data = pd.DataFrame([{
        'Is_Peak_Window': int(p_win),
        'Corridor_ID': corr,
        'NiMet_Rain_Prob': r_prob,
        'Est_Vehicle_Volume': vol,
        'Hour_Sin': h_sin,
        'Hour_Cos': h_cos
    }])
    input_scaled = scaler.transform(input_data)
    return model.predict(input_scaled)[0]

# 4. Process Inputs and Run Inference Engine
if st.button("📊 Calculate Traffic Severity Index"):
    predicted_hours = run_inference(hour, is_peak, corridor_id, rain_prob, vehicle_volume)
    
    st.write("---")
    st.subheader(f"Results for {corridor_name} at {hour}:00")
    
    if predicted_hours < 1.0:
        st.success(f"🟢 **LIGHT TRAFFIC** | Est. Delay: {predicted_hours:.2f} hours")
        st.balloons()
    elif 1.0 <= predicted_hours < 2.5:
        st.warning(f"🟡 **MODERATE CONGESTION** | Est. Delay: {predicted_hours:.2f} hours")
    else:
        st.error(f"🔴 **HEAVY GRIDLOCK** | Est. Delay: {predicted_hours:.2f} hours")
        
    # --- FIXED INTERACTIVE CHART SECTION ---
    st.write("---")
    st.subheader("📈 24-Hour Corridor Delay Trend")
    st.markdown("This live chart shows how your selected environmental parameters impact travel delays across a full day cycle.")
    
    chart_data = []
    for hr in range(24):
        hr_is_peak = (6 <= hr <= 10 or 16 <= hr <= 21)
        delay_prediction = run_inference(hr, hr_is_peak, corridor_id, rain_prob, vehicle_volume)
        
        # Using plain integer hr as the axis index breaks alphabetical constraints
        chart_data.append({"Hour of Day": hr, "Estimated Delay (Hours)": round(delay_prediction, 2)})
        
    trend_df = pd.DataFrame(chart_data).set_index("Hour of Day")
    
    # Display line chart
    st.line_chart(trend_df)
    st.info("💡 *Insight:* The dual spikes correspond perfectly to your morning and evening peak traffic rush periods combined with your custom weather settings.")
