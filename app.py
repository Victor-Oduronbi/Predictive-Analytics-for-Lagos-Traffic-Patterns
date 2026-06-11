import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration Setup
st.set_page_config(page_title="Lagos Traffic Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Lagos Traffic Congestion Engine")
st.markdown("Predicting transit severity and lost man-hours along major economic corridors.")

# 2. Load the trained machine learning components
@st.cache_resource
def load_artifacts():
    model = joblib.load('lagos_stack_model.pkl')
    scaler = joblib.load('data_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception:
    st.error("Please run 'train_stack.py' first to generate your model artifacts!")
    st.stop()

# 3. Create Sidebar Input Controls (User-Friendly Interface)
st.sidebar.header("🕹️ Corridor Parameters")

corridor_name = st.sidebar.selectbox("Select Target Route Corridor", ["Ikotun-Ipaja Road", "Lekki-Epe Expressway"])
corridor_id = 0 if corridor_name == "Ikotun-Ipaja Road" else 1

hour = st.sidebar.slider("Hour of Day (24hr Clock)", 0, 23, 8)
is_peak = st.sidebar.checkbox("Is Peak Commuter Window?", value=(6 <= hour <= 10 or 16 <= hour <= 21))

st.sidebar.header("🌤️ Environmental Factors")
rain_prob = st.sidebar.slider("NiMet Precipitation Probability (%)", 0.0, 100.0, 75.0)
vehicle_volume = st.sidebar.number_input("Estimated Vehicle Density (Volume Counts)", min_value=100, max_value=5000, value=1800)

# 4. Process Inputs and Run Inference Engine
if st.button("📊 Calculate Traffic Severity Index"):
    # Cyclic Time Transformations
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    
    # Structure features identically to training columns
    input_data = pd.DataFrame([{
        'Is_Peak_Window': int(is_peak),
        'Corridor_ID': corridor_id,
        'NiMet_Rain_Prob': rain_prob,
        'Est_Vehicle_Volume': vehicle_volume,
        'Hour_Sin': hour_sin,
        'Hour_Cos': hour_cos
    }])
    
    # Scale inputs and predict using our stacked ensemble
    input_scaled = scaler.transform(input_data)
    predicted_hours = model.predict(input_scaled)[0]
    
    # 5. Render Beautiful Status Cards
    st.write("---")
    st.subheader(f"Results for {corridor_name} at {hour}:00")
    
    if predicted_hours < 1.0:
        st.success(f"🟢 **LIGHT TRAFFIC** | Est. Delay: {predicted_hours:.2f} hours")
        st.balloons()
    elif 1.0 <= predicted_hours < 2.5:
        st.warning(f"🟡 **MODERATE CONGESTION** | Est. Delay: {predicted_hours:.2f} hours")
    else:
        st.error(f"🔴 **HEAVY GRIDLOCK** | Est. Delay: {predicted_hours:.2f} hours")
        
    st.info("💡 *Insight:* Stacking ensembles combine multiple algorithm geometries to optimize structural accuracy and reduce metric errors.")
