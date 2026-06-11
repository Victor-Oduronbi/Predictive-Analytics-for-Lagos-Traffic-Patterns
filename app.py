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

# 2. Hybrid Model Loader / Auto-Trainer (Bypasses Python 3.14 Cloud Bugs)
@st.cache_resource
def get_model_and_scaler():
    """
    Attempts to read data and build a fresh streaming model directly inside the cloud 
    container to guarantee zero version mismatches.
    """
    try:
        # Load the CSV data directly from your repo root
        df = pd.read_csv('lagos_traffic_proxies.csv')
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour_Of_Day'] / 24.0)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour_Of_Day'] / 24.0)

        X = df.drop(columns=['Hour_Of_Day', 'Target_Lost_Man_Hours'])
        y = df['Target_Lost_Man_Hours']

        # Scale features natively
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Build and fit the exact same multi-model ensemble stack on the fly
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

# 3. Create Sidebar Input Controls
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
        
    st.info("💡 *Insight:* This production engine trains a dynamic Stacking Regressor on the fly to bypass cloud environment serialization mismatches.")
