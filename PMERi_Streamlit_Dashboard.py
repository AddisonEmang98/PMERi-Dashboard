import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# =====================================
# Load Model
# =====================================
model = joblib.load("PMERi_RandomForest_Model.pkl")

LOG_FILE = "predictions_log.csv"

# =====================================
# Normalization (DOSH-CALIBRATED + CLIPPED)
# =====================================
def normalize(value, vmin, vmax):
    norm = (value - vmin) / (vmax - vmin)
    return max(0.0, min(1.0, norm))  # clip to [0,1]

# =====================================
# Page Config
# =====================================
st.set_page_config(
    page_title="PMERi Dashboard",
    page_icon="🏭",
    layout="centered"
)

st.title("🏭 PMERi Dashboard")
st.write("Input sensor values into monitoring system.")

# =====================================
# INPUTS
# =====================================
st.header("Air Quality Stressors")

col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 300.0, 50.0, 1.0)
    pm10 = st.number_input("PM10 (µg/m³)", 0.0, 300.0, 80.0, 1.0)

with col2:
    co2 = st.number_input("CO₂ (ppm)", 0.0, 5000.0, 600.0, 1.0)
    hcho = st.number_input("HCHO (ppm)", 0.0, 0.3, 0.05, 0.001, format="%.3f")

st.header("Thermal Stressors")

col3, col4 = st.columns(2)

with col3:
    temp_raw = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0, 0.1)

with col4:
    humidity_raw = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, 0.1)

st.header("Physical Stressors")

col5, col6 = st.columns(2)

with col5:
    noise_raw = st.number_input("Noise (dBA)", 0.0, 120.0, 70.0, 0.1)

with col6:
    lighting_raw = st.number_input("Lighting (Lux)", 0.0, 1000.0, 300.0, 1.0)

# =====================================
# Prediction
# =====================================
if st.button("Predict Risk"):

    # ================================
    # AQI Construction
    # ================================
    pm25_n = normalize(pm25, 15, 75)
    pm10_n = normalize(pm10, 15, 70)
    co2_n = normalize(co2, 0, 1000)
    hcho_n = normalize(hcho, 0, 0.100)

    aqi = (pm25_n + pm10_n + co2_n + hcho_n) / 4

    # ================================
    # Thermal
    # ================================
    temp = normalize(temp_raw, 23, 32.5)
    humidity = normalize(humidity_raw, 40, 70)

    # ================================
    # Physical
    # ================================
    noise = normalize(noise_raw, 0, 85)
    lighting = normalize(lighting_raw, 200, 750)

    # ================================
    # Model Input
    # ================================
    input_data = pd.DataFrame({
        "Air Quality Index (AQI)": [aqi],
        "Temperature (C)": [temp],
        "Humidity (%)": [humidity],
        "Noise (dBA)": [noise],
        "Lighting (LUX)": [lighting]
    })

    # ================================
    # Raw Display
    # ================================
    st.subheader("Raw Sensor Inputs")

    st.dataframe(pd.DataFrame([{
        "PM2.5": pm25,
        "PM10": pm10,
        "CO2": co2,
        "HCHO": hcho,
        "Temperature": temp_raw,
        "Humidity": humidity_raw,
        "Noise": noise_raw,
        "Lighting": lighting_raw
    }]))

    # ================================
    # Prediction (FINAL DECISION)
    # ================================
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    risk_mapping = {0: "Low", 1: "Moderate", 2: "High"}
    predicted_label = risk_mapping.get(prediction[0], "Unknown")

    # ================================
    # PMERi (SUPPORTING ONLY)
    # ================================
    pmeri_live = input_data.mean(axis=1).iloc[0]

    st.subheader("PMERi Index (Supporting Metric)")
    st.metric("PMERi (Normalized)", f"{pmeri_live:.3f}")

    if pmeri_live < 0.4:
        pmeri_cat = "Low Risk"
    elif pmeri_live < 0.7:
        pmeri_cat = "Moderate Risk"
    else:
        pmeri_cat = "High Risk"

    st.write(f"PMERi Category: **{pmeri_cat}**")

    # ================================
    # FINAL RISK (RF ONLY)
    # ================================
    st.subheader("Final Risk Assessment (Random Forest)")

    if predicted_label == "Low":
        st.success(f"FINAL RISK: {predicted_label}")
    elif predicted_label == "Moderate":
        st.warning(f"FINAL RISK: {predicted_label}")
    else:
        st.error(f"FINAL RISK: {predicted_label}")

    st.write("Model Confidence:")

    for cls, prob in zip(model.classes_, probabilities[0]):
        label = risk_mapping.get(cls, str(cls))
        st.write(f"{label}: {prob*100:.3f}%")
        st.progress(prob)

    # ================================
    # Logging
    # ================================
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = pd.DataFrame([{
        "Timestamp": timestamp,
        "PM2.5": pm25,
        "PM10": pm10,
        "CO2": co2,
        "HCHO": hcho,
        "Temperature": temp_raw,
        "Humidity": humidity_raw,
        "Noise": noise_raw,
        "Lighting": lighting_raw,
        "AQI": aqi,
        "Prediction": predicted_label,
        "PMERi": pmeri_live,
        "PMERi Category": pmeri_cat
    }])

    if os.path.exists(LOG_FILE):
        log_entry.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        log_entry.to_csv(LOG_FILE, index=False)

    st.success("Data logged successfully.")

# =====================================
# Feature Importance
# =====================================
st.markdown("---")
st.subheader("Feature Importance")
st.image("Feature_Importance.png", use_container_width=True)

# =====================================
# Footer
# =====================================
st.markdown("---")

st.write(
    "Developed as part of the Bachelor of Mechanical Engineering (Honours) "
    "at University Malaysia Sarawak (UNIMAS)."
)

st.write(
    "Supervisor: Ts. Mohd. Azrin bin Mohd. Said"
)

st.write(
    "FYP Student: Addison Ding Emang (83044)"
)