import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("PMERi_RandomForest_Model.pkl")

st.title("PMERi Environmental Risk Index Dashboard")

st.header("Environmental Inputs")

aqi = st.number_input("Air Quality Index (AQI)", value=0.50)
temp = st.number_input("Temperature (°C)", value=0.50)
humidity = st.number_input("Humidity (%)", value=0.50)
noise = st.number_input("Noise (dBA)", value=0.50)
lighting = st.number_input("Lighting (LUX)", value=0.50)

if st.button("Predict Risk"):

    input_data = pd.DataFrame({
        "Air Quality Index (AQI)": [aqi],
        "Temperature (C)": [temp],
        "Humidity (%)": [humidity],
        "Noise (dBA)": [noise],
        "Lighting (LUX)": [lighting]
    })

    prediction = model.predict(input_data)

    risk_mapping = {
        0: "Low Risk",
        1: "Moderate Risk",
        2: "High Risk"
    }

    st.success(
        f"Predicted Risk: {risk_mapping[prediction[0]]}"
    )