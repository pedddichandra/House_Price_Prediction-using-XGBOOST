import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("best_xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="🏠 House Price Predictor", layout="centered")

st.title("🏠 House Price Prediction App")
st.write("Enter the details below to predict the house price using **XGBoost** model.")

# Sidebar for user input
st.sidebar.header("Input Features")

area = st.sidebar.number_input("Area (sq ft)", min_value=500, max_value=10000, step=100)
bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=1, max_value=10, step=1)
stories = st.sidebar.number_input("Stories", min_value=1, max_value=5, step=1)
parking = st.sidebar.number_input("Parking spaces", min_value=0, max_value=5, step=1)

# Binary inputs
mainroad = st.sidebar.selectbox("Mainroad access", ["Yes", "No"])
guestroom = st.sidebar.selectbox("Guestroom", ["Yes", "No"])
basement = st.sidebar.selectbox("Basement", ["Yes", "No"])
hotwaterheating = st.sidebar.selectbox("Hot Water Heating", ["Yes", "No"])
airconditioning = st.sidebar.selectbox("Air Conditioning", ["Yes", "No"])
prefarea = st.sidebar.selectbox("Preferred Area", ["Yes", "No"])

# Furnishing status
furnishingstatus = st.sidebar.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

# Convert categorical to binary
binary_map = {"Yes": 1, "No": 0}
mainroad = binary_map[mainroad]
guestroom = binary_map[guestroom]
basement = binary_map[basement]
hotwaterheating = binary_map[hotwaterheating]
airconditioning = binary_map[airconditioning]
prefarea = binary_map[prefarea]

# One-hot encoding for furnishing status
furnishingstatus_furnished = 1 if furnishingstatus == "furnished" else 0
furnishingstatus_semi = 1 if furnishingstatus == "semi-furnished" else 0
furnishingstatus_unfurnished = 1 if furnishingstatus == "unfurnished" else 0

# Prepare features for scaling (only numerical features)
numerical_features = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "stories": [stories],
    "parking": [parking]
})

# Scale numerical features
scaled_features = scaler.transform(numerical_features)

# Combine into a single row DataFrame with correct feature names
input_data = pd.DataFrame({
    "area": [scaled_features[0][0]],
    "bedrooms": [scaled_features[0][1]],
    "bathrooms": [scaled_features[0][2]],
    "stories": [scaled_features[0][3]],
    "mainroad": [mainroad],
    "guestroom": [guestroom],
    "basement": [basement],
    "hotwaterheating": [hotwaterheating],
    "airconditioning": [airconditioning],
    "parking": [scaled_features[0][4]],
    "prefarea": [prefarea],
    "furnishingstatus_semi-furnished": [furnishingstatus_semi],
    "furnishingstatus_unfurnished": [furnishingstatus_unfurnished]
})

# Prediction button
if st.button("🔍 Predict House Price"):
    prediction = model.predict(input_data)
    st.success(f"🏡 Predicted House Price: ₹{prediction[0]:,.2f}")

st.write("---")
st.write("Developed with ❤️ using Streamlit and XGBoost.")
