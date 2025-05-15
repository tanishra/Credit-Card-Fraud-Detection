import streamlit as st
import pickle
import numpy as np

# Load the trained XGBoost model
with open("saved_models/xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to check if it's fraudulent.")

# Input fields for 30 features
time = st.number_input("Time (seconds since first transaction)", min_value=0.0)
amount = st.number_input("Transaction Amount", min_value=0.0)

# Create sliders/inputs for V1 to V28
v_features = []
for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0, step=0.1)
    v_features.append(val)

# Combine all features into an array
input_data = np.array([[time, amount] + v_features])

if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write("### 🔎 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {probability:.4f})")
    else:
        st.success(f"✅ Legitimate Transaction (Probability of Fraud: {probability:.4f})")
