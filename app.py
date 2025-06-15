import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load pre-trained scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("rmodel.pkl")
st.set_page_config(page_title="Restaurant Rating Predictor", layout="wide")
st.title("🍽️ Restaurant Rating Predictor")
st.caption("💡 Predict the customer review rating class based on restaurant features")
st.divider()
st.subheader(" Restaurant Information")
col1, col2 = st.columns(2)
with col1:
    averagecost = st.number_input(
        " Average Cost for Two (₹)", 
        min_value=50, max_value=99999, 
        value=1000, step=100
    )
    pricerange = st.selectbox(
        " Price Range (1 = Cheapest, 4 = Most Expensive)",
        [1, 2, 3, 4]
    )

with col2:
    tablebooking = st.radio(" Table Booking Available?", ["Yes", "No"])
    onlinedelivery = st.radio(" Online Delivery Available?", ["Yes", "No"])

bookingstatus = 1 if tablebooking == "Yes" else 0
deliverystatus = 1 if onlinedelivery == "Yes" else 0

features = np.array([[averagecost, bookingstatus, deliverystatus, pricerange]])
scaled_features = scaler.transform(features)

rating_map = {
    0: "😒 Poor",
    1: "😔 Average",
    2: "😁 Good",
    3: "😍 Excellent"
}
st.divider()
predictbutton = st.button(" Predicting Restaurant Rating")

if predictbutton:
    prediction = model.predict(scaled_features)[0]
    label = rating_map.get(int(prediction), "Unknown")
    st.success(f"✅ Predicted Rating Class: {int(prediction)} - {label}")
    st.balloons()

