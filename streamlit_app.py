import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("car_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Page settings
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Title
st.title("🚗 Used Car Price Predictor")
st.write("### Machine Learning Based Price Estimation")
st.write(
    "Enter the details of your used car to estimate its selling price."
)

st.divider()

# Input section
st.subheader("📋 Enter Car Details")

year = st.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2026,
    value=2018,
    step=1
)

present_price = st.number_input(
    "Present Price (Lakh)",
    min_value=0.1,
    value=8.5,
    step=0.1
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=25000,
    step=1000
)

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

owner = st.selectbox(
    "Previous Owners",
    [0, 1, 2, 3]
)

st.divider()

# Prediction button
if st.button("🔮 Predict Price", use_container_width=True):

    # Create input data
    input_data = {
        "Year": year,
        "Present_Price": present_price,
        "Kms_Driven": kms_driven,
        "Owner": owner,
        "Transmission_Manual": 1 if transmission == "Manual" else 0,
        "Fuel_Type_Petrol": 1 if fuel_type == "Petrol" else 0,
        "Fuel_Type_Diesel": 1 if fuel_type == "Diesel" else 0,
        "Seller_Type_Individual": 1 if seller_type == "Individual" else 0
    }

    input_df = pd.DataFrame([input_data])

    # Match training columns
    input_df = input_df.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Display result
    st.success("Prediction Completed!")

    st.metric(
        "Estimated Selling Price",
        f"₹ {prediction:.2f} Lakh"
    )

    st.write(
        f"Approximately **₹ {prediction * 100000:,.0f}**"
    )

st.divider()

st.caption(
    "Used Car Price Predictor | Random Forest Regression | Python"
)
