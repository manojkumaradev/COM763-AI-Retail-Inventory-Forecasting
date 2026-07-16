
import streamlit as st
import pandas as pd
import pickle
import numpy as np

@st.cache_resource
def load_model():
    with open('inventory_app_model.pkl', 'rb') as f:
        return pickle.load(f)

data_bundle = load_model()
model = data_bundle['model']
encoders = data_bundle['encoders']
features = data_bundle['features']

st.title("📈 Retail Inventory Forecasting")

st.sidebar.header("Input Features")
store_id = st.sidebar.selectbox("Store ID", encoders['Store ID'].classes_)
product_id = st.sidebar.selectbox("Product ID", encoders['Product ID'].classes_)
category = st.sidebar.selectbox("Category", encoders['Category'].classes_)
region = st.sidebar.selectbox("Region", encoders['Region'].classes_)
inventory = st.sidebar.number_input("Inventory Level", value=100)
ordered = st.sidebar.number_input("Units Ordered", value=50)
forecast = st.sidebar.number_input("Demand Forecast", value=100.0)
price = st.sidebar.number_input("Price", value=50.0)
discount = st.sidebar.slider("Discount (%)", 0, 100, 10)
weather = st.sidebar.selectbox("Weather Condition", encoders['Weather Condition'].classes_)
holiday = st.sidebar.selectbox("Holiday/Promotion", [0, 1])
comp_price = st.sidebar.number_input("Competitor Pricing", value=45.0)
seasonality = st.sidebar.selectbox("Seasonality", encoders['Seasonality'].classes_)
forecast_date = st.sidebar.date_input("Forecast Date")

input_data = pd.DataFrame([{
    'Store ID': encoders['Store ID'].transform([store_id])[0],
    'Product ID': encoders['Product ID'].transform([product_id])[0],
    'Category': encoders['Category'].transform([category])[0],
    'Region': encoders['Region'].transform([region])[0],
    'Inventory Level': inventory,
    'Units Ordered': ordered,
    'Demand Forecast': forecast,
    'Price': price,
    'Discount': discount,
    'Weather Condition': encoders['Weather Condition'].transform([weather])[0],
    'Holiday/Promotion': holiday,
    'Competitor Pricing': comp_price,
    'Seasonality': encoders['Seasonality'].transform([seasonality])[0],
    'Year': forecast_date.year,
    'Month': forecast_date.month,
    'Day': forecast_date.day,
    'DayOfWeek': forecast_date.weekday()
}])

if st.button("Predict"):
    prediction = model.predict(input_data[features])
    st.success(f"Predicted Units Sold: {round(prediction[0], 2)}")
