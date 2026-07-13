import streamlit as st
import pandas as pd
import pickle

# Load the trained model pipeline
try:
    with open('model.pkl', 'rb') as file:
        model_pipeline = pickle.load(file)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}\nPlease ensure 'model.pkl' is in the same directory.")
    model_pipeline = None

st.title('Retail Inventory Units Sold Predictor')
st.write('Enter the details below to predict the number of units sold.')

if model_pipeline is not None:
    # Define input fields for all features except 'Date' and 'Units Sold'
    # Based on X = df.drop(['Units Sold', 'Date'], axis=1)
    
    st.header('Store and Product Information')
    store_id = st.text_input('Store ID', 'S001')
    product_id = st.text_input('Product ID', 'P0001')
    category = st.selectbox('Category', ['Groceries', 'Toys', 'Electronics', 'Clothing', 'Furniture'])
    region = st.selectbox('Region', ['North', 'South', 'East', 'West'])

    st.header('Inventory and Sales Metrics')
    inventory_level = st.number_input('Inventory Level', min_value=0, value=200)
    units_ordered = st.number_input('Units Ordered', min_value=0, value=100)
    demand_forecast = st.number_input('Demand Forecast', min_value=0.0, value=150.0)
    price = st.number_input('Price', min_value=0.0, value=50.0)
    discount = st.number_input('Discount (%)', min_value=0, max_value=100, value=10)

    st.header('External Factors')
    weather_condition = st.selectbox('Weather Condition', ['Sunny', 'Rainy', 'Cloudy', 'Snowy'])
    holiday_promotion = st.selectbox('Holiday/Promotion', [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
    competitor_pricing = st.number_input('Competitor Pricing', min_value=0.0, value=45.0)
    seasonality = st.selectbox('Seasonality', ['Autumn', 'Summer', 'Spring', 'Winter'])

    if st.button('Predict Units Sold'):
        # Create a DataFrame from the input values
        input_data = pd.DataFrame([
            {
                'Store ID': store_id,
                'Product ID': product_id,
                'Category': category,
                'Region': region,
                'Inventory Level': inventory_level,
                'Units Ordered': units_ordered,
                'Demand Forecast': demand_forecast,
                'Price': price,
                'Discount': discount,
                'Weather Condition': weather_condition,
                'Holiday/Promotion': holiday_promotion,
                'Competitor Pricing': competitor_pricing,
                'Seasonality': seasonality
            }
        ])

        try:
            prediction = model_pipeline.predict(input_data)
            st.success(f"Predicted Units Sold: {prediction[0]:.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.warning("Model could not be loaded. Prediction functionality is disabled.")
