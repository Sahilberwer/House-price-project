import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("data.csv")

# Features & Target
X = data[['area', 'bedrooms', 'bathrooms', 'age']]
y = data['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# UI
st.title("🏠 House Price Predictor")

st.write("Enter house details below:")

area = st.number_input("Area (sq ft)", min_value=0)
bedrooms = st.number_input("Bedrooms", min_value=0)
bathrooms = st.number_input("Bathrooms", min_value=0)
age = st.number_input("Age of house", min_value=0)

# Predict button
if st.button("Predict Price"):
    prediction = model.predict([[area, bedrooms, bathrooms, age]])
    st.success(f"Estimated Price: ₹ {int(prediction[0])}")