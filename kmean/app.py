import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Mapping cluster IDs to class labels (adjust if your mapping is different)
label_mapping = {0: 'Versicolor', 1: 'Setosa', 2: 'Virginica'}

st.title("Flower Cluster Predictor (K-Means)")

st.write("Enter flower measurements below and click **Predict** to see its cluster and closest Iris class.")

# Input form
with st.form("iris_form"):
    sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1)
    sepal_width  = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.5)
    petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4)
    petal_width  = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2)
    
    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    
    cluster = model.predict(input_scaled)[0]
    predicted_class = label_mapping.get(cluster, "Unknown")

    st.subheader("🌼 Prediction Results")
    st.write(f"**Cluster ID**: {cluster}")
    st.write(f"**Closest Iris Class**: {predicted_class}")

