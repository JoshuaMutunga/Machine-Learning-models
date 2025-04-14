# app.py
import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_iris

# Load model and data info
model = joblib.load('naive_bayes_model.pkl')
iris = load_iris()

st.title("Naive Bayes: Flower Classification")

st.write("Enter the flower measurements below:")

# Inputs
sepal_length = st.number_input("Sepal length (cm)", 0.0, 10.0, step=0.1)
sepal_width  = st.number_input("Sepal width (cm)", 0.0, 10.0, step=0.1)
petal_length = st.number_input("Petal length (cm)", 0.0, 10.0, step=0.1)
petal_width  = st.number_input("Petal width (cm)", 0.0, 10.0, step=0.1)

# Prediction
if st.button("Predict"):
    X_new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(X_new)[0]
    class_name = iris.target_names[prediction]
    st.success(f"The predicted Iris flower is: **{class_name}** 🌼")
