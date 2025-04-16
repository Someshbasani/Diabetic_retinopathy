import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load(r"C:\Users\basan\Desktop\datascience class\project123\best_random_forest_model.pkl")

st.set_page_config(page_title="Disease Prediction", layout="centered")
st.title("Disease Prediction using Random Forest")

st.markdown("Enter patient features to predict the prognosis.")

# Example feature inputs (you can customize this)
# Assuming your model was trained on 4 numerical features as an example:
feature_1 = st.number_input("Age", min_value=0.0, max_value=100.0)
feature_2 = st.number_input("Systolic_BP", min_value=0.0, max_value=180.0)
feature_3 = st.number_input("Diastolic_BP", min_value=0.0, max_value=200.0)
feature_4 = st.number_input("Cholesterol", min_value=0.0, max_value=200.0)

# Add more fields based on your actual model's features

# Prediction button
st.subheader("Prediction Retinopathy")
if st.button("Submit"):
    input_data = np.array([[feature_1, feature_2, feature_3, feature_4]])
    prediction = model.predict(input_data)
    st.subheader("Prediction Retinopathy")
    st.write("Retinopathy" if prediction[0] == 1 else "Non_Retinopathy")
