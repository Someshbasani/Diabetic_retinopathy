import streamlit as st
#import joblib
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Diabetic Retinopathy Prediction", layout="centered")
st.title("Diabetic Retinopathy Prediction using Random Forest")

# Load the trained model with error handling
try:
    # Check if the model file exists
    if not os.path.exists("best_random_forest_model.pkl"):
        st.error("Model file not found. Please ensure 'best_random_forest_model.pkl' is in the same directory.")
        st.stop()
    
    model = joblib.load("best_random_forest_model.pkl")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

st.markdown("Enter patient features to predict diabetic retinopathy risk.")

# Feature inputs - adjust these based on your actual model's requirements
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=45)
    systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=50, max_value=250, value=120)
    hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=20.0, value=6.5, step=0.1)

with col2:
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    diabetes_duration = st.number_input("Diabetes Duration (years)", min_value=0, max_value=50, value=5)

# Prediction button
if st.button("Predict Diabetic Retinopathy Risk"):
    try:
        # Prepare input data - adjust feature order to match your model's training data
        input_data = np.array([[
            age,
            systolic_bp,
            diastolic_bp,
            hba1c,
            bmi,
            diabetes_duration
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        # Display results
        st.subheader("Prediction Results")
        
        if prediction[0] == 1:
            st.error(f"High Risk of Diabetic Retinopathy ({prediction_proba[0][1]*100:.1f}% probability)")
            st.warning("Recommendation: Please consult an ophthalmologist for further evaluation.")
        else:
            st.success(f"Low Risk of Diabetic Retinopathy ({prediction_proba[0][0]*100:.1f}% probability)")
            st.info("Recommendation: Continue regular diabetes management and annual eye exams.")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
