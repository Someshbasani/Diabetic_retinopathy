import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("best_random_forest_model.pkl")

st.set_page_config(page_title="Disease Prediction", layout="centered")
st.title("Disease Prediction using Random Forest")

st.markdown("Enter patient features to predict the prognosis.")

# Example feature inputs (you can customize this)
# Assuming your model was trained on 4 numerical features as an example:
feature_1 = st.number_input("Age", min_value=0.0, max_value=100.0)
feature_2 = st.number_input("Systolic_BP (Normal: 90-120 mmHg)", min_value=0.0, max_value=180.0)
feature_3 = st.number_input("Diastolic_BP (Normal: 60-80 mmHg)", min_value=0.0, max_value=200.0)
feature_4 = st.number_input("Cholesterol (Normal: <200 mg/dL)", min_value=0.0, max_value=280.0)

# Add more fields based on your actual model's features
# Prediction button
st.subheader("Prediction Retinopathy")
if st.button("Submit"):
    input_data = np.array([[feature_1, feature_2, feature_3, feature_4]])
    prediction = model.predict(input_data)
    st.subheader("Prediction Retinopathy")
    st.write("Retinopathy" if prediction[0] == 1 else "Non_Retinopathy")
# Blood pressure reference table
st.markdown("### 🩸 Blood Pressure Categories")

st.markdown("""
<style>
.table-container {
    overflow-x: auto;
}
.custom-table {
    border-collapse: collapse;
    width: 100%;
    text-align: center;
}
.custom-table th, .custom-table td {
    border: 1px solid #ccc;
    padding: 10px;
}
.custom-table th {
    background-color: #f2f2f2;
}
</style>

<div class="table-container">
<table class="custom-table">
    <tr>
        <th>Category</th>
        <th>Systolic (mm Hg)</th>
        <th>Diastolic (mm Hg)</th>
    </tr>
    <tr>
        <td>Normal</td>
        <td>Less than 120</td>
        <td>Less than 80</td>
    </tr>
    <tr>
        <td>Elevated</td>
        <td>120–129</td>
        <td>Less than 80</td>
    </tr>
    <tr>
        <td>High BP (Stage 1)</td>
        <td>130–139</td>
        <td>80–89</td>
    </tr>
    <tr>
        <td>High BP (Stage 2)</td>
        <td>140 or higher</td>
        <td>90 or higher</td>
    </tr>
    <tr>
        <td>Hypertensive Crisis</td>
        <td>Higher than 180</td>
        <td>Higher than 120</td>
    </tr>
</table>
</div>
""", unsafe_allow_html=True)

