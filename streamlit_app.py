import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# Load the trained model
model = joblib.load("best_XGBoost_model.pkl")

st.set_page_config(page_title="Disease Prediction", layout="centered")
st.title("Disease Prediction using XGBoost")

st.markdown("Enter patient features to predict the prognosis.")

# user inputs
feature_1 = st.number_input("Age", min_value=0.0, max_value=100.0)
feature_2 = st.number_input("Systolic_BP (Normal: 90-120 mmHg)", min_value=0.0, max_value=180.0)
feature_3 = st.number_input("Diastolic_BP (Normal: 60-80 mmHg)", min_value=0.0, max_value=200.0)
feature_4 = st.number_input("Cholesterol (Normal: <200 mg/dL)", min_value=0.0, max_value=280.0)

# Prediction button
st.subheader("Prediction Retinopathy Disease")

if st.button("Submit"):
    input_data = np.array([[feature_1, feature_2, feature_3, feature_4]])
    
    # Predict the class
    prediction = model.predict(input_data)
    
    # Predict probabilities
    probabilities = model.predict_proba(input_data)
    retinopathy_prob = probabilities[0][1] * 100  # probability for class 1

    result = "Retinopathy" if prediction[0] == 1 else "Non_Retinopathy"

    st.subheader("Prediction Result:")
    st.write(f"🩺 Disease Status: **{result}**")
    st.write(f"📈 Prediction Confidence: **{retinopathy_prob:.2f}%**")

    # Save the input data + prediction into DataFrame
    df = pd.DataFrame({
        "Age": [feature_1],
        "Systolic_BP": [feature_2],
        "Diastolic_BP": [feature_3],
        "Cholesterol": [feature_4],
        "Prediction": [result],
        "Confidence (%)": [retinopathy_prob]
    })

    # Download as CSV
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Prediction as CSV",
        data=csv,
        file_name="prediction_data.csv",
        mime="text/csv",
    )

    # Generate and download as image
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('tight')
    ax.axis('off')
    table_data = [
        ["Feature", "Value"],
        ["Age", feature_1],
        ["Systolic_BP", feature_2],
        ["Diastolic_BP", feature_3],
        ["Cholesterol", feature_4],
        ["Prediction", result],
        ["Confidence (%)", f"{retinopathy_prob:.2f}%"]
    ]
    table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    st.download_button(
        label="📸 Download Prediction as Image",
        data=buf.getvalue(),
        file_name="prediction_result.png",
        mime="image/png",
    )

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
