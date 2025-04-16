
# Diabetic Retinopathy Prediction

This project is a **binary classification machine learning application** developed to predict whether a patient is likely to suffer from **diabetic retinopathy** based on various blood test-related features.

🔗 **Live Demo**: [Streamlit App](https://diabeticretinopathy-ss.streamlit.app/)

---

##  Business Objective

The main goal of this project is to build a machine learning model that predicts **diabetic retinopathy prognosis** (0: No, 1: Yes) based on patient data. This helps healthcare professionals assess the risk early and take preventive measures.

---

##  Dataset Details

- **Total Rows**: 6,000  
- **Total Features**: 6  

| Feature         | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `ID`            | Unique identifier (numeric)                                                |
| `age`           | Age of the patient                                                         |
| `systolic_bp`   | Systolic blood pressure (normal < 120 mmHg)                                |
| `diastolic_bp`  | Diastolic blood pressure (normal < 80 mmHg)                                |
| `cholesterol`   | Cholesterol level (normal: 125–200 mg/dl)                                  |
| `prognosis`     | Target variable: 0 = No retinopathy, 1 = Has retinopathy (binary class)     |

---
##  Machine Learning Pipeline

- **EDA & Preprocessing**: Handled missing values, feature analysis, scaling
- **Model Used**: Logistic Regression (and optionally tried Random Forest, SVM)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
- **Deployment Framework**: [Streamlit](https://streamlit.io)

---

## Model Deployment

The final model is deployed using **Streamlit**, allowing users to input patient parameters and receive instant predictions.

Try it live here: [https://diabeticretinopathy-ss.streamlit.app/](https://diabeticretinopathy-ss.streamlit.app/)

---

##  Project Structure

```
📁 diabetic-retinopathy-prediction
├──  data/
│   └── diabetic_retinopathy.csv
├── notebooks/
│   └── eda_and_modeling.ipynb
├──  model/
│   └── logistic_model.pkl
├── app.py               # Streamlit app
├── requirements.txt     # Python dependencies
├── README.md            # Project overview
└── final_presentation.pdf
```

---

## ⚙️ How to Run Locally

1. Clone the repo  
```bash
git clone https://github.com/yourusername/diabetic-retinopathy-prediction.git
cd diabetic-retinopathy-prediction
```

2. Install dependencies  
```bash
pip install -r requirements.txt
```

3. Run Streamlit App  
```bash
streamlit run app.py
```

---

##Final Submission Includes

- Source Code  
- Streamlit App  
- Final Presentation (PDF)  
- README and Documentation  

---

## 👥 Contributors

- Basani somesh – Data Science Engineer

---

## 📧 Contact

For any queries, reach out via basanisomesh9959@gmail.com

---

Let me know if you'd like a version with badge icons, contribution guidelines, or deployment instructions for other platforms (like Heroku or Hugging Face).
