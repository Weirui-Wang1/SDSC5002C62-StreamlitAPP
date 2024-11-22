import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from openai import OpenAI
client = OpenAI(api_key='sk-xU154GlXJPiJt7dmU3OUtP9eUBJEkL5CSQvVdLbdXWy0UfFH', base_url="https://api.gpts.vin/v1")
# client = OpenAI(api_key='sk-svcacct-yUj8UDhHgaDIUPaB2LmSqZ1aQA8Vn3B9NOmwcyQEmIsT3BlbkFJtT8zLkBndfFBzFD97Y3XflVcvvFTL2gNxPynllDaHZgA')

# Load the model
try:
    model = joblib.load('XGBoost.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure the files are in the correct directory.")
    st.stop()

# Define feature options
cp_options = {
    1: 'Typical angina (1)',
    2: 'Atypical angina (2)',
    3: 'Non-anginal pain (3)',
    4: 'Asymptomatic (4)'
}

restecg_options = {
    0: 'Normal (0)',
    1: 'ST-T wave abnormality (1)',
    2: 'Left ventricular hypertrophy (2)'
}

slope_options = {
    1: 'Upsloping (1)',
    2: 'Flat (2)',
    3: 'Downsloping (3)'
}

thal_options = {
    3: 'Normal (3)',
    6: 'Fixed defect (6)',
    7: 'Reversible defect (7)'
}

# Define feature names
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    "age_chol_interaction", "thalach_trestbps_ratio", "age_bp_ratio", "heart_reserve"
]

# Streamlit user interface
st.title("Heart Disease Predictor")

# age: numerical input
age = st.number_input("Age:", min_value=1, max_value=120, value=50)

# sex: categorical selection
sex = st.selectbox("Sex (0=Female, 1=Male):", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')

# cp: categorical selection
cp = st.selectbox("Chest pain type:", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])

# trestbps: numerical input
trestbps = st.number_input("Resting blood pressure (trestbps):", min_value=50, max_value=200, value=120)

# chol: numerical input
chol = st.number_input("Serum cholesterol in mg/dl (chol):", min_value=100, max_value=600, value=200)

# fbs: categorical selection
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs):", options=[0, 1], format_func=lambda x: 'False (0)' if x == 0 else 'True (1)')

# restecg: categorical selection
restecg = st.selectbox("Resting electrocardiographic results:", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])

# thalach: numerical input
thalach = st.number_input("Maximum heart rate achieved (thalach):", min_value=50, max_value=250, value=150)

# exang: categorical selection
exang = st.selectbox("Exercise induced angina (exang):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# oldpeak: numerical input
oldpeak = st.number_input("ST depression induced by exercise relative to rest (oldpeak):", min_value=0.0, max_value=10.0, value=1.0)

# slope: categorical selection
slope = st.selectbox("Slope of the peak exercise ST segment (slope):", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])

# ca: numerical input
ca = st.number_input("Number of major vessels colored by fluoroscopy (ca):", min_value=0, max_value=4, value=0)

# thal: categorical selection
thal = st.selectbox("Thal (thal):", options=list(thal_options.keys()), format_func=lambda x: thal_options[x])

# Process inputs and make predictions
feature_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Create new features based on domain knowledge
age_chol_interaction = age * chol
thalach_trestbps_ratio = thalach / (trestbps + 1)
age_bp_ratio = age / (trestbps + 1)
heart_reserve = 220 - age - thalach

# Append new features to the feature values
feature_values.extend([age_chol_interaction, thalach_trestbps_ratio, age_bp_ratio, heart_reserve])
features = np.array([feature_values])

# Standardize the features using the loaded scaler
features_scaled = scaler.transform(features)

if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features_scaled)[0]
    predicted_proba = model.predict_proba(features_scaled)[0]

    # Display prediction results
    class_text = "No Heart Disease" if predicted_class == 0 else "Heart Disease"

    st.write(f"**Predicted Class:** {class_text}")
    st.write(f"**Probability of No Heart Disease:** {predicted_proba[0]:.2f}")
    st.write(f"**Probability of Heart Disease:** {predicted_proba[1]:.2f}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100


    prompt = f"""
    You are an experienced cardiologist. Based on the following patient's health data and the heart disease prediction result, please provide a detailed comprehensive diagnostic opinion, including potential health risks, recommended lifestyle changes, and possible treatment options.

    - **Age**: 
    {age}
    - **Sex**: 
    {'Female' if sex == 0 else 'Male'}
    - **Chest Pain Type**: 
    {cp_options[cp]}
    - **Resting Blood Pressure (trestbps)**: 
    {trestbps}
     mmHg
    - **Serum Cholesterol (chol)**: 
    {chol}
     mg/dl
    - **Fasting Blood Sugar > 120 mg/dl (fbs)**: 
    {'No' if fbs == 0 else 'Yes'}
    - **Resting Electrocardiographic Results (restecg)**: 
    {restecg_options[restecg]}
    - **Maximum Heart Rate Achieved (thalach)**: 
    {thalach}
     bpm
    - **Exercise Induced Angina (exang)**: 
    {'No' if exang == 0 else 'Yes'}
    - **ST Depression Induced by Exercise Relative to Rest (oldpeak)**: 
    {oldpeak}
    - **Slope of the Peak Exercise ST Segment (slope)**: 
    {slope_options[slope]}
    - **Number of Major Vessels Colored by Fluoroscopy (ca)**: 
    {ca}
    - **Thalassemia (thal)**: 
    {thal_options[thal]}

    **Predicted Probability of Heart Disease**: 
    {100 - probability:.2f}
    %

    Based on the above information, assess the patient's cardiac health and provide specific recommendations.
    """

    print(prompt)

    completion = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "user", "content": prompt}, ],
    )

    advice = completion.choices[0].message.content

    # if predicted_class == 1:
    #     advice = (
    #         f"Based on our model, you are assessed to have a high risk of heart disease. "
    #         f"The model estimates that your probability of developing heart disease is {probability:.1f}%. "
    #         "While this figure serves as an approximation, it indicates that you may be at considerable risk. "
    #         "I strongly recommend consulting a cardiologist at your earliest convenience for further evaluation "
    #         "to ensure accurate diagnosis and appropriate treatment. "
    #     )
    # else:
    #     advice = (
    #         f"Based on our model, you are assessed to have a low risk of heart disease. "
    #         f"The model estimates that your probability of being free from heart disease is {probability:.1f}%. "
    #         "Nevertheless, it remains crucial to maintain a healthy lifestyle.  "
    #         "I recommend regular check-ups to monitor your cardiovascular health "
    #         "and advise seeking medical attention promptly should you experience any symptoms. "
    #     )

    st.write(advice)

    # Calculate SHAP values and display force plot
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

    st.image("shap_force_plot.png")
