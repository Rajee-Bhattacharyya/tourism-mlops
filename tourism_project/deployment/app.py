import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from huggingface_hub import hf_hub_download
import os

# 1. Page Configuration
st.set_page_config(page_title="Visit with Us - Wellness Package Predictor", layout="centered")

# 2. Load the model from Hugging Face Hub
@st.cache_resource
def load_model():
    # Replace with actual model repo ID
    repo_id = "RajeeBhattacharyya/tourism-prediction-model"
    model_path = hf_hub_download(repo_id=repo_id, filename="model.json")

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

model = load_model()

# 3. App Header
st.title("Wellness Tourism Package Predictor")
st.write("Enter customer details below to predict the likelihood of a package purchase.")

# 4. User Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        duration = st.number_input("Duration of Pitch", min_value=0, value=15)
        gender = st.selectbox("Gender", ["Male", "Female"])
        num_trips = st.number_input("Number of Trips", min_value=0, value=3)

    with col2:
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        passport = st.selectbox("Has Passport?", ["Yes", "No"])
        own_car = st.selectbox("Owns Car?", ["Yes", "No"])
        monthly_income = st.number_input("Monthly Income", min_value=0, value=25000)

    # Manual Encoding Mappings (matching the LabelEncoder from training)
    gender_map = {"Female": 0, "Male": 1}
    occ_map = {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3}
    marital_map = {"Divorced": 0, "Married": 1, "Single": 2, "Unmarried": 3}
    binary_map = {"Yes": 1, "No": 0}

    # Submit Button
    submit = st.form_submit_button("Predict Purchase")

# 5. Prediction Logic
if submit:
    # Constructing the dataframe with all features used in training
    input_data = pd.DataFrame({
        'Age': [age],
        'TypeofContact': [0],
        'CityTier': [city_tier],
        'DurationOfPitch': [duration],
        'Occupation': [occ_map[occupation]],
        'Gender': [gender_map[gender]],
        'NumberOfPersonVisiting': [2],
        'NumberOfFollowups': [3],
        'ProductPitched': [0],
        'PreferredPropertyStar': [3],
        'MaritalStatus': [marital_map[marital_status]],
        'NumberOfTrips': [num_trips],
        'Passport': [binary_map[passport]],
        'PitchSatisfactionScore': [3],
        'OwnCar': [binary_map[own_car]],
        'NumberOfChildrenVisiting': [1],
        'Designation': [0],
        'MonthlyIncome': [monthly_income]
    })

    # Making Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    if prediction == 1:
        st.success(f"### Prediction: Likely to Purchase!")
        st.write(f"Confidence Level: {probability:.2%}")
    else:
        st.error(f"### Prediction: Unlikely to Purchase.")
        st.write(f"Confidence Level: {(1-probability):.2%}")
