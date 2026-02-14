
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the trained tourism model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="vallabbharath/tourism-model",
    filename="best_tourism_model_v1.joblib"
)

# Load model
model = joblib.load(model_path)

# UI
st.title("Wellness Tourism Package Purchase Prediction")
st.write("Enter customer details to predict whether they will purchase the Wellness Tourism Package.")

# ===== User Inputs =====

Age = st.number_input("Age", min_value=18, max_value=80, value=30)

TypeofContact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

CityTier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

Occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Free Lancer", "Large Business"]
)

Gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

NumberOfPersonVisiting = st.number_input(
    "Number of People Visiting",
    min_value=1, max_value=10, value=2
)

PreferredPropertyStar = st.selectbox(
    "Preferred Hotel Star Rating",
    [1, 2, 3, 4, 5]
)

MaritalStatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

NumberOfTrips = st.number_input(
    "Number of Trips per Year",
    min_value=0, max_value=20, value=2
)

Passport = st.selectbox("Has Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns Car?", ["Yes", "No"])

NumberOfChildrenVisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0, max_value=5, value=0
)

Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

MonthlyIncome = st.number_input(
    "Monthly Income",
    min_value=0, value=30000
)

PitchSatisfactionScore = st.slider(
    "Pitch Satisfaction Score",
    min_value=1, max_value=5, value=3
)

ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

NumberOfFollowups = st.number_input(
    "Number of Follow-ups",
    min_value=0, max_value=10, value=2
)

DurationOfPitch = st.number_input(
    "Duration of Pitch (minutes)",
    min_value=0, value=15
)

# ===== Create dataframe =====

input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch
}])

classification_threshold = 0.45

# ===== Prediction =====

if st.button("Predict Purchase"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)

    if prediction == 1:
        st.success("This customer is likely to PURCHASE the Wellness Tourism Package.")
    else:
        st.warning("This customer is NOT likely to purchase the Wellness Tourism Package.")
