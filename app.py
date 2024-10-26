import streamlit as st
import joblib
import pandas as pd

# Load pre-trained models
xgb_model = joblib.load('xgb_model.pkl')
brf_model = joblib.load('brf_model.pkl')

# Load the list of expected features from a reference dataset or the model itself
# This could be a saved DataFrame or list of column names used in training
expected_features = xgb_model.get_booster().feature_names

st.title("Insurance Claim Prediction")

st.write("""
This app uses an ensemble of XGBoost and Balanced Random Forest models to predict the likelihood of an insurance claim.
""")

# Define input fields for user features
st.header("Enter the Policy and Vehicle Details:")

# Define input fields based on the model's expected features (modify as per your features)
age_of_policyholder = st.slider("Age of Policyholder", 18, 100, 30)
policy_tenure = st.slider("Policy Tenure (in years)", 1, 20, 5)
age_of_car = st.slider("Age of Car (in years)", 0, 30, 5)
is_esc = st.selectbox("Electronic Stability Control (ESC)", ["Yes", "No"])
is_tpms = st.selectbox("Tire Pressure Monitoring System (TPMS)", ["Yes", "No"])

# Map binary features to match model's expected inputs
is_esc = 1 if is_esc == "Yes" else 0
is_tpms = 1 if is_tpms == "Yes" else 0

# Construct a DataFrame with all expected features
input_data = pd.DataFrame({
    'age_of_policyholder': [age_of_policyholder],
    'policy_tenure': [policy_tenure],
    'age_of_car': [age_of_car],
    'is_esc': [is_esc],
    'is_tpms': [is_tpms]
})

# Add any missing columns from the expected features and set them to zero
for feature in expected_features:
    if feature not in input_data.columns:
        input_data[feature] = 0

# Order the input columns to match the model's training data
input_data = input_data[expected_features]

# Predict button
if st.button("Predict Claim Probability"):
    # Get probabilities from each model
    xgb_proba = xgb_model.predict_proba(input_data)[:, 1]
    brf_proba = brf_model.predict_proba(input_data)[:, 1]

    # Ensemble prediction with weighted averaging
    ensemble_proba = (0.6 * xgb_proba + 0.4 * brf_proba)

    # Display results
    st.write(f"Predicted Probability of a Claim: {ensemble_proba[0]:.2f}")
    st.write("Based on your input data, there is a high likelihood of a claim" if ensemble_proba[0] > 0.5 else "Based on your input data, there is a low likelihood of a claim.")



