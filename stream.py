import streamlit as st
import pandas as pd
import pickle

# Load model and data
ab = pd.read_csv("final_health_data.csv")
model = pickle.load(open("disease_model.pkl", "rb"))

# Extract symptom columns
symptom_columns = [col for col in ab.columns if col not in ['disease', 'symtoms_score']]

# UI
st.title("🧠 Health Care Prediction")
st.markdown("### Select your symptoms from the list below:")

# Multiselect dropdown
selected_symptoms = st.multiselect("Choose Symptoms", options=symptom_columns)

# Prediction button
if st.button("Predict"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom to predict the disease.")
    else:
        input_data = {col: 1 if col in selected_symptoms else 0 for col in symptom_columns}
        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df).max()

        st.success(f"Predicted Disease: {prediction}")
        st.info(f"Risk Score: {round(probability * 100, 2)}%")
