
import streamlit as st
import numpy as np
import joblib

# Load your model
model = joblib.load('model.pkl')

st.title("Postpartum Depression Risk Predictor")

# Input from user
st.subheader("Enter the following details:")

age = st.slider("Mother's Age", 18, 45, 25)
children = st.selectbox("Number of Children", [0, 1, 2, 3, "More"])
employment = st.radio("Employment Status", ["Employed", "Unemployed"])
support = st.selectbox("Social Support Level", ["Low", "Medium", "High"])

# Convert inputs to numbers
def preprocess(age, children, employment, support):
    if children == "More":
        children = 4
    else:
        children = int(children)

    emp = 1 if employment == "Employed" else 0
    support_score = {"Low": 0, "Medium": 1, "High": 2}
    return np.array([[age, children, emp, support_score[support]]])

# Predict button
if st.button("Predict"):
    input_data = preprocess(age, children, employment, support)
    prediction = model.predict(input_data)
    st.success(f"Predicted PPD Risk Level: {prediction[0]}")
