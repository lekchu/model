import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Train a basic model inside the app
def train_model():
    X = np.array([
        [25, 1, 1, 2],
        [30, 2, 0, 1],
        [22, 0, 1, 0],
        [35, 3, 0, 2],
        [28, 2, 1, 1]
    ])
    y = ['Mild', 'Moderate', 'Mild', 'Severe', 'Moderate']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_model()

st.title("Postpartum Depression Risk Predictor")

age = st.slider("Mother's Age", 18, 45, 25)
children = st.selectbox("Number of Children", [0, 1, 2, 3, "More"])
employment = st.radio("Employment Status", ["Employed", "Unemployed"])
support = st.selectbox("Social Support Level", ["Low", "Medium", "High"])

def preprocess(age, children, employment, support):
    children = 4 if children == "More" else int(children)
    emp = 1 if employment == "Employed" else 0
    support_score = {"Low": 0, "Medium": 1, "High": 2}
    return np.array([[age, children, emp, support_score[support]]])

if st.button("Predict"):
    input_data = preprocess(age, children, employment, support)
    prediction = model.predict(input_data)
    st.success(f"Predicted PPD Risk Level: {prediction[0]}")

