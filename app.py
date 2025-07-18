import streamlit as st
import numpy as np
import pandas as pd
import joblib


# Load model and label encoder
model = joblib.load("ffnn_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ------------------ UI Layout ------------------

st.set_page_config(page_title="PPD Risk Predictor", layout="centered")
st.title("🧠 Postpartum Depression Risk Predictor")
st.markdown("This app predicts the risk level of postpartum depression based on questionnaire responses and personal information.")

st.divider()
st.header("👩 Personal & Family Info")

col1, col2 = st.columns(2)
with col1:
    Age = st.slider("Age", 18, 45, 25)
    Employment = st.selectbox("Employment Status", ["Employed", "Unemployed"])
    Children = st.selectbox("Number of Children", [0, 1, 2, 3, "More"])
with col2:
    Pregnancy = st.selectbox("Currently Pregnant?", ["Select...", "Yes", "No"])
    Delivery = st.selectbox("Recently Given Birth?", ["Select...", "Yes", "No"])
    FamilySupport = st.selectbox("Family Support Level", ["Low", "Medium", "High"])

# ------------------ Questionnaire ------------------

st.divider()
st.header("📝 EPDS Questionnaire (Last 7 Days)")

# Question mapping
questions = {
    "Q1": ("I have been able to laugh and see the funny side of things.",
           {"As much as I always could": 0, "Not quite so much now": 1,
            "Definitely not so much now": 2, "Not at all": 3}),
    "Q2": ("I have looked forward with enjoyment to things",
           {"As much as I ever did": 0, "Rather less than I used to": 1,
            "Definitely less than I used to": 2, "Hardly at all": 3}),
    "Q3": ("I have blamed myself unnecessarily when things went wrong",
           {"No, never": 0, "Not very often": 1, "Yes, some of the time": 2, "Yes, most of the time": 3}),
    "Q4": ("I have been anxious or worried for no good reason",
           {"No, not at all": 0, "Hardly ever": 1, "Yes, sometimes": 2, "Yes, very often": 3}),
    "Q5": ("I have felt scared or panicky for no very good reason",
           {"No, not at all": 0, "No, not much": 1, "Yes, sometimes": 2, "Yes, quite a lot": 3}),
    "Q6": ("Things have been getting on top of me",
           {"No, I have been coping as well as ever": 0, "No, most of the time I have coped quite well": 1,
            "Yes, sometimes I haven't been coping as well as usual": 2, "Yes, most of the time I haven't been able to cope at all": 3}),
    "Q7": ("I have been so unhappy that I have had difficulty sleeping",
           {"No, not at all": 0, "Not very often": 1, "Yes, sometimes": 2, "Yes, most of the time": 3}),
    "Q8": ("I have felt sad or miserable",
           {"No, not at all": 0, "Not very often": 1, "Yes, quite often": 2, "Yes, most of the time": 3}),
    "Q9": ("I have been so unhappy that I have been crying",
           {"No, never": 0, "Only occasionally": 1, "Yes, quite often": 2, "Yes, most of the time": 3}),
    "Q10": ("The thought of harming myself has occurred to me",
            {"Never": 0, "Hardly ever": 1, "Sometimes": 2, "Yes, quite often": 3})
}

responses = {}
for q_id, (question, options) in questions.items():
    responses[q_id] = st.selectbox(f"{q_id}. {question}", list(options.keys()), key=q_id)

# ------------------ Prediction ------------------

if st.button("🔍 Predict Risk Level"):
    # Encode categorical values
    employment = 1 if Employment == "Employed" else 0
    children = 4 if Children == "More" else int(Children)
    preg = 1 if Pregnancy == "Yes" else 0
    delivery = 1 if Delivery == "Yes" else 0
    support_map = {"Low": 0, "Medium": 1, "High": 2}

    # Get EPDS scores
    epds_scores = [questions[q][1][responses[q]] for q in questions]

    # Form input DataFrame
    input_data = pd.DataFrame([{
        "Age": Age,
        "Employment": employment,
        "Children": children,
        "Pregnancy": preg,
        "Delivery": delivery,
        "FamilySupport": support_map[FamilySupport],
        **{q: val for q, val in zip(questions.keys(), epds_scores)}
    }])

    # Predict
    pred_encoded = model.predict(input_data)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    st.success(f"🎯 Predicted Postpartum Depression Risk: **{pred_label}**")

    # Visualize
    st.divider()
    st.subheader("📊 Risk Level Chart")
    ax.bar(["Predicted Risk"], [pred_encoded], color="tomato")
    ax.set_ylim([0, 3])
    ax.set_ylabel("Risk Level (0 = Mild → 3 = Profound)")
    st.pyplot(fig)
