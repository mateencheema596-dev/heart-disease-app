import streamlit as st
import joblib
import numpy as np

model = joblib.load("heart_model.pkl")

st.title("Heart Disease Prediction")
st.write("Enter the patient's details:")

age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox(
    "Chest pain type",
    options=[1, 2, 3, 4],
    format_func=lambda x: {1: "Typical angina", 2: "Atypical angina",
                           3: "Non-anginal pain", 4: "Asymptomatic"}[x],
)
thalach = st.number_input("Maximum heart rate achieved", min_value=60, max_value=220, value=150)

if st.button("Predict"):
    pred = model.predict(np.array([[age, sex, cp, thalach]]))[0]
    if pred == 1:
        st.error("Heart disease likely.")
    else:
        st.success("Heart disease unlikely.")