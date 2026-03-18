import streamlit as st
import numpy as np
import pickle
import os

# Page config (IMPORTANT for professional look)
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# Train model if not exists
if not os.path.exists("model.pkl"):
    import train_model

# Load model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------- HEADER ----------
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🩺 Diabetes Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-based health risk prediction for early diagnosis</p>", unsafe_allow_html=True)

st.divider()

# ---------- SIDEBAR ----------
st.sidebar.header("📝 Patient Details")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20)
glucose = st.sidebar.number_input("Glucose Level", 0, 200)
bp = st.sidebar.number_input("Blood Pressure", 0, 150)
skin = st.sidebar.number_input("Skin Thickness", 0, 100)
insulin = st.sidebar.number_input("Insulin", 0, 900)
bmi = st.sidebar.number_input("BMI", 0.0, 50.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.sidebar.number_input("Age", 1, 120)

predict_btn = st.sidebar.button("🔍 Predict")

# ---------- MAIN LAYOUT ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Input Summary")
    st.write(f"**Pregnancies:** {pregnancies}")
    st.write(f"**Glucose:** {glucose}")
    st.write(f"**Blood Pressure:** {bp}")
    st.write(f"**BMI:** {bmi}")
    st.write(f"**Age:** {age}")

with col2:
    st.subheader("🧠 Prediction Result")

    if predict_btn:
        data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        data = scaler.transform(data)

        prediction = model.predict(data)

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Diabetes")
            st.markdown("**Recommendation:** Please consult a doctor immediately.")
        else:
            st.success("✅ Low Risk of Diabetes")
            st.markdown("**Recommendation:** Maintain a healthy lifestyle.")

# ---------- METRICS ----------
st.divider()
st.subheader("📈 Health Indicators")

m1, m2, m3 = st.columns(3)

m1.metric("Glucose", glucose)
m2.metric("BMI", bmi)
m3.metric("Age", age)

# ---------- FOOTER ----------
st.divider()
st.markdown(
    "<p style='text-align: center; font-size: 14px;'>Final Year Project | Developed using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)