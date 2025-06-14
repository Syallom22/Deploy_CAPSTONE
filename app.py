import streamlit as st
import pandas as pd
import joblib

# Load model dan fitur yang digunakan
model = joblib.load("random_forest_model.pkl")

# Judul
st.title("Prediksi Kategori Obesitas (NObeyesdad)")

# Input pengguna
st.header("Masukkan Data Pengguna")

age = st.slider("Umur", 10, 100, 25)
height = st.number_input("Tinggi Badan (meter)", value=1.70)
weight = st.number_input("Berat Badan (kg)", value=70.0)

gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
family_history = st.selectbox("Riwayat Keluarga Obesitas", ["yes", "no"])

# Encode input sesuai format training
gender_encoded = 1 if gender == "Male" else 0
family_history_encoded = 1 if family_history == "yes" else 0

# Susun ke DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "Height": height,
    "Weight": weight,
    "Gender": gender_encoded,
    "family_history_with_overweight": family_history_encoded
}])

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_data)
    st.subheader("Hasil Prediksi:")
    st.success(f"Kategori Obesitas: {prediction[0]}")
