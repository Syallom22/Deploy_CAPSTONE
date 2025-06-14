import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('random_forest_model.pkl')

# Judul aplikasi
st.title("Prediksi Kategori Obesitas - NObeyesdad")

# Input pengguna
st.header("Masukkan Data Pengguna")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 10, 100)
height = st.number_input("Height (meters)", value=1.70)
weight = st.number_input("Weight (kg)", value=70.0)
family_history = st.selectbox("Family History with Overweight", ["yes", "no"])
favc = st.selectbox("Frequent consumption of high caloric food (FCVC)", ["yes", "no"])
fcvc = st.slider("Frequency of vegetable consumption (FCVC)", 1.0, 3.0, step=0.1)
ncp = st.slider("Number of main meals (NCP)", 1.0, 4.0, step=0.5)
caec = st.selectbox("Consumption of food between meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Do you smoke?", ["yes", "no"])
ch2o = st.slider("Daily water consumption (CH2O)", 0.0, 3.0, step=0.1)
scc = st.selectbox("Calories consumption monitoring (SCC)", ["yes", "no"])
faf = st.slider("Physical activity frequency (FAF)", 0.0, 3.0, step=0.1)
tue = st.slider("Time using technology devices (TUE)", 0.0, 3.0, step=0.1)
calc = st.selectbox("Consumption of alcohol (CALC)", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Transportation used", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

# Buat DataFrame dari input
input_dict = {
    "Gender": gender,
    "Age": age,
    "Height": height,
    "Weight": weight,
    "family_history_with_overweight": family_history,
    "FAVC": favc,
    "FCVC": fcvc,
    "NCP": ncp,
    "CAEC": caec,
    "SMOKE": smoke,
    "CH2O": ch2o,
    "SCC": scc,
    "FAF": faf,
    "TUE": tue,
    "CALC": calc,
    "MTRANS": mtrans
}

input_df = pd.DataFrame([input_dict])

# Encoding sama seperti training
input_encoded = pd.get_dummies(input_df)

# Pastikan kolom cocok dengan training set
# Load kolom training dari file/fitur asli
columns_needed = joblib.load('columns_list.pkl')  # ← Simpan ini sebelumnya saat training
for col in columns_needed:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[columns_needed]

# Prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_encoded)
    st.subheader("Hasil Prediksi:")
    st.success(f"Kategori Obesitas: {prediction[0]}")
