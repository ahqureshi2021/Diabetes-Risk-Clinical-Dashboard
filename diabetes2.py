import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes Clinical Dashboard", layout="wide")

st.title("🏥 Diabetes Risk Clinical Dashboard")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    return pd.read_csv(url, names=columns)

data = load_data()

# Preprocessing
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_with_zero] = data[cols_with_zero].replace(0, np.nan)
data.fillna(data.median(), inplace=True)

X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# UI Layout (Hospital Style)
# -----------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🧾 Patient Inputs")

    glucose_type = st.selectbox("Glucose Type", ["Fasting", "Random (Non-Fasting)"])
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose (mg/dL)", 50, 300, 120)
    hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.5)
    bp = st.number_input("Blood Pressure", 40, 140, 70)
    bmi = st.number_input("BMI", 10.0, 60.0, 30.0)
    age = st.number_input("Age", 10, 100, 30)

    predict_btn = st.button("Run Clinical Assessment")

with col2:
    st.subheader("📊 Clinical Output")

    if predict_btn:
        # Adjust glucose based on type
        adjusted_glucose = glucose
        if glucose_type == "Random (Non-Fasting)":
            adjusted_glucose = glucose * 0.85  # normalize random glucose approx

        input_data = [pregnancies, adjusted_glucose, bp, 20, 80, bmi, 0.5, age]

        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        # HbA1c adjustment (clinical override)
        if hba1c >= 6.5:
            prob = max(prob, 0.85)
            prediction = 1
        elif 5.7 <= hba1c < 6.5:
            prob = max(prob, 0.6)

        # Metrics display
        m1, m2, m3 = st.columns(3)
        m1.metric("Diabetes Risk", f"{prob*100:.1f}%")
        m2.metric("HbA1c", f"{hba1c}%")
        m3.metric("Glucose", f"{glucose} mg/dL")

        st.divider()

        if prediction == 1:
            st.error("⚠️ Clinical Interpretation: High likelihood of Diabetes")
        else:
            st.success("✅ Clinical Interpretation: Low likelihood of Diabetes")

        # Detailed interpretation
        st.subheader("🧠 Clinical Insights")

        if hba1c >= 6.5:
            st.write("• HbA1c indicates Diabetes (≥ 6.5%)")
        elif hba1c >= 5.7:
            st.write("• HbA1c indicates Prediabetes")
        else:
            st.write("• HbA1c is within normal range")

        if glucose_type == "Fasting":
            if glucose >= 126:
                st.write("• Fasting glucose in diabetic range")
        else:
            if glucose >= 200:
                st.write("• Random glucose in diabetic range")

# -----------------------------
# Model Performance
# -----------------------------
st.subheader("📈 Model Performance")
y_pred = model.predict(X_test)
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# -----------------------------
# Data Preview
# -----------------------------
st.subheader("📂 Dataset Preview")
st.dataframe(data.head())
