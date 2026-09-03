import streamlit as st
import pandas as pd
import joblib

# ---------------- Load saved artifacts ----------------
model = joblib.load("income_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")
category_options = joblib.load("category_options.pkl")
numeric_ranges = joblib.load("numeric_ranges.pkl")
fnlwgt_default = joblib.load("fnlwgt_default.pkl")
sex_mapping = joblib.load("sex_mapping.pkl")

st.set_page_config(page_title="Income Predictor", page_icon="💰")
st.title("💰 US Adult Income Predictor")
st.write(
    "Fill in the details below and the model will predict whether this person's "
    "yearly income is **<=50K** or **>50K**, based on the 1994 US Census (Adult Income) dataset."
)

# ---------------- Input form ----------------
with st.form("income_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", numeric_ranges["Age"][0], numeric_ranges["Age"][1], 35)
        education = st.selectbox("Education", category_options["Education"])
        education_num = st.slider("Years of Education", numeric_ranges["Education_Num"][0],
                                   numeric_ranges["Education_Num"][1], 10)
        workclass = st.selectbox("Workclass", category_options["Workclass"])
        occupation = st.selectbox("Occupation", category_options["Occupation"])
        hours_per_week = st.slider("Hours per Week", numeric_ranges["Hours_per_week"][0],
                                    numeric_ranges["Hours_per_week"][1], 40)

    with col2:
        marital_status = st.selectbox("Marital Status", category_options["Martial_Status"])
        relationship = st.selectbox("Relationship", category_options["Relationship"])
        race = st.selectbox("Race", category_options["Race"])
        sex = st.selectbox("Sex", category_options["Sex"])
        country = st.selectbox("Native Country", category_options["Country"],
                                index=category_options["Country"].index("United-States")
                                if "United-States" in category_options["Country"] else 0)
        capital_gain = st.number_input("Capital Gain", numeric_ranges["Capital_Gain"][0],
                                        numeric_ranges["Capital_Gain"][1], 0)
        capital_loss = st.number_input("Capital Loss", numeric_ranges["Capital_Loss"][0],
                                        numeric_ranges["Capital_Loss"][1], 0)

    submitted = st.form_submit_button("Predict Income")

# ---------------- Prediction ----------------
if submitted:
    # build a single-row dataframe with the raw inputs
    row = pd.DataFrame([{
        "Age": age,
        "Workclass": workclass,
        "fnlwgt": fnlwgt_default,  # technical census sampling weight, not user-facing
        "Education": education,
        "Education_Num": education_num,
        "Martial_Status": marital_status,
        "Occupation": occupation,
        "Relationship": relationship,
        "Race": race,
        "Sex": sex_mapping[sex],
        "Capital_Gain": capital_gain,
        "Capital_Loss": capital_loss,
        "Hours_per_week": hours_per_week,
        "Country": country,
    }])

    # same feature engineering as training
    row["Age_Group"] = pd.cut(row["Age"], bins=[0, 25, 40, 60, 100],
                               labels=["Young", "Adult", "Middle_Aged", "Senior"])
    row["Work_Hours_Group"] = pd.cut(row["Hours_per_week"], bins=[0, 35, 45, 100],
                                      labels=["Part_Time", "Full_Time", "Over_Time"])

    row = pd.get_dummies(row, columns=["Workclass", "Education", "Martial_Status", "Occupation",
                                        "Relationship", "Race", "Country",
                                        "Age_Group", "Work_Hours_Group"])

    # align columns with the ones the model was trained on, then scale
    row = row.reindex(columns=model_columns, fill_value=0)
    row_scaled = scaler.transform(row)

    prediction = model.predict(row_scaled)[0]
    probability = model.predict_proba(row_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.success(f"Predicted income: **>50K** (probability: {probability:.1%})")
    else:
        st.info(f"Predicted income: **<=50K** (probability of >50K: {probability:.1%})")

st.caption("Model: Logistic Regression trained on the 1994 US Census Adult Income dataset (Barry Becker).")
