import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime

# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="Income Classification System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# Custom styling
# =========================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .result-high {
        background-color: #ecfdf5;
        border-left: 6px solid #10b981;
    }
    .result-low {
        background-color: #eff6ff;
        border-left: 6px solid #3b82f6;
    }
    .result-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .result-sub {
        color: #4b5563;
        font-size: 0.95rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.4rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# Load artifacts
# =========================================================
@st.cache_resource
def load_artifacts():
    return {
        "model": joblib.load("income_model.pkl"),
        "scaler": joblib.load("scaler.pkl"),
        "model_columns": joblib.load("model_columns.pkl"),
        "category_options": joblib.load("category_options.pkl"),
        "numeric_ranges": joblib.load("numeric_ranges.pkl"),
        "fnlwgt_default": joblib.load("fnlwgt_default.pkl"),
        "sex_mapping": joblib.load("sex_mapping.pkl"),
        "metrics": joblib.load("metrics.pkl"),
        "coefficients": joblib.load("coefficients.pkl"),
    }

artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
model_columns = artifacts["model_columns"]
category_options = artifacts["category_options"]
numeric_ranges = artifacts["numeric_ranges"]
fnlwgt_default = artifacts["fnlwgt_default"]
sex_mapping = artifacts["sex_mapping"]
metrics = artifacts["metrics"]
coefficients = artifacts["coefficients"]

if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.markdown("### Income Classification System")
    st.caption("Logistic Regression · 1994 US Census (Adult) Dataset")
    st.divider()

    st.markdown("**Model Performance**")
    m1, m2 = st.columns(2)
    m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    m2.metric("F1 Score", f"{metrics['f1']*100:.1f}%")
    m3, m4 = st.columns(2)
    m3.metric("Precision", f"{metrics['precision']*100:.1f}%")
    m4.metric("Recall", f"{metrics['recall']*100:.1f}%")

    st.caption(f"Trained on {metrics['n_train']:,} records · tested on {metrics['n_test']:,} records")

    st.divider()
    with st.expander("About this dataset"):
        st.write(
            "Extracted by Barry Becker from the 1994 US Census database. "
            "14 demographic and employment attributes are used to predict whether "
            "a person's annual income is above or below $50,000."
        )

    if st.session_state.history:
        st.divider()
        st.markdown("**This session**")
        st.caption(f"{len(st.session_state.history)} prediction(s) made")
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# =========================================================
# Header
# =========================================================
st.markdown('<p class="main-header">Income Classification System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Predict whether an individual\'s annual income is above or below '
    '$50,000 based on census attributes.</p>',
    unsafe_allow_html=True,
)

tab_predict, tab_insights, tab_about = st.tabs(["Predictor", "Model Insights", "About"])

# =========================================================
# TAB 1 — Predictor
# =========================================================
with tab_predict:
    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<p class="section-title">Individual Profile</p>', unsafe_allow_html=True)
        with st.form("income_form"):
            c1, c2 = st.columns(2)

            with c1:
                age = st.slider("Age", numeric_ranges["Age"][0], numeric_ranges["Age"][1], 35)
                education = st.selectbox("Education", category_options["Education"])
                education_num = st.slider(
                    "Years of Education",
                    numeric_ranges["Education_Num"][0],
                    numeric_ranges["Education_Num"][1],
                    10,
                )
                workclass = st.selectbox("Workclass", category_options["Workclass"])
                occupation = st.selectbox("Occupation", category_options["Occupation"])
                hours_per_week = st.slider(
                    "Hours per Week",
                    numeric_ranges["Hours_per_week"][0],
                    numeric_ranges["Hours_per_week"][1],
                    40,
                )

            with c2:
                marital_status = st.selectbox("Marital Status", category_options["Martial_Status"])
                relationship = st.selectbox("Relationship", category_options["Relationship"])
                race = st.selectbox("Race", category_options["Race"])
                sex = st.selectbox("Sex", category_options["Sex"])
                default_country_index = (
                    category_options["Country"].index("United-States")
                    if "United-States" in category_options["Country"] else 0
                )
                country = st.selectbox("Native Country", category_options["Country"], index=default_country_index)
                capital_gain = st.number_input(
                    "Capital Gain",
                    numeric_ranges["Capital_Gain"][0],
                    numeric_ranges["Capital_Gain"][1],
                    0,
                )
                capital_loss = st.number_input(
                    "Capital Loss",
                    numeric_ranges["Capital_Loss"][0],
                    numeric_ranges["Capital_Loss"][1],
                    0,
                )

            submitted = st.form_submit_button("Run Prediction", use_container_width=True, type="primary")

    with right:
        st.markdown('<p class="section-title">Prediction Result</p>', unsafe_allow_html=True)

        if submitted:
            row = pd.DataFrame([{
                "Age": age,
                "Workclass": workclass,
                "fnlwgt": fnlwgt_default,
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

            row["Age_Group"] = pd.cut(
                row["Age"], bins=[0, 25, 40, 60, 100],
                labels=["Young", "Adult", "Middle_Aged", "Senior"]
            )
            row["Work_Hours_Group"] = pd.cut(
                row["Hours_per_week"], bins=[0, 35, 45, 100],
                labels=["Part_Time", "Full_Time", "Over_Time"]
            )

            row = pd.get_dummies(row, columns=[
                "Workclass", "Education", "Martial_Status", "Occupation",
                "Relationship", "Race", "Country", "Age_Group", "Work_Hours_Group"
            ])
            row = row.reindex(columns=model_columns, fill_value=0)
            row_scaled = scaler.transform(row)

            prediction = model.predict(row_scaled)[0]
            probability = model.predict_proba(row_scaled)[0][1]

            st.session_state.history.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Age": age,
                "Occupation": occupation,
                "Education": education,
                "Hours/Week": hours_per_week,
                "Prediction": ">50K" if prediction == 1 else "<=50K",
                "Probability": round(probability, 3),
            })

            if prediction == 1:
                st.markdown(f"""
                <div class="result-card result-high">
                    <div class="result-title">Predicted Income: Above $50,000</div>
                    <div class="result-sub">Model confidence: {probability:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card result-low">
                    <div class="result-title">Predicted Income: $50,000 or Below</div>
                    <div class="result-sub">Confidence in "Above 50K": {probability:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%"},
                title={"text": "Probability of Income > $50,000"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#10b981" if probability >= 0.5 else "#3b82f6"},
                    "steps": [
                        {"range": [0, 50], "color": "#eff6ff"},
                        {"range": [50, 100], "color": "#ecfdf5"},
                    ],
                    "threshold": {
                        "line": {"color": "#1a1a2e", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(gauge, use_container_width=True)
        else:
            st.info("Fill in the profile on the left and click **Run Prediction** to see the result here.")

    if st.session_state.history:
        st.markdown('<p class="section-title">Session History</p>', unsafe_allow_html=True)
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download history as CSV",
            history_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

# =========================================================
# TAB 2 — Model Insights
# =========================================================
with tab_insights:
    st.markdown('<p class="section-title">What Drives the Prediction</p>', unsafe_allow_html=True)
    st.write(
        "Each bar shows how strongly a feature pushes the model's prediction toward "
        "**>50K** (positive, right) or **<=50K** (negative, left), holding other features fixed."
    )

    top_n = 15
    top_features = pd.concat([coefficients.head(top_n // 2), coefficients.tail(top_n // 2)])
    top_features = top_features.sort_values()

    bar = go.Figure(go.Bar(
        x=top_features.values,
        y=top_features.index,
        orientation="h",
        marker_color=["#3b82f6" if v < 0 else "#10b981" for v in top_features.values],
    ))
    bar.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Effect on log-odds of earning > $50K",
    )
    st.plotly_chart(bar, use_container_width=True)

    st.markdown('<p class="section-title">Confusion Matrix (Test Set)</p>', unsafe_allow_html=True)
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual <=50K", "Actual >50K"],
        columns=["Predicted <=50K", "Predicted >50K"],
    )
    st.dataframe(cm_df, use_container_width=True)

# =========================================================
# TAB 3 — About
# =========================================================
with tab_about:
    st.markdown('<p class="section-title">About This System</p>', unsafe_allow_html=True)
    st.write(
        "This application predicts whether an individual's annual income exceeds $50,000, "
        "based on the 1994 US Census Adult Income dataset compiled by Barry Becker. "
        "The task is framed as binary classification."
    )

    st.markdown("**Pipeline**")
    st.write(
        "- Missing values were kept as their own category or dropped\n"
        "- Outliers in numeric fields were capped using the IQR method\n"
        "- Categorical features were encoded (Label Encoding for binary fields, One-Hot Encoding otherwise)\n"
        "- New engineered features: `Age_Group` and `Work_Hours_Group`\n"
        "- Class imbalance was handled with `class_weight=\"balanced\"`\n"
        "- Final model: Logistic Regression, chosen for its small footprint and strong performance"
    )

    st.markdown("**Limitations**")
    st.write(
        "This model is trained on 1994 census data and reflects the economic conditions and "
        "demographics of that period — it is a demonstration of a machine learning pipeline, "
        "not a real-world income estimator."
    )
