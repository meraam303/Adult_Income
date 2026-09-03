import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression

# ---- Load & merge (same steps as the notebook) ----
train = pd.read_csv("adult_train.csv")
test = pd.read_csv("adult_test.csv", skiprows=[1])
test["Target"] = test["Target"].str.replace(".", "", regex=False)

df = pd.concat([train, test], ignore_index=True)
df = df.rename(columns={"Target": "Income"})

object_cols = df.select_dtypes(include="object").columns
for col in object_cols:
    df[col] = df[col].str.strip()

df = df.dropna()
df = df.drop_duplicates()

# save the raw category options + numeric ranges for the Streamlit form, BEFORE encoding
category_options = {
    "Workclass": sorted(df["Workclass"].unique().tolist()),
    "Education": sorted(df["Education"].unique().tolist()),
    "Martial_Status": sorted(df["Martial_Status"].unique().tolist()),
    "Occupation": sorted(df["Occupation"].unique().tolist()),
    "Relationship": sorted(df["Relationship"].unique().tolist()),
    "Race": sorted(df["Race"].unique().tolist()),
    "Sex": sorted(df["Sex"].unique().tolist()),
    "Country": sorted(df["Country"].unique().tolist()),
}
numeric_ranges = {
    "Age": (int(df["Age"].min()), int(df["Age"].max())),
    "Education_Num": (int(df["Education_Num"].min()), int(df["Education_Num"].max())),
    "Capital_Gain": (int(df["Capital_Gain"].min()), int(df["Capital_Gain"].max())),
    "Capital_Loss": (int(df["Capital_Loss"].min()), int(df["Capital_Loss"].max())),
    "Hours_per_week": (int(df["Hours_per_week"].min()), int(df["Hours_per_week"].max())),
}
fnlwgt_default = int(df["fnlwgt"].median())

# ---- Feature engineering (same as notebook) ----
df["Age_Group"] = pd.cut(df["Age"], bins=[0, 25, 40, 60, 100],
                          labels=["Young", "Adult", "Middle_Aged", "Senior"])
df["Work_Hours_Group"] = pd.cut(df["Hours_per_week"], bins=[0, 35, 45, 100],
                                 labels=["Part_Time", "Full_Time", "Over_Time"])

df["Income"] = df["Income"].map({"<=50K": 0, ">50K": 1})

le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
sex_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

df = pd.get_dummies(df, columns=["Workclass", "Education", "Martial_Status", "Occupation",
                                  "Relationship", "Race", "Country",
                                  "Age_Group", "Work_Hours_Group"])

# ---- Train / test split + scaling ----
X = df.drop("Income", axis=1)
y = df["Income"]
model_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ---- Train the final model ----
# Logistic Regression is used for deployment: it's tiny (a few KB vs ~10-200MB for a
# Random Forest), fast to load, and still performs well on this dataset.
log_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
log_model.fit(X_train_scaled, y_train)

from sklearn.metrics import accuracy_score, f1_score
X_test_scaled = scaler.transform(X_test)
preds = log_model.predict(X_test_scaled)
print("Test accuracy:", accuracy_score(y_test, preds))
print("Test F1:", f1_score(y_test, preds))

joblib.dump(log_model, "income_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(model_columns, "model_columns.pkl")
joblib.dump(category_options, "category_options.pkl")
joblib.dump(numeric_ranges, "numeric_ranges.pkl")
joblib.dump(fnlwgt_default, "fnlwgt_default.pkl")
joblib.dump(sex_mapping, "sex_mapping.pkl")

print("All artifacts saved.")
