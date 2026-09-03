# Income Predictor - Streamlit App

Predicts whether a person's income is <=50K or >50K based on the
1994 US Census Adult Income dataset, using a Random Forest model.

## Files
- `app.py` — the Streamlit app (the UI)
- `income_model.pkl` — the trained Random Forest model
- `scaler.pkl` — the StandardScaler fitted on the training data
- `model_columns.pkl` — the exact column order the model expects
- `category_options.pkl` — dropdown options for categorical fields
- `numeric_ranges.pkl` — min/max used for the sliders
- `fnlwgt_default.pkl` — a fixed default for the `fnlwgt` census weight (not user-facing)
- `sex_mapping.pkl` — how "Male"/"Female" were label-encoded during training
- `train_export.py` — the script that trained the model and produced all the .pkl files
  (re-run it if you want to retrain the model)

## How to run locally
1. Install the requirements:
   `pip install -r requirements.txt`
2. Run the app:
   `streamlit run app.py`
3. It opens automatically in your browser (usually http://localhost:8501)

## How to deploy for free (Streamlit Community Cloud)
1. Push this whole folder to a GitHub repository (keep the .pkl files in it — the model
   file is about 10 MB, well under GitHub's 100 MB limit).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app", pick your repo/branch, and set the main file to `app.py`.
4. Click "Deploy" — you'll get a public URL to share.
