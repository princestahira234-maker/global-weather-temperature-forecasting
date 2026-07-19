import streamlit as st
import pandas as pd
from xgboost import XGBRegressor

st.set_page_config(
    page_title="Weather Temperature Prediction",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ Weather Temperature Prediction")
st.write("Upload a weather dataset and generate temperature predictions.")

# Load Model
@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model("weather_model.json")
    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Input Data")
    st.dataframe(df.head())

    try:
        required_features = model.get_booster().feature_names

        X = df[required_features].copy()

        categorical_cols = [
            "country",
            "location_name",
            "condition_text",
            "wind_direction"
        ]

        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype("category")

        predictions = model.predict(X)

        df["predicted_temperature_celsius"] = predictions

        st.subheader("Prediction Results")
        st.dataframe(df.head())

        csv = df.to_csv(index=False)

        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

        st.success("Predictions generated successfully!")

    except Exception as e:
        st.error(f"Error: {e}")