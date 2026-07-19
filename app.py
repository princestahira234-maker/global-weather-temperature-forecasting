import streamlit as st
import pandas as pd
import plotly.express as px
from xgboost import XGBRegressor

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="WeatherVision AI™",
    page_icon="🌦️",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.metric-card {
    background-color: #0f172a;
    padding: 15px;
    border-radius: 12px;
    color: white;
    text-align: center;
}

.hero {
    padding: 35px;
    border-radius: 20px;
    background: linear-gradient(90deg,#0f172a,#2563eb);
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

.footer {
    text-align:center;
    color:gray;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero">

<h1>🌦️ WeatherVision AI™</h1>

<h3>Intelligent Climate & Temperature Forecasting Platform</h3>

<p>
Advanced AI-powered weather analytics and forecasting system
</p>

</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🌦️ WeatherVision AI")

    st.markdown("---")

    st.success("Model Status: Ready")

    st.markdown("""
### Platform Overview

🤖 AI Model: XGBoost

📈 Accuracy: 95.32%

🌍 Dataset: 152K+ Records

☁️ Forecasting Engine: Active
""")

    st.markdown("---")

    st.info(
        "Upload a weather dataset (.csv) to generate temperature forecasts."
    )

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model("weather_model.json")
    return model

model = load_model()

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

st.subheader("📂 Upload Weather Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

# --------------------------------------------------
# PREDICTION PIPELINE
# --------------------------------------------------

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Input Dataset")

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

        # --------------------------------------
        # KPI CARDS
        # --------------------------------------

        avg_temp = round(
            df["predicted_temperature_celsius"].mean(),
            2
        )

        max_temp = round(
            df["predicted_temperature_celsius"].max(),
            2
        )

        min_temp = round(
            df["predicted_temperature_celsius"].min(),
            2
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("📊 Records", len(df))
        c2.metric("🌡️ Average °C", avg_temp)
        c3.metric("🔥 Maximum °C", max_temp)
        c4.metric("❄️ Minimum °C", min_temp)

        # --------------------------------------
        # CHART
        # --------------------------------------

        st.subheader("📈 Temperature Forecast Trend")

        chart_df = df.reset_index()

        fig = px.line(
            chart_df,
            x=chart_df.index,
            y="predicted_temperature_celsius",
            markers=True,
            title="Predicted Temperature Trend"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # --------------------------------------
        # AI INSIGHTS
        # --------------------------------------

        st.subheader("🤖 AI Insights")

        st.info(f"""
Average Forecast Temperature: {avg_temp}°C

Highest Forecast Temperature: {max_temp}°C

Lowest Forecast Temperature: {min_temp}°C

Records Analysed: {len(df)}
""")

        # --------------------------------------
        # RESULTS
        # --------------------------------------

        st.subheader("📋 Prediction Results")

        st.dataframe(df.head())

        csv = df.to_csv(index=False)

        st.download_button(
            label="📥 Download Prediction Results",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

        st.success(
            "✅ Predictions generated successfully."
        )

    except Exception as e:

        st.error(f"Error: {e}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown("""
<div class="footer">

WeatherVision AI™

Intelligent Climate & Temperature Forecasting Platform

Powered by XGBoost • Streamlit • Machine Learning

</div>
""", unsafe_allow_html=True)
