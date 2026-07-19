import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
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

.hero-title {
    text-align:center;
    color:#0f172a;
    font-size:42px;
    font-weight:700;
}

.hero-subtitle {
    text-align:center;
    color:#475569;
    font-size:18px;
}

.metric-container {
    background:#f8fafc;
    padding:10px;
    border-radius:12px;
    border:1px solid #e2e8f0;
}

.footer {
    text-align:center;
    color:#64748b;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

try:
    logo = Image.open("assets/logo.png")
    st.sidebar.image(logo, width=180)
except:
    pass

st.sidebar.title("WeatherVision AI™")

st.sidebar.success("✅ Model Status: Ready")

st.sidebar.info("""
Algorithm: XGBoost Regressor

Task: Temperature Forecasting

Deployment: Production Ready
""")

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Platform Features

• AI Temperature Forecasting

• CSV Upload

• Interactive Analytics

• Download Predictions

• Model Insights
""")

# --------------------------------------------------
# HERO BANNER
# --------------------------------------------------

try:
    st.image("assets/banner.jpg", use_container_width=True)
except:
    pass

st.markdown("""
<div class='hero-title'>
WeatherVision AI™
</div>

<div class='hero-subtitle'>
Intelligent Climate & Temperature Forecasting Platform
</div>
""", unsafe_allow_html=True)

st.write("")

# --------------------------------------------------
# LOAD MODEL
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
    "Choose a CSV File",
    type=["csv"]
)

# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Dataset Preview")

    rows, cols = df.shape

    c1, c2 = st.columns(2)

    c1.metric("Rows", rows)
    c2.metric("Columns", cols)

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

        # ------------------------------------------
        # KPI CARDS
        # ------------------------------------------

        avg_temp = round(
            df["predicted_temperature_celsius"].mean(), 2
        )

        max_temp = round(
            df["predicted_temperature_celsius"].max(), 2
        )

        min_temp = round(
            df["predicted_temperature_celsius"].min(), 2
        )

        st.subheader("📊 Forecast Summary")

        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Records",
            len(df)
        )

        k2.metric(
            "Average °C",
            avg_temp
        )

        k3.metric(
            "Maximum °C",
            max_temp
        )

        k4.metric(
            "Minimum °C",
            min_temp
        )

        # ------------------------------------------
        # CHART
        # ------------------------------------------

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

        # ------------------------------------------
        # HISTOGRAM
        # ------------------------------------------

        st.subheader("📉 Temperature Distribution")

        hist_fig = px.histogram(
            df,
            x="predicted_temperature_celsius",
            nbins=20
        )

        st.plotly_chart(
            hist_fig,
            use_container_width=True
        )

        # ------------------------------------------
        # AI INSIGHTS
        # ------------------------------------------

        st.subheader("🤖 AI Forecast Insights")

        st.info(f"""
Average predicted temperature is {avg_temp}°C.

Highest predicted temperature is {max_temp}°C.

Lowest predicted temperature is {min_temp}°C.

Total records analysed: {len(df)}.
""")

        # ------------------------------------------
        # RESULTS
        # ------------------------------------------

        st.subheader("📋 Prediction Results")

        st.dataframe(df.head(20))

        # ------------------------------------------
        # DOWNLOAD
        # ------------------------------------------

        csv = df.to_csv(index=False)

        st.download_button(
            label="📥 Download Prediction Results",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

        st.success(
            "Predictions generated successfully!"
        )

        # ------------------------------------------
        # ASSETS GALLERY
        # ------------------------------------------

        st.subheader("📚 Model Insights")

        col1, col2 = st.columns(2)

        try:
            col1.image(
                "assets/feature_importance.png",
                caption="Feature Importance"
            )
        except:
            pass

        try:
            col2.image(
                "assets/correlation_heatmap.png",
                caption="Correlation Heatmap"
            )
        except:
            pass

        col3, col4 = st.columns(2)

        try:
            col3.image(
                "assets/actual_vs_predicted.png",
                caption="Actual vs Predicted"
            )
        except:
            pass

        try:
            col4.image(
                "assets/temperature_distribution.png",
                caption="Temperature Distribution"
            )
        except:
            pass

    except Exception as e:

        st.error(f"Error: {e}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown("""
<div class='footer'>

WeatherVision AI™

Intelligent Climate & Temperature Forecasting Platform

Powered by XGBoost • Streamlit • Machine Learning

</div>
""", unsafe_allow_html=True)
