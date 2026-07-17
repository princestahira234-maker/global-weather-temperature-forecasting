Weather Temperature Prediction Model
Overview
This repository contains a machine learning solution for weather temperature forecasting using XGBoost Regression.
The model predicts air temperature in Celsius (temperature_celsius) using meteorological measurements, geographical information, air-quality indicators, and temporal features. The complete pipeline has been validated and tested to ensure reliable predictive performance and deployment readiness.
________________


Project Objective
The objective of this project is to accurately estimate temperature values based on weather and environmental conditions.
The model leverages:
* Weather observations
* Atmospheric measurements
* Air quality indicators
* Geographic coordinates
* Temporal information
to generate robust and accurate temperature predictions.
________________


Dataset Information
Attribute
	Value
	Dataset Size
	152,802 Records
	Total Dataset Features
	44
	Features Used for Training
	26
	Target Variable
	temperature_celsius
	Data Domain
	Global Weather & Air Quality
	Dataset Highlights
* 152,802 weather observations
* Multiple countries and locations
* Air-quality measurements
* Weather condition indicators
* Geographic coordinates
* Time-based weather records
________________


Model Information
Attribute
	Value
	Algorithm
	XGBoost Regressor (XGBRegressor)
	Learning Type
	Supervised Learning
	Task
	Regression
	Target Variable
	Temperature (°C)
	Input Features
	26
	Model Format
	Pickle (.pkl)
	Deployment Status
	Ready
	________________


Model Performance
Metric
	Value
	Mean Absolute Error (MAE)
	1.51
	Root Mean Squared Error (RMSE)
	2.05
	R² Score
	0.9532
	Performance Summary
* Explains approximately 95.3% of variance in temperature values.
* Average prediction error remains close to 1.5°C.
* Low RMSE indicates strong prediction consistency.
* Demonstrates excellent predictive performance.
________________


Feature Importance Analysis
Feature importance analysis was performed to identify the most influential predictors affecting temperature estimation.
Rank
	Feature
	Importance
	1
	uv_index
	0.2814
	2
	latitude
	0.2334
	3
	pressure_mb
	0.1412
	4
	month
	0.0894
	5
	humidity
	0.0359
	6
	longitude
	0.0315
	7
	year
	0.0206
	8
	location_name
	0.0196
	9
	air_quality_Sulphur_dioxide
	0.0168
	10
	country
	0.0151
	  

Key Insights
* UV Index emerged as the strongest predictor.
* Geographic location significantly influences temperature.
* Atmospheric pressure contributes strongly to model predictions.
* Seasonal patterns captured through the month feature improve forecasting performance.
________________


Correlation Analysis
A correlation study was conducted to understand relationships among major weather variables.
  

Findings
* Temperature exhibits meaningful relationships with atmospheric variables.
* Environmental indicators provide complementary predictive information.
* Correlation analysis supported feature selection and model development decisions.
________________


Input Feature Schema
The model requires exactly 26 input features.
Categorical Features
* country
* location_name
* condition_text
* wind_direction
Numerical Features
* latitude
* longitude
* wind_kph
* wind_degree
* pressure_mb
* precip_mm
* humidity
* cloud
* visibility_km
* uv_index
* air_quality_Carbon_Monoxide
* air_quality_Ozone
* air_quality_Nitrogen_dioxide
* air_quality_Sulphur_dioxide
* air_quality_PM2.5
* air_quality_PM10
* air_quality_us-epa-index
* air_quality_gb-defra-index
* year
* month
* day
* hour
Important: Feature names and ordering must exactly match the training schema.
________________


Installation
Install required dependencies:
pip install -r requirements.txt


Or install manually:
pip install pandas numpy scikit-learn xgboost matplotlib seaborn


________________


Loading the Model
import pickle


with open("weather_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)


________________


Data Preparation
Generate Time-Based Features
import pandas as pd


df["last_updated"] = pd.to_datetime(df["last_updated"])


df["year"] = df["last_updated"].dt.year
df["month"] = df["last_updated"].dt.month
df["day"] = df["last_updated"].dt.day
df["hour"] = df["last_updated"].dt.hour


Prepare Input Features
required_features = model.get_booster().feature_names


X = df[required_features].copy()


Convert Categorical Features
categorical_columns = [
    "country",
    "location_name",
    "condition_text",
    "wind_direction"
]


for col in categorical_columns:
    X[col] = X[col].astype("category")


________________


Running Predictions
predictions = model.predict(X)


print(predictions[:5])


Example Output:
[22.536125 15.67361 24.182901 10.451522 25.986174]


________________


Model Validation
The model successfully passed the following validation checks:
* Model loading verification
* Feature count verification
* Feature name matching
* Missing value verification
* Categorical feature handling
* Prediction generation testing
* End-to-end inference validation
Validation Check
	Status
	Model Loading
	✅ Passed
	Feature Matching
	✅ Passed
	Missing Values
	✅ Passed
	Prediction Generation
	✅ Passed
	Inference Pipeline
	✅ Passed
	Validation Results
Expected Features : 26
Provided Features : 26
Feature Match     : True
Missing Values    : 0
Prediction Status : Successful


________________


Actual vs Predicted Performance
Model predictions were compared against actual temperature observations.
  

The close alignment of points around the reference line demonstrates strong predictive capability and supports the achieved R² score of 0.9532.
________________


Temperature Distribution
The distribution of temperature observations within the dataset was analyzed to understand data coverage and variability.
  

The visualization confirms that the model was trained on a diverse range of temperature values, supporting better generalization across different weather conditions.
________________


Data Visualization Methodology
Exploratory Data Analysis (EDA) and model evaluation visualizations were performed to understand dataset characteristics, validate feature relationships, and assess model performance.
Libraries Used
* Pandas
* NumPy
* Matplotlib
* Seaborn
* XGBoost
Visualization Techniques
Feature Importance Analysis
A horizontal bar chart was generated using XGBoost feature importance scores to identify the most influential variables affecting temperature prediction.
Correlation Heatmap
A correlation matrix was created and visualized using Seaborn to explore relationships among weather-related numerical features.
Actual vs Predicted Analysis
Scatter plots were used to compare actual and predicted temperature values, providing a visual assessment of model accuracy.
Temperature Distribution Analysis
Histograms were generated to analyze the distribution and coverage of temperature observations within the dataset.
Visualization Workflow
1. Data Cleaning and Validation
2. Feature Engineering
3. Exploratory Data Analysis (EDA)
4. Correlation Analysis
5. Model Training
6. Feature Importance Evaluation
7. Model Validation
8. Prediction Analysis
These visualizations improve interpretability, provide validation evidence, and support the overall reliability of the developed machine learning model.
________________


Repository Structure
weather-temperature-prediction/
│
├── weather_prediction_model.pkl
├── README.md
├── requirements.txt
├── app.py
├── .gitignore
└── assets/
    ├── feature_importance.png
    ├── correlation_heatmap.png
    ├── actual_vs_predicted.png
    └── temperature_distribution.png


________________


Technologies Used
* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Matplotlib
* Seaborn
________________


Deployment Readiness
Completed Validation:
✔ Model Serialization Verified
✔ Feature Compatibility Verified
✔ Data Integrity Verified
✔ Inference Pipeline Tested
✔ Performance Evaluated
✔ Documentation Prepared
The model is ready for:
* Client-side testing
* Application integration
* API deployment
* Research and analytics workflows
* Production deployment with appropriate monitoring
________________


Conclusion
This project presents a production-ready weather temperature prediction system powered by XGBoost Regression. Trained on more than 152,000 weather records and validated through comprehensive testing, the model delivers strong predictive performance with an R² score of 0.9532, low prediction error, and a fully validated inference pipeline, making it suitable for practical forecasting, analytics, and deployment scenarios.
