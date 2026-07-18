# Weather Temperature Prediction Model

## Overview

This repository contains a machine learning solution for weather temperature forecasting using **XGBoost Regression**. The model predicts air temperature in Celsius (`temperature_celsius`) using meteorological measurements, geographical information, air-quality indicators, and temporal features.

The complete pipeline has been validated and tested and is deployed through a Streamlit web application.

---

## Model Information

| Attribute         | Value                            |
| ----------------- | -------------------------------- |
| Algorithm         | XGBoost Regressor (XGBRegressor) |
| Learning Type     | Supervised Learning              |
| Task              | Regression                       |
| Target Variable   | temperature_celsius              |
| Input Features    | 26                               |
| Model Format      | XGBoost JSON (.json)             |
| Deployment Status | Ready                            |

---

## Model Performance

| Metric                         | Value  |
| ------------------------------ | ------ |
| Mean Absolute Error (MAE)      | 1.51   |
| Root Mean Squared Error (RMSE) | 2.05   |
| R² Score                       | 0.9532 |

### Performance Summary

* Explains approximately 95.3% of variance in temperature values.
* Average prediction error remains close to 1.5°C.
* Low RMSE indicates strong prediction consistency.
* Demonstrates excellent predictive performance.

---

## Loading the Model

```python
from xgboost import XGBRegressor

model = XGBRegressor()
model.load_model("weather_model.json")
```

---

## Running Predictions

```python
predictions = model.predict(X)

print(predictions[:5])
```

Example Output:

```python
[22.53, 15.67, 24.18, 10.45, 25.98]
```

---

## Streamlit Application

Run the application locally:

```bash
streamlit run app.py
```

The application allows users to:

* Upload weather datasets (.csv)
* Generate temperature predictions
* View prediction results
* Download prediction outputs

---

## Repository Structure

```text
weather-temperature-prediction/
│
├── app.py
├── weather_model.json
├── requirements.txt
├── README.md
├── .gitignore
└── assets/
    ├── feature_importance.png
    ├── correlation_heatmap.png
    ├── actual_vs_predicted.png
    └── temperature_distribution.png
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Streamlit
* Matplotlib
* Seaborn

---

## Deployment Readiness

✔ Model Validation Completed

✔ Feature Compatibility Verified

✔ Streamlit Application Tested

✔ Prediction Pipeline Verified

✔ JSON Model Successfully Loaded

✔ Deployment Ready

---

## Conclusion

This project presents a production-ready weather temperature prediction system powered by XGBoost Regression. Trained on more than 152,000 weather records and validated through comprehensive testing, the model achieves strong predictive performance with an R² score of 0.9532 and is suitable for analytics, forecasting, and deployment scenarios.
