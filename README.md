# Rainfall Prediction Using Machine Learning with Python

## Project Overview

Rainfall Prediction is a machine learning project that predicts whether rainfall will occur on the following day using historical weather data.

The project uses Python, Scikit-learn, and Streamlit to develop an interactive application for data analysis, machine learning model comparison, and rainfall prediction.

## Objectives

- Analyse historical weather data.
- Perform data cleaning and preprocessing.
- Explore weather patterns through visualizations.
- Develop and compare machine learning classification models.
- Evaluate model performance using standard classification metrics.
- Build an interactive rainfall prediction application.

## Dataset Description

- **Dataset Name:** Rain in Australia
- **Dataset File:** `weatherAUS.csv`
- **Target Variable:** `RainTomorrow`
- **Prediction Classes:** Yes / No
- **Features:** Temperature, humidity, rainfall, wind speed, pressure, sunshine, and cloud measurements.

**Dataset Source:**  
https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package

## Technologies Used

pandas==2.2.3
numpy==2.1.3
matplotlib==3.9.2
seaborn==0.13.2
scikit-learn==1.5.2
scipy==1.14.1
joblib==1.4.2

## Machine Learning Models

1. Dummy Classifier (Baseline)
2. Logistic Regression
3. Decision Tree
4. Random Forest

## Application Modules

- Overview and Data Cleaning
- Exploratory Data Analysis
- Model Comparison
- Rainfall Prediction

## Project Structure

```text
Rainfall-Prediction-Using-Machine-Learning/
│
├── app.py
├── rain.py
├── weatherAUS.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run the Project

Clone the repository:

```bash
git clone https://github.com/paturu2529/Rainfall-Prediction-Using-Machine-Learning.git
```

Open the project directory:

```bash
cd Rainfall-Prediction-Using-Machine-Learning
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Limitations

- The model uses historical Australian weather data.
- Predictions may not be reliable for other geographical regions.
- Predictions are for educational and demonstration purposes and are not official weather forecasts.

## Author

**Paturu Bhargava Naidu**

M.Tech Data Science  
Lovely Professional University

GitHub: https://github.com/paturu2529
