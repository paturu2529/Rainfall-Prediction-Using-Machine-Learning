# Application Architecture

## Project Overview
The Rainfall Prediction application uses Python and machine learning to predict whether rainfall will occur the following day.

## Main Components

### 1. Dataset
The application uses the Rain in Australia dataset stored in `weatherAUS.csv`.

### 2. Data Preprocessing
The dataset is cleaned, duplicate records are removed, dates are processed, and missing numerical values are handled.

### 3. Exploratory Data Analysis
Statistical analysis and visualizations are used to understand weather patterns.

### 4. Model Training
Dummy Classifier, Logistic Regression, Decision Tree, and Random Forest models are developed and evaluated.

### 5. Prediction
The trained Random Forest pipeline generates rainfall predictions from weather measurements.

### 6. User Interface
Streamlit provides the interface for data exploration, model comparison, and rainfall prediction.

## Technologies
Python, Pandas, NumPy, Scikit-learn, Matplotlib, SciPy, Joblib, and Streamlit.