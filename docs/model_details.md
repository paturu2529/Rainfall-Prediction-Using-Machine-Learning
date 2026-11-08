# Machine Learning Model Details

## 1. Dummy Classifier
A baseline model that predicts the most frequent class. It is used as a reference to evaluate other machine learning models.

## 2. Logistic Regression
A classification algorithm used to estimate the probability of rainfall based on weather features.

## 3. Decision Tree Classifier
A tree-based algorithm that makes predictions using a sequence of feature-based decisions.

## 4. Random Forest Classifier
An ensemble learning algorithm that combines multiple decision trees to produce a classification result.

## Model Comparison
The four models are evaluated using the same training and testing split.

The evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## Application
The trained Random Forest pipeline is used by the Streamlit application to predict next-day rainfall based on user-provided weather measurements.