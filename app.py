import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")

# ============================================================
# RAINFALL PREDICTION USING MACHINE LEARNING
# Streamlit app aligned with the CSE570 assignment rubric
# ============================================================

st.set_page_config(
    page_title="Rainfall Prediction | ML with Python",
    page_icon="🌧️",
    layout="wide"
)

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weatherAUS.csv")
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rainfall_model.pkl")

FEATURES = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustSpeed", "Humidity9am", "Humidity3pm", "Pressure9am",
    "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"
]
TARGET = "RainTomorrow"
DATE_COL = "Date"

UNITS = {
    "MinTemp": "°C", "MaxTemp": "°C", "Rainfall": "mm",
    "Evaporation": "mm", "Sunshine": "hours",
    "WindGustSpeed": "km/h", "Humidity9am": "%",
    "Humidity3pm": "%", "Pressure9am": "hPa",
    "Pressure3pm": "hPa", "Cloud9am": "oktas (0–8)",
    "Cloud3pm": "oktas (0–8)", "Temp9am": "°C", "Temp3pm": "°C"
}

st.title("🌧️ Rainfall Prediction Using Machine Learning")
st.caption("CSE570 — Machine Learning with Python | Skill-Based Assignment")
st.markdown(
    "This application explores historical weather data, visualizes patterns, "
    "compares machine-learning models, and predicts whether rain may occur "
    "the following day. The supplied Rain in Australia dataset is Australian "
    "data; it is a learning demonstration, not an official local forecast."
)

@st.cache_data
def load_and_clean_data(path, file_mtime):
    df = pd.read_csv(path)
    required = [DATE_COL, TARGET] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    initial_rows = len(df)
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TARGET]).copy()
    df = df[df[TARGET].isin(["Yes", "No"])].copy()

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(DATE_COL).reset_index(drop=True)
    cleaning_summary = {
        "initial_rows": initial_rows,
        "duplicates_removed": duplicate_rows,
        "rows_after_cleaning": len(df),
        "rows_removed_total": initial_rows - len(df)
    }
    return df, cleaning_summary

@st.cache_resource
def fit_models(data_fingerprint, X_train, y_train):
    # All imputation/scaling happens inside the pipelines, fitted only on training data.
    models = {
        "Baseline (most frequent)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DummyClassifier(strategy="most_frequent"))
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1500, class_weight="balanced", random_state=42
            ))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=20,
                class_weight="balanced", random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=200, max_depth=None,
                min_samples_leaf=2, class_weight="balanced",
                random_state=42, n_jobs=-1
            ))
        ])
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models

if not os.path.exists(DATA_FILE):
    st.error(
        f"Could not find `weatherAUS.csv` at: {DATA_FILE}. "
        "Place the CSV file in the same folder as this app.py file."
    )
    st.stop()

try:
    df, cleaning = load_and_clean_data(DATA_FILE, os.path.getmtime(DATA_FILE))
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if len(df) < 50:
    st.error("The dataset has too few valid rows for a meaningful train/test split.")
    st.stop()

df["RainTomorrowBinary"] = df[TARGET].map({"No": 0, "Yes": 1})
tab_overview, tab_eda, tab_models, tab_predict, tab_report = st.tabs([
    "📌 Overview & Data Cleaning",
    "📊 EDA & Statistical Analysis",
    "🤖 Model Comparison",
    "🌦️ Predict Rainfall",
    "📝 Assignment Report Guide"
])

# -------------------- OVERVIEW --------------------
with tab_overview:
    st.header("1. Problem and Dataset")
    st.write(
        "**Problem statement:** Predict whether rainfall will occur tomorrow "
        "using the weather measurements available for a given day. This is a "
        "binary supervised-classification problem."
    )
    st.write(
        "**Target variable:** `RainTomorrow` (Yes = rain, No = no rain). "
        "Intended impact: demonstrate how weather observations can support "
        "planning. The model is not intended for operational safety decisions."
    )

    a, b, c, d = st.columns(4)
    a.metric("Valid records", f"{len(df):,}")
    b.metric("Input features", len(FEATURES))
    c.metric("Duplicate rows removed", f"{cleaning['duplicates_removed']:,}")
    d.metric("Missing feature cells", f"{int(df[FEATURES].isna().sum().sum()):,}")

    st.subheader("Dataset source")
    st.markdown(
        "[Rain in Australia — Kaggle dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)"
    )
    st.write(
        "The exact record count and missing-value summary below are calculated "
        "from the CSV file supplied by the student; dataset versions may differ."
    )

    st.subheader("Cleaning summary")
    st.dataframe(pd.DataFrame({
        "Measure": [
            "Rows before cleaning", "Duplicate rows found/removed",
            "Rows after cleaning", "Rows removed overall"
        ],
        "Value": [
            cleaning["initial_rows"], cleaning["duplicates_removed"],
            cleaning["rows_after_cleaning"], cleaning["rows_removed_total"]
        ]
    }), use_container_width=True)

    st.subheader("Data dictionary")
    dictionary = {
        "MinTemp": "Minimum temperature (°C)",
        "MaxTemp": "Maximum temperature (°C)",
        "Rainfall": "Rainfall recorded that day (mm)",
        "Evaporation": "Evaporation (mm)",
        "Sunshine": "Sunshine duration (hours)",
        "WindGustSpeed": "Maximum wind gust speed (km/h)",
        "Humidity9am": "Humidity at 9 AM (%)",
        "Humidity3pm": "Humidity at 3 PM (%)",
        "Pressure9am": "Atmospheric pressure at 9 AM (hPa)",
        "Pressure3pm": "Atmospheric pressure at 3 PM (hPa)",
        "Cloud9am": "Cloud cover at 9 AM (oktas)",
        "Cloud3pm": "Cloud cover at 3 PM (oktas)",
        "Temp9am": "Temperature at 9 AM (°C)",
        "Temp3pm": "Temperature at 3 PM (°C)",
        "RainTomorrow": "Target: Yes or No"
    }
    st.dataframe(pd.DataFrame(
        [{"Feature": k, "Description": v} for k, v in dictionary.items()]
    ), use_container_width=True)

    st.subheader("Target/class distribution")
    target_counts = df[TARGET].value_counts().reindex(["No", "Yes"], fill_value=0)
    fig, ax = plt.subplots()
    ax.bar(target_counts.index, target_counts.values)
    ax.set_title("RainTomorrow Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of records")
    st.pyplot(fig)
    plt.close(fig)
    st.write("Class proportions:")
    st.dataframe((target_counts / target_counts.sum() * 100).round(2).rename("Percentage").reset_index().rename(columns={"index": "RainTomorrow"}), use_container_width=True)

    st.subheader("Missing values by feature")
    missing_counts = df[FEATURES].isna().sum().sort_values(ascending=False)
    missing_counts = missing_counts[missing_counts > 0]

    # Build a feature-wise preprocessing summary before model fitting.
    missing_summary = pd.DataFrame({
        "Feature": FEATURES,
        "Missing Values": [int(df[col].isna().sum()) for col in FEATURES],
        "Missing Percentage": [
            round(float(df[col].isna().mean() * 100), 2) for col in FEATURES
        ],
        "Imputation Strategy": ["Median (inside model pipeline)"] * len(FEATURES)
    }).sort_values("Missing Percentage", ascending=False)

    total_missing_cells = int(df[FEATURES].isna().sum().sum())
    affected_features = int((df[FEATURES].isna().sum() > 0).sum())
    highest_missing_feature = (
        missing_summary.iloc[0]["Feature"]
        if not missing_summary.empty else "None"
    )

    prep_col1, prep_col2, prep_col3 = st.columns(3)
    prep_col1.metric("Missing Feature Cells", f"{total_missing_cells:,}")
    prep_col2.metric("Features with Missing Values", affected_features)
    prep_col3.metric("Highest Missing Feature", highest_missing_feature)

    st.subheader("Preprocessing Summary")
    st.dataframe(missing_summary, use_container_width=True, hide_index=True)
    st.info(
        "Missing numeric values are not filled during dataset loading. "
        "Each model pipeline applies median imputation after the chronological "
        "train/test split, fitting the imputer on training data only. "
        "This helps prevent information leakage from the test set."
    )
    if len(missing_counts):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(missing_counts.index, missing_counts.values)
        ax.set_title("Missing Values by Feature")
        ax.set_ylabel("Missing cell count")
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.success("No missing values were found in the selected numeric features.")

    st.subheader("Preview of cleaned data")
    st.dataframe(df[[DATE_COL] + FEATURES + [TARGET]].head(20), use_container_width=True)

# -------------------- EDA --------------------
with tab_eda:
    st.header("2. Exploratory Data Analysis (EDA)")
    st.write(
        "Use the plots below to inspect individual feature distributions, "
        "relationships between variables, and differences between rain/no-rain days."
    )

    chosen_feature = st.selectbox("Select a feature for distribution analysis", FEATURES)
    fig, ax = plt.subplots()
    values = df[chosen_feature].dropna()
    ax.hist(values, bins=30)
    ax.set_title(f"Distribution of {chosen_feature}")
    ax.set_xlabel(f"{chosen_feature} ({UNITS[chosen_feature]})")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    plt.close(fig)
    st.write(values.describe().to_frame("Descriptive statistics").round(3))

    st.subheader("Rain vs no-rain comparison")
    compare_feature = st.selectbox(
        "Choose a feature to compare by target",
        ["Humidity3pm", "Humidity9am", "Pressure3pm", "Pressure9am",
         "Sunshine", "WindGustSpeed", "Cloud3pm", "Rainfall"],
        key="compare_feature"
    )
    groups = [
        df.loc[df[TARGET] == label, compare_feature].dropna()
        for label in ["No", "Yes"]
    ]
    fig, ax = plt.subplots()
    ax.boxplot(groups, tick_labels=["No Rain", "Rain"], showfliers=False)
    ax.set_title(f"{compare_feature} by RainTomorrow")
    ax.set_ylabel(f"{compare_feature} ({UNITS[compare_feature]})")
    st.pyplot(fig)
    plt.close(fig)

    no_rain_median = groups[0].median() if len(groups[0]) else np.nan
    rain_median = groups[1].median() if len(groups[1]) else np.nan
    st.write(f"Median {compare_feature} on no-rain days: **{no_rain_median:.3f}**")
    st.write(f"Median {compare_feature} on rain days: **{rain_median:.3f}**")

    st.subheader("Correlation analysis")
    corr = df[FEATURES].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    image = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation Matrix of Weather Features")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Correlation indicates association, not causation.")

    st.subheader("Statistical test: Humidity3pm on rain vs no-rain days")
    no_rain_humidity = df.loc[df[TARGET] == "No", "Humidity3pm"].dropna()
    rain_humidity = df.loc[df[TARGET] == "Yes", "Humidity3pm"].dropna()
    if len(no_rain_humidity) > 0 and len(rain_humidity) > 0:
        stat, p_value = mannwhitneyu(
            no_rain_humidity, rain_humidity, alternative="two-sided"
        )
        st.write("**Test:** Two-sided Mann–Whitney U test (independent groups).")
        st.write(f"U statistic: **{stat:,.2f}**")
        st.write(f"p-value: **{p_value:.6g}**")
        if p_value < 0.05:
            st.write(
                "Interpretation: At the 5% significance level, the observed "
                "Humidity3pm distributions differ statistically between the "
                "rain and no-rain groups. This does not prove causation."
            )
        else:
            st.write(
                "Interpretation: At the 5% significance level, the test does "
                "not provide sufficient evidence of a difference between groups."
            )
    else:
        st.info("Not enough non-missing values to run the selected test.")

    st.subheader("Group-wise summary")
    group_summary = df.groupby(TARGET)[FEATURES].median().T
    st.dataframe(group_summary.round(3), use_container_width=True)

# -------------------- MODEL COMPARISON --------------------
with tab_models:
    st.header("3. Model Development and Evaluation")
    st.write(
        "The data is sorted by date. The earliest 80% is used for training and "
        "the latest 20% is held out for testing. Imputation and scaling are "
        "inside model pipelines to reduce data leakage."
    )

    split_index = int(len(df) * 0.80)
    X = df[FEATURES]
    y = df[TARGET].map({"No": 0, "Yes": 1})
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        st.error("Train/test split must contain both target classes.")
    else:
        with st.spinner("Training and evaluating four models..."):
            models = fit_models(
                f"{len(df)}-{os.path.getmtime(DATA_FILE)}",
                X_train, y_train
            )

        results = []
        predictions_by_model = {}
        for name, fitted_model in models.items():
            pred = fitted_model.predict(X_test)
            predictions_by_model[name] = pred
            row = {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision (Rain)": precision_score(y_test, pred, zero_division=0),
                "Recall (Rain)": recall_score(y_test, pred, zero_division=0),
                "F1-score (Rain)": f1_score(y_test, pred, zero_division=0)
            }
            try:
                proba = fitted_model.predict_proba(X_test)[:, list(fitted_model.named_steps["classifier"].classes_).index(1)]
                row["ROC-AUC"] = roc_auc_score(y_test, proba)
            except Exception:
                row["ROC-AUC"] = np.nan
            results.append(row)

        results_df = pd.DataFrame(results).set_index("Model")
        st.subheader("Model comparison table")
        st.dataframe(results_df.style.format("{:.3f}"), use_container_width=True)

        metric = st.selectbox(
            "Choose metric to visualize",
            ["Accuracy", "Precision (Rain)", "Recall (Rain)", "F1-score (Rain)", "ROC-AUC"]
        )
        fig, ax = plt.subplots()
        ax.bar(results_df.index, results_df[metric])
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric)
        ax.set_title(f"Model Comparison by {metric}")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        best_name = results_df["F1-score (Rain)"].idxmax()
        st.success(
            f"Best model by Rain-class F1-score on the held-out test set: {best_name}"
        )
        st.caption(
            "Choose a final model based on the task's priorities, not accuracy alone. "
            "If missing rain is costly, pay special attention to Rain recall."
        )

        selected_model_name = st.selectbox(
            "Inspect detailed results for a model",
            list(models.keys()),
            index=list(models.keys()).index("Random Forest")
        )
        selected_model = models[selected_model_name]
        selected_pred = predictions_by_model[selected_model_name]

        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_test, selected_pred, labels=[0, 1])
        fig, ax = plt.subplots()
        image = ax.imshow(cm)
        ax.set_xticks([0, 1], labels=["Predicted No", "Predicted Rain"])
        ax.set_yticks([0, 1], labels=["Actual No", "Actual Rain"])
        ax.set_title(f"Confusion Matrix — {selected_model_name}")
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center")
        fig.colorbar(image, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Classification report")
        report = classification_report(
            y_test, selected_pred, labels=[0, 1],
            target_names=["No Rain", "Rain"], output_dict=True,
            zero_division=0
        )
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

        st.subheader("ROC curves")
        fig, ax = plt.subplots()
        for name, fitted_model in models.items():
            try:
                class_list = list(fitted_model.named_steps["classifier"].classes_)
                scores = fitted_model.predict_proba(X_test)[:, class_list.index(1)]
                RocCurveDisplay.from_predictions(
                    y_test, scores, name=name, ax=ax
                )
            except Exception:
                pass
        ax.set_title("ROC Curves on Held-out Test Data")
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Feature importance / interpretation")
        classifier = selected_model.named_steps["classifier"]
        if hasattr(classifier, "feature_importances_"):
            importance = pd.Series(
                classifier.feature_importances_, index=FEATURES
            ).sort_values(ascending=False)
            st.dataframe(
                importance.rename("Importance").to_frame().round(4),
                use_container_width=True
            )
            fig, ax = plt.subplots()
            importance.sort_values().plot(kind="barh", ax=ax)
            ax.set_title(f"Feature Importance — {selected_model_name}")
            ax.set_xlabel("Importance")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        elif hasattr(classifier, "coef_"):
            coefficients = pd.Series(
                classifier.coef_[0], index=FEATURES
            ).sort_values(key=abs, ascending=False)
            st.dataframe(
                coefficients.rename("Coefficient").to_frame().round(4),
                use_container_width=True
            )
            st.caption(
                "For Logistic Regression, coefficients show direction and "
                "strength on the scaled feature values; they are not causal effects."
            )
        else:
            st.write(
                "The baseline classifier does not provide feature importance."
            )

        # Save the random forest pipeline for prediction.
        if "Random Forest" in models:
            joblib.dump(
                {"model": models["Random Forest"], "features": FEATURES},
                MODEL_FILE
            )
            st.info(
                f"Random Forest model saved to `{MODEL_FILE}` for the prediction tab."
            )

# -------------------- PREDICTION --------------------
with tab_predict:
    st.header("4. Predict Rainfall")
    st.warning(
        "Enter measurements for one observation. The model was trained on "
        "Australian weather data, so these predictions should not be treated "
        "as reliable forecasts for India."
    )

    with st.form("prediction_form"):
        weather = {}
        left, right = st.columns(2)
        defaults = {
            "MinTemp": 22.0, "MaxTemp": 32.0, "Rainfall": 0.0,
            "Evaporation": 5.0, "Sunshine": 6.0, "WindGustSpeed": 35.0,
            "Humidity9am": 75.0, "Humidity3pm": 55.0,
            "Pressure9am": 1012.0, "Pressure3pm": 1010.0,
            "Cloud9am": 4.0, "Cloud3pm": 5.0, "Temp9am": 25.0, "Temp3pm": 30.0
        }
        limits = {
            "MinTemp": (-50.0, 60.0), "MaxTemp": (-50.0, 60.0),
            "Rainfall": (0.0, 1000.0), "Evaporation": (0.0, 100.0),
            "Sunshine": (0.0, 24.0), "WindGustSpeed": (0.0, 300.0),
            "Humidity9am": (0.0, 100.0), "Humidity3pm": (0.0, 100.0),
            "Pressure9am": (800.0, 1100.0), "Pressure3pm": (800.0, 1100.0),
            "Cloud9am": (0.0, 8.0), "Cloud3pm": (0.0, 8.0),
            "Temp9am": (-50.0, 60.0), "Temp3pm": (-50.0, 60.0)
        }
        for i, feature in enumerate(FEATURES):
            with left if i % 2 == 0 else right:
                low, high = limits[feature]
                weather[feature] = st.number_input(
                    f"{feature} ({UNITS[feature]})",
                    min_value=low, max_value=high,
                    value=defaults[feature], step=1.0,
                    key=f"predict_{feature}"
                )
        predict_clicked = st.form_submit_button("Predict Rainfall")

    if predict_clicked:
        input_df = pd.DataFrame(
            [[weather[f] for f in FEATURES]], columns=FEATURES
        )
        # Use the fitted Random Forest if available; otherwise train/load it.
        try:
            if os.path.exists(MODEL_FILE):
                saved = joblib.load(MODEL_FILE)
                model = saved["model"]
            else:
                st.info("Train the models in the Model Comparison tab first.")
                st.stop()

            prediction = int(model.predict(input_df)[0])
            class_list = list(model.named_steps["classifier"].classes_)
            proba = model.predict_proba(input_df)[0]
            rain_probability = float(proba[class_list.index(1)]) * 100

            c1, c2 = st.columns(2)
            with c1:
                if prediction == 1:
                    st.error("🌧️ Rain predicted")
                else:
                    st.success("☀️ No rain predicted")
            with c2:
                st.metric("Estimated rain probability", f"{rain_probability:.2f}%")
                st.progress(min(max(rain_probability / 100, 0), 1))
            st.dataframe(
                input_df.T.rename(columns={0: "Entered value"}),
                use_container_width=True
            )
            st.caption(
                "The displayed probability is a model estimate and may not be "
                "calibrated. It is not an official forecast."
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# -------------------- REPORT GUIDE --------------------
with tab_report:
    st.header("5. Assignment Report Guide")
    st.write(
        "Use this structure for the written report required by the assignment. "
        "Insert actual figures and measured results from the tabs above; do not "
        "invent accuracy values or findings."
    )
    st.markdown("""
### Suggested report structure

1. **Title page**
   - Rainfall Prediction Using Machine Learning
   - Course: CSE570 — Machine Learning with Python
   - Student name, registration number, programme, faculty, submission date

2. **Problem statement and objectives**
   - Define the binary classification problem and `RainTomorrow` target.
   - Explain intended users, measurable objectives, assumptions, and limitations.

3. **Dataset description**
   - Source: Rain in Australia dataset (Kaggle link shown in the Overview tab).
   - Report the exact row/feature counts calculated from your downloaded CSV.
   - Include target distribution, data dictionary, and missing-value summary.

4. **Methodology / workflow**
   - Load data → remove duplicates → correct data types → chronological split
     → imputation/scaling inside pipelines → train models → evaluate → predict.
   - Explain why time ordering is used for the train/test split.

5. **Data cleaning and EDA**
   - Include target distribution, missing-value chart, feature histogram,
     rain/no-rain boxplot, correlation matrix, group-wise summary, and
     Mann–Whitney U test.
   - Interpret each figure in words, not only paste the chart.

6. **Model development and evaluation**
   - Baseline, Logistic Regression, Decision Tree, Random Forest.
   - Compare accuracy, Rain precision, Rain recall, Rain F1-score, ROC-AUC,
     confusion matrix, and ROC curves.
   - Explain the selected model and feature importance/coefficients.
   - Record real metrics generated by your run.

7. **Results and discussion**
   - State the strongest model for your chosen metric.
   - Discuss class imbalance, false negatives, false positives, and generalization.

8. **Conclusion, limitations, and future work**
   - Australian data may not generalize to Indian weather.
   - Manual input values and the model's estimated probability are not official forecasts.
   - Future work: validate with suitable Indian weather data and add reliable forecast data.

9. **References and evidence**
   - Dataset link, Python, Pandas, Scikit-learn, Streamlit documentation.
   - GitHub repository URL and LinkedIn post URL/screenshot after you create them.
""")
