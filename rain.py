import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ----------------------------------------
# ML RAIN FORECASTING CLI
# ----------------------------------------

print("\n" + "*" * 50)
print("     RAINFALL PREDICTION ENGINE     ")
print("*" * 50 + "\n")

work_dir = os.path.dirname(os.path.abspath(__file__))

CSV_LOC = os.path.join(work_dir, "weatherAUS.csv")
MODEL_LOC = os.path.join(work_dir, "rainfall_model.pkl")

PREDICTOR_COLS = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation",
    "Sunshine", "WindGustSpeed", "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm",
    "Temp9am", "Temp3pm"
]

UNIT_DICT = {
    "MinTemp": "°C", "MaxTemp": "°C", "Rainfall": "mm",
    "Evaporation": "mm", "Sunshine": "hours",
    "WindGustSpeed": "km/h", "Humidity9am": "%",
    "Humidity3pm": "%", "Pressure9am": "hPa",
    "Pressure3pm": "hPa", "Cloud9am": "oktas (0-8)",
    "Cloud3pm": "oktas (0-8)", "Temp9am": "°C",
    "Temp3pm": "°C"
}

def run_prediction_interface(trained_pipeline):
    """Gathers user metrics via CLI and returns a model prediction."""
    
    print("\n--- INPUT WEATHER METRICS ---")
    print("Please provide values in the specified units.\n")

    user_data = {}

    for metric in PREDICTOR_COLS:
        while True:
            try:
                num_val = float(input(f"Enter {metric} ({UNIT_DICT[metric]}): "))

                if not -100 <= num_val <= 1000:
                    print(">> Value out of bounds. Try again.")
                    continue

                if metric.startswith("Humidity") and not 0 <= num_val <= 100:
                    print(">> Humidity scale is 0 to 100.")
                    continue

                if metric.startswith("Cloud") and not 0 <= num_val <= 8:
                    print(">> Cloud okta scale is 0 to 8.")
                    continue

                if metric == "Sunshine" and not 0 <= num_val <= 24:
                    print(">> Sunshine cannot exceed 24 hours.")
                    continue

                if "Temp" in metric and not -50 <= num_val <= 60:
                    print(">> Unrealistic temperature entered.")
                    continue

                if metric in ["Rainfall", "Evaporation", "WindGustSpeed"] and num_val < 0:
                    print(">> Metric cannot be a negative number.")
                    continue

                user_data[metric] = num_val
                break

            except ValueError:
                print(">> Invalid character. Numeric input required.")

    live_df = pd.DataFrame([[user_data[c] for c in PREDICTOR_COLS]], columns=PREDICTOR_COLS)

    pred_class = int(trained_pipeline.predict(live_df)[0])
    
    prob_scores = trained_pipeline.predict_proba(live_df)[0]
    class_labels = list(trained_pipeline.named_steps["rf_clf"].classes_)
    chance_of_rain = prob_scores[class_labels.index(1)] * 100

    print("\n" + "=" * 40)
    print("          SYSTEM OUTPUT")
    print("=" * 40)

    if pred_class == 1:
        print(">> VERDICT: RAIN EXPECTED")
    else:
        print(">> VERDICT: NO RAIN")

    print(f">> Confidence/Probability: {chance_of_rain:.2f}%")
    print("\n* Disclaimer: Educational ML output, not official meteorology.")


def build_and_train_model():
    """Reads CSV, structures train/test splits, and trains a Random Forest pipeline."""
    
    if not os.path.exists(CSV_LOC):
        print(f"\nFATAL: Dataset missing at {CSV_LOC}")
        return None

    print("\nReading weather dataset...")
    df = pd.read_csv(CSV_LOC)

    vital_cols = ["Date", "RainTomorrow"] + PREDICTOR_COLS
    absent = [c for c in vital_cols if c not in df.columns]

    if absent:
        print("FATAL: Dataset missing required columns:", absent)
        return None

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "RainTomorrow"])
    df = df[df["RainTomorrow"].isin(["Yes", "No"])]
    df = df.sort_values("Date").reset_index(drop=True)

    features = df[PREDICTOR_COLS]
    target = df["RainTomorrow"].map({"No": 0, "Yes": 1})

    split_idx = int(len(df) * 0.80)

    x_train_set = features.iloc[:split_idx]
    x_test_set = features.iloc[split_idx:]
    y_train_set = target.iloc[:split_idx]
    y_test_set = target.iloc[split_idx:]

    if len(x_train_set) == 0 or len(x_test_set) == 0:
        print("FATAL: Insufficient rows for splitting data.")
        return None

    if y_train_set.nunique() < 2 or y_test_set.nunique() < 2:
        print("FATAL: Needs both positive and negative cases in splits.")
        return None

    print(f"Train samples: {len(x_train_set)}")
    print(f"Test samples:  {len(x_test_set)}")
    print("\nBuilding and fitting pipeline...")

    rf_pipeline = Pipeline([
        ("missing_fill", SimpleImputer(strategy="median")),
        ("rf_clf", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    rf_pipeline.fit(x_train_set, y_train_set)

    test_preds = rf_pipeline.predict(x_test_set)

    print("\n--- MODEL PERFORMANCE REPORT ---")
    
    acc = accuracy_score(y_test_set, test_preds)
    print(f"General Accuracy: {acc * 100:.2f}%")

    print("\nDetailed Report:")
    print(classification_report(
        y_test_set, test_preds,
        labels=[0, 1],
        target_names=["Dry", "Wet"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test_set, test_preds, labels=[0, 1]))

    joblib.dump(
        {"model": rf_pipeline, "features": PREDICTOR_COLS},
        MODEL_LOC
    )

    print(f"\n>> Pipeline saved to {MODEL_LOC}")
    return rf_pipeline


def start_app():
    """Main execution loop for the CLI."""
    
    active_pipeline = build_and_train_model()

    if active_pipeline is None:
        print("\nHalting execution. Fix data issues.")
        return

    while True:
        print("\n-- MAIN MENU --")
        print("1. Forecast tomorrow's rain")
        print("2. Quit application")

        user_choice = input("Select an option (1/2): ").strip()

        if user_choice == "1":
            run_prediction_interface(active_pipeline)
        elif user_choice == "2":
            print("\nShutting down predictor. Goodbye!")
            break
        else:
            print(">> Unrecognized input.")

if __name__ == "__main__":
    start_app()