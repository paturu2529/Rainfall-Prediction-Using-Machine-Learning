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

# ----------------------------------------------------
# ML-BASED RAINFALL PREDICTOR
# ----------------------------------------------------

st.set_page_config(
    page_title="Rain Predictor | ML App",
    page_icon="💧",
    layout="wide"
)

current_dir = os.path.dirname(os.path.abspath(__file__))
path_to_csv = os.path.join(current_dir, "weatherAUS.csv")
model_save_path = os.path.join(current_dir, "rainfall_model.pkl")

weather_cols = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustSpeed", "Humidity9am", "Humidity3pm", "Pressure9am",
    "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"
]
predict_target = "RainTomorrow"
time_col = "Date"

metric_units = {
    "MinTemp": "°C", "MaxTemp": "°C", "Rainfall": "mm",
    "Evaporation": "mm", "Sunshine": "hrs",
    "WindGustSpeed": "km/h", "Humidity9am": "%",
    "Humidity3pm": "%", "Pressure9am": "hPa",
    "Pressure3pm": "hPa", "Cloud9am": "oktas (0-8)",
    "Cloud3pm": "oktas (0-8)", "Temp9am": "°C", "Temp3pm": "°C"
}

# --- Functions (Reordered) ---

@st.cache_resource
def train_and_evaluate(data_id_string, train_features, train_labels):
    """Builds and fits multiple classification pipelines."""
    
    ml_pipelines = {
        "Simple Baseline": Pipeline([
            ("fill_missing", SimpleImputer(strategy="median")),
            ("clf", DummyClassifier(strategy="most_frequent"))
        ]),
        "Logistics Reg": Pipeline([
            ("fill_missing", SimpleImputer(strategy="median")),
            ("scale_data", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1500, class_weight="balanced", random_state=42
            ))
        ]),
        "Tree Model": Pipeline([
            ("fill_missing", SimpleImputer(strategy="median")),
            ("clf", DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=20,
                class_weight="balanced", random_state=42
            ))
        ]),
        "Forest Model": Pipeline([
            ("fill_missing", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=None,
                min_samples_leaf=2, class_weight="balanced",
                random_state=42, n_jobs=-1
            ))
        ])
    }
    
    for algo_name, pipeline in ml_pipelines.items():
        pipeline.fit(train_features, train_labels)
        
    return ml_pipelines


def validate_dataset(data_frame):
    """Validate the raw dataset before preprocessing."""
    required_cols = [time_col, predict_target] + weather_cols

    if data_frame.empty:
        raise ValueError("The dataset is empty.")

    missing_cols = [col for col in required_cols if col not in data_frame.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    invalid_targets = int(
        (~data_frame[predict_target].isin(["Yes", "No"])).sum()
    )
    missing_dates = int(data_frame[time_col].isna().sum())

    return {
        "invalid_targets": invalid_targets,
        "missing_dates": missing_dates,
    }


@st.cache_data
def prep_dataset(file_location, modified_time):
    """Reads the CSV file, removes duplicates, and drops bad rows."""
    
    data_frame = pd.read_csv(file_location)
    validation_report = validate_dataset(data_frame)
    
    needed_cols = [time_col, predict_target] + weather_cols
    missing_cols = [c for c in needed_cols if c not in data_frame.columns]
    
    if missing_cols:
        raise ValueError(f"Missing crucial columns: {missing_cols}")

    start_len = len(data_frame)
    dupe_count = int(data_frame.duplicated().sum())
    
    data_frame = data_frame.drop_duplicates().copy()
    data_frame[time_col] = pd.to_datetime(data_frame[time_col], errors="coerce")
    data_frame = data_frame.dropna(subset=[time_col, predict_target]).copy()
    data_frame = data_frame[data_frame[predict_target].isin(["Yes", "No"])].copy()

    for col in weather_cols:
        data_frame[col] = pd.to_numeric(data_frame[col], errors="coerce")

    data_frame = data_frame.sort_values(time_col).reset_index(drop=True)
    
    stats = {
        "start_len": start_len,
        "dupe_count": dupe_count,
        "final_len": len(data_frame),
        "total_dropped": start_len - len(data_frame),
        "invalid_targets": validation_report["invalid_targets"],
        "missing_dates": validation_report["missing_dates"],
        "validation_status": "Passed"
    }
    
    return data_frame, stats

# --- Main App ---

st.title("💧 ML-Based Rainfall Predictor")
st.caption("CSE570 — Python Machine Learning | Assignment")
st.markdown(
    "This tool analyzes past weather measurements, visualizes trends, "
    "tests various classification algorithms, and attempts to forecast rain. "
    "Note: This relies on an Australian dataset for educational purposes."
)

if not os.path.exists(path_to_csv):
    st.error(f"Cannot locate dataset at: {path_to_csv}. Please ensure it is in the same folder.")
    st.stop()

try:
    weather_data, clean_stats = prep_dataset(path_to_csv, os.path.getmtime(path_to_csv))
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()

if len(weather_data) < 50:
    st.error("Insufficient data rows to perform train/test splits.")
    st.stop()

info_tab, analysis_tab, ml_tab, forecast_tab = st.tabs([
    "📝 Overview & Data Cleaning",
    "📈 Statistics & EDA",
    "⚙️ Model comparison",
    "🔮 Predict Rainfall"
])

with info_tab:
    st.header("1. Overview")
    st.write(
        "**Objective:** Build a binary classifier to predict tomorrow's rain status "
        "based on today's weather metrics."
    )
    st.write(
        "**Target:** `RainTomorrow` (Yes / No). Not designed for real-world emergency use."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Usable Rows", f"{len(weather_data):,}")
    col2.metric("Predictors", len(weather_cols))
    col3.metric("Dupes Dropped", f"{clean_stats['dupe_count']:,}")
    col4.metric("Blank Cells", f"{int(weather_data[weather_cols].isna().sum().sum()):,}")

    st.subheader("Dataset Validation Report")
    st.success("Dataset validation completed successfully.")

    val_col1, val_col2, val_col3 = st.columns(3)
    val_col1.metric("Invalid Target Records", f"{clean_stats['invalid_targets']:,}")
    val_col2.metric("Missing Date Records", f"{clean_stats['missing_dates']:,}")
    val_col3.metric("Validation Status", clean_stats["validation_status"])

    st.subheader("Data Cleaning Stats")
    st.dataframe(pd.DataFrame({
        "Metric": [
            "Initial Rows", "Duplicates Found",
            "Final Rows", "Total Rows Dropped"
        ],
        "Count": [
            clean_stats["start_len"], clean_stats["dupe_count"],
            clean_stats["final_len"], clean_stats["total_dropped"]
        ]
    }), use_container_width=True)

    st.subheader("Target Balance")
    class_totals = weather_data[predict_target].value_counts().reindex(["No", "Yes"], fill_value=0)
    
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(class_totals.index, class_totals.values)
    ax_bar.set_title("Target Distribution")
    ax_bar.set_xlabel("Rain Outcome")
    ax_bar.set_ylabel("Count")
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    st.subheader("Clean Data Glimpse")
    st.dataframe(weather_data[[time_col] + weather_cols + [predict_target]].head(20), use_container_width=True)


with analysis_tab:
    st.header("2. Data Exploration")
    st.write("Inspect how different metrics distribute and relate to rainfall.")

    focus_var = st.selectbox("Pick a metric for distribution", weather_cols)
    fig_hist, ax_hist = plt.subplots()
    val_series = weather_data[focus_var].dropna()
    ax_hist.hist(val_series, bins=30)
    ax_hist.set_title(f"{focus_var} Histogram")
    ax_hist.set_xlabel(f"{focus_var} ({metric_units[focus_var]})")
    st.pyplot(fig_hist)
    plt.close(fig_hist)

    st.subheader("Boxplot: Rain vs Dry Days")
    box_var = st.selectbox(
        "Select metric to split by outcome",
        ["Humidity3pm", "Humidity9am", "Pressure3pm", "Pressure9am",
         "Sunshine", "WindGustSpeed", "Cloud3pm", "Rainfall"]
    )
    
    split_data = [
        weather_data.loc[weather_data[predict_target] == out, box_var].dropna()
        for out in ["No", "Yes"]
    ]
    
    fig_box, ax_box = plt.subplots()
    ax_box.boxplot(split_data, tick_labels=["Dry", "Wet"], showfliers=False)
    ax_box.set_ylabel(f"{box_var} ({metric_units[box_var]})")
    st.pyplot(fig_box)
    plt.close(fig_box)

    st.subheader("Relationships (Correlation Matrix)")
    corr_df = weather_data[weather_cols].corr(numeric_only=True)
    fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
    img = ax_corr.imshow(corr_df, aspect="auto", vmin=-1, vmax=1)
    ax_corr.set_xticks(range(len(corr_df.columns)))
    ax_corr.set_xticklabels(corr_df.columns, rotation=90)
    ax_corr.set_yticks(range(len(corr_df.index)))
    ax_corr.set_yticklabels(corr_df.index)
    fig_corr.colorbar(img, ax=ax_corr)
    fig_corr.tight_layout()
    st.pyplot(fig_corr)
    plt.close(fig_corr)

    st.subheader("Significance Test")
    dry_hum = weather_data.loc[weather_data[predict_target] == "No", "Humidity3pm"].dropna()
    wet_hum = weather_data.loc[weather_data[predict_target] == "Yes", "Humidity3pm"].dropna()
    
    if len(dry_hum) > 0 and len(wet_hum) > 0:
        u_val, p_val = mannwhitneyu(dry_hum, wet_hum, alternative="two-sided")
        st.write(f"Mann–Whitney U: **{u_val:,.2f}** | p-value: **{p_val:.6g}**")


with ml_tab:
    st.header("3. Machine Learning Comparison")
    st.write("Models are trained on the oldest 80% of data and evaluated on the newest 20%.")

    cut_off = int(len(weather_data) * 0.80)
    
    X_data = weather_data[weather_cols]
    y_data = weather_data[predict_target].map({"No": 0, "Yes": 1})
    
    train_x, test_x = X_data.iloc[:cut_off], X_data.iloc[cut_off:]
    train_y, test_y = y_data.iloc[:cut_off], y_data.iloc[cut_off:]

    if train_y.nunique() < 2 or test_y.nunique() < 2:
        st.error("Need both classes in train and test splits!")
    else:
        with st.spinner("Generating models..."):
            fitted_algos = train_and_evaluate(
                f"{len(weather_data)}-{os.path.getmtime(path_to_csv)}",
                train_x, train_y
            )

        metrics_list = []
        model_preds = {}
        
        for m_name, m_pipe in fitted_algos.items():
            y_hat = m_pipe.predict(test_x)
            model_preds[m_name] = y_hat
            
            row_data = {
                "Algorithm": m_name,
                "Accuracy": accuracy_score(test_y, y_hat),
                "Precision": precision_score(test_y, y_hat, zero_division=0),
                "Recall": recall_score(test_y, y_hat, zero_division=0),
                "F1 Score": f1_score(test_y, y_hat, zero_division=0)
            }
            
            try:
                probs = m_pipe.predict_proba(test_x)[:, list(m_pipe.named_steps["clf"].classes_).index(1)]
                row_data["ROC AUC"] = roc_auc_score(test_y, probs)
            except Exception:
                row_data["ROC AUC"] = np.nan
                
            metrics_list.append(row_data)

        score_table = pd.DataFrame(metrics_list).set_index("Algorithm")
        st.subheader("Performance Table")
        st.dataframe(score_table.style.format("{:.3f}"), use_container_width=True)

        st.subheader("Inspect a Model")
        pick_algo = st.selectbox(
            "Select algorithm for deep dive:",
            list(fitted_algos.keys()),
            index=list(fitted_algos.keys()).index("Forest Model")
        )
        
        active_pipe = fitted_algos[pick_algo]
        active_preds = model_preds[pick_algo]

        c_mat = confusion_matrix(test_y, active_preds, labels=[0, 1])
        fig_cm, ax_cm = plt.subplots()
        cm_img = ax_cm.imshow(c_mat)
        ax_cm.set_xticks([0, 1], labels=["Pred: No", "Pred: Yes"])
        ax_cm.set_yticks([0, 1], labels=["True: No", "True: Yes"])
        
        for (row_idx, col_idx), val in np.ndenumerate(c_mat):
            ax_cm.text(col_idx, row_idx, str(val), ha="center", va="center")
            
        fig_cm.colorbar(cm_img, ax=ax_cm)
        st.pyplot(fig_cm)
        plt.close(fig_cm)

        if "Forest Model" in fitted_algos:
            joblib.dump(
                {"model": fitted_algos["Forest Model"], "features": weather_cols},
                model_save_path
            )


with forecast_tab:
    st.header("4. Make a Prediction")
    
    with st.form("custom_predict_form"):
        user_inputs = {}
        col_a, col_b = st.columns(2)
        
        base_vals = {
            "MinTemp": 22.0, "MaxTemp": 32.0, "Rainfall": 0.0,
            "Evaporation": 5.0, "Sunshine": 6.0, "WindGustSpeed": 35.0,
            "Humidity9am": 75.0, "Humidity3pm": 55.0,
            "Pressure9am": 1012.0, "Pressure3pm": 1010.0,
            "Cloud9am": 4.0, "Cloud3pm": 5.0, "Temp9am": 25.0, "Temp3pm": 30.0
        }
        
        boundaries = {
            "MinTemp": (-50.0, 60.0), "MaxTemp": (-50.0, 60.0),
            "Rainfall": (0.0, 1000.0), "Evaporation": (0.0, 100.0),
            "Sunshine": (0.0, 24.0), "WindGustSpeed": (0.0, 300.0),
            "Humidity9am": (0.0, 100.0), "Humidity3pm": (0.0, 100.0),
            "Pressure9am": (800.0, 1100.0), "Pressure3pm": (800.0, 1100.0),
            "Cloud9am": (0.0, 8.0), "Cloud3pm": (0.0, 8.0),
            "Temp9am": (-50.0, 60.0), "Temp3pm": (-50.0, 60.0)
        }
        
        for idx, f_name in enumerate(weather_cols):
            with col_a if idx % 2 == 0 else col_b:
                min_v, max_v = boundaries[f_name]
                user_inputs[f_name] = st.number_input(
                    f"{f_name} ({metric_units[f_name]})",
                    min_value=min_v, max_value=max_v,
                    value=base_vals[f_name], step=1.0
                )
                
        run_btn = st.form_submit_button("Forecast Now")

    if run_btn:
        df_new = pd.DataFrame([[user_inputs[c] for c in weather_cols]], columns=weather_cols)
        
        try:
            if os.path.exists(model_save_path):
                disk_data = joblib.load(model_save_path)
                live_model = disk_data["model"]
            else:
                st.info("Please train algorithms in the previous tab first.")
                st.stop()

            outcome = int(live_model.predict(df_new)[0])
            c_list = list(live_model.named_steps["clf"].classes_)
            prob_array = live_model.predict_proba(df_new)[0]
            rain_chance = float(prob_array[c_list.index(1)]) * 100

            res_c1, res_c2 = st.columns(2)
            
            with res_c1:
                if outcome == 1:
                    st.error("🌧️ Outlook: Wet (Rain Expected)")
                else:
                    st.success("☀️ Outlook: Dry (No Rain Expected)")
                    
            with res_c2:
                st.metric("Probability of Rain", f"{rain_chance:.1f}%")
                st.progress(min(max(rain_chance / 100, 0), 1))
                
        except Exception as e:
            st.error(f"Error during forecasting: {e}")