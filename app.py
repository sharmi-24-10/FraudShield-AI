
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import hashlib
import io

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. CUSTOM CSS
# ==================================================

st.markdown("""
<style>

/* Main application background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #F1F5F9 0%,
        #FFFFFF 55%,
        #E0F2FE 100%
    );
}

/* Main content text */
[data-testid="stMain"] {
    color: #1E293B !important;
}

[data-testid="stMain"] .stMarkdown,
[data-testid="stMain"] .stMarkdown p,
[data-testid="stMain"] .stMarkdown li,
[data-testid="stMain"] .stMarkdown label {
    color: #334155 !important;
}

/* Main headings */
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] h5,
[data-testid="stMain"] h6 {
    color: #172554 !important;
    font-weight: 750 !important;
}

/* Custom title */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #172554 !important;
    margin-bottom: 0;
}

.subtitle {
    font-size: 16px;
    color: #475569 !important;
    margin-bottom: 25px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #101D3A 0%,
        #172554 100%
    );
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Sidebar captions */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #CBD5E1 !important;
}

/* Information cards */
.info-card {
    background: #FFFFFF;
    padding: 22px;
    border-radius: 15px;
    border-left: 5px solid #2563EB;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
    margin-bottom: 12px;
}

.info-card h3 {
    color: #334155 !important;
    font-size: 15px;
    margin: 0;
}

.info-card h1 {
    color: #2563EB !important;
    font-size: 30px;
    margin: 8px 0;
}

.info-card p {
    color: #64748B !important;
    margin: 0;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0;
    border-radius: 13px;
    padding: 15px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.06);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: #475569 !important;
}

[data-testid="stMetricValue"] {
    color: #1D4ED8 !important;
}

/* Input fields */
[data-testid="stMain"] input,
[data-testid="stMain"] textarea {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    border: 1px solid #CBD5E1 !important;
}

/* Dropdowns */
[data-testid="stMain"] [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    border-color: #CBD5E1 !important;
}

[data-testid="stMain"] [data-baseweb="select"] * {
    color: #1E293B !important;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    background: linear-gradient(
        90deg,
        #2563EB,
        #0891B2
    ) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    padding: 8px 18px;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background: linear-gradient(
        90deg,
        #1D4ED8,
        #0E7490
    ) !important;
    color: #FFFFFF !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 2px dashed #60A5FA !important;
    border-radius: 12px !important;
    padding: 15px !important;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label p,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section div,
[data-testid="stFileUploader"] section span,
[data-testid="stFileUploader"] section small,
[data-testid="stFileUploaderDropzone"] {
    color: #1E293B !important;
}

[data-testid="stFileUploader"] section {
    background: #FFFFFF !important;
}

[data-testid="stFileUploader"] button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
}

[data-testid="stFileUploader"] button span {
    color: #FFFFFF !important;
}

[data-testid="stFileUploader"] button:hover {
    background: #1D4ED8 !important;
}

/* Uploaded file name */
[data-testid="stFileUploaderFileName"] {
    color: #1E293B !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Captions */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {
    color: #64748B !important;
}

/* Links */
[data-testid="stMain"] a {
    color: #2563EB !important;
}

/* Header */
[data-testid="stHeader"] {
    background: transparent;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. PROJECT PATHS AND MODEL CONFIGURATION
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")

COMPARISON_FILE = os.path.join(
    BASE_DIR,
    "model_comparison.csv"
)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "SVM": "svm.pkl"
}

# Credit card fraud dataset input features
FEATURES = (
    ["Time"] +
    [f"V{i}" for i in range(1, 29)] +
    ["Amount"]
)


# ==================================================
# 4. LOAD MODELS ONLY WHEN REQUIRED
# ==================================================

@st.cache_resource(show_spinner="Loading trained models...")
def load_models():

    loaded_models = {}

    for name, filename in MODEL_FILES.items():

        model_path = os.path.join(
            MODEL_DIR,
            filename
        )

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"Missing model file: {model_path}"
            )

        loaded_models[name] = joblib.load(
            model_path
        )

    return loaded_models


# ==================================================
# 5. GET EXACT MODEL FEATURE ORDER
# ==================================================

def get_model_features(models):

    reference_features = None

    for name, model in models.items():

        model_features = getattr(
            model,
            "feature_names_in_",
            None
        )

        if model_features is None:

            # Fallback when the model was trained
            # using NumPy arrays instead of DataFrames
            model_features = FEATURES

        else:

            model_features = list(
                model_features
            )

        if reference_features is None:

            reference_features = model_features

        elif set(model_features) != set(reference_features):

            raise ValueError(
                f"Feature columns differ in model: {name}"
            )

    return reference_features


# ==================================================
# 6. LOAD MODEL COMPARISON RESULTS
# ==================================================

@st.cache_data(show_spinner=False)
def load_comparison():

    if not os.path.exists(COMPARISON_FILE):

        return pd.DataFrame()

    return pd.read_csv(
        COMPARISON_FILE
    )


# ==================================================
# 7. CACHE UPLOADED CSV
# ==================================================

@st.cache_data(show_spinner=False)
def read_uploaded_csv(file_content):

    return pd.read_csv(
        io.BytesIO(file_content)
    )


# ==================================================
# 8. CALCULATE MODEL METRICS IN BATCHES
# ==================================================

def calculate_metrics(model, X, y):

    batch_size = 1000

    prediction_parts = []
    score_parts = []

    has_probability = hasattr(
        model,
        "predict_proba"
    )

    has_decision = hasattr(
        model,
        "decision_function"
    )

    for start in range(0, len(X), batch_size):

        X_batch = X.iloc[
            start:start + batch_size
        ]

        predictions_batch = model.predict(
            X_batch
        )

        prediction_parts.append(
            predictions_batch
        )

        if has_probability:

            scores_batch = model.predict_proba(
                X_batch
            )[:, 1]

            score_parts.append(
                scores_batch
            )

        elif has_decision:

            scores_batch = model.decision_function(
                X_batch
            )

            score_parts.append(
                scores_batch
            )

    predictions = np.concatenate(
        prediction_parts
    )

    scores = (
        np.concatenate(score_parts)
        if score_parts
        else None
    )

    metrics = {
        "Accuracy": accuracy_score(
            y,
            predictions
        ),

        "Precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),

        "F1-Score": f1_score(
            y,
            predictions,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y,
            predictions
        )
    }

    if (
        scores is not None
        and len(np.unique(y)) == 2
    ):

        metrics["ROC-AUC"] = roc_auc_score(
            y,
            scores
        )

        metrics["PR-AUC"] = average_precision_score(
            y,
            scores
        )

    else:

        metrics["ROC-AUC"] = np.nan
        metrics["PR-AUC"] = np.nan

    return predictions, metrics


# ==================================================
# 9. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.markdown(
    "# 🛡️ FraudShield AI"
)

st.sidebar.markdown(
    "### Financial Fraud Detection"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "📊 Model Comparison",
        "🧪 Evaluate Models",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Machine Learning Assignment 2"
)

st.sidebar.caption(
    "Powered by Scikit-learn & Streamlit"
)


# ==================================================
# 10. DASHBOARD
# ==================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">'
        '🛡️ FraudShield AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Intelligent Financial Fraud Detection '
        'and Model Evaluation Platform'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "## Welcome to FraudShield AI"
    )

    st.write(
        "FraudShield AI uses machine learning algorithms "
        "to identify suspicious financial transactions "
        "and distinguish fraudulent transactions "
        "from legitimate ones."
    )

    st.markdown(
        "### Project Overview"
    )

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        ("Dataset", "Credit Card", "Fraud Detection"),
        ("Total Features", "30", "Transaction attributes"),
        ("ML Models", "5", "Classification algorithms"),
        ("Problem Type", "Binary", "Fraud / Legitimate")
    ]

    for col, (heading, value, description) in zip(
        [c1, c2, c3, c4],
        cards
    ):

        with col:

            st.markdown(
                f"""
                <div class="info-card">
                    <h3>{heading}</h3>
                    <h1>{value}</h1>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        "### Machine Learning Models"
    )

    model_cols = st.columns(5)

    model_descriptions = {
        "Logistic Regression": "Linear classification",
        "Decision Tree": "Rule-based classification",
        "KNN": "Neighbour-based classification",
        "Naive Bayes": "Probability-based classification",
        "SVM": "Margin-based classification"
    }

    for col, (name, description) in zip(
        model_cols,
        model_descriptions.items()
    ):

        with col:

            st.markdown(
                f"""
                <div class="info-card">
                    <h3>{name}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        "### Model Performance Snapshot"
    )

    comparison_df = load_comparison()

    if not comparison_df.empty:

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Results loaded from the model_comparison.csv "
            "generated by the notebook."
        )

    else:

        st.warning(
            "model_comparison.csv is not available. "
            "Run the notebook comparison step first."
        )

    st.info(
        "Use the navigation menu to compare models "
        "or evaluate an uploaded dataset."
    )


# ==================================================
# 11. MODEL COMPARISON
# ==================================================

elif page == "📊 Model Comparison":

    st.markdown(
        "## 📊 Machine Learning Model Comparison"
    )

    comparison_df = load_comparison()

    if comparison_df.empty:

        st.error(
            "model_comparison.csv is missing "
            "or empty."
        )

        st.stop()

    st.write(
        "Compare the performance of all five trained "
        "models using multiple evaluation metrics."
    )

    st.markdown(
        "### Performance Table"
    )

    st.dataframe(
        comparison_df.style.format(
            {
                col: "{:.4f}"
                for col in comparison_df.select_dtypes(
                    include="number"
                ).columns
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    numeric_columns = comparison_df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:

        metric = st.selectbox(
            "Select metric to compare",
            numeric_columns
        )

        if "Model" in comparison_df.columns:

            chart_data = comparison_df.set_index(
                "Model"
            )[[metric]]

            st.markdown(
                f"### {metric} Comparison"
            )

            st.bar_chart(
                chart_data,
                color="#2563EB"
            )

            valid_values = comparison_df[
                metric
            ].dropna()

            if not valid_values.empty:

                best_index = comparison_df[
                    metric
                ].idxmax()

                best_model = comparison_df.loc[
                    best_index,
                    "Model"
                ]

                best_value = comparison_df.loc[
                    best_index,
                    metric
                ]

                st.success(
                    f"Highest {metric}: {best_model} "
                    f"({best_value:.4f})"
                )

    st.caption(
        "For fraud detection, consider Recall, F1, MCC "
        "and PR-AUC alongside Accuracy and ROC-AUC."
    )

    st.download_button(
        "⬇️ Download Comparison CSV",
        data=comparison_df.to_csv(
            index=False
        ),
        file_name="model_comparison.csv",
        mime="text/csv"
    )


# ==================================================
# 12. EVALUATE MODELS
# ==================================================

elif page == "🧪 Evaluate Models":

    st.markdown(
        "## 🧪 Evaluate Machine Learning Models"
    )

    st.write(
        "Upload a labelled CSV dataset containing "
        "the 30 transaction features and Class column."
    )

    uploaded_file = st.file_uploader(
        "Upload test dataset (CSV)",
        type=["csv"],
        key="evaluation_upload"
    )

    if uploaded_file is not None:

        try:

            # Read uploaded file
            file_bytes = uploaded_file.getvalue()

            file_hash = hashlib.md5(
                file_bytes
            ).hexdigest()

            data = read_uploaded_csv(
                file_bytes
            )

            st.success(
                f"Dataset loaded successfully: "
                f"{len(data):,} rows and "
                f"{len(data.columns)} columns."
            )

            # Preview dataset
            with st.expander(
                "Preview uploaded dataset"
            ):

                st.dataframe(
                    data.head(5),
                    use_container_width=True,
                    hide_index=True
                )

            # Validate Class column
            if "Class" not in data.columns:

                st.error(
                    "The uploaded CSV must contain "
                    "a 'Class' column."
                )

                st.stop()

            # Validate feature columns
            missing_columns = [
                col for col in FEATURES
                if col not in data.columns
            ]

            if missing_columns:

                st.error(
                    f"Missing required columns: "
                    f"{missing_columns}"
                )

                st.stop()

            # Convert input features to numeric
            X_upload = data[FEATURES].apply(
                pd.to_numeric,
                errors="coerce"
            )

            y_upload = pd.to_numeric(
                data["Class"],
                errors="coerce"
            )

            # Check missing values
            if (
                X_upload.isnull().any().any()
                or y_upload.isnull().any()
            ):

                st.error(
                    "Missing or non-numeric values found. "
                    "Please clean your dataset."
                )

                st.stop()

            # Check infinite values
            if not np.isfinite(
                X_upload.to_numpy(dtype=float)
            ).all():

                st.error(
                    "Infinite values found in the input features."
                )

                st.stop()

            # Validate target values
            if not set(
                y_upload.unique()
            ).issubset({0, 1}):

                st.error(
                    "Class must contain only 0 "
                    "(Legitimate) and 1 (Fraudulent)."
                )

                st.stop()

            if len(y_upload.unique()) != 2:

                st.error(
                    "Both legitimate and fraudulent "
                    "transactions must be present."
                )

                st.stop()

            # Convert target to integer
            y_upload = y_upload.astype(int)

            if len(data) < 2:

                st.error(
                    "The dataset must contain at least "
                    "two rows."
                )

                st.stop()

            st.markdown("---")

            # Evaluation settings
            st.markdown(
                "### Evaluation Settings"
            )

            sample_options = sorted(
                set([
                    min(5000, len(data)),
                    min(10000, len(data)),
                    len(data)
                ])
            )

            sample_size = st.selectbox(
                "Number of transactions to evaluate",
                sample_options,
                index=0,
                format_func=lambda n: (
                    f"Full dataset ({n:,} rows)"
                    if n == len(data)
                    else f"{n:,} rows - Faster evaluation"
                )
            )

            st.caption(
                "Smaller evaluation sizes use stratified "
                "sampling to preserve the class distribution."
            )

            evaluation_key = (
                f"{file_hash}_{sample_size}"
            )

            # Evaluate button
            if st.button(
                "🚀 Evaluate All Five Models",
                type="primary",
                use_container_width=True
            ):

                try:

                    # Load models
                    models = load_models()

                    # Get exact feature order from trained models
                    model_features = get_model_features(
                        models
                    )

                    # Check that all required model features exist
                    missing_model_features = [
                        col for col in model_features
                        if col not in data.columns
                    ]

                    if missing_model_features:

                        st.error(
                            "The uploaded CSV is missing model "
                            f"features: {missing_model_features}"
                        )

                        st.stop()

                    # Stratified sampling
                    if sample_size < len(data):

                        evaluation_data, _ = train_test_split(
                            data,
                            train_size=sample_size,
                            stratify=y_upload,
                            random_state=42
                        )

                    else:

                        evaluation_data = data.copy()

                    # Arrange features in exact training order
                    X_eval = evaluation_data[
                        model_features
                    ].apply(
                        pd.to_numeric,
                        errors="coerce"
                    )

                    y_eval = pd.to_numeric(
                        evaluation_data["Class"],
                        errors="coerce"
                    ).astype(int)

                    if (
                        X_eval.isnull().any().any()
                        or not np.isfinite(
                            X_eval.to_numpy(dtype=float)
                        ).all()
                    ):

                        st.error(
                            "Invalid feature values found "
                            "in the selected evaluation data."
                        )

                        st.stop()

                    if len(y_eval.unique()) != 2:

                        st.error(
                            "The selected sample does not "
                            "contain both classes."
                        )

                        st.stop()

                    # Check progress
                    evaluation_results = []
                    all_predictions = {}

                    total_models = len(models)

                    progress = st.progress(0)

                    status = st.empty()

                    # Evaluate each model
                    with st.spinner(
                        "Evaluating models. Please wait..."
                    ):

                        for index, (name, model) in enumerate(
                            models.items()
                        ):

                            status.info(
                                f"Evaluating {name} "
                                f"({index + 1}/{total_models})..."
                            )

                            # Ensure model feature order is respected
                            model_specific_features = getattr(
                                model,
                                "feature_names_in_",
                                model_features
                            )

                            X_model = X_eval[
                                list(model_specific_features)
                            ]

                            predictions, metrics = calculate_metrics(
                                model,
                                X_model,
                                y_eval
                            )

                            row = {
                                "Model": name
                            }

                            row.update(metrics)

                            evaluation_results.append(
                                row
                            )

                            all_predictions[name] = predictions

                            progress.progress(
                                (index + 1) / total_models
                            )

                    # Save evaluation results
                    st.session_state[
                        "evaluation_key"
                    ] = evaluation_key

                    st.session_state[
                        "evaluation_df"
                    ] = pd.DataFrame(
                        evaluation_results
                    )

                    st.session_state[
                        "evaluation_predictions"
                    ] = all_predictions

                    st.session_state[
                        "evaluation_y"
                    ] = y_eval.reset_index(
                        drop=True
                    )

                    st.session_state[
                        "evaluation_data"
                    ] = evaluation_data.reset_index(
                        drop=True
                    )

                    status.success(
                        "All five models evaluated successfully!"
                    )

                except Exception as e:

                    st.error(
                        f"Model evaluation failed: {e}"
                    )

            # Display results only when they match current upload
            if (
                st.session_state.get("evaluation_key")
                == evaluation_key
            ):

                evaluation_df = st.session_state[
                    "evaluation_df"
                ]

                all_predictions = st.session_state[
                    "evaluation_predictions"
                ]

                y_eval = st.session_state[
                    "evaluation_y"
                ]

                evaluation_data = st.session_state[
                    "evaluation_data"
                ]

                st.markdown("---")

                st.markdown(
                    "### 📊 Model Evaluation Results"
                )

                st.dataframe(
                    evaluation_df.style.format(
                        {
                            col: "{:.4f}"
                            for col in evaluation_df.select_dtypes(
                                include="number"
                            ).columns
                        }
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                # Confusion matrix
                st.markdown(
                    "### Confusion Matrix"
                )

                selected_model = st.selectbox(
                    "Choose model",
                    list(MODEL_FILES.keys()),
                    key="evaluation_model"
                )

                selected_predictions = all_predictions[
                    selected_model
                ]

                import matplotlib.pyplot as plt
                import seaborn as sns

                cm = confusion_matrix(
                    y_eval,
                    selected_predictions,
                    labels=[0, 1]
                )

                fig, ax = plt.subplots(
                    figsize=(5, 3.5)
                )

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=[
                        "Legitimate",
                        "Fraud"
                    ],
                    yticklabels=[
                        "Legitimate",
                        "Fraud"
                    ],
                    ax=ax
                )

                ax.set_xlabel(
                    "Predicted Class"
                )

                ax.set_ylabel(
                    "Actual Class"
                )

                ax.set_title(
                    f"{selected_model} Confusion Matrix"
                )

                fig.tight_layout()

                st.pyplot(fig)

                plt.close(fig)

                # Classification report
                st.markdown(
                    "### Classification Report"
                )

                report = classification_report(
                    y_eval,
                    selected_predictions,
                    labels=[0, 1],
                    target_names=[
                        "Legitimate",
                        "Fraudulent"
                    ],
                    output_dict=True,
                    zero_division=0
                )

                report_df = pd.DataFrame(
                    report
                ).transpose()

                st.dataframe(
                    report_df.round(4),
                    use_container_width=True
                )

                # Download results
                st.markdown(
                    "### Download Evaluation Results"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.download_button(
                        "⬇️ Download Metrics",
                        data=evaluation_df.to_csv(
                            index=False
                        ),
                        file_name="evaluation_results.csv",
                        mime="text/csv"
                    )

                with col2:

                    prediction_data = evaluation_data.copy()

                    for name, predictions in all_predictions.items():

                        prediction_data[
                            f"{name} Prediction"
                        ] = predictions

                    st.download_button(
                        "⬇️ Download Predictions",
                        data=prediction_data.to_csv(
                            index=False
                        ),
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:

            st.error(
                f"Error processing uploaded dataset: {e}"
            )


# ==================================================
# 13. ABOUT PROJECT
# ==================================================

elif page == "ℹ️ About Project":

    st.markdown(
        "## ℹ️ About FraudShield AI"
    )

    st.markdown(
        "### Problem Statement"
    )

    st.write(
        "Financial fraud is a major challenge in digital "
        "payment systems. FraudShield AI uses supervised "
        "machine learning algorithms to classify credit "
        "card transactions as legitimate or fraudulent."
    )

    st.markdown(
        "### Dataset"
    )

    st.write(
        "Credit Card Fraud Detection dataset from the "
        "ULB Machine Learning Group, available on Kaggle."
    )

    st.markdown(
        "### Dataset Characteristics"
    )

    st.write("""
    - Total records: 284,807
    - Input features: 30
    - Target column: Class
    - Class 0: Legitimate transaction
    - Class 1: Fraudulent transaction
    - Problem type: Binary classification
    """)

    st.markdown(
        "### Algorithms Implemented"
    )

    st.write("""
    1. Logistic Regression
    2. Decision Tree
    3. K-Nearest Neighbours
    4. Gaussian Naive Bayes
    5. Support Vector Machine (LinearSVC)
    """)

    st.markdown(
        "### Evaluation Metrics"
    )

    st.write("""
    - Accuracy
    - ROC-AUC
    - Precision
    - Recall
    - F1-score
    - Matthews Correlation Coefficient (MCC)
    - Precision-Recall AUC (PR-AUC)
    """)

    st.markdown(
        "### Project Technology"
    )

    st.write(
        "Python | Pandas | NumPy | Scikit-learn | "
        "Matplotlib | Seaborn | Joblib | Streamlit"
    )

    st.markdown(
        "### Dataset Source"
    )

    st.markdown(
        "[Kaggle - Credit Card Fraud Detection]"
        "(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)"
    )

    st.success(
        "FraudShield AI - Machine Learning Assignment 2"
    )
