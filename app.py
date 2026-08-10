"""
AI-Based Smart E-Commerce Profit Prediction and Business Intelligence System.
Streamlit Multi-Page Enterprise AI Web Dashboard with MLR, XGBoost, and TensorFlow DFFNN.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys

# Import custom src modules
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent

# Directories
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Create directories if they don't exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
from src.utils import logger, safe_divide
from src.data_loader import DataLoader
from src.data_merger import DataMerger
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.eda import EDAAnalyzer
from src.train_mlr import MLRTrainer
from src.train_xgb import XGBoostTrainer
from src.train_dffnn import DFFNNTrainer
from src.evaluate import ModelEvaluator
from src.predict import ProfitPredictor
from src.business_insights import BusinessInsightsEngine
from src.explain_model import SHAPExplainer

# -----------------------------------------------------------------------------
# Streamlit Page Config & Professional CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI E-Commerce Profit Prediction System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #10B981 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "df_raw_list" not in st.session_state:
    st.session_state["df_raw_list"] = []
if "df_merged" not in st.session_state:
    st.session_state["df_merged"] = None
if "df_cleaned" not in st.session_state:
    st.session_state["df_cleaned"] = None
if "df_engineered" not in st.session_state:
    st.session_state["df_engineered"] = None
if "preprocessor" not in st.session_state:
    st.session_state["preprocessor"] = DataPreprocessor()
if "mlr_model" not in st.session_state:
    st.session_state["mlr_model"] = None
if "xgb_model" not in st.session_state:
    st.session_state["xgb_model"] = None
if "dffnn_model" not in st.session_state:
    st.session_state["dffnn_model"] = None
if "model_metrics" not in st.session_state:
    st.session_state["model_metrics"] = []
if "best_model_name" not in st.session_state:
    st.session_state["best_model_name"] = None

# Initialize Core Classes
loader = DataLoader()
merger = DataMerger()
engineer = FeatureEngineer()
analyzer = EDAAnalyzer()
evaluator = ModelEvaluator()
insights_engine = BusinessInsightsEngine()

# -----------------------------------------------------------------------------
# Sidebar Navigation (15 Pages)
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/isometric-line/100/shopping-bag.png", width=70)
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page:",
    [
        "🏠 Home",
        "📂 Upload Dataset",
        "📋 Data Preview",
        "🧹 Data Cleaning",
        "⚙️ Feature Engineering",
        "📊 Exploratory Data Analysis",
        "🤖 Train MLR",
        "🌲 Train XGBoost",
        "🧠 Train DFFNN",
        "📈 Model Evaluation",
        "⚖️ Model Comparison",
        "💰 Profit Prediction",
        "🔬 Explainable AI",
        "💡 Business Insights",
        "📄 Download Reports"
    ]
)

st.sidebar.divider()
st.sidebar.caption("🤖 **Models**: MLR, XGBoost & TensorFlow DFFNN")
st.sidebar.caption("💡 **Features**: Auto-Merge, SHAP XAI, Business Intelligence")

def check_data_ready():
    if st.session_state["df_merged"] is None and not st.session_state["df_raw_list"]:
        st.warning("⚠️ No dataset loaded yet. Please go to '📂 Upload Dataset' or load the sample dataset to proceed.")
        st.stop()

def get_active_dataframe():
    if st.session_state["df_engineered"] is not None:
        return st.session_state["df_engineered"]
    if st.session_state["df_cleaned"] is not None:
        return st.session_state["df_cleaned"]
    if st.session_state["df_merged"] is not None:
        return st.session_state["df_merged"]
    if st.session_state["df_raw_list"]:
        return st.session_state["df_raw_list"][0]
    return None

# -----------------------------------------------------------------------------
# Page 1: Home
# -----------------------------------------------------------------------------
if page == "🏠 Home":
    st.markdown('<div class="main-title">AI-Based Smart E-Commerce Profit Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multiple Linear Regression (MLR) vs. XGBoost vs. TensorFlow Deep Feedforward Neural Network (DFFNN)</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predictive Models", "3 Models", "MLR, XGBoost, DFFNN")
    c2.metric("Feature Engine", "8+ Business KPIs", "Margin, ROI, Conversion")
    c3.metric("Multi-CSV Merger", "Auto Key Join", "Outer / Inner Joins")
    c4.metric("Explainable AI", "SHAP Engine", "Global & Local XAI")

    st.divider()
    st.subheader("📌 Project Capabilities & Key Features")
    st.markdown("""
    - **📂 Multi-CSV Smart Merger**: Auto-detects key join columns (`product_id`, `order_id`, etc.) across raw Kaggle CSV files.
    - **🧹 Production Preprocessing**: Automated median/mode imputation, IQR outlier clipping, One-Hot Encoding, and `StandardScaler` saved via `joblib`.
    - **⚙️ Target Leakage Prevention**: Computes Profit, Profit Margin, Advertising ROI, Delivery Efficiency, Conversion Rate, and Customer Engagement without target leakage.
    - **🤖 Triple-Model Suite**: Fits **Multiple Linear Regression (MLR)**, **XGBoost Regressor**, and a **TensorFlow 5-layer Deep Neural Network** (`Dense(128)->Dropout(0.3)->Dense(64)->Dropout(0.2)->Dense(32)->Dense(16)->Dense(1)`).
    - **🔬 SHAP Explainability**: Explains predictions using global beeswarm plots, feature importance bar charts, and individual prediction text attributions.
    - **💡 Business Intelligence Engine**: Translates numbers into actionable commercial strategies for category expansion, discounting, and ad spend efficiency.
    """)

# -----------------------------------------------------------------------------
# Page 2: Upload Dataset
# -----------------------------------------------------------------------------
elif page == "📂 Upload Dataset":
    st.header("📂 Upload Dataset & Multi-CSV Loader")

    uploaded_files = st.file_uploader(
        "Upload Raw Dataset File(s) (CSV, XLSX):",
        type=["csv", "xlsx"],
        accept_multiple_files=True
    )

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        if uploaded_files:
            dfs = []
            for f in uploaded_files:
                df, status = loader.load_dataset(f, file_name=f.name)
                if status["success"]:
                    dfs.append(df)
            st.session_state["df_raw_list"] = dfs
            st.success(f"Successfully loaded {len(dfs)} dataset file(s).")

    with col_u2:
        if st.button("🚀 Load Sample Kaggle-Style Dataset (1,000 Products)", use_container_width=True):
            sample_df = loader.generate_sample_ecommerce_dataset(1000)
            st.session_state["df_raw_list"] = [sample_df]
            st.session_state["df_merged"] = sample_df
            st.session_state["df_cleaned"] = None
            st.session_state["df_engineered"] = None
            st.success("Sample 1,000-product dataset loaded successfully!")

    if st.session_state["df_raw_list"]:
        st.divider()
        st.subheader(f"Uploaded Files Count: {len(st.session_state['df_raw_list'])}")
        for idx, d in enumerate(st.session_state["df_raw_list"]):
            st.write(f"**File #{idx+1} Shape:** {d.shape[0]} rows, {d.shape[1]} columns")
            st.dataframe(d.head(3), use_container_width=True)

# -----------------------------------------------------------------------------
# Page 3: Data Preview
# -----------------------------------------------------------------------------
elif page == "📋 Data Preview":
    check_data_ready()
    st.header("📋 Data Quality & Inspection Preview")

    df_curr = get_active_dataframe()
    report = loader.generate_quality_report(df_curr)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Rows", f"{report['total_rows']:,}")
    m2.metric("Total Columns", report['total_columns'])
    m3.metric("Numeric Cols", len(report['numeric_columns']))
    m4.metric("Categorical Cols", len(report['categorical_columns']))
    m5.metric("Duplicates", report['duplicate_rows'])

    st.subheader("DataFrame Head Preview")
    st.dataframe(df_curr.head(10), use_container_width=True)

    with st.expander("🔍 View Detailed Column Data Types & Missing Values"):
        col_df = pd.DataFrame({
            "Column": df_curr.columns,
            "Type": [str(t) for t in df_curr.dtypes],
            "Missing Count": df_curr.isnull().sum().values,
            "Missing (%)": (df_curr.isnull().sum().values / len(df_curr) * 100).round(2)
        })
        st.dataframe(col_df, use_container_width=True)

# -----------------------------------------------------------------------------
# Page 4: Data Cleaning & Merging
# -----------------------------------------------------------------------------
elif page == "🧹 Data Cleaning":
    check_data_ready()
    st.header("🧹 Multi-CSV Merging & Production Data Cleaning")

    raw_list = st.session_state["df_raw_list"]

    if len(raw_list) > 1:
        st.subheader("1. Multi-CSV Dataset Merger")
        join_type = st.selectbox("Select Join Method:", ["outer", "inner", "left"], index=0)
        if st.button("🔀 Auto-Merge Datasets on Key Columns", type="primary"):
            merged_df, msg = merger.merge_datasets(raw_list, how=join_type)
            st.session_state["df_merged"] = merged_df
            st.success(f"Datasets merged successfully! Merged shape: {merged_df.shape}")
            st.info(msg)
    else:
        if st.session_state["df_merged"] is None and raw_list:
            st.session_state["df_merged"] = raw_list[0].copy()

    df_to_clean = st.session_state["df_merged"] if st.session_state["df_merged"] is not None else raw_list[0]

    st.divider()
    st.subheader("2. Run Automated Data Cleaning")
    if st.button("🧹 Clean Currency Fields, Remove Duplicates & Outliers", type="primary"):
        preprocessor = st.session_state["preprocessor"]
        df_mapped = preprocessor.map_columns(df_to_clean)
        df_clean = preprocessor.clean_data(df_mapped)
        df_clean = preprocessor.handle_outliers(df_clean)
        st.session_state["df_cleaned"] = df_clean
        st.success(f"Data cleaning completed successfully! Cleaned shape: {df_clean.shape}")

    if st.session_state["df_cleaned"] is not None:
        st.write("##### Cleaned DataFrame Preview")
        st.dataframe(st.session_state["df_cleaned"].head(5), use_container_width=True)

# -----------------------------------------------------------------------------
# Page 5: Feature Engineering
# -----------------------------------------------------------------------------
elif page == "⚙️ Feature Engineering":
    check_data_ready()
    st.header("⚙️ Business Feature Engineering Engine")
    st.write("Generates Profit Margin, Revenue, Advertising ROI, Delivery Efficiency, Customer Engagement, Discount Effectiveness, Inventory Turnover, and Conversion Rate.")

    df_base = st.session_state["df_cleaned"] if st.session_state["df_cleaned"] is not None else get_active_dataframe()

    if st.button("⚡ Compute All Business Features & Target Variables", type="primary"):
        eng = FeatureEngineer()
        df_engineered = eng.create_features(df_base)
        st.session_state["df_engineered"] = df_engineered
        st.success(f"Feature engineering complete! Total features & targets: {df_engineered.shape[1]}")

    if st.session_state["df_engineered"] is not None:
        df_feat = st.session_state["df_engineered"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Revenue", f"₹{df_feat['revenue'].mean():,.2f}")
        c2.metric("Avg Profit", f"₹{df_feat['profit'].mean():,.2f}")
        c3.metric("Avg Profit Margin", f"{(df_feat['profit_margin'].mean()*100):.1f}%")
        c4.metric("Avg Advertising ROI", f"{df_feat['advertising_roi'].mean():.2f}x")

        st.dataframe(df_feat.head(10), use_container_width=True)

# -----------------------------------------------------------------------------
# Page 6: Exploratory Data Analysis
# -----------------------------------------------------------------------------
elif page == "📊 Exploratory Data Analysis":
    check_data_ready()
    st.header("📊 Exploratory Data Analysis (EDA)")

    df_curr = get_active_dataframe()

    tab_e1, tab_e2, tab_e3 = st.tabs(["💰 Target Profit Dynamics", "🔥 Correlation Matrix", "🏷️ Category Analysis"])

    with tab_e1:
        st.pyplot(analyzer.plot_and_save_profit_distribution(df_curr))

    with tab_e2:
        st.pyplot(analyzer.plot_and_save_correlation_matrix(df_curr))

    with tab_e3:
        st.plotly_chart(analyzer.plot_category_profitability(df_curr), use_container_width=True)

# -----------------------------------------------------------------------------
# Page 7: Train MLR
# -----------------------------------------------------------------------------
elif page == "🤖 Train MLR":
    check_data_ready()
    st.header("🤖 Train Multiple Linear Regression (MLR) Model")

    if st.session_state["df_engineered"] is None:
        eng = FeatureEngineer()
        st.session_state["df_engineered"] = eng.create_features(get_active_dataframe())

    df_feat = st.session_state["df_engineered"]
    eng = FeatureEngineer()
    X, y_profit, _ = eng.get_features_and_targets(df_feat)
    preprocessor = st.session_state["preprocessor"]

    alpha = st.slider("Select Ridge Penalty (Alpha):", 0.01, 50.0, 1.0)

    if st.button("🚀 Train Multiple Linear Regression Model", type="primary"):
        X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(X, y_profit)
        mlr = MLRTrainer(alpha=alpha)
        mlr.fit(X_train, y_train, feature_names=preprocessor.feature_columns)
        eval_metrics = mlr.evaluate(X_test, y_test)
        eval_metrics["Model"] = "Multiple Linear Regression (MLR)"

        st.session_state["mlr_model"] = mlr
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["mlr_metrics"] = eval_metrics
        st.success("MLR Model trained and saved as 'models/mlr_model.pkl'!")

    if st.session_state.get("mlr_metrics"):
        st.divider()
        st.subheader("📊 Individual MLR Evaluation Results")
        m = st.session_state["mlr_metrics"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("R² Score", f"{m.get('R2', 0.0):.4f}")
        m2.metric("Adjusted R²", f"{m.get('Adjusted_R2', 0.0):.4f}")
        m3.metric("RMSE Error", f"₹{m.get('RMSE', 0.0):,.2f}")
        m4.metric("MAE Error", f"₹{m.get('MAE', 0.0):,.2f}")
        m5.metric("MSE Error", f"{m.get('MSE', 0.0):,.2f}")

        mlr = st.session_state["mlr_model"]
        y_test = st.session_state["y_test"]
        X_test = st.session_state["X_test"]
        y_preds = mlr.predict(X_test)

        tab_m1, tab_m2 = st.tabs(["🎯 Actual vs. Predicted (MLR)", "📉 Residual Analysis (MLR)"])
        with tab_m1:
            st.plotly_chart(evaluator.plot_actual_vs_predicted({"MLR": (y_test, y_preds)}), use_container_width=True)
        with tab_m2:
            st.plotly_chart(evaluator.plot_residuals({"MLR": (y_test, y_preds)}), use_container_width=True)

# -----------------------------------------------------------------------------
# Page 8: Train XGBoost
# -----------------------------------------------------------------------------
elif page == "🌲 Train XGBoost":
    check_data_ready()
    st.header("🌲 Train XGBoost Regressor Model")

    if st.session_state["df_engineered"] is None:
        eng = FeatureEngineer()
        st.session_state["df_engineered"] = eng.create_features(get_active_dataframe())

    df_feat = st.session_state["df_engineered"]
    eng = FeatureEngineer()
    X, y_profit, _ = eng.get_features_and_targets(df_feat)
    preprocessor = st.session_state["preprocessor"]

    c_x1, c_x2 = st.columns(2)
    with c_x1:
        n_est = st.slider("n_estimators", 10, 500, 150)
        max_d = st.slider("max_depth", 2, 12, 6)
    with c_x2:
        lr_rate = st.slider("Learning Rate", 0.01, 0.30, 0.05)

    if st.button("🌲 Train XGBoost Regressor Model", type="primary"):
        X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(X, y_profit)
        xgb_mod = XGBoostTrainer({"n_estimators": n_est, "max_depth": max_d, "learning_rate": lr_rate})
        xgb_mod.fit(X_train, y_train, eval_set=(X_test, y_test), feature_names=preprocessor.feature_columns)
        eval_metrics = xgb_mod.evaluate(X_test, y_test)
        eval_metrics["Model"] = "XGBoost Regressor"

        st.session_state["xgb_model"] = xgb_mod
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["xgb_metrics"] = eval_metrics
        st.success("XGBoost Model trained and saved as 'models/xgb_model.pkl'!")

    if st.session_state.get("xgb_metrics"):
        st.divider()
        st.subheader("📊 Individual XGBoost Evaluation Results")
        m = st.session_state["xgb_metrics"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("R² Score", f"{m.get('R2', 0.0):.4f}")
        m2.metric("Adjusted R²", f"{m.get('Adjusted_R2', 0.0):.4f}")
        m3.metric("RMSE Error", f"₹{m.get('RMSE', 0.0):,.2f}")
        m4.metric("MAE Error", f"₹{m.get('MAE', 0.0):,.2f}")
        m5.metric("MSE Error", f"{m.get('MSE', 0.0):,.2f}")

        xgb_mod = st.session_state["xgb_model"]
        y_test = st.session_state["y_test"]
        X_test = st.session_state["X_test"]
        y_preds = xgb_mod.predict(X_test)

        tab_x1, tab_x2, tab_x3 = st.tabs(["🔥 Feature Importances", "🎯 Actual vs. Predicted (XGBoost)", "📉 Residual Analysis (XGBoost)"])
        with tab_x1:
            st.dataframe(xgb_mod.get_feature_importances(), use_container_width=True)
        with tab_x2:
            st.plotly_chart(evaluator.plot_actual_vs_predicted({"XGBoost": (y_test, y_preds)}), use_container_width=True)
        with tab_x3:
            st.plotly_chart(evaluator.plot_residuals({"XGBoost": (y_test, y_preds)}), use_container_width=True)

# -----------------------------------------------------------------------------
# Page 9: Train DFFNN
# -----------------------------------------------------------------------------
elif page == "🧠 Train DFFNN":
    check_data_ready()
    st.header("🧠 Train Deep Learning Engine (Tabular ResNet Architecture)")
    st.markdown("""
        **Architecture Highlights**:
        - **Projection Layer**: `Dense(128) + Batch Normalization + Swish`
        - **Residual Block 1**: `[Dense(128) -> BatchNorm -> Swish -> Dropout(0.1) -> Dense(128) -> BatchNorm] + Skip Connection`
        - **Intermediate Projection**: `Dense(64) + Batch Normalization + Swish`
        - **Residual Block 2**: `[Dense(64) -> BatchNorm -> Swish -> Dropout(0.1) -> Dense(64) -> BatchNorm] + Skip Connection`
        - **Output Head**: `Dense(32, Swish) -> Dense(16, Swish) -> Output(1, Linear)`
    """)

    if st.session_state["df_engineered"] is None:
        eng = FeatureEngineer()
        st.session_state["df_engineered"] = eng.create_features(get_active_dataframe())

    df_feat = st.session_state["df_engineered"]
    eng = FeatureEngineer()
    X, y_profit, _ = eng.get_features_and_targets(df_feat)
    preprocessor = st.session_state["preprocessor"]

    c_d1, c_d2 = st.columns(2)
    with c_d1:
        epochs = st.slider("Training Epochs", 5, 200, 40)
    with c_d2:
        batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)

    if st.button("🧠 Train Tabular ResNet Deep Learning Model", type="primary"):
        X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split(X, y_profit)
        
        with st.spinner("Training TensorFlow / Keras Tabular ResNet Model..."):
            dffnn = DFFNNTrainer({"epochs": epochs, "batch_size": batch_size})
            dffnn.fit(X_train, y_train, eval_set=(X_test, y_test), feature_names=preprocessor.feature_columns)
            eval_metrics = dffnn.evaluate(X_test, y_test)
            eval_metrics["Model"] = "Tabular Residual Neural Network (Tabular ResNet)"

            st.session_state["dffnn_model"] = dffnn
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.session_state["dffnn_metrics"] = eval_metrics
        st.success("Tabular ResNet Model trained and saved as 'models/dffnn_model.keras'!")

    if st.session_state.get("dffnn_metrics"):
        st.divider()
        st.subheader("📊 Individual DFFNN Evaluation Results")
        m = st.session_state["dffnn_metrics"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("R² Score", f"{m.get('R2', 0.0):.4f}")
        m2.metric("Adjusted R²", f"{m.get('Adjusted_R2', 0.0):.4f}")
        m3.metric("RMSE Error", f"₹{m.get('RMSE', 0.0):,.2f}")
        m4.metric("MAE Error", f"₹{m.get('MAE', 0.0):,.2f}")
        m5.metric("MSE Error", f"{m.get('MSE', 0.0):,.2f}")

        dffnn = st.session_state["dffnn_model"]
        y_test = st.session_state["y_test"]
        X_test = st.session_state["X_test"]
        y_preds = dffnn.predict(X_test)

        tab_d1, tab_d2, tab_d3 = st.tabs(["⚡ Loss Curve", "🎯 Actual vs. Predicted (DFFNN)", "📉 Residual Analysis (DFFNN)"])
        with tab_d1:
            st.plotly_chart(evaluator.plot_dffnn_loss_curve(dffnn.history), use_container_width=True)
        with tab_d2:
            st.plotly_chart(evaluator.plot_actual_vs_predicted({"DFFNN": (y_test, y_preds)}), use_container_width=True)
        with tab_d3:
            st.plotly_chart(evaluator.plot_residuals({"DFFNN": (y_test, y_preds)}), use_container_width=True)

# -----------------------------------------------------------------------------
# Page 10: Model Evaluation
# -----------------------------------------------------------------------------
elif page == "📈 Model Evaluation":
    st.header("📈 Model Diagnostic Evaluation & Plots")

    if not any(k in st.session_state for k in ["mlr_metrics", "xgb_metrics", "dffnn_metrics"]):
        st.warning("⚠️ Train MLR, XGBoost, or DFFNN models first.")
        st.stop()

    results_dict = {}
    if st.session_state.get("mlr_model"):
        mlr = st.session_state["mlr_model"]
        results_dict["MLR"] = (st.session_state["y_test"], mlr.predict(st.session_state["X_test"]))
    if st.session_state.get("xgb_model"):
        xgb_mod = st.session_state["xgb_model"]
        results_dict["XGBoost"] = (st.session_state["y_test"], xgb_mod.predict(st.session_state["X_test"]))
    if st.session_state.get("dffnn_model"):
        dffnn = st.session_state["dffnn_model"]
        results_dict["DFFNN"] = (st.session_state["y_test"], dffnn.predict(st.session_state["X_test"]))

    tab_v1, tab_v2, tab_v3 = st.tabs(["🎯 Prediction vs. Actual", "📉 Residual Plot", "⚡ DFFNN Loss Curve"])

    with tab_v1:
        st.plotly_chart(evaluator.plot_actual_vs_predicted(results_dict), use_container_width=True)

    with tab_v2:
        st.plotly_chart(evaluator.plot_residuals(results_dict), use_container_width=True)

    with tab_v3:
        if st.session_state.get("dffnn_model"):
            dffnn = st.session_state["dffnn_model"]
            st.plotly_chart(evaluator.plot_dffnn_loss_curve(dffnn.history), use_container_width=True)

# -----------------------------------------------------------------------------
# Page 11: Model Comparison
# -----------------------------------------------------------------------------
elif page == "⚖️ Model Comparison":
    st.header("⚖️ Model Benchmark Performance & Individual Metric Breakdown")

    metrics_list = []
    if st.session_state.get("mlr_metrics"):
        metrics_list.append(st.session_state["mlr_metrics"])
    if st.session_state.get("xgb_metrics"):
        metrics_list.append(st.session_state["xgb_metrics"])
    if st.session_state.get("dffnn_metrics"):
        metrics_list.append(st.session_state["dffnn_metrics"])

    if metrics_list:
        comp_df = evaluator.build_comparison_table(metrics_list)
        best_name, _ = evaluator.select_best_model(comp_df)
        st.session_state["best_model_name"] = best_name

        st.subheader(f"🏆 Auto-Selected Best Model: `{best_name}`")
        st.dataframe(comp_df, use_container_width=True)

        # --- Individual Model Metrics Section ---
        st.divider()
        st.subheader("📊 Individual Model Performance Metrics")

        for m_dict in metrics_list:
            m_name = m_dict.get("Model", "Model")
            with st.expander(f"📌 Individual Metrics: {m_name}", expanded=True):
                im1, im2, im3, im4, im5 = st.columns(5)
                im1.metric("R² Score", f"{m_dict.get('R2', 0.0):.4f}")
                im2.metric("Adjusted R²", f"{m_dict.get('Adjusted_R2', 0.0):.4f}")
                im3.metric("RMSE Error", f"₹{m_dict.get('RMSE', 0.0):,.2f}")
                im4.metric("MAE Error", f"₹{m_dict.get('MAE', 0.0):,.2f}")
                im5.metric("MSE Error", f"{m_dict.get('MSE', 0.0):,.2f}")
    else:
        st.warning("⚠️ No trained models available for comparison. Train MLR, XGBoost, or DFFNN first.")

# -----------------------------------------------------------------------------
# Page 12: Profit Prediction
# -----------------------------------------------------------------------------
elif page == "💰 Profit Prediction":
    st.header("💰 Product Net Profit & Revenue Prediction")

    model_options = {}
    if st.session_state.get("mlr_model"):
        model_options["Multiple Linear Regression (MLR)"] = st.session_state["mlr_model"]
    if st.session_state.get("xgb_model"):
        model_options["XGBoost Regressor"] = st.session_state["xgb_model"]
    if st.session_state.get("dffnn_model"):
        model_options["Deep Feedforward Neural Network (DFFNN)"] = st.session_state["dffnn_model"]

    if not model_options:
        st.warning("Train MLR, XGBoost, or DFFNN models first.")
        st.stop()

    selected_model_name = st.selectbox("Select Model:", list(model_options.keys()))
    chosen_model = model_options[selected_model_name]
    preprocessor = st.session_state["preprocessor"]

    predictor = ProfitPredictor(preprocessor, chosen_model)

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        sp = st.number_input("Selling Price (₹)", min_value=10.0, value=1200.0)
        cp = st.number_input("Cost Price (₹)", min_value=5.0, value=700.0)
        units = st.number_input("Units Sold", min_value=1, value=150)
    with col_in2:
        ad = st.number_input("Ad Spend (₹)", min_value=0.0, value=4000.0)
        ship = st.number_input("Shipping Cost (₹)", min_value=0.0, value=80.0)
        ret = st.slider("Return Rate (%)", 0.0, 25.0, 5.0) / 100.0
    with col_in3:
        cat = st.selectbox("Category", ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports"])
        brand = st.text_input("Brand", "TechPulse")
        rating = st.slider("Rating", 1.0, 5.0, 4.3)

    single_input = {
        "selling_price": sp,
        "cost_price": cp,
        "units_sold": units,
        "ad_cost": ad,
        "shipping_cost": ship,
        "return_rate": ret,
        "category": cat,
        "brand": brand,
        "rating": rating,
        "website_visits": units * 25
    }

    if st.button("🔮 Predict Net Profit", type="primary", use_container_width=True):
        res = predictor.predict_single(single_input)
        st.session_state["latest_single_input"] = single_input
        st.session_state["latest_prediction"] = res

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Revenue", f"₹{res['predicted_revenue']:,.2f}")
        m2.metric("Predicted Net Profit", f"₹{res['predicted_profit']:,.2f}")
        m3.metric("Profit Margin", f"{res['profit_margin_pct']}%")
        m4.metric("Advertising ROI", f"{res['roi_pct']}%")

        # --- Multi-Model Comparison & Best Model Selection ---
        st.divider()
        st.subheader("🏆 Multi-Model Comparison & Best Model Recommendation")

        # Collect predictions and metrics across MLR, XGBoost, and DFFNN
        X_single = predictor.preprocessor.transform_single(single_input)
        all_model_preds = []

        if st.session_state.get("mlr_model"):
            m_mlr = st.session_state["mlr_model"]
            p_val = float(m_mlr.predict(X_single)[0])
            met = st.session_state.get("mlr_metrics", {})
            all_model_preds.append({
                "Model": "Multiple Linear Regression (MLR)",
                "Predicted Profit (₹)": f"₹{p_val:,.2f}",
                "raw_profit": p_val,
                "Test R² Score": met.get("R2", 0.0),
                "RMSE (₹)": met.get("RMSE", 0.0),
                "MAE (₹)": met.get("MAE", 0.0)
            })

        if st.session_state.get("xgb_model"):
            m_xgb = st.session_state["xgb_model"]
            p_val = float(m_xgb.predict(X_single)[0])
            met = st.session_state.get("xgb_metrics", {})
            all_model_preds.append({
                "Model": "XGBoost Regressor",
                "Predicted Profit (₹)": f"₹{p_val:,.2f}",
                "raw_profit": p_val,
                "Test R² Score": met.get("R2", 0.0),
                "RMSE (₹)": met.get("RMSE", 0.0),
                "MAE (₹)": met.get("MAE", 0.0)
            })

        if st.session_state.get("dffnn_model"):
            m_dffnn = st.session_state["dffnn_model"]
            p_val = float(m_dffnn.predict(X_single)[0])
            met = st.session_state.get("dffnn_metrics", {})
            all_model_preds.append({
                "Model": "Deep Feedforward Neural Network (DFFNN)",
                "Predicted Profit (₹)": f"₹{p_val:,.2f}",
                "raw_profit": p_val,
                "Test R² Score": met.get("R2", 0.0),
                "RMSE (₹)": met.get("RMSE", 0.0),
                "MAE (₹)": met.get("MAE", 0.0)
            })

        if all_model_preds:
            df_pred_comp = pd.DataFrame(all_model_preds)
            df_pred_comp = df_pred_comp.sort_values(by="Test R² Score", ascending=False).reset_index(drop=True)
            
            best_model_info = df_pred_comp.iloc[0]
            best_model_name = best_model_info["Model"]
            best_r2 = best_model_info["Test R² Score"]
            best_rmse = best_model_info["RMSE (₹)"]
            best_pred = best_model_info["Predicted Profit (₹)"]

            # Highlight Best Model
            df_pred_comp["Status"] = ["🏆 Best Model" if i == 0 else "Competitive" for i in range(len(df_pred_comp))]
            
            st.success(
                f"**🏆 Recommended Best Model: `{best_model_name}`**\n\n"
                f"- **Highest Test R² Score**: `{best_r2:.4f}`\n"
                f"- **Lowest RMSE Error**: `₹{best_rmse:,.2f}`\n"
                f"- **Recommended Profit Prediction**: `{best_pred}`"
            )

            display_cols = ["Model", "Predicted Profit (₹)", "Test R² Score", "RMSE (₹)", "Status"]
            st.dataframe(df_pred_comp[display_cols], use_container_width=True)

            st.info(
                f"💡 **Business Intelligence Note**: `{best_model_name}` is selected as the superior model because it achieves the "
                f"highest empirical test $R^2$ score (`{best_r2:.4f}`) and lowest prediction variance (`₹{best_rmse:,.2f}`). "
                f"Use this prediction for commercial pricing decisions."
            )

            # --- Quantile Risk & Profit Bounds (P10 - P50 - P90) ---
            try:
                from src.train_quantile import QuantileRegressorTrainer
                q_tr = QuantileRegressorTrainer()
                q_tr.fit(st.session_state["X_test"][:500], st.session_state["y_test"][:500])
                q_bounds = q_tr.predict_bounds(X_single)

                p10_v = float(q_bounds["p10"][0])
                p50_v = float(q_bounds["p50"][0])
                p90_v = float(q_bounds["p90"][0])

                st.divider()
                st.subheader("🛡️ Quantile Risk & Profit Confidence Bounds (P10 - P50 - P90)")
                q1, q2, q3 = st.columns(3)
                q1.metric("⚠️ Bearish / Worst-Case Profit (P10)", f"₹{p10_v:,.2f}")
                q2.metric("🎯 Expected Baseline Profit (P50)", f"₹{p50_v:,.2f}")
                q3.metric("🚀 Bullish / Best-Case Profit (P90)", f"₹{p90_v:,.2f}")
            except Exception as e:
                pass

# -----------------------------------------------------------------------------
# Page 13: Explainable AI
# -----------------------------------------------------------------------------
elif page == "🔬 Explainable AI":
    st.header("🔬 Explainable AI (SHAP Explanations)")

    if "X_test" not in st.session_state:
        st.warning("Train a model first.")
        st.stop()

    model_options = {}
    if st.session_state.get("mlr_model"):
        model_options["MLR"] = st.session_state["mlr_model"]
    if st.session_state.get("xgb_model"):
        model_options["XGBoost"] = st.session_state["xgb_model"]
    if st.session_state.get("dffnn_model"):
        model_options["DFFNN"] = st.session_state["dffnn_model"]

    sel_m = st.selectbox("Select Model for SHAP:", list(model_options.keys()))
    chosen_m = model_options[sel_m]
    preprocessor = st.session_state["preprocessor"]
    X_test = st.session_state["X_test"]

    explainer = SHAPExplainer(chosen_m, preprocessor.feature_columns)

    with st.spinner("Computing SHAP attributions..."):
        shap_mat, shap_vals = explainer.compute_shap_values(X_test[:20])

    if shap_vals is not None:
        tab_x1, tab_x2, tab_x3 = st.tabs(["🔥 Global Beeswarm", "📊 Feature Importance", "💬 Local Explanation"])
        with tab_x1:
            st.pyplot(explainer.generate_summary_plot(shap_vals, X_test[:20]))
        with tab_x2:
            st.pyplot(explainer.generate_bar_plot(shap_vals))
        with tab_x3:
            if "latest_single_input" in st.session_state:
                local_exps = explainer.generate_local_explanation(st.session_state["latest_single_input"], shap_mat[0])
                for item in local_exps:
                    st.write(f"- {item['text']}")

# -----------------------------------------------------------------------------
# Page 14: Business Insights
# -----------------------------------------------------------------------------
elif page == "💡 Business Insights":
    check_data_ready()
    st.header("💡 Automated Commercial Business Insights")

    df_curr = get_active_dataframe()
    insights = insights_engine.generate_full_insights(df_curr)

    st.subheader(f"🏆 Top Performing Category: {insights['top_category']}")

    st.markdown("### 📌 Executive Action Recommendations")
    for rec in insights["recommendations"]:
        st.markdown(f"- {rec}")

# -----------------------------------------------------------------------------
# Page 15: Download Reports
# -----------------------------------------------------------------------------
elif page == "📄 Download Reports":
    st.header("📄 Download Reports & Processed Data")

    df_curr = get_active_dataframe()

    if df_curr is not None:
        csv_bytes = df_curr.to_csv(index=False).encode('utf-8')
        st.download_button(
            "💾 Download Processed Dataset (CSV)",
            data=csv_bytes,
            file_name="processed_ecommerce_dataset.csv",
            mime="text/csv"
        )
