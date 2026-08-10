# AI-Based Smart E-Commerce Profit Prediction and Business Intelligence System

An enterprise-grade AI application designed to predict e-commerce product net profitability prior to sales using **Multiple Linear Regression (MLR)**, **XGBoost Regressor**, and **TensorFlow / Keras Deep Feedforward Neural Networks (DFFNN)**, supported by **SHAP Explainable AI** and an interactive 15-page **Streamlit** dashboard.

---

## 📁 Directory Structure

```
hv/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── scaler.pkl
│   ├── encoder.pkl
│   ├── num_imputer.pkl
│   ├── cat_imputer.pkl
│   ├── feature_columns.pkl
│   ├── mlr_model.pkl
│   ├── xgb_model.pkl
│   └── dffnn_model.keras
├── outputs/
│   ├── graphs/
│   ├── reports/
│   └── predictions/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Paths, hyperparams, auto-folder creator
│   ├── utils.py                  # Logger, joblib helpers, regression metrics
│   ├── data_loader.py            # CSV/Excel smart loader & data quality report
│   ├── data_merger.py            # Multi-CSV intelligent key join engine
│   ├── preprocessing.py          # Data cleaning, imputation, scaling, joblib saver
│   ├── feature_engineering.py    # Profit Margin, ROI, Conversion Rate, Engagement Score
│   ├── eda.py                    # Plotly & Matplotlib graph generator & file saver
│   ├── train_mlr.py              # MLR model trainer (mlr_model.pkl)
│   ├── train_xgb.py              # XGBoost model trainer (xgb_model.pkl)
│   ├── train_dffnn.py            # TensorFlow / Keras DFFNN trainer (dffnn_model.keras)
│   ├── evaluate.py               # Benchmark comparison, residual plots, loss curves
│   ├── predict.py                # Single & batch prediction engine
│   ├── business_insights.py      # Automated commercial recommendations
│   └── explain_model.py          # SHAP global & local explainer
├── app.py                        # 15-Page Streamlit Enterprise Dashboard
├── requirements.txt              # Dependency specification
└── README.md                     # Documentation & Quick Start Guide
```

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Streamlit Application
```bash
streamlit run app.py
```

---

## ⚡ Tech Stack & Key Features

- **Data Merger**: `src/data_merger.py` automatically merges multi-table Kaggle CSV datasets on key join fields (`product_id`, `order_id`, etc.).
- **Preprocessing & Joblib Artifacts**: Preprocessing imputes median/mode, clips outliers, one-hot encodes categoricals, standardizes features, and serializes `scaler.pkl`, `encoder.pkl`, `feature_columns.pkl`.
- **Multiple Linear Regression**: `src/train_mlr.py` fits Scikit-Learn Ridge/Linear Regression (`models/mlr_model.pkl`).
- **XGBoost Regressor**: `src/train_xgb.py` fits XGBoost gradient boosted trees (`models/xgb_model.pkl`).
- **TensorFlow DFFNN**: `src/train_dffnn.py` fits Keras `Dense(128)->Dropout(0.3)->Dense(64)->Dropout(0.2)->Dense(32)->Dense(16)->Dense(1)` model with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint (`models/dffnn_model.keras`).
- **Explainable AI (SHAP)**: Global beeswarm summary, bar plot importances, and local prediction attributions.
"# AI-Ecommerce-Profit-Prediction" 
