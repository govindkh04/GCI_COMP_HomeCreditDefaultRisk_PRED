# Home Credit Default Risk Prediction

A machine learning project focused on predicting the probability of loan default using customer demographic, financial, and employment-related data.

## Overview

The objective is to identify customers who are likely to experience difficulties repaying their loans by analyzing historical customer information and developing predictive classification models.

**Problem Type:** Binary Classification  
**Target Variable:** `TARGET`  
**Evaluation Metric:** ROC-AUC

## Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Boosting Algorithms:** LightGBM, CatBoost
- **Hyperparameter Optimization:** Optuna
- **Environment:** Jupyter Notebook / Google Colab

## Project Workflow

### 1. Data Loading & Exploration
- Loaded and explored training and testing datasets.
- Examined feature distributions, data types, and missing values.
- Identified categorical variables and potential anomalies.

### 2. Exploratory Data Analysis (EDA)
- Analyzed target variable distribution and class imbalance.
- Visualized numerical and categorical features.
- Investigated outliers in income, credit amounts, and car ownership age.
- Explored relationships between financial characteristics and loan default risk.

### 3. Feature Engineering
Created additional features to capture meaningful financial and demographic relationships:

- `CREDIT_INCOME_RATIO`
- `ANNUITY_INCOME_RATIO`
- `CREDIT_GOODS_RATIO`
- `INCOME_PER_PERSON`
- `CREDIT_PER_PERSON`
- `AGE_YEARS`
- `EMPLOYED_TO_AGE_RATIO`
- `EXT_SOURCES_MEAN`
- `EXT_SOURCES_PROD`

### 4. Data Preprocessing
- Handled missing values using appropriate imputation strategies.
- Identified and treated anomalous employment duration values.
- Applied Label Encoding and Count Encoding to categorical variables.
- Used One-Hot Encoding for grouped car ownership age.
- Standardized numerical features using StandardScaler.

### 5. Model Development

Implemented and experimented with multiple classification algorithms:

- Logistic Regression
- Random Forest Classifier
- LightGBM
- CatBoost

Additional techniques:
- Optuna-based hyperparameter optimization.
- Stratified K-Fold cross-validation.
- Weighted ensemble prediction blending.

### 6. Prediction & Submission
- Generated probability predictions for the test dataset.
- Prepared predictions in the required submission format.
- Exported final predictions as a CSV file.

## Repository Structure

```text
Home-Credit-Default-Risk/
│
├── input/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── output/
│   └── submission.csv
│
├── Home_Credit_Default_Risk.ipynb
│
└── README.md
