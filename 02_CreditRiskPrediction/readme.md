# Credit Risk Prediction

---

## 🎯 Objective

The goal of this project is to **predict whether a loan applicant is likely to default on a loan** using supervised machine learning models. The project focuses on addressing key challenges such as:
- Handling **class imbalance**
- Preventing **model overfitting**
- Ensuring interpretability and performance of the models

---

## 🛠️ Tools & Libraries Used

- **Programming Language:** Python  
- **Data Processing:** `Pandas`, `NumPy`  
- **Modeling & Evaluation:** `Scikit-learn`  
- **Visualization:** `Matplotlib`, `Seaborn`  

---

## 🤖 Models Implemented

- **Logistic Regression**  
- **Random Forest Classifier**  

Each model was evaluated on classification performance, and insights were derived to recommend the most effective solution.

---

## 📌 Key Contributions

- Performed **data cleaning and preprocessing**
- Addressed **imbalanced classes** using appropriate techniques
- Built and compared **baseline ML models**
- Visualized model performance and insights

---

# Loan Default Prediction Project

This project focuses on building a machine learning model to predict loan defaults based on various applicant and loan characteristics. The goal is to assist financial institutions in assessing credit risk more accurately.

## Project Overview

The core objective is to classify loan applications into two categories: "Default" (Class 1) or "No Default" (Class 0). This is a binary classification problem, with a particular emphasis on correctly identifying potential defaulters to mitigate financial risk.

## Dataset

The dataset used in this project contains 148,670 loan records with 10 selected features and a target variable.

**Features:**
* `income`: Applicant's income.
* `dtir1`: Debt-to-income ratio.
* `loan_amount`: The exact loan amount.
* `rate_of_interest`: The interest rate charged.
* `Credit_Worthiness`: Lender's assessment of credit risk (encoded).
* `LTV`: Loan-to-Value ratio.
* `loan_purpose`: The reason for the loan (encoded).
* `term`: The loan's repayment period.
* `age`: Applicant's age group (encoded).
* `loan_type`: The type of loan (encoded).

**Target Variable:**
* `default`: Binary indicator (0 for No Default, 1 for Default).

## Data Preprocessing

The raw data underwent several preprocessing steps to prepare it for machine learning:

1.  **Feature Selection:** Identified and selected 10 optimal features based on domain relevance and initial correlation analysis. Irrelevant columns like `ID` and `defaultID`, and the original `status` column (which was a different target) were dropped.
2.  **Missing Value Imputation:**
    * Numerical features (`rate_of_interest`, `dtir1`, `LTV`, `income`, `term`) were imputed using their **median** values. For `rate_of_interest`, `dtir1`, `LTV`, and `income`, additional **missing indicator columns** were created to capture potential predictive signals from missingness.
    * Categorical features (`age`, `loan_purpose`) were imputed using their **mode** (most frequent category). Special care was taken to convert non-standard missing representations (like the string 'nan', empty strings) into `np.nan` before imputation.
3.  **Feature Encoding:**
    * **One-Hot Encoding** was applied to nominal categorical features: `loan_type`, `Credit_Worthiness`, and `loan_purpose`. This creates new binary columns for each category.
    * **Ordinal Encoding** was applied to the `age` column. A specific order was defined (`<25`, `25-34`, ..., `>74`) to map age ranges to numerical values, preserving their inherent order.

## Model Training

1.  **Data Splitting:** The preprocessed data was split into training (80%) and testing (20%) sets, ensuring stratification to maintain the original class distribution in both sets.
2.  **Class Imbalance Handling:** The training data exhibited class imbalance (approximately 24.6% defaults). **SMOTE (Synthetic Minority Oversampling Technique)** was applied *only* to the training set to generate synthetic samples for the minority class (`default=1`), thereby balancing the dataset for training.
3.  **Model Selection:** A **RandomForestClassifier** was chosen for its ability to handle complex, non-linear relationships and its robust performance on tabular data.
4.  **Training:** The RandomForestClassifier was trained on the SMOTE-resampled training data.

## Model Performance

The RandomForestClassifier achieved strong performance on the unseen test set:

| Metric            | Score (Class 1 - Default) | Score (Overall) |
| :---------------- | :------------------------ | :-------------- |
| **Accuracy** | -                         | 0.92            |
| **Precision** | 0.82                      | -               |
| **Recall** | 0.87                      | -               |
| **F1-Score** | 0.84                      | -               |
| **ROC AUC Score** | (Not explicitly shown above, but typically high with these metrics) |
| **Confusion Matrix:** |                           |                 |
| True Positives (TP) | 6356                      |                 |
| False Negatives (FN)| 972                       |                 |
| True Negatives (TN) | 21031                     |                 |
| False Positives (FP)| 1375                      |                 |

**Interpretation:**
* The model correctly identifies 87% of actual defaulters (high Recall), which is crucial for risk management.
* When the model predicts a default, it is correct 82% of the time (good Precision), reducing unnecessary loan denials.
* The overall accuracy of 92% and the balanced F1-score for the minority class indicate a robust and effective model.

## Saved Model Components

For deployment and future inference, the following components have been saved:

* `random_forest_model.joblib`: The trained RandomForestClassifier model.
* `one_hot_encoder.joblib`: The fitted OneHotEncoder used for nominal categorical features.
* `ordinal_encoder_age.joblib`: The fitted OrdinalEncoder used for the `age` column.

## How to Use for Inference

To make predictions on new, unseen loan application data:

1.  **Load Components:** Load the saved model and encoders using `joblib.load()`.
2.  **Load New Data:** Load your new loan application data into a Pandas DataFrame.
3.  **Preprocessing New Data:** Apply the exact same preprocessing steps (missing value handling, encoding) to the new data as was done during training, using the *loaded* encoders' `transform()` method.
4.  **Predict:** Use the `predict()` or `predict_proba()` method of the loaded RandomForestClassifier model on the fully preprocessed new data.




False Positives (FP)

1375



Interpretation:

The model correctly identifies 87% of actual defaulters (high Recall), which is crucial for risk management.

When the model predicts a default, it is correct 82% of the time (good Precision), reducing unnecessary loan denials.

The overall accuracy of 92% and the balanced F1-score for the minority class indicate a robust and effective model.

Saved Model Components
For deployment and future inference, the following components have been saved:

random_forest_model.joblib: The trained RandomForestClassifier model.

one_hot_encoder.joblib: The fitted OneHotEncoder used for nominal categorical features.

ordinal_encoder_age.joblib: The fitted OrdinalEncoder used for the age column.

How to Use for Inference
To make predictions on new, unseen loan application data:

Load Components: Load the saved model and encoders using joblib.load().

Load New Data: Load your new loan application data into a Pandas DataFrame.

Preprocessing New Data: Apply the exact same preprocessing steps (missing value handling, encoding) to the new data as was done during training, using the loaded encoders' transform() method.

Predict: Use the predict() or predict_proba() method of the loaded RandomForestClassifier model on the fully preprocessed new data.

---

