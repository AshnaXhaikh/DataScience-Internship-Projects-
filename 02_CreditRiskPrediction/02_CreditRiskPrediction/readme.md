Here’s a **refined and professional version** of your project documentation with improved flow, formatting, and clarity — perfect for your README or project page:

---

# Loan Default Prediction using Machine Learning

This project presents a streamlined pipeline to predict the likelihood of loan defaults using financial and demographic data. The goal is to help financial institutions assess credit risk more effectively and reduce potential losses.

---

## 🔍 Objective

Classify loan applications as either:

* **Default (1)** — High credit risk
* **No Default (0)** — Low credit risk

This binary classification task emphasizes correctly identifying defaulters (high recall) to minimize financial risk.

---

## 📊 Dataset Overview

* **Total Records:** 148,670
* **Features:** 10 selected predictors
* **Target:** `default` (0 = No Default, 1 = Default)

### Features Used:

* `income` — Monthly income
* `dtir1` — Debt-to-Income Ratio
* `loan_amount` — Requested loan amount
* `rate_of_interest` — Annual interest rate
* `Credit_Worthiness` — Credit rating (encoded)
* `LTV` — Loan-to-Value ratio
* `loan_purpose` — Purpose of loan (encoded)
* `term` — Loan term in months
* `age` — Age group (encoded)
* `loan_type` — Loan category (encoded)

---

## 🛠️ Data Preprocessing

1. **Feature Selection:**
   Removed irrelevant fields (`ID`, `defaultID`, and `status`). Retained 10 essential predictors based on correlation and business logic.

2. **Handling Missing Values:**

   * Median imputation for numeric fields (`income`, `rate_of_interest`, `dtir1`, `LTV`, `term`)
   * Mode imputation for categorical fields (`age`, `loan_purpose`)
   * Special handling for non-standard missing formats (e.g., empty strings, `'nan'`)

3. **Encoding:**

   * **One-Hot Encoding:** `loan_type`, `loan_purpose`, `Credit_Worthiness`
   * **Ordinal Encoding:** `age` (ordered from `<25` to `>74`)

---

## 🧠 Model Development

* **Algorithm Used:** `RandomForestClassifier`
* **Training/Test Split:** 80% training (stratified), 20% test
* **Class Imbalance:**
  Applied **SMOTE** on the training data to balance class distribution (\~50:50). This added \~47,500 synthetic samples for the minority class (`default=1`).

---

## ✅ Model Performance (on Test Set)

| Metric        | Score for Default Class | Overall Score |
| ------------- | ----------------------- | ------------- |
| **Accuracy**  | -                       | 0.92          |
| **Precision** | 0.82                    | -             |
| **Recall**    | 0.87                    | -             |
| **F1 Score**  | 0.84                    | -             |

### Confusion Matrix:

* **True Positives (TP):** 6,356
* **False Negatives (FN):** 972
* **True Negatives (TN):** 21,031
* **False Positives (FP):** 1,375

### 📌 Key Insights:

* **87%** of actual defaulters were correctly identified
* **82%** precision in predicting defaults — minimizes false alarms
* Robust performance with balanced metrics ensures reliability in practice

---

## 🧳 Deployment Assets

Saved model artifacts for future use:

* `random_forest_model.joblib` — Trained model
* `one_hot_encoder.joblib` — For categorical features
* `ordinal_encoder_age.joblib` — For `age` column

---

## 🔮 How to Make Predictions

1. **Load Components:**
   Use `joblib.load()` to load model and encoders.

2. **Input New Data:**
   Load new applicant data into a DataFrame.

3. **Preprocess:**
   Apply the *same transformations* as training:

   * Missing value imputation
   * Encoding with fitted encoders

4. **Predict:**
   Use `model.predict()` or `predict_proba()` on processed input.

---

Let me know if you'd like a one-line summary for a project card or a version tailored for GitHub's README.
