# Personal Loan Acceptance Prediction

**Objective:** Predict which customers are likely to accept a personal loan offer using the Bank Marketing Dataset (UCI ML Repository).

**Goal:** Help the bank identify target customer segments and improve marketing efficiency.

**Approach:**
- Data exploration and visualization
- Build classification models (Logistic Regression, Decision Tree)
- Extract actionable business insights

---

## 📊 **Bank Marketing Dataset (UCI Repository)**

### 📝 **Description**

This dataset contains data collected from **direct telemarketing campaigns** of a Portuguese banking institution. The objective of the campaigns was to promote **term deposit subscriptions**. The dataset includes various attributes related to the client's personal and banking information, as well as campaign-specific features.

It is widely used in machine learning and statistical modeling for **binary classification tasks**, particularly to predict whether a client will subscribe to a term deposit.

---
## 🎓 Problem Statement

Financial institutions aim to identify the likelihood of customers accepting loan offers. This helps optimize marketing strategies, reduce costs, and better serve clients. The goal here is to build a **binary classification model** to predict **loan acceptance (1) or rejection (0)**.

---
### 📁 **Dataset Details**

* **Total Instances**: 45,211
* **Total Features**: 17 input variables + 1 output variable
* **Target Variable**: `y` (binary: `yes` or `no`) – whether the client subscribed to a term deposit

---

### 🔍 **Features Overview**

**Client Attributes**

* `age`: Age of the client
* `job`: Type of job
* `marital`: Marital status
* `education`: Education level
* `default`: Has credit in default?
* `balance`: Average yearly account balance (EUR)
* `housing`: Has a housing loan?
* `loan`: Has a personal loan?

**Contact and Campaign Attributes**

* `contact`: Contact communication type (cellular, telephone)
* `day`: Last contact day of the month
* `month`: Last contact month
* `duration`: Duration of the last contact (in seconds)
* `campaign`: Number of contacts performed during the campaign
* `pdays`: Days since last contact from a previous campaign (-1 means not previously contacted)
* `previous`: Number of contacts before this campaign
* `poutcome`: Outcome of the previous marketing campaign

**Output**

* `y`: Term deposit subscription (binary: `yes` or `no`)

---

## 🚀 ML Pipeline & Modeling

### Preprocessing:

* `StandardScaler` applied to numerical features.
* `OneHotEncoder` used for categorical features.
* `ColumnTransformer` used to handle both in one pipeline.

### Model:

* **LightGBMClassifier** with `class_weight='balanced'` to handle class imbalance.
* Integrated into a `Pipeline`.

---

## ⚖️ Model Evaluation

Threshold optimization was conducted to improve recall for the minority class (loan accepted).

### Final Performance:

| Metric    | Class 0 (Reject) | Class 1 (Accept) |
| --------- | ---------------- | ---------------- |
| Precision | 0.94             | 0.35             |
| Recall    | 0.85             | 0.62             |
| F1-Score  | 0.89             | 0.45             |
| Accuracy  | **82%** overall  |                  |

Threshold used for best F1: **0.64**

---
## 🔍 Future Improvements

* Add SHAP/ELI5 for model explainability.
* Deploy using Hugging Face Spaces or Streamlit Cloud.
* Add logging and user authentication for enterprise use.

---

## 🚀 Try It Out

> Run the app locally:

```bash
streamlit run app.py
```

---


