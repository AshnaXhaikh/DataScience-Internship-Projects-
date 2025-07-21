## 🩺 Problem Type

This is a **regression problem**, where the goal is to predict a **continuous numeric value** — the **medical insurance claim amount**.

---

## 📘 Problem Statement

“Given the information about a customer and an incident, **predict the expected medical insurance claim amount**.”

The features include:
- **Demographic details** (age, sex, BMI)
- **Insurance-related info** (smoker status, region)
- **Claim-specific attributes** (number of children, etc.)

The target variable is:
- `charges` — the total amount billed by the insurer.

This problem helps insurance companies:
- Estimate risk and premium pricing
- Detect anomalies in claims
- Improve customer segmentation

---
## **view live app**
[streamlit live demo](https://medicalcostinsurace.streamlit.app/)

### 🛠️ Approach

1. **Polynomial Features** (degree=2): Captures nonlinear interactions between features.
2. **StandardScaler**: Ensures features are standardized before modeling.
3. **LassoCV**: Automatically selects the best regularization parameter `alpha` via cross-validation.
4. **Final Model**: Once the best alpha is found, a fresh Lasso model is trained with it for final predictions.

---

### 🧪 Steps Performed

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LassoCV, Lasso

# 1. Cross-validation to find best alpha
lasso_cv = LassoCV(cv=5, random_state=42)
pipeline_cv = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), lasso_cv)
pipeline_cv.fit(X_train, y_train)

# 2. Train final model using best alpha
best_alpha = lasso_cv.alpha_
final_model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), Lasso(alpha=best_alpha))
final_model.fit(X_train, y_train)

# 3. Make predictions
y_pred = final_model.predict(X_test)
```

---

### 💾 Saving the Final Model

```python
import joblib
joblib.dump(final_model, 'lasso_poly_model.pkl')
```

You can now finalize this model as your deployed solution.

---

### 📌 Notes (Tailored to Your Questions)

1. **Why fit two times?**
   First fit: To find the best alpha using `LassoCV`.
   Second fit: To build a clean final model with that alpha.

2. **Why no manual scaling?**
   `StandardScaler()` is already in the pipeline. It scales during `.fit()` and `.predict()` automatically.

3. **Why use `LassoCV` before `Lasso`?**
   `LassoCV` helps tune the model. Once the best alpha is known, we lock it into a fresh `Lasso`.

4. **What is a good RMSE?**
   There’s no universal "best" RMSE — lower is better, but it must be compared in context of the problem and scale of your target variable.

5. **Best alpha meaning**
   Best alpha balances bias and variance — it reduces overfitting while keeping good predictive performance.

6. **Pipeline Advantage**
   It ensures consistent transformation (scaling, feature expansion) across both training and testing data — critical when saving the model.

---

### 📦 Requirements

```txt
scikit-learn
joblib
numpy
pandas
```
