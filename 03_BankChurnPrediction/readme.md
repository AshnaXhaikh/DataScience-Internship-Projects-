# 💼 Customer Churn Prediction

This project aims to predict customer churn using a machine learning classification model. Churn prediction helps businesses proactively identify customers at risk of leaving and take steps to retain them.
---
🚀 Try the App
Access the live Streamlit app here:
🔗[Open Streamlit App](https://churnpredictions-createdbyashna.streamlit.app)

## 🚀 Features

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training (Random Forest)
- Hyperparameter Tuning
- Model Evaluation (Confusion Matrix, Accuracy, etc.)
- Model and Scaler Serialization (Pickle)
- Ready for Deployment (FastAPI, Streamlit, etc.)

---
📄 Read the Full Documentation
Learn more about the project, its data pipeline, model architecture, evaluation, and more in the detailed documentation:
📘 [Read Full Gamma Documentation](https://bank-customer-churn-pred-8tjb9fk.gamma.site/)
---
## 📁 Project Structure
BankChurn/
│
├── app.py # FastAPI or Streamlit app for deployment
├── churn_model_rf.pkl # Trained Random Forest model
├── rf_scaler.pkl # StandardScaler used during preprocessing
├── requirements.txt # Project dependencies
└── README.md # Project overview and instructions


---

## 📌 Key Steps

### 1. Exploratory Data Analysis (EDA)

- Countplots by churn status
- Distribution of age, credit score, and balance
- Heatmap for correlation analysis

### 2. Feature Engineering

- Encoded categorical features (country, gender)
- Scaled numerical features using `StandardScaler`

### 3. Model Training

- Trained a Random Forest Classifier
- Evaluated using accuracy, classification report, and confusion matrix

### 4. Model Saving

- Saved both model and scaler using `pickle`

```python
import pickle
pickle.dump(model, open('models/churn_model.pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
```
5. Deployment Options
✅ Kaggle Notebook (Public)

✅ Streamlit (free app hosting)

✅ FastAPI (with deployment to Render/Glitch/Replit)

🚀 Future Improvements

Integrate SHAP for model explainability

Add frontend for predictions

## 📥 Cloning the Repository

To clone the repository to your local machine, run:

```bash
git clone https://github.com/AshnaXhaikh/Churn_Predictions/BankChurn.git
cd BankChurn
```

