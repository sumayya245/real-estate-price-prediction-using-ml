# real-estate-price-prediction-using-ml
# 🏡 Real Estate Price Prediction using Machine Learning

This project predicts house prices using multiple machine learning models based on features such as location, square footage, number of bedrooms, and more. A Flask web app is provided for user interaction and real-time predictions.

---

## 📌 Project Overview

The workflow includes:

- Data preprocessing and outlier removal
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training (multiple regression models)
- Web app deployment with Flask

---

## 📊 Dataset Features

The dataset contains the following features:

- `Location`
- `Square Footage`
- `Number of Bedrooms (BHK)`
- `Number of Bathrooms`
- `Property Type`
- `Year Built`
- `Parking`
- `Price`

---

## ⚙️ Technologies Used

- **Python (Anaconda environment)**
- **Pandas, NumPy** – Data processing
- **Matplotlib, Seaborn** – Visualization
- **Scikit-learn** – ML models and evaluation
- **XGBoost** – Advanced tree-based model
- **Flask** – Web deployment
- **HTML/CSS** – Frontend

---

## 🧠 Machine Learning Models Used

The following models are implemented and compared in the file `model_training.py`:

1. **Linear Regression**
2. **Random Forest Regressor**
3. **XGBoost Regressor**
4. **Support Vector Regressor (SVR)**

These models are trained, evaluated, and compared using standard regression metrics.

### Evaluation Metrics:
- **R² Score** (Accuracy of the model)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
