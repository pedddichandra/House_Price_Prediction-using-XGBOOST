# 🏠 House Price Prediction using XGBoost

## Overview
Predict house prices based on features like area, bedrooms, bathrooms, parking, and more using **XGBoost Regression**. Demonstrates data preprocessing, feature encoding, model training, hyperparameter tuning, and evaluation.

## Features
- Encoded categorical variables (`Yes/No` → `1/0`)
- Compared Linear Regression, Ridge/Lasso, and XGBoost
- Hyperparameter tuning with `GridSearchCV`
- Evaluated models using **R² Score** and **RMSE**
- Deployed interactive **Streamlit app** for predictions

## Dataset
- **Columns:** `['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']`
- **Target:** `price`
- **Samples:** 545

## Model Performance
| Model             | R² Score | RMSE        |
|------------------|-----------|------------|
| Linear Regression | 0.65      | 1,324,506  |
| XGBoost (Tuned)   | 0.63      | 1,371,601  |

## Tech Stack
- Python 3.x
- Pandas, NumPy, Scikit-Learn
- XGBoost
- Matplotlib, Seaborn
- Streamlit (for deployment)

## Run Locally
1. Clone the repo:
```bash
git clone https://github.com/pedddichandra/House_Price_Prediction-using-XGBOOST
cd House_Price_Prediction-using-XGBOOST
streamlit run house_pred.py


Future Improvements

Feature importance visualization

Deploy API using FastAPI

Deploy live on Streamlit Cloud or Render



Author

Sathvik Chandra Peddi
