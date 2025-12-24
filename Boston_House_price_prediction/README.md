	# Beginner Level Task 02 - Boston House Price Prediction

This project predicts Boston house prices (MEDV) using machine learning regression models.

## Dataset
- File: HousingData.csv
- Rows: 506
- Columns: 14
- Target Column: MEDV (Median house value)

## Steps Performed
- Data loading
- Handling missing values
- Exploratory Data Analysis (EDA)
- Correlation heatmap
- Train–test split
- Feature scaling
- Model training (Linear Regression, Random Forest, Gradient Boosting)
- Model comparison
- Saving best model (best_model.joblib)

## Best Model
Gradient Boosting Regressor  
Test R² ≈ 0.90  
Lowest MAE and RMSE

## How to Run
pip install -r requirements.txt  
python boston_house_price_prediction.py  

## Files
boston_house_price_prediction.py  
HousingData.csv  
best_model.joblib  
requirements.txt  
README.md  
