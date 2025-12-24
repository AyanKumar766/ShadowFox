import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# 1) Load dataset
df = pd.read_csv("HousingData.csv")

# 2) Data preprocessing
numeric_df = df.select_dtypes(include=[np.number])   # keep numeric columns
X = numeric_df.drop(columns=["MEDV"])                # features (CRIM, RM, etc.)
y = numeric_df["MEDV"]                               # target (house price)

imputer = SimpleImputer(strategy="mean")             # fill missing values
X = imputer.fit_transform(X)

scaler = StandardScaler()                            # scale features
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split( # train/test split
    X, y, test_size=0.2, random_state=42
)

# 3) Model selection + training + evaluation
def evaluate(model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    return r2, mae, rmse, model

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
}

results = {}

print("Model performance:")
for name, model in models.items():
    r2, mae, rmse, trained = evaluate(model)
    results[name] = {"r2": r2, "mae": mae, "rmse": rmse, "model": trained}
    print(f"{name:18s}  R2={r2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}")

# choose best model based on R²
best_name = max(results, key=lambda k: results[k]["r2"])
best_model = results[best_name]["model"]

# 4) Save best model + preprocessing objects
joblib.dump(
    {"model": best_model, "imputer": imputer, "scaler": scaler},
    "best_model.joblib"
)

print(f"\nBest model: {best_name}")
print("Training complete. Best model saved as best_model.joblib.")
