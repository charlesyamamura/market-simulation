import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_excel("data1319.xlsx")

# -----------------------------
# 2. Temporal split
# -----------------------------
train_df = df[df['year'] <= 2018].copy()
test_df  = df[df['year'] == 2019].copy()

y_train = train_df['share']
y_test  = test_df['share']

exclude_cols = ["share", "mo", "year", "model"]
X_train = train_df.drop(columns=exclude_cols)
X_test  = test_df.drop(columns=exclude_cols)

# -----------------------------
# 3. Encoding (no leakage)
# -----------------------------
for col in X_train.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col]  = le.transform(X_test[col])

# -----------------------------
# 4. TimeSeries CV
# -----------------------------
tscv = TimeSeriesSplit(n_splits=4)

# -----------------------------
# 5. XGBOOST (Improved CV)
# -----------------------------
xgb_model = XGBRegressor(random_state=42)

xgb_param_grid = {
    "n_estimators": [300, 500],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.05],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

xgb_grid = GridSearchCV(
    xgb_model,
    xgb_param_grid,
    cv=tscv,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_

print("\nBest XGBoost Params:", xgb_grid.best_params_)

# -----------------------------
# 6. RANDOM FOREST (NEW)
# -----------------------------
rf_model = RandomForestRegressor(random_state=42)

rf_param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"]
}

rf_grid = GridSearchCV(
    rf_model,
    rf_param_grid,
    cv=tscv,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_

print("\nBest Random Forest Params:", rf_grid.best_params_)

# -----------------------------
# 7. Evaluation function
# -----------------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)
    
    return {
        "Model": name,
        "Train R2": r2_score(y_train, y_train_pred),
        "Test R2": r2_score(y_test, y_test_pred),
        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
    }

# -----------------------------
# 8. Compare models
# -----------------------------
results = []

results.append(evaluate_model("XGBoost", xgb_best, X_train, X_test, y_train, y_test))
results.append(evaluate_model("Random Forest", rf_best, X_train, X_test, y_train, y_test))

results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.sort_values(by="Test R2", ascending=False))