import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import GridSearchCV

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_excel("data1319.xlsx")

# -----------------------------
# 2. Temporal train-test split
# -----------------------------
train_df = df[df['year'] <= 2018].copy()
test_df = df[df['year'] == 2019].copy()

# Define target
y_train = train_df['share']
y_test = test_df['share']

# Define features (exclude mo, year, model, target)
exclude_cols = ["share", "mo", "year", "model"]
X_train = train_df.drop(columns=exclude_cols)
X_test = test_df.drop(columns=exclude_cols)

# -----------------------------
# 3. Encode categorical features (after split to avoid leakage)
# -----------------------------
for col in X_train.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# -----------------------------
# 4. GridSearchCV to tune XGBoost
# -----------------------------
xgb_base = XGBRegressor(random_state=42)

param_grid = {
    "n_estimators": [200, 400, 600],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.03, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    cv=5,  # 5-fold CV on 2013-2018 data
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest parameters found:")
print(grid_search.best_params_)
print("\nBest CV score (R² on training folds):")
print(grid_search.best_score_)

# Use the best model
model = grid_search.best_estimator_

# -----------------------------
# 5. Train metrics and test evaluation
# -----------------------------
def get_regression_metrics(X_train, X_test, y_train, y_test, model):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics_dict = {
        "train R2": r2_score(y_train, y_train_pred),
        "test R2": r2_score(y_test, y_test_pred),
        "train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "train MAE": np.mean(np.abs(y_train - y_train_pred)),
        "test MAE": np.mean(np.abs(y_test - y_test_pred))
    }
    return metrics_dict

metrics = get_regression_metrics(X_train, X_test, y_train, y_test, model)
print("\nRegression metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.3f}")

# -----------------------------
# 6. Feature importance (highlight top 5)
# -----------------------------
booster = model.get_booster()
importance = booster.get_score(importance_type='gain')
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

print("\nTop features by gain:")
top_features = sorted_importance[:5]
for feat, val in top_features:
    print(f"{feat}: {val:.3f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar([feat for feat, _ in sorted_importance],
        [val for _, val in sorted_importance],
        color='skyblue')
plt.title("Feature Importance (Gain)")
plt.ylabel("Gain")
plt.xticks(rotation=45, ha='right')

# Highlight top features in orange
for feat, _ in top_features:
    plt.bar(feat, importance[feat], color='orange')

plt.tight_layout()
plt.show()

# -----------------------------
# 7. Example predictions on new data
# -----------------------------
new_predictions = model.predict(X_test.iloc[:5])
print("\nExample predictions (first 5 rows of 2019):")
print(new_predictions)

# -----------------------------
# 8. Predicted vs Actual plot for 2019
# -----------------------------
y_test_pred = model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.7)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linestyle='--', linewidth=2)
plt.xlabel("Actual Share (2019)")
plt.ylabel("Predicted Share (2019)")
plt.title("Predicted vs Actual Share for 2019")
plt.grid(True)
plt.tight_layout()
plt.show()