import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor, plot_tree


# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_excel("data1319.xlsx")


# -----------------------------
# 2. Define target
# -----------------------------
y = df["share"]


# -----------------------------
# 3. Define features (exclude mo, year, model)
# -----------------------------
exclude_cols = ["share", "mo", "year", "model"]
X = df.drop(columns=exclude_cols)


# -----------------------------
# 4. Encode categorical features
# -----------------------------
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])


# -----------------------------
# 5. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 6. Define XGBoost model
# -----------------------------
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


# -----------------------------
# 7. Train model
# -----------------------------
model.fit(X_train, y_train)


# -----------------------------
# 8. Predictions
# -----------------------------
predictions = model.predict(X_test)


# -----------------------------
# 9. Evaluate model
# -----------------------------
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("RMSE:", rmse)
print("R²:", r2)


# -----------------------------
# 10. Predict new data
# -----------------------------
# Example: predict the first 5 rows
new_predictions = model.predict(X.iloc[:5])
print("Example predictions:")
print(new_predictions)

# -----------------------------
# 11. K-Fold cross validation
# -----------------------------
scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print("R2 scores:", scores)
print("Mean R2:", np.mean(scores))

# =============================
# 12. Explain the model
# =============================
booster = model.get_booster()

# ---- 12a. Print all trees ----
tree_dumps = booster.get_dump(with_stats=True)
for i, tree in enumerate(tree_dumps):
    print(f"\n===== Tree {i} =====\n")
    print(tree)

# ---- 12b. Plot the first few trees ----
N = min(5, len(tree_dumps))  # number of trees to visualize
for i in range(N):
    plt.figure(figsize=(20, 10))
    plot_tree(model, num_trees=i, rankdir='LR')
    plt.title(f"Tree {i}")
    plt.show()

# ---- 12c. Feature importance ----
importance = booster.get_score(importance_type='gain')
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
print("\nFeature Importance (by gain):")
for feat, val in sorted_importance:
    print(f"{feat}: {val:.3f}")

# ---- 12d. Summary figure highlighting top features ----
top_features = [feat for feat, _ in sorted_importance[:5]]  # top 5 features

plt.figure(figsize=(12, 6))
plt.bar([feat for feat, _ in sorted_importance], [val for _, val in sorted_importance], color='skyblue')
plt.title("Feature Importance (Gain)")
plt.ylabel("Gain")
plt.xticks(rotation=45, ha='right')

# Highlight top features in orange
for feat in top_features:
    plt.bar(feat, importance[feat], color='orange')

plt.tight_layout()
plt.show()