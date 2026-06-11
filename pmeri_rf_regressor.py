import pandas as pd
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# -------------------
# 1. Load data
# -------------------
df = pd.read_csv("Corrected_PMERi_Data.csv")

# -------------------
# 2. Split features/target
# -------------------
X = df.drop(columns=["PMERi Score", "Risk Label"])
y = df["PMERi Score"]

print("\nFeatures Used:")
print(X.columns.tolist())

# -------------------
# 3. Train-test split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------
# 4. Base model
# -------------------
rf = RandomForestRegressor(
    random_state=42
)

# -------------------
# 5. Hyperparameter grid
# -------------------
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# -------------------
# 6. GridSearchCV
# -------------------
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# -------------------
# 7. Best model
# -------------------
best_model = grid_search.best_estimator_

print("\n========== GRID SEARCH RESULTS ==========")

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation R²:")
print(grid_search.best_score_)

# -------------------
# 8. Prediction
# -------------------
y_pred = best_model.predict(X_test)

# -------------------
# 9. Evaluation
# -------------------
r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

print("\n========== TEST SET RESULTS ==========")

print("\nR² Score:")
print(r2)

print("\nMean Absolute Error (MAE):")
print(mae)

print("\nRoot Mean Squared Error (RMSE):")
print(rmse)

# -------------------
# 10. Feature Importance
# -------------------
feature_importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)

feature_importance.to_csv(
    "PMERi_Regressor_Feature_Importance.csv",
    header=["Importance"]
)

# -------------------
# 11. Save Trained Model
# -------------------
joblib.dump(
    best_model,
    "PMERi_RF_Regressor.pkl"
)

print("\nModel saved successfully as:")
print("PMERi_RF_Regressor.pkl")

print("\nFeature importance saved as:")
print("PMERi_Regressor_Feature_Importance.csv")