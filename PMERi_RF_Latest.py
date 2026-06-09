import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -------------------
# 1. Load data
# -------------------
df = pd.read_csv("Corrected_PMERi_Data.csv")

# -------------------
# 2. Split features/target
# -------------------
# Remove PMERi Score to create a true predictive model
X = df.drop(columns=["Risk Label", "PMERi Score"])
y = df["Risk Label"]

print("\nFeatures Used:")
print(X.columns.tolist())

# -------------------
# 3. Train-test split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------
# 4. Base model
# -------------------
rf = RandomForestClassifier(
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
    scoring="f1_macro",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# -------------------
# 7. Best model
# -------------------
best_model = grid_search.best_estimator_

print("\n========== GRID SEARCH RESULTS ==========")

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1-Macro:")
print(grid_search.best_score_)

# -------------------
# Improvement 4:
# Cross-validation standard deviation
# -------------------
results = pd.DataFrame(grid_search.cv_results_)

best_index = grid_search.best_index_

cv_std = results.loc[best_index, "std_test_score"]

print("\nCross-Validation Standard Deviation:")
print(cv_std)

# -------------------
# 8. Prediction
# -------------------
y_pred = best_model.predict(X_test)

y_prob = best_model.predict_proba(X_test)

print("\nFirst 10 Prediction Probabilities:")
print(y_prob[:10])

# -------------------
# 9. Evaluation
# -------------------
print("\n========== TEST SET RESULTS ==========")

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

report_text = classification_report(y_test, y_pred)

print("\nClassification Report:")
print(report_text)

# -------------------
# Improvement 2:
# Save Classification Report
# -------------------
report_dict = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose()

report_df.to_csv(
    "Classification_Report.csv"
)

# -------------------
# Improvement 3:
# Save Confusion Matrix
# -------------------
cm_df = pd.DataFrame(cm)

cm_df.to_csv(
    "Confusion_Matrix.csv",
    index=False
)

# -------------------
# 10. Feature Importance
# -------------------
feature_importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)

# Save feature importance
feature_importance.to_csv(
    "PMERi_Feature_Importance.csv",
    header=["Importance"]
)

# -------------------
# Improvement 1:
# Feature Importance Plot
# -------------------
plt.figure(figsize=(8, 5))

feature_importance.plot(kind="bar")

plt.title("Random Forest Feature Importance")
plt.xlabel("Environmental Variables")
plt.ylabel("Importance Score")

plt.tight_layout()

plt.savefig(
    "Feature_Importance.png",
    dpi=300
)

plt.show()

# -------------------
# 11. Save Trained Model
# -------------------
joblib.dump(
    best_model,
    "PMERi_RandomForest_Model.pkl"
)

# -------------------
# 12. Summary
# -------------------
print("\n========== FILES GENERATED ==========")

print("\nModel:")
print("PMERi_RandomForest_Model.pkl")

print("\nFeature Importance:")
print("PMERi_Feature_Importance.csv")

print("\nFeature Importance Figure:")
print("Feature_Importance.png")

print("\nClassification Report:")
print("Classification_Report.csv")

print("\nConfusion Matrix:")
print("Confusion_Matrix.csv")

print("\nTraining completed successfully.")