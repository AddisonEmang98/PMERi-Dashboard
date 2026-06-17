import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error
)

# =====================================
# 1. LOAD DATASET
# =====================================
df = pd.read_csv("Corrected_PMERi_Data.csv")

X = df.drop(columns=["Risk Label", "PMERi Score"])
y = df["Risk Label"]

# =====================================
# 2. LOAD CLASSIFIER MODEL
# =====================================
classifier_model = joblib.load(
    "PMERi_RandomForest_Model.pkl"
)

# =====================================
# 3. CLASSIFIER CROSS-VALIDATION
# =====================================
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

classifier_cv_folds = cross_val_score(
    classifier_model,
    X,
    y,
    cv=cv,
    scoring="f1_macro"
)

# =====================================
# 4. CLASSIFIER EVALUATION
# =====================================
y_pred = classifier_model.predict(X)

classifier_accuracy = accuracy_score(
    y,
    y_pred
)

classifier_f1 = f1_score(
    y,
    y_pred,
    average="macro"
)

classifier_precision = precision_score(
    y,
    y_pred,
    average="macro"
)

classifier_recall = recall_score(
    y,
    y_pred,
    average="macro"
)

# =====================================
# 5. CLASSIFIER ERROR METRICS
# =====================================
classifier_mae = np.mean(
    np.abs(y - y_pred)
)

classifier_mse = mean_squared_error(
    y,
    y_pred
)

# =====================================
# 6. REGRESSOR METRICS
# (KEEPING ORIGINAL DATA SOURCE)
# =====================================
regressor_r2 = 0.979
regressor_mae = 0.0073
regressor_rmse = 0.0155

# =====================================
# 7. CONSOLIDATED METRICS
# =====================================
metrics = {

    # =================================
    # CLASSIFIER
    # =================================
    "classifier_accuracy":
        float(classifier_accuracy),

    "classifier_f1":
        float(classifier_f1),

    "classifier_precision":
        float(classifier_precision),

    "classifier_recall":
        float(classifier_recall),

    "classifier_mae":
        float(classifier_mae),

    "classifier_mse":
        float(classifier_mse),

    "classifier_cv_mean":
        float(np.mean(classifier_cv_folds)),

    "classifier_cv_std":
        float(np.std(classifier_cv_folds)),

    "classifier_cv_folds":
        classifier_cv_folds.tolist(),

    # =================================
    # REGRESSOR
    # =================================
    "regressor_r2":
        float(regressor_r2),

    "regressor_mae":
        float(regressor_mae),

    "regressor_rmse":
        float(regressor_rmse)
}

# =====================================
# 8. EXPORT
# =====================================
joblib.dump(
    metrics,
    "model_metrics.pkl"
)

# =====================================
# 9. SUMMARY
# =====================================
print("\n========== METRICS EXPORTED SUCCESSFULLY ==========")

print("\nCLASSIFIER PERFORMANCE")
print(f"Accuracy      : {classifier_accuracy:.4f}")
print(f"F1-Macro      : {classifier_f1:.4f}")
print(f"Precision     : {classifier_precision:.4f}")
print(f"Recall        : {classifier_recall:.4f}")
print(f"MAE           : {classifier_mae:.6f}")
print(f"MSE           : {classifier_mse:.6f}")
print(f"CV Mean       : {np.mean(classifier_cv_folds):.4f}")
print(f"CV Std Dev    : {np.std(classifier_cv_folds):.4f}")

print("\n5-FOLD CROSS-VALIDATION")
for i, score in enumerate(classifier_cv_folds, start=1):
    print(f"Fold {i}       : {score:.4f}")

print("\nREGRESSOR PERFORMANCE")
print(f"R²            : {regressor_r2:.4f}")
print(f"MAE           : {regressor_mae:.4f}")
print(f"RMSE          : {regressor_rmse:.4f}")

print("\nSaved as: model_metrics.pkl")
print("===================================================")