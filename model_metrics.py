import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("Corrected_PMERi_Data.csv")

X = df.drop(columns=["Risk Label", "PMERi Score"])
y = df["Risk Label"]

# =========================
# 2. Load trained classifier model
# =========================
model = joblib.load("PMERi_RandomForest_Model.pkl")

# =========================
# 3. Cross-validation (F1 Macro)
# =========================
cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="f1_macro"
)

# =========================
# 4. Generate training-set style metrics (for thesis reporting)
# =========================
y_pred = model.predict(X)

classifier_f1 = f1_score(y, y_pred, average="macro")
classifier_acc = accuracy_score(y, y_pred)

# =========================
# 5. Metrics dictionary (FINAL THESIS FORMAT)
# =========================
metrics = {
    "classifier_f1": float(classifier_f1),
    "classifier_accuracy": float(classifier_acc),
    "classifier_cv_mean": float(cv_scores.mean()),
    "classifier_cv_folds": cv_scores.tolist(),

    # regressor values (from your regression script)
    "regressor_r2": 0.979,
    "regressor_mae": 0.0073,
    "regressor_rmse": 0.0155
}

# =========================
# 6. Save metrics
# =========================
joblib.dump(metrics, "model_metrics.pkl")

print("\n========== MODEL METRICS SAVED ==========")
print("Classifier F1:", classifier_f1)
print("CV Mean F1:", cv_scores.mean())
print("CV Folds:", cv_scores.tolist())
print("Metrics saved to model_metrics.pkl")