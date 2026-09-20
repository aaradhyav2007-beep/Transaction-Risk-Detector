import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from preprocessing import prepare_data

data = prepare_data()

X_train_scaled = data["X_train_scaled"]
X_val_scaled = data["X_val_scaled"]
X_test_scaled = data["X_test_scaled"]

y_train_np = data["y_train"]
y_val_np = data["y_val"]
y_test_np = data["y_test"]

feature_columns = data["feature_columns"]
scaler = data["scaler"]

# =========================================================
# CONFIGURATION
# =========================================================

FRAUD_WEIGHT = 4
REGULARIZATION_C = 100
MAX_ITER = 2000

# =========================================================
# 1. CREATE SCIKIT-LEARN MODEL
# =========================================================

sklearn_model = LogisticRegression(
    class_weight={
        0: 1,
        1: FRAUD_WEIGHT
    },
    C=REGULARIZATION_C,
    max_iter=MAX_ITER,
    random_state=42
)

# =========================================================
# 2. TRAIN MODEL
# =========================================================

sklearn_model.fit(
    X_train_scaled,
    y_train_np
)

# =========================================================
# 3. VALIDATION PREDICTIONS
# =========================================================

sklearn_val_probabilities = (
    sklearn_model
    .predict_proba(X_val_scaled)[:, 1]
)

# =========================================================
# 4. SELECT DECISION THRESHOLD
# =========================================================

best_threshold = 0.0
best_validation_f1 = 0.0

for threshold in np.arange(
    0.10,
    0.90,
    0.01
):

    validation_predictions = (
        sklearn_val_probabilities >= threshold
    ).astype(int)

    validation_f1 = f1_score(
        y_val_np,
        validation_predictions,
        zero_division=0
    )

    if validation_f1 > best_validation_f1:

        best_validation_f1 = validation_f1
        best_threshold = threshold


print("\nSCIKIT-LEARN")
print("----------------")

print(
    "Best validation threshold:",
    best_threshold
)

print(
    "Validation F1:",
    best_validation_f1
)

# =========================================================
# 5. FINAL TEST PREDICTIONS
# =========================================================

sklearn_test_probabilities = (
    sklearn_model
    .predict_proba(X_test_scaled)[:, 1]
)

sklearn_test_predictions = (
    sklearn_test_probabilities >= best_threshold
).astype(int)

# =========================================================
# 6. EVALUATION
# =========================================================

sklearn_precision = precision_score(
    y_test_np,
    sklearn_test_predictions,
    zero_division=0
)

sklearn_recall = recall_score(
    y_test_np,
    sklearn_test_predictions,
    zero_division=0
)

sklearn_f1 = f1_score(
    y_test_np,
    sklearn_test_predictions,
    zero_division=0
)

sklearn_accuracy = np.mean(
    sklearn_test_predictions == y_test_np
)

sklearn_cm = confusion_matrix(
    y_test_np,
    sklearn_test_predictions
)

# =========================================================
# 7. DISPLAY FINAL RESULTS
# =========================================================

print("\nFINAL SCIKIT-LEARN TEST RESULTS")
print("--------------------------------")

print(
    "Threshold:",
    best_threshold
)

print(
    "Accuracy:",
    sklearn_accuracy
)

print(
    "Precision:",
    sklearn_precision
)

print(
    "Recall:",
    sklearn_recall
)

print(
    "F1 Score:",
    sklearn_f1
)

print("\nConfusion Matrix:")

print(
    sklearn_cm
)