import numpy as np
import pandas as pd    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from app import X_test_scaled, X_train_scaled, X_val_scaled , y_train_np, y_val_np, y_test_np, b, w, alpha, num_iters, lambda_, sample_weights, cost_history
sklearn_model = LogisticRegression(
    class_weight = {0:1, 1:4},  # Weights for each class
    C = 100,   #strength of regularization, smaller values specify stronger regularization
    max_iter = 2000,
    random_state = 42
)
sklearn_model.fit(X_train_scaled, y_train_np)
sklearn_val_probabilities = sklearn_model.predict_proba(X_val_scaled)[:,1]

best_sklearn_threshold = 0
best_sklearn_f1 = 0

# Find the best threshold using validation data
for threshold in np.arange(0.1, 0.9, 0.01):

    val_predictions = (
        sklearn_val_probabilities >= threshold
    ).astype(int)

    f1 = f1_score(
        y_val_np,
        val_predictions,
        zero_division=0
    )

    if f1 > best_sklearn_f1:
        best_sklearn_f1 = f1
        best_sklearn_threshold = threshold


print("\nSCIKIT-LEARN")
print("----------------")
print("Best validation threshold:", best_sklearn_threshold)
print("Validation F1:", best_sklearn_f1)


# -------------------------------------------------
# Final evaluation on TEST data
# -------------------------------------------------

sklearn_test_probabilities = sklearn_model.predict_proba(
    X_test_scaled
)[:, 1]

sklearn_final_predictions = (
    sklearn_test_probabilities >= best_sklearn_threshold
).astype(int)


sklearn_precision = precision_score(
    y_test_np,
    sklearn_final_predictions,
    zero_division=0
)

sklearn_recall = recall_score(
    y_test_np,
    sklearn_final_predictions,
    zero_division=0
)

sklearn_f1 = f1_score(
    y_test_np,
    sklearn_final_predictions,
    zero_division=0
)

sklearn_accuracy = np.mean(
    sklearn_final_predictions == y_test_np
)

sklearn_cm = confusion_matrix(
    y_test_np,
    sklearn_final_predictions
)


print("\nFINAL SCIKIT-LEARN TEST RESULTS")
print("--------------------------------")
print("Threshold:", best_sklearn_threshold)
print("Accuracy:", sklearn_accuracy)
print("Precision:", sklearn_precision)
print("Recall:", sklearn_recall)
print("F1 Score:", sklearn_f1)

print("\nConfusion Matrix:")
print(sklearn_cm)

print("\nMODEL COMPARISON")
print("----------------")
print("From-scratch bias:", b)
print("Scikit-Learn bias:", sklearn_model.intercept_[0])
print("From-scratch weights:", w)
print("Scikit-Learn weights:", sklearn_model.coef_[0])