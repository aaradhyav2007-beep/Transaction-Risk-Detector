import os

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

from from_scratch import (
    train_from_scratch,
    predict_probability,
    find_best_threshold,
    X_val_scaled,
    X_test_scaled,
    y_val_np,
    y_test_np
)


# =========================================================
# CREATE RESULTS DIRECTORY
# =========================================================

os.makedirs("results", exist_ok=True)


# =========================================================
# TRAIN MODEL
# =========================================================

w, b, cost_history = train_from_scratch()


# =========================================================
# VALIDATION PROBABILITIES
# =========================================================

val_probabilities = predict_probability(
    X_val_scaled,
    w,
    b
)


# =========================================================
# TEST PROBABILITIES
# =========================================================

test_probabilities = predict_probability(
    X_test_scaled,
    w,
    b
)


# =========================================================
# FIND BEST VALIDATION THRESHOLD
# =========================================================

best_threshold, best_validation_f1 = find_best_threshold(
    y_val_np,
    val_probabilities
)


print("Best validation threshold:", best_threshold)
print("Validation F1:", best_validation_f1)


# =========================================================
# 1. TRAINING COST CURVE
# =========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(cost_history) + 1),
    cost_history
)

plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Logistic Regression Training Cost")

plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "results/training_cost.png",
    dpi=200
)

plt.close()


# =========================================================
# 2. CONFUSION MATRIX
# =========================================================

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)


cm = confusion_matrix(
    y_test_np,
    test_predictions
)


plt.figure(figsize=(6, 5))

plt.imshow(cm, cmap="Blues")

plt.title(
    f"Confusion Matrix — Threshold {best_threshold:.2f}"
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Normal", "Fraud"]
)

plt.yticks(
    [0, 1],
    ["Normal", "Fraud"]
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=14
        )


plt.colorbar()

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=200
)

plt.close()


# =========================================================
# 3. THRESHOLD ANALYSIS
# =========================================================

thresholds = np.arange(
    0.10,
    0.91,
    0.01
)


precisions = []
recalls = []
f1_scores = []


for threshold in thresholds:

    predictions = (
        val_probabilities >= threshold
    ).astype(int)


    precision = precision_score(
        y_val_np,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_val_np,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_val_np,
        predictions,
        zero_division=0
    )


    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)


# =========================================================
# THRESHOLD PLOT
# =========================================================

plt.figure(figsize=(9, 5))

plt.plot(
    thresholds,
    precisions,
    label="Precision"
)

plt.plot(
    thresholds,
    recalls,
    label="Recall"
)

plt.plot(
    thresholds,
    f1_scores,
    label="F1 Score"
)


plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Selected threshold = {best_threshold:.2f}"
)


plt.xlabel("Decision Threshold")
plt.ylabel("Score")

plt.title(
    "Threshold Analysis on Validation Data"
)

plt.legend()

plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    "results/threshold_analysis.png",
    dpi=200
)

plt.close()


# =========================================================
# FINAL TEST METRICS
# =========================================================

test_precision = precision_score(
    y_test_np,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test_np,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test_np,
    test_predictions,
    zero_division=0
)


print("\nFINAL TEST RESULTS")
print("------------------")

print(
    "Threshold:",
    best_threshold
)

print(
    "Precision:",
    test_precision
)

print(
    "Recall:",
    test_recall
)

print(
    "F1 Score:",
    test_f1
)

print("\nResults saved to:")

print("results/training_cost.png")
print("results/confusion_matrix.png")
print("results/threshold_analysis.png")
print("results/final_test_metrics.txt")