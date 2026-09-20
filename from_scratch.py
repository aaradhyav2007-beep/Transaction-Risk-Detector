import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing import prepare_data

# =========================================================
# CONFIGURATION
# =========================================================

ALPHA = 0.1
LAMBDA = 0.01
NUM_ITERS = 2000
FRAUD_WEIGHT = 4

EPS = 1e-10

# =========================================================
# PREPARE DATA
# =========================================================

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
# 1. SIGMOID FUNCTION
# =========================================================

def sigmoid(z):
    """
    Convert model scores into probabilities between 0 and 1.
    """

    return 1 / (1 + np.exp(-z))

# =========================================================
# 2. WEIGHTED COST FUNCTION
# =========================================================

def compute_cost_weighted(
    X,
    y,
    w,
    b,
    lambda_,
    sample_weights
):
    """
    Calculate weighted Logistic Regression cost
    with L2 regularization.
    """

    m = X.shape[0]

    # Model prediction
    predictions = sigmoid(
        np.dot(X, w) + b
    )

    # Prevent log(0)
    predictions = np.clip(
        predictions,
        EPS,
        1 - EPS
    )

    # Logistic loss
    loss = -(
        y * np.log(predictions)
        + (1 - y) * np.log(1 - predictions)
    )

    # Give fraud transactions more importance
    weighted_loss = sample_weights * loss

    # Average loss
    cost = (
        1 / m
    ) * np.sum(weighted_loss)

    # L2 regularization
    regularization = (
        lambda_ / (2 * m)
    ) * np.sum(w ** 2)

    return cost + regularization

# =========================================================
# 3. WEIGHTED GRADIENT
# =========================================================

def compute_gradient_weighted(
    X,
    y,
    w,
    b,
    lambda_,
    sample_weights
):
    """
    Calculate gradients for weights and bias.
    """

    m = X.shape[0]

    # Model prediction
    predictions = sigmoid(
        np.dot(X, w) + b
    )

    # Prediction error
    error = predictions - y

    # Give fraud errors more importance
    weighted_error = (
        sample_weights * error
    )

    # Gradient of weights
    dj_dw = (
        1 / m
    ) * np.dot(
        X.T,
        weighted_error
    )

    # Gradient of bias
    dj_db = (
        1 / m
    ) * np.sum(
        weighted_error
    )

    # L2 regularization
    # Bias is NOT regularized.
    dj_dw = dj_dw + (
        lambda_ / m
    ) * w

    return dj_dw, dj_db

# =========================================================
# 4. GRADIENT DESCENT
# =========================================================

def gradient_descent_weighted(
    X,
    y,
    w,
    b,
    alpha,
    num_iters,
    lambda_,
    sample_weights
):
    """
    Train Logistic Regression using
    weighted gradient descent.
    """

    cost_history = []

    for _ in range(num_iters):

        # Calculate gradients
        dj_dw, dj_db = compute_gradient_weighted(
            X,
            y,
            w,
            b,
            lambda_,
            sample_weights
        )

        # Update weights
        w = w - alpha * dj_dw

        # Update bias
        b = b - alpha * dj_db

        # Calculate cost after update
        cost = compute_cost_weighted(
            X,
            y,
            w,
            b,
            lambda_,
            sample_weights
        )

        cost_history.append(cost)

    return w, b, cost_history

# =========================================================
# 5. TRAIN MODEL FROM SCRATCH
# =========================================================

def train_from_scratch():
    """
    Train the weighted Logistic Regression model.
    """

    # Start with zero weights
    w = np.zeros(
        X_train_scaled.shape[1]
    )

    # Start with zero bias
    b = 0.0

    # Assign higher weight to fraud transactions
    sample_weights = np.where(
        y_train_np == 1,
        FRAUD_WEIGHT,
        1
    )

    # Train using gradient descent
    w, b, cost_history = gradient_descent_weighted(
        X_train_scaled,
        y_train_np,
        w,
        b,
        ALPHA,
        NUM_ITERS,
        LAMBDA,
        sample_weights
    )

    return w, b, cost_history

# =========================================================
# 6. PREDICT PROBABILITY
# =========================================================

def predict_probability(X, w, b):
    """
    Return the probability of fraud.
    """

    return sigmoid(
        np.dot(X, w) + b
    )

# =========================================================
# 7. MAKE CLASS PREDICTIONS
# =========================================================

def predict(
    X,
    w,
    b,
    threshold
):
    """
    Convert probabilities into
    binary predictions.
    """

    probabilities = predict_probability(
        X,
        w,
        b
    )

    return (
        probabilities >= threshold
    ).astype(int)

# =========================================================
# 8. FIND BEST THRESHOLD
# =========================================================

def find_best_threshold(
    y_true,
    probabilities
):
    """
    Find the threshold that produces
    the highest F1 score on validation data.
    """

    best_threshold = 0.0
    best_f1 = 0.0

    for threshold in np.arange(
        0.10,
        0.90,
        0.01
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1

# =========================================================
# 9. EVALUATE MODEL
# =========================================================

def evaluate_model(
    y_true,
    predictions
):
    """
    Calculate classification metrics.
    """

    confusion = confusion_matrix(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    accuracy = np.mean(
        predictions == y_true
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion
    }

# =========================================================
# 10. TRAIN AND EVALUATE
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    w, b, cost_history = train_from_scratch()

    print(
        "Final cost:",
        cost_history[-1]
    )


    # -----------------------------------------------------
    # Find threshold using validation data
    # -----------------------------------------------------

    val_probabilities = predict_probability(
        X_val_scaled,
        w,
        b
    )

    best_threshold, validation_f1 = find_best_threshold(
        y_val_np,
        val_probabilities
    )

    print("\nVALIDATION RESULTS")
    print("-------------------")

    print(
        "Best threshold:",
        best_threshold
    )

    print(
        "Validation F1:",
        validation_f1
    )

    # -----------------------------------------------------
    # Final evaluation on test data
    # -----------------------------------------------------

    test_predictions = predict(
        X_test_scaled,
        w,
        b,
        best_threshold
    )

    results = evaluate_model(
        y_test_np,
        test_predictions
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\nFINAL TEST RESULTS")
    print("-------------------")

    print(
        "Threshold:",
        best_threshold
    )

    print(
        "Accuracy:",
        results["accuracy"]
    )

    print(
        "Precision:",
        results["precision"]
    )

    print(
        "Recall:",
        results["recall"]
    )

    print(
        "F1 Score:",
        results["f1"]
    )

    print("\nConfusion Matrix:")

    print(
        results["confusion_matrix"]
    )