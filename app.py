import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear.model import LogisticRegression

# --------my-------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/transaction_risk_detector_modified.csv")

# Uncomment to inspect class balance
# df["is_fraud"].value_counts().plot(kind="bar")
# plt.title("Fraud vs Normal Transaction")
# plt.xlabel("Transaction type")
# plt.ylabel("No of Transactions")
# plt.xticks([0, 1], ["Normal", "Fraud"], rotation=0)
# plt.show()

df = df.drop("transaction_id", axis=1)

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# One-hot encode categoricals
X = pd.get_dummies(X, columns=["merchant_category", "payment_method"], dtype=int)

# -------------------------------------------------
# Feature Engineering
# -------------------------------------------------

# How large is this transaction compared
# with the customer's recent average?
X["amount_ratio"] = (
    X["transaction_amount"] /
    X["avg_transaction_amount_30d"].replace(0, np.nan)
)

X["amount_ratio"] = X["amount_ratio"].fillna(0)

# Represent hour cyclically
X["hour_sin"] = np.sin(
    2 * np.pi * X["transaction_hour"] / 24
)
X["hour_cos"] = np.cos(
    2 * np.pi * X["transaction_hour"] / 24
)
# Original hour is no longer needed
X = X.drop("transaction_hour", axis=1)

# ---------------------------------------------------------------------------
# 2. Train/test split
# ---------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
# ---------------------------------------------------------------------------
# 3. Scale (fit only on train -> avoid data leakage)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

y_train_np = y_train.to_numpy()
y_val_np = y_val.to_numpy()
y_test_np = y_test.to_numpy()

EPS = 1e-10  # numerical stability guard for log()

normal_weight = 1
fraud_weight = 4

sample_wights = np.where(y_train_np == 1,4,1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, w, b):
    m = X.shape[0]
    f_wb = np.clip(sigmoid(np.dot(X, w) + b), EPS, 1 - EPS)
    cost = -(1 / m) * np.sum(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))
    return cost

def compute_cost_weighted(X,y,w,b,lambda_,sample_weights):
    m = X.shape[0]
    f_wb = np.clip(sigmoid(np.dot(X,w)+b),EPS,1-EPS)
    loss = -(y*np.log(f_wb)+(1-y)*np.log(1-f_wb))
    weighted_loss = sample_weights * loss
    cost = (1/m) * np.sum(weighted_loss)
    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)
    return cost + reg_cost

def compute_gradient_weighted(X, y, w, b, lambda_, sample_weights):
    m = X.shape[0]
    f_wb = sigmoid(np.dot(X,w)+b)
    error = f_wb - y
    weighted_error = sample_weights * error
    dj_dw = (1 / m) * np.dot(X.T, weighted_error)
    dj_db = (1 / m) * np.sum(weighted_error)
    dj_dw = dj_dw + (lambda_ / m) * w  # regularize weights only, not bias
    return dj_dw, dj_db

def gradient_descent_weighted(X, y, w, b, alpha, num_iters, lambda_, sample_weights):
    cost_history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_weighted(X, y, w, b, lambda_, sample_weights)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost = compute_cost_weighted(X, y, w, b, lambda_, sample_weights)
        cost_history.append(cost)
    return w, b, cost_history

def compute_cost_reg(X, y, w, b, lambda_):
    m = X.shape[0]
    f_wb = np.clip(sigmoid(np.dot(X, w) + b), EPS, 1 - EPS)
    cost = -(1 / m) * np.sum(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))
    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)
    return cost + reg_cost


def compute_gradient_reg(X, y, w, b, lambda_):
    m = X.shape[0]
    f_wb = sigmoid(np.dot(X, w) + b)
    error = f_wb - y

    dj_dw = (1 / m) * np.dot(X.T, error)
    dj_db = (1 / m) * np.sum(error)

    dj_dw = dj_dw + (lambda_ / m) * w  # regularize weights only, not bias

    return dj_dw, dj_db


def gradient_descent(X, y, w, b, alpha, num_iters, lambda_):
    cost_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_reg(X, y, w, b, lambda_)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost_reg(X, y, w, b, lambda_)
        cost_history.append(cost)

    return w, b, cost_history

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
alpha = 0.1    
lambda_ = 0.01
num_iters = 2000

w = np.zeros(X_train_scaled.shape[1])
b = 0.0

sample_weights = np.where(y_train_np == 1,4,1)
w, b, cost_history = gradient_descent_weighted(
    X_train_scaled, y_train_np, w, b, alpha, num_iters, lambda_, sample_weights
)

print("Final cost:", cost_history[-1])

# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------
def predict_probability(X, w, b):
    return sigmoid(np.dot(X, w) + b)


# compute validation probabilities (kept for inspection if needed)
val_probabilities = predict_probability(
    X_val_scaled,
    w,
    b
)

# -------------------------------------------------
# Final evaluation on untouched test set
# -------------------------------------------------
best_threshold = 0.38  # You can adjust this threshold based on your validation results
test_probabilities = predict_probability(
    X_test_scaled,
    w,
    b
)

final_predictions = (
    test_probabilities >= best_threshold
).astype(int)

cm = confusion_matrix(
    y_test_np,
    final_predictions
)

precision = precision_score(
    y_test_np,
    final_predictions,
    zero_division=0
)

recall = recall_score(
    y_test_np,
    final_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test_np,
    final_predictions,
    zero_division=0
)

accuracy = np.mean(
    final_predictions == y_test_np
)

print("\nFINAL TEST RESULTS")
print("-------------------")
print("Threshold:", best_threshold)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nConfusion Matrix:")
print(cm)