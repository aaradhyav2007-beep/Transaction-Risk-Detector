import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = "data/transaction_risk_detector_modified.csv"

# =========================================================
# 1. LOAD AND PREPARE FEATURES
# =========================================================

def load_and_prepare_data():
    """
    Load the transaction dataset and perform
    feature engineering and one-hot encoding.
    """

    df = pd.read_csv(DATA_PATH)

    # Transaction ID is not useful for prediction.
    df = df.drop(
        "transaction_id",
        axis=1
    )

    X = df.drop(
        "is_fraud",
        axis=1
    )

    y = df["is_fraud"]


    # -----------------------------------------------------
    # Amount ratio
    # -----------------------------------------------------

    X["amount_ratio"] = (
        X["transaction_amount"]
        / X["avg_transaction_amount_30d"].replace(
            0,
            np.nan
        )
    )

    X["amount_ratio"] = X["amount_ratio"].fillna(0)


    # -----------------------------------------------------
    # Cyclical hour features
    # -----------------------------------------------------

    X["hour_sin"] = np.sin(
        2 * np.pi * X["transaction_hour"] / 24
    )

    X["hour_cos"] = np.cos(
        2 * np.pi * X["transaction_hour"] / 24
    )


    # Original hour is no longer required.
    X = X.drop(
        "transaction_hour",
        axis=1
    )


    # -----------------------------------------------------
    # One-hot encoding
    # -----------------------------------------------------

    X = pd.get_dummies(
        X,
        columns=[
            "merchant_category",
            "payment_method"
        ],
        dtype=int
    )


    # Save feature order.
    feature_columns = X.columns.tolist()

    return X, y, feature_columns


# =========================================================
# 2. SPLIT DATA
# =========================================================

def split_data(X, y):
    """
    Split the dataset into training,
    validation, and test sets.
    """

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# =========================================================
# 3. SCALE DATA
# =========================================================

def scale_data(
    X_train,
    X_val,
    X_test
):
    """
    Scale features using StandardScaler.

    The scaler is fitted only on training data
    to prevent data leakage.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    )


# =========================================================
# 4. COMPLETE PREPROCESSING PIPELINE
# =========================================================

def prepare_data():
    """
    Complete preprocessing pipeline.

    Returns all prepared data required
    by the models and application.
    """

    X, y, feature_columns = load_and_prepare_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(X, y)

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        scaler
    ) = scale_data(
        X_train,
        X_val,
        X_test
    )

    return {
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,

        "y_train": y_train.to_numpy(),
        "y_val": y_val.to_numpy(),
        "y_test": y_test.to_numpy(),

        "feature_columns": feature_columns,

        "scaler": scaler
    }

# =========================================================
# 5. PREPROCESS A NEW TRANSACTION
# =========================================================

def preprocess_transaction(
    transaction,
    feature_columns,
    scaler
):
    """
    Apply the same feature engineering and preprocessing
    used during training to a new transaction.
    """

    transaction = transaction.copy()

    # -----------------------------------------------------
    # Amount ratio
    # -----------------------------------------------------

    transaction["amount_ratio"] = (
        transaction["transaction_amount"]
        / transaction["avg_transaction_amount_30d"].replace(
            0,
            np.nan
        )
    )

    transaction["amount_ratio"] = (
        transaction["amount_ratio"].fillna(0)
    )


    # -----------------------------------------------------
    # Cyclical hour features
    # -----------------------------------------------------

    transaction["hour_sin"] = np.sin(
        2 * np.pi * transaction["transaction_hour"] / 24
    )

    transaction["hour_cos"] = np.cos(
        2 * np.pi * transaction["transaction_hour"] / 24
    )


    # Original hour is no longer needed.
    transaction = transaction.drop(
        "transaction_hour",
        axis=1
    )


    # -----------------------------------------------------
    # One-hot encoding
    # -----------------------------------------------------

    transaction = pd.get_dummies(
        transaction,
        columns=[
            "merchant_category",
            "payment_method"
        ],
        dtype=int
    )


    # -----------------------------------------------------
    # Match training feature columns
    # -----------------------------------------------------

    for column in feature_columns:

        if column not in transaction.columns:
            transaction[column] = 0


    # Remove unexpected columns and
    # preserve the exact training order.
    transaction = transaction[
        feature_columns
    ]


    # -----------------------------------------------------
    # Scaling
    # -----------------------------------------------------

    transaction_scaled = scaler.transform(
        transaction
    )

    return transaction_scaled