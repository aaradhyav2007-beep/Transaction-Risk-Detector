import streamlit as st
import pandas as pd
import numpy as np

from from_scratch import (
    train_from_scratch,
    scaler,
    feature_columns,
)
# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="Transaction Risk Detector",
    layout="centered"
)

# 2. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    w, b, cost_history = train_from_scratch()
    return w, b, cost_history


w, b, cost_history = load_model()

# 3. TITLE

st.title("Transaction Risk Detector")

st.write(
    "A Logistic Regression based system that estimates "
    "the probability that a transaction is suspicious."
)


# =========================================================
# 4. TRANSACTION INPUTS
# =========================================================

st.header("Transaction Details")


transaction_amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0
)


transaction_hour = st.slider(
    "Transaction Hour",
    min_value=0,
    max_value=23,
    value=12
)


is_weekend = st.selectbox(
    "Is Weekend?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)


acc_age_days = st.number_input(
    "Account Age (days)",
    min_value=0,
    value=365
)


num_transactions_last_24h = st.number_input(
    "Number of Transactions in Last 24 Hours",
    min_value=0,
    value=5
)


avg_transaction_amount_30d = st.number_input(
    "Average Transaction Amount (30 days)",
    min_value=0.0,
    value=100.0
)


distance_from_home_km = st.number_input(
    "Distance From Home (km)",
    min_value=0.0,
    value=5.0
)


is_foreign_transaction = st.selectbox(
    "Foreign Transaction?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)


merchant_category = st.selectbox(
    "Merchant Category",
    [
        "Electronics",
        "Clothing",
        "Food",
        "Travel"
    ]
)


payment_method = st.selectbox(
    "Payment Method",
    [
        "Credit Card",
        "Debit Card",
        "Cash",
        "Online Transfer"
    ]
)


num_failed_attempts_last_hour = st.number_input(
    "Failed Attempts in Last Hour",
    min_value=0,
    value=0
)


is_new_device = st.selectbox(
    "New Device?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)


is_new_merchant = st.selectbox(
    "New Merchant?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)


card_present = st.selectbox(
    "Card Present?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# 5. PREDICTION

if st.button("Analyze Transaction"):

    # Create raw transaction

    transaction = pd.DataFrame({
        "transaction_amount": [transaction_amount],
        "transaction_hour": [transaction_hour],
        "is_weekend": [is_weekend],
        "acc_age_days": [acc_age_days],
        "num_transactions_last_24h": [
            num_transactions_last_24h
        ],
        "avg_transaction_amount_30d": [
            avg_transaction_amount_30d
        ],
        "distance_from_home_km": [
            distance_from_home_km
        ],
        "is_foreign_transaction": [
            is_foreign_transaction
        ],
        "merchant_category": [
            merchant_category
        ],
        "payment_method": [
            payment_method
        ],
        "num_failed_attempts_last_hour": [
            num_failed_attempts_last_hour
        ],
        "is_new_device": [
            is_new_device
        ],
        "is_new_merchant": [
            is_new_merchant
        ],
        "card_present": [
            card_present
        ]
    })

    # Feature Engineering

    # 1. Transaction amount compared with user's average
    if avg_transaction_amount_30d > 0:
        transaction["amount_ratio"] = (
            transaction["transaction_amount"]
            / transaction["avg_transaction_amount_30d"]
        )
    else:
        transaction["amount_ratio"] = 0


    # 2. Cyclical representation of transaction hour
    transaction["hour_sin"] = np.sin(
        2 * np.pi * transaction["transaction_hour"] / 24
    )

    transaction["hour_cos"] = np.cos(
        2 * np.pi * transaction["transaction_hour"] / 24
    )

    # We no longer need the original hour
    transaction = transaction.drop(
        "transaction_hour",
        axis=1
    )

    # One-hot encode categorical features

    transaction = pd.get_dummies(
        transaction,
        columns=[
            "merchant_category",
            "payment_method"
        ],
        dtype=int
    )

    # Make feature columns identical to training data

    for column in feature_columns:

        if column not in transaction.columns:
            transaction[column] = 0


    # Remove any unexpected columns
    transaction = transaction[feature_columns]


    # -----------------------------------------------------
    # Scale using the SAME scaler used during training
    # -----------------------------------------------------

    transaction_scaled = scaler.transform(
        transaction
    )

    # Logistic Regression prediction

    z = np.dot(
        transaction_scaled,
        w
    ) + b

    probability = 1 / (
        1 + np.exp(-z)
    )

    probability = float(probability[0])

    # Classification threshold

    threshold = 0.38

    prediction = int(
        probability >= threshold
    )
    # 6. DISPLAY RESULTS

    st.divider()

    st.subheader("Prediction Results")


    st.metric(
        "Risk Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:

        st.error(
            "⚠️ SUSPICIOUS TRANSACTION"
        )

    else:

        st.success(
            "✅ NORMAL TRANSACTION"
        )


    st.write(
        f"Decision threshold: **{threshold}**"
    )