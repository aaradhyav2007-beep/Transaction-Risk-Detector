import numpy as np
import pandas as pd
import streamlit as st

from from_scratch import (
    train_from_scratch,
    predict_probability,
    find_best_threshold,
    X_val_scaled,
    y_val_np,
    scaler,
    feature_columns
)

from preprocessing import preprocess_transaction

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Transaction Risk Detector",
    layout="centered"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    """
    Train and cache the Logistic Regression model.
    """

    w, b, cost_history = train_from_scratch()

    # Select decision threshold using validation data.
    val_probabilities = predict_probability(
        X_val_scaled,
        w,
        b
    )

    threshold, validation_f1 = find_best_threshold(
        y_val_np,
        val_probabilities
    )

    return (
        w,
        b,
        cost_history,
        threshold,
        validation_f1
    )


(
    w,
    b,
    cost_history,
    threshold,
    validation_f1
) = load_model()


# =========================================================
# TITLE
# =========================================================

st.title("Transaction Risk Detector")

st.write(
    "A Logistic Regression based system that estimates "
    "the probability that a transaction is suspicious."
)


# =========================================================
# TRANSACTION INPUTS
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

# =========================================================
# PREDICTION
# =========================================================

if st.button("Analyze Transaction"):

    # -----------------------------------------------------
    # Create transaction DataFrame
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Apply training preprocessing
    # -----------------------------------------------------

    transaction_scaled = preprocess_transaction(
        transaction,
        feature_columns,
        scaler
    )

    # -----------------------------------------------------
    # Calculate fraud probability
    # -----------------------------------------------------

    probability = predict_probability(
        transaction_scaled,
        w,
        b
    )

    probability = float(
        probability[0]
    )

    # -----------------------------------------------------
    # Convert probability into prediction
    # -----------------------------------------------------

    prediction = int(
        probability >= threshold
    )


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

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
        f"Decision threshold: **{threshold:.2f}**"
    )