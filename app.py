import pandas as pd
import streamlit as st

from textwrap import dedent

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
# HELPER — RENDER HTML
# =========================================================

def render_html(html):
    st.html(dedent(html))

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Transaction Risk Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background-color: #0b0f14;
        color: #f3f4f6;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1,
    h2,
    h3 {
        color: #f8fafc;
    }

    p {
        color: #aab4c0;
    }


    /* =====================================================
       HEADER
       ===================================================== */

    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;

        padding: 14px 18px;

        border: 1px solid #202936;
        border-radius: 12px;

        background: #111720;

        margin-bottom: 28px;
    }

    .brand {
        font-size: 14px;
        font-weight: 700;

        letter-spacing: 1.4px;

        color: #f8fafc;
    }

    .brand-subtitle {
        font-size: 11px;

        color: #718096;

        margin-top: 3px;

        letter-spacing: 1px;
    }

    .model-status {
        font-size: 11px;

        color: #79d99b;

        border: 1px solid #27553a;

        background: #102218;

        padding: 7px 11px;

        border-radius: 20px;

        letter-spacing: 0.7px;
    }


    /* =====================================================
       HERO
       ===================================================== */

    .hero-title {
        font-size: 38px;
        font-weight: 750;

        letter-spacing: -1px;

        margin-bottom: 5px;

        color: #f8fafc;
    }

    .hero-description {
        color: #8c98a8;

        font-size: 15px;

        max-width: 750px;

        line-height: 1.6;

        margin-bottom: 25px;
    }


    /* =====================================================
       MODEL INFORMATION CARDS
       ===================================================== */

    .info-card {
        background: #111720;

        border: 1px solid #202936;

        border-radius: 12px;

        padding: 16px;

        min-height: 86px;
    }

    .info-label {
        font-size: 10px;

        color: #6f7c8d;

        text-transform: uppercase;

        letter-spacing: 1.1px;

        margin-bottom: 8px;
    }

    .info-value {
        font-size: 20px;

        font-weight: 650;

        color: #f3f4f6;
    }

    .info-small {
        font-size: 11px;

        color: #7d8999;

        margin-top: 4px;
    }


    /* =====================================================
       SECTION HEADINGS
       ===================================================== */

    .section-heading {
        font-size: 13px;

        font-weight: 700;

        text-transform: uppercase;

        letter-spacing: 1.2px;

        color: #8f9aaa;

        margin-top: 32px;

        margin-bottom: 14px;
    }


    /* =====================================================
       RESULT PANEL
       ===================================================== */

    .result-panel {
        background: #111720;

        border: 1px solid #202936;

        border-radius: 16px;

        padding: 28px;

        margin-top: 20px;
    }

    .result-label {
        color: #7f8b9b;

        text-transform: uppercase;

        letter-spacing: 1.2px;

        font-size: 10px;
    }

    .risk-number {
        font-size: 58px;

        font-weight: 750;

        letter-spacing: -2px;

        line-height: 1;

        margin-top: 8px;
    }

    .risk-normal {
        color: #79d99b;
    }

    .risk-suspicious {
        color: #ff6b6b;
    }

    .decision-normal {
        color: #79d99b;

        font-weight: 700;

        font-size: 17px;

        margin-top: 12px;
    }

    .decision-suspicious {
        color: #ff6b6b;

        font-weight: 700;

        font-size: 17px;

        margin-top: 12px;
    }


    /* =====================================================
       RISK BAR
       ===================================================== */

    .risk-bar-container {
        margin-top: 25px;
    }

    .risk-bar {
        height: 8px;

        background: #202936;

        border-radius: 20px;

        position: relative;

        overflow: visible;
    }

    .risk-fill {
        height: 8px;

        border-radius: 20px;

        background: #4d8dff;
    }

    .threshold-marker {
        position: absolute;

        top: -6px;

        width: 2px;

        height: 20px;

        background: #f2c94c;
    }

    .risk-scale {
        display: flex;

        justify-content: space-between;

        margin-top: 9px;

        color: #657182;

        font-size: 10px;
    }


    /* =====================================================
       TRANSACTION SUMMARY CARDS
       ===================================================== */

    .signal-card {
        background: #0d131b;

        border: 1px solid #202936;

        border-radius: 10px;

        padding: 13px 15px;

        margin-bottom: 10px;
    }

    .signal-title {
        color: #d8dee7;

        font-size: 12px;

        font-weight: 650;
    }

    .signal-value {
        color: #7f8b9b;

        font-size: 11px;

        margin-top: 4px;
    }


    /* =====================================================
       BUTTON
       ===================================================== */

    .stButton > button {
        width: 100%;

        border-radius: 9px;

        border: 1px solid #326edb;

        background: #1d5fd1;

        color: white;

        font-weight: 700;

        padding: 12px;

        letter-spacing: 0.4px;
    }

    .stButton > button:hover {
        background: #2670ec;

        border-color: #4b8df0;
    }


    /* =====================================================
       INPUT LABELS
       ===================================================== */

    label {
        color: #c4ccd7 !important;
    }


    /* =====================================================
       EXPANDERS
       ===================================================== */

    [data-testid="stExpander"] {
        background: #111720;

        border: 1px solid #202936;

        border-radius: 12px;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {
        text-align: center;

        color: #4f5b6a;

        font-size: 10px;

        margin-top: 45px;

        letter-spacing: 0.5px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    """
    Train and cache the Logistic Regression model.

    The decision threshold is selected using
    validation data.
    """

    w, b, cost_history = train_from_scratch()

    # Get validation probabilities.
    val_probabilities = predict_probability(
        X_val_scaled,
        w,
        b
    )

    # Select threshold using validation F1.
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
# HEADER
# =========================================================

render_html(
    """
    <div class="top-bar">

        <div>
            <div class="brand">
                🛡 TRANSACTION RISK DETECTOR
            </div>

            <div class="brand-subtitle">
                FRAUD INTELLIGENCE CONSOLE
            </div>
        </div>

        <div class="model-status">
            ● MODEL ONLINE
        </div>

    </div>
    """
)


# =========================================================
# HERO
# =========================================================

render_html(
    """
    <div class="hero-title">
        Analyze a transaction.
    </div>

    <div class="hero-description">
        Evaluate transaction risk using a class-weighted
        Logistic Regression model with engineered behavioral
        and transaction features.
    </div>
    """
)


# =========================================================
# MODEL STATUS
# =========================================================

render_html(
    """
    <div class="section-heading">
        Model Status
    </div>
    """
)

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(
    4,
    gap="medium"
)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

with summary_col1:

    render_html(
        """
        <div class="info-card">

            <div class="info-label">
                Model
            </div>

            <div class="info-value">
                Logistic Regression
            </div>

            <div class="info-small">
                Implemented from scratch
            </div>

        </div>
        """
    )


# ---------------------------------------------------------
# FRAUD WEIGHT
# ---------------------------------------------------------

with summary_col2:

    render_html(
        """
        <div class="info-card">

            <div class="info-label">
                Fraud Weight
            </div>

            <div class="info-value">
                4×
            </div>

            <div class="info-small">
                Class-weighted training
            </div>

        </div>
        """
    )


# ---------------------------------------------------------
# THRESHOLD
# ---------------------------------------------------------

with summary_col3:

    render_html(
        f"""
        <div class="info-card">

            <div class="info-label">
                Threshold
            </div>

            <div class="info-value">
                {threshold:.2f}
            </div>

            <div class="info-small">
                Selected using validation F1
            </div>

        </div>
        """
    )


# ---------------------------------------------------------
# VALIDATION F1
# ---------------------------------------------------------

with summary_col4:

    render_html(
        f"""
        <div class="info-card">

            <div class="info-label">
                Validation F1
            </div>

            <div class="info-value">
                {validation_f1:.3f}
            </div>

            <div class="info-small">
                Held-out validation set
            </div>

        </div>
        """
    )


# =========================================================
# TRANSACTION PROFILE
# =========================================================

render_html(
    """
    <div class="section-heading">
        Transaction Profile
    </div>
    """
)


# =========================================================
# TRANSACTION DETAILS
# =========================================================

with st.expander(
    "💳  Transaction",
    expanded=True
):

    col1, col2, col3 = st.columns(3)

    with col1:

        transaction_amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=100.0
        )

    with col2:

        transaction_hour = st.slider(
            "Transaction Hour",
            min_value=0,
            max_value=23,
            value=12
        )

    with col3:

        is_weekend = st.selectbox(
            "Weekend?",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )


# =========================================================
# ACCOUNT ACTIVITY
# =========================================================

with st.expander(
    "👤  Account Activity",
    expanded=True
):

    col1, col2, col3 = st.columns(3)

    with col1:

        acc_age_days = st.number_input(
            "Account Age (days)",
            min_value=0,
            value=365
        )

    with col2:

        num_transactions_last_24h = st.number_input(
            "Transactions — Last 24h",
            min_value=0,
            value=5
        )

    with col3:

        avg_transaction_amount_30d = st.number_input(
            "Average Amount — 30d",
            min_value=0.0,
            value=100.0
        )


# =========================================================
# LOCATION & PAYMENT
# =========================================================

with st.expander(
    "🌍  Location & Payment",
    expanded=False
):

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        distance_from_home_km = st.number_input(
            "Distance From Home (km)",
            min_value=0.0,
            value=5.0
        )

    with col2:

        is_foreign_transaction = st.selectbox(
            "Foreign?",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

    with col3:

        merchant_category = st.selectbox(
            "Merchant",
            [
                "Electronics",
                "Clothing",
                "Food",
                "Travel"
            ]
        )

    with col4:

        payment_method = st.selectbox(
            "Payment",
            [
                "Credit Card",
                "Debit Card",
                "Cash",
                "Online Transfer"
            ]
        )


# =========================================================
# SECURITY SIGNALS
# =========================================================

with st.expander(
    "🔐  Security Signals",
    expanded=False
):

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        num_failed_attempts_last_hour = st.number_input(
            "Failed Attempts",
            min_value=0,
            value=0
        )

    with col2:

        is_new_device = st.selectbox(
            "New Device?",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

    with col3:

        is_new_merchant = st.selectbox(
            "New Merchant?",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

    with col4:

        card_present = st.selectbox(
            "Card Present?",
            [0, 1],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )


# =========================================================
# ANALYZE BUTTON
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

analyze = st.button(
    "ANALYZE TRANSACTION"
)


# =========================================================
# PREDICTION
# =========================================================

if analyze:

    # -----------------------------------------------------
    # Create transaction DataFrame
    # -----------------------------------------------------

    transaction = pd.DataFrame({

        "transaction_amount": [
            transaction_amount
        ],

        "transaction_hour": [
            transaction_hour
        ],

        "is_weekend": [
            is_weekend
        ],

        "acc_age_days": [
            acc_age_days
        ],

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
    # Calculate probability
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
    # Classification
    # -----------------------------------------------------

    prediction = int(
        probability >= threshold
    )


    # =====================================================
    # RISK ASSESSMENT
    # =====================================================

    render_html(
        """
        <div class="section-heading">
            Risk Assessment
        </div>
        """
    )


    if prediction == 1:

        risk_class = "risk-suspicious"

        decision_class = "decision-suspicious"

        decision_text = (
            "⚠ SUSPICIOUS TRANSACTION"
        )

    else:

        risk_class = "risk-normal"

        decision_class = "decision-normal"

        decision_text = (
            "✓ NORMAL TRANSACTION"
        )


    # -----------------------------------------------------
    # Risk result panel
    # -----------------------------------------------------

    render_html(
        f"""
        <div class="result-panel">

            <div class="result-label">
                Estimated Fraud Probability
            </div>

            <div class="risk-number {risk_class}">
                {probability * 100:.2f}%
            </div>

            <div class="{decision_class}">
                {decision_text}
            </div>

            <div class="risk-bar-container">

                <div class="risk-bar">

                    <div
                        class="risk-fill"
                        style="width: {probability * 100:.2f}%"
                    ></div>

                    <div
                        class="threshold-marker"
                        style="left: {threshold * 100:.2f}%"
                    ></div>

                </div>

                <div class="risk-scale">

                    <span>
                        0%
                    </span>

                    <span>
                        Decision threshold: {threshold:.0%}
                    </span>

                    <span>
                        100%
                    </span>

                </div>

            </div>

        </div>
        """
    )

# =====================================================
# TRANSACTION SIGNALS
# =====================================================

render_html(
    """
    <div class="section-heading">
        Transaction Signals
    </div>
    """
)


# Calculate the transaction's amount ratio.
if avg_transaction_amount_30d > 0:
    amount_ratio = (
        transaction_amount
        / avg_transaction_amount_30d
    )
else:
    amount_ratio = 0


signal_col1, signal_col2, signal_col3 = st.columns(3)


# -----------------------------------------------------
# Amount Ratio
# -----------------------------------------------------

with signal_col1:

    if amount_ratio >= 3:
        amount_status = "Above typical amount"
    else:
        amount_status = "Within typical range"

    render_html(
        f"""
        <div class="signal-card">

            <div class="signal-title">
                Transaction / Average Amount
            </div>

            <div class="signal-value">
                {amount_ratio:.2f}×
            </div>

            <div class="signal-value">
                {amount_status}
            </div>

        </div>
        """
    )


# -----------------------------------------------------
# Device Status
# -----------------------------------------------------

with signal_col2:

    device_status = (
        "New device detected"
        if is_new_device == 1
        else "Existing device"
    )

    render_html(
        f"""
        <div class="signal-card">

            <div class="signal-title">
                Device Status
            </div>

            <div class="signal-value">
                {device_status}
            </div>

        </div>
        """
    )


# -----------------------------------------------------
# Foreign Transaction
# -----------------------------------------------------

with signal_col3:

    foreign_status = (
        "Foreign transaction"
        if is_foreign_transaction == 1
        else "Domestic transaction"
    )

    render_html(
        f"""
        <div class="signal-card">

            <div class="signal-title">
                Transaction Location
            </div>

            <div class="signal-value">
                {foreign_status}
            </div>

        </div>
        """
    )


signal_col4, signal_col5, signal_col6 = st.columns(3)


# -----------------------------------------------------
# Failed Attempts
# -----------------------------------------------------

with signal_col4:

    render_html(
        f"""
        <div class="signal-card">

            <div class="signal-title">
                Failed Attempts — Last Hour
            </div>

            <div class="signal-value">
                {num_failed_attempts_last_hour}
            </div>

        </div>
        """
    )


# -----------------------------------------------------
# New Merchant
# -----------------------------------------------------

with signal_col5:

    merchant_status = (
        "New merchant"
        if is_new_merchant == 1
        else "Existing merchant"
    )

    render_html(
        f"""
        <div class="signal-card">

            <div class="signal-title">
                Merchant Status
            </div>

            <div class="signal-value">
                {merchant_status}
            </div>

        </div>
        """
    )


# -----------------------------------------------------
# Distance From Home
# -----------------------------------------------------

with signal_col6:

    render_html(
        f"""
        <div class="signal-card">

            <div class="signal-title">
                Distance From Home
            </div>

            <div class="signal-value">
                {distance_from_home_km:.1f} km
            </div>

        </div>
        """
    )
    # =====================================================
    # TRANSACTION SUMMARY
    # =====================================================

    render_html(
        """
        <div class="section-heading">
            Transaction Summary
        </div>
        """
    )


    col1, col2, col3, col4 = st.columns(4)


    # -----------------------------------------------------
    # Amount
    # -----------------------------------------------------

    with col1:

        render_html(
            f"""
            <div class="signal-card">

                <div class="signal-title">
                    Transaction Amount
                </div>

                <div class="signal-value">
                    ₹ {transaction_amount:,.2f}
                </div>

            </div>
            """
        )


    # -----------------------------------------------------
    # Merchant
    # -----------------------------------------------------

    with col2:

        render_html(
            f"""
            <div class="signal-card">

                <div class="signal-title">
                    Merchant
                </div>

                <div class="signal-value">
                    {merchant_category}
                </div>

            </div>
            """
        )


    # -----------------------------------------------------
    # Payment
    # -----------------------------------------------------

    with col3:

        render_html(
            f"""
            <div class="signal-card">

                <div class="signal-title">
                    Payment Method
                </div>

                <div class="signal-value">
                    {payment_method}
                </div>

            </div>
            """
        )


    # -----------------------------------------------------
    # Hour
    # -----------------------------------------------------

    with col4:

        render_html(
            f"""
            <div class="signal-card">

                <div class="signal-title">
                    Transaction Hour
                </div>

                <div class="signal-value">
                    {transaction_hour:02d}:00
                </div>

            </div>
            """
        )


    # =====================================================
    # MODEL NOTE
    # =====================================================

    st.caption(
        "The probability is produced by the trained Logistic "
        "Regression model. The decision is determined by the "
        "validation-selected threshold."
    )


# =========================================================
# FOOTER
# =========================================================

render_html(
    """
    <div class="footer">
        TRANSACTION RISK DETECTOR • LOGISTIC REGRESSION •
        EVIDENCE-DRIVEN CLASSIFICATION
    </div>
    """
)