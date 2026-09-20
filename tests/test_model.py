import numpy as np
import pytest
from from_scratch import (
    sigmoid,
    train_from_scratch,
    predict_probability,
    predict
)


# =========================================================
# TEST 1 — SIGMOID
# =========================================================

def test_sigmoid_zero():

    result = sigmoid(0)

    assert np.isclose(result, 0.5)


def test_sigmoid_range():

    values = np.array([
        -10,
        -1,
        0,
        1,
        10
    ])

    probabilities = sigmoid(values)

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


# =========================================================
# TEST 2 — MODEL TRAINING
# =========================================================

def test_model_training():

    w, b, cost_history = train_from_scratch()

    assert w is not None
    assert b is not None

    assert len(cost_history) > 0


# =========================================================
# TEST 3 — PREDICTION PROBABILITIES
# =========================================================

def test_prediction_probability_range():

    w, b, _ = train_from_scratch()

    X_sample = np.array([
        [0] * len(w)
    ])

    probabilities = predict_probability(
        X_sample,
        w,
        b
    )

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


# =========================================================
# TEST 4 — PREDICTION OUTPUT
# =========================================================

def test_prediction_output():

    w, b, _ = train_from_scratch()

    X_sample = np.array([
        [0] * len(w)
    ])

    predictions = predict(
        X_sample,
        w,
        b,
        threshold=0.38
    )

    assert predictions.shape == (1,)

    assert predictions[0] in [0, 1]