# Transaction Risk Detector

A machine learning system for detecting potentially fraudulent financial transactions using Logistic Regression, with both a from-scratch implementation and a scikit-learn reference model.

The project focuses on understanding the complete machine learning workflow rather than treating the model as a black box:

**Data → Feature Engineering → Preprocessing → Logistic Regression → Threshold Selection → Evaluation → Streamlit Application**

---

## Overview

Transaction fraud detection is a binary classification problem where fraudulent transactions are relatively rare compared with normal transactions.

This project explores how Logistic Regression can be used for fraud detection while addressing practical machine learning concerns such as:

- Class imbalance
- Missing values
- Categorical features
- Feature scaling
- Feature engineering
- L2 regularization
- Class weighting
- Decision threshold selection
- Precision/Recall trade-offs
- Model evaluation
- Reproducible preprocessing
- Automated testing
- Deployment through Streamlit

The Logistic Regression model is implemented **from scratch using NumPy** to understand the underlying mathematics and optimization process. A scikit-learn implementation is also trained as a reference point.

---

## Key Objectives

The project was designed to answer several practical questions:

1. How does Logistic Regression work internally?
2. How does class imbalance affect fraud detection?
3. Why can accuracy be misleading for imbalanced classification?
4. How does class weighting affect the model?
5. How should a classification threshold be selected?
6. Can a from-scratch implementation reproduce the behavior of a trusted library implementation?
7. How can the trained model be integrated into an interactive application?

---
<img width="1326" height="492" alt="Screenshot 2026-09-20 at 6 54 38 PM" src="https://github.com/user-attachments/assets/913c6ab8-9f0a-40a0-85c5-915adac5f43c" />
<img width="1098" height="771" alt="Screenshot 2026-09-20 at 6 50 22 PM" src="https://github.com/user-attachments/assets/3c7ebc56-0432-48fb-b793-29a02f5dc8bb" />
<img width="1125" height="757" alt="Screenshot 2026-09-20 at 6 51 07 PM" src="https://github.com/user-attachments/assets/773e25b1-360a-4517-b89e-68b0ca41ce65" />
<img width="1179" height="409" alt="Screenshot 2026-09-20 at 6 56 41 PM" src="https://github.com/user-attachments/assets/1214aea1-5a57-4ded-9e94-b4facbaf9961" />


## System Architecture

```text
                    Transaction Data
                           │
                           ▼
                 ┌────────────────────┐
                 │ Data Preprocessing │
                 └─────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Feature Engineering │
                │                     │
                │ • Amount Ratio      │
                │ • Cyclical Time     │
                │ • One-Hot Encoding  │
                └──────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │   Train / Val /    │
                 │       Test Split   │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │   StandardScaler   │
                 │  Fit on Train Only │
                 └─────────┬──────────┘
                           │
                ┌──────────┴───────────┐
                ▼                      ▼
      ┌──────────────────┐   ┌──────────────────┐
      │ From-Scratch     │   │ Scikit-Learn     │
      │ Logistic         │   │ Logistic         │
      │ Regression       │   │ Regression       │
      └────────┬─────────┘   └────────┬─────────┘
               │                      │
               └──────────┬───────────┘
                          ▼
                ┌─────────────────────┐
                │ Validation          │
                │ Threshold Selection │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Locked Test         │
                │ Evaluation          │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Streamlit           │
                │ Application         │
                └─────────────────────┘

Dataset
The dataset contains transaction-level information used to classify transactions as normal or fraudulent.
Features
Feature	Description
transaction_id	Unique transaction identifier
transaction_amount	Transaction amount
transaction_hour	Hour at which the transaction occurred
is_weekend	Whether the transaction occurred on a weekend
account_age_days	Age of the account in days
num_transactions_last_24h	Number of recent transactions
avg_transaction_amount_30d	Average transaction amount over the previous 30 days
distance_from_home_km	Distance from the account holder's home
is_foreign_transaction	Whether the transaction is foreign
merchant_category	Merchant category
payment_method	Payment method
num_failed_attempts_last_hour	Failed attempts during the previous hour
is_new_device	Whether a new device was used
is_new_merchant	Whether the merchant is new
card_present	Whether the physical card was present
is_fraud	Target variable


Target:
0 = Normal transaction
1 = Fraudulent transaction

Class Distribution
The dataset contains:
- 4437 normal transactions
- 563 fraudulent transactions
- 11.26% fraud rate
This imbalance makes accuracy alone an insufficient metric for evaluating the model.
For example, a model could obtain high accuracy while detecting very few fraudulent transactions.

Therefore, this project focuses primarily on:
- Precision
- Recall
- F1 Score
- Confusion Matrix
Feature Engineering
Several features are transformed before training.
Transaction Amount Ratio

A relative spending feature is created:
amount_ratio =
transaction_amount / avg_transaction_amount_30d
This provides information about how unusual the current transaction amount is relative to the customer's recent behavior.
Zero denominators are handled safely during preprocessing.
Cyclical Time Encoding
Transaction hour is converted into two cyclical features:
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
This represents the circular nature of time.
For example:
23:00 → close to 00:00
A simple numerical representation of the hour would incorrectly treat these times as far apart.
The original transaction_hour feature is removed after the transformation.
Categorical Encoding
Categorical features are converted using one-hot encoding:
merchant_category
payment_method
This allows them to be used by the Logistic Regression model.
Data Preprocessing
The preprocessing pipeline follows these steps:
1. Load dataset
2. Remove transaction_id
3. Create engineered features
4. Encode categorical features
5. Split into train / validation / test sets
6. Fit StandardScaler on training data
7. Transform validation and test data
The dataset is split into:
80% Training
10% Validation
10% Test
The split is stratified to preserve the class distribution.
A fixed random seed is used for reproducibility.
Feature Scaling
Numerical features have different ranges.
For example:
transaction_amount
account_age_days
distance_from_home_km
num_transactions_last_24h
Standardization is therefore applied using:
z = (x - μ) / σ
The scaler is fitted only on the training data.
Validation and test data are transformed using the parameters learned from the training set.
This prevents information from the validation or test sets from influencing preprocessing.
Logistic Regression From Scratch
The core Logistic Regression model is implemented using NumPy.
The model calculates:
z = Xw + b
where:
- X = input features
- w = model weights
- b = bias
The probability is then calculated using the sigmoid function.
Sigmoid Function
σ(z) = 1 / (1 + e^(-z))
The sigmoid converts the model's raw score into a probability between 0 and 1.
z → sigmoid(z) → probability
For example:
0.10 → low estimated fraud probability
0.80 → high estimated fraud probability
Binary Cross-Entropy Loss
The base Logistic Regression loss is:
J =
-(1/m) Σ [
    y log(f)
    +
    (1-y) log(1-f)
]
where:
- m = number of training examples
- y = actual label
- f = predicted probability
The implementation clips probabilities using a small epsilon value before applying logarithms to avoid numerical issues.
L2 Regularization
L2 regularization is added to reduce excessively large model weights.
The regularized objective is:
J_reg =
J +
(λ / 2m) Σ w²
The regularization term is applied to the feature weights.
The bias term is not regularized.
The corresponding gradient contribution is:
(λ / m) w
Class Weighting
Fraudulent transactions are the minority class.
To make the model pay more attention to fraudulent examples, class weighting is used:
class_weight = {
    0: 1,
    1: 4
}

Therefore:
Normal transaction → weight 1
Fraudulent transaction → weight 4
This changes the contribution of each training example during optimization.
The goal is not simply to maximize overall accuracy, but to improve the model's ability to identify the minority fraud class.
Gradient Descent
The model parameters are optimized using gradient descent.
The prediction error is:
error = f - y
The weight gradient is:
dj_dw = (1/m) Xᵀ(error)
The bias gradient is:
dj_db = mean(error)
With L2 regularization:
dj_dw += (λ/m)w
The parameters are updated using:
w = w - α × dj_dw

b = b - α × dj_db
where:
α = learning rate
Model Configuration
The current from-scratch model uses:
Learning rate (alpha): 0.1
L2 regularization (lambda): 0.01
Training iterations: 2000
Fraud class weight: 4
Normal class weight: 1
These values are part of the current experiment configuration and are not presented as universally optimal hyperparameters.
Decision Threshold
Logistic Regression produces probabilities.
The default classification rule is commonly:
probability >= 0.50 → class 1
probability <  0.50 → class 0
For fraud detection, the threshold can be changed depending on the desired precision/recall trade-off.
Instead of selecting the threshold using the test set, this project uses the validation set.
Procedure
Train model
     ↓
Generate validation probabilities
     ↓
Evaluate multiple thresholds
     ↓
Select threshold using validation F1
     ↓
Lock threshold
     ↓
Evaluate once on test set
The selected validation threshold for the current experiment is:
0.38
The test set was not used to select this threshold.
Evaluation
Final test-set evaluation at threshold 0.38:
Metric	Result
Accuracy	0.7020
Precision	0.2215
Recall	0.5738
F1 Score	0.3196


These results show the trade-off involved in fraud detection.
The model identifies a meaningful portion of fraudulent transactions, but the relatively low precision means that a significant number of normal transactions are also classified as suspicious.
This is an important limitation of the current model rather than something hidden by reporting accuracy alone.
Confusion Matrix
The current test confusion matrix is:
                  Predicted
                Normal   Fraud
Actual Normal     316     123
Actual Fraud       26      35
Interpretation:
316 → Normal transactions correctly classified
123 → Normal transactions incorrectly flagged
26  → Fraudulent transactions missed
35  → Fraudulent transactions correctly detected
The confusion matrix is useful because it exposes the types of mistakes that a single accuracy value cannot show.
Threshold Analysis
The project evaluates multiple classification thresholds on the validation set.
The analysis compares:
Precision
Recall
F1 Score
across different threshold values.
The selected threshold is:
0.38
because it produced the highest validation F1 score among the evaluated thresholds.
The threshold is then kept fixed for final test evaluation.
Training Convergence
The training process records the loss during optimization.
The resulting training curve is stored at:
results/training_cost.png
The curve shows a rapid reduction in the initial loss followed by a relatively stable region as optimization progresses.
This provides a visual check that the optimization process is behaving as expected.
From-Scratch vs Scikit-Learn
A second Logistic Regression implementation using scikit-learn is included as a reference implementation.
The purpose is not to replace the from-scratch implementation, but to provide an independent comparison.
The from-scratch implementation and the scikit-learn implementation produce closely aligned model parameters and evaluation behavior under comparable preprocessing and class-weighting settings.
This provides a practical validation that the mathematical implementation is behaving consistently with a widely used machine learning library.
Streamlit Application
The project includes an interactive Streamlit interface.
The application provides a dark fintech-style interface for entering transaction information and receiving a model prediction.
The interface includes:
Transaction
- Transaction amount
- Transaction hour
- Weekend status
Account Activity
- Account age
- Number of recent transactions
- Average transaction amount
Location & Payment
- Distance from home
- Foreign transaction status
- Merchant category
- Payment method
Security Signals
- Failed attempts
- New device
- New merchant
- Card present
The application displays:
- Estimated fraud probability
- Suspicious / normal classification
- Risk visualization
- Decision threshold
- Transaction summary
- Transaction signals
The displayed signals describe transaction characteristics and are not presented as model feature-attribution explanations.
Project Structure
Transaction Risk Detector/
│
├── app.py
├── from_scratch.py
├── preprocessing.py
├── sklearn_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── transaction_risk_detector_modified.csv
│
├── results/
│   ├── __init__.py
│   ├── generate_results.py
│   ├── training_cost.png
│   ├── confusion_matrix.png
│   ├── threshold_analysis.png
│   └── final_test_metrics.txt
│
└── tests/
    └── test_model.py
File Responsibilities
preprocessing.py
Contains shared preprocessing and feature engineering logic.

Responsibilities include:
- Feature creation
- Categorical encoding
- Train/validation/test splitting
- Feature scaling
- Feature column management
from_scratch.py
Contains the NumPy-based Logistic Regression implementation.
Responsibilities include:
- Sigmoid
- Loss calculation
- L2 regularization
- Gradient calculation
- Gradient descent
- Class weighting
- Probability prediction
- Threshold selection
sklearn_model.py(It is used to compare the custom implementation against a standard library implementation.)
app.py
Contains the Streamlit application.
It provides an interactive interface for submitting transaction information and viewing model predictions.
results/generate_results.py

Reproduces the model evaluation workflow and generates:
training_cost.png
confusion_matrix.png
threshold_analysis.png
final_test_metrics.txt
tests/test_model.py
Contains automated tests for important model behavior.
Current tests cover:
- Sigmoid behavior
- Sigmoid output range
- Probability output
- Prediction shape
- Binary prediction values
Testing
The project uses pytest.
Run the complete test suite with:
python3 -m pytest
Current test suite:
5 passed
The tests provide basic automated verification of the model implementation.

Installation
Clone the repository:
git clone <YOUR_REPOSITORY_URL>
cd Transaction-Risk-Detector
Create a virtual environment:
python3 -m venv .venv
Activate it:
source .venv/bin/activate
Install dependencies:
python3 -m pip install -r requirements.txt
Running the Application
Start the Streamlit application with:
python3 -m streamlit run app.py
Streamlit will provide a local URL where the application can be opened in a browser.

Reproducing the Results
To regenerate the evaluation artifacts:
python3 -m results.generate_results
This generates:
results/
├── training_cost.png
├── confusion_matrix.png
├── threshold_analysis.png
└── final_test_metrics.txt
Technical Stack
Python
NumPy
Pandas
Scikit-learn
Matplotlib
Streamlit
Pytest
Engineering Practices

This project emphasizes several engineering practices alongside model development:
- Reproducible train/validation/test splitting
- Shared preprocessing logic
- Training-only scaler fitting
- Explicit validation-based threshold selection
- Separation of validation and test evaluation
- Automated unit tests
- Reference implementation using scikit-learn
- Reproducible result generation
- Clear project structure
- Explicit documentation of limitations
- No secrets or credentials stored in the repository

Limitations
This project is an educational and portfolio machine learning system rather than a production fraud detection platform.

Important limitations include:
Dataset Size
The dataset is relatively small for a real-world financial fraud detection system.
Feature Availability
Real fraud detection systems may use substantially more information, including historical behavioral patterns, merchant intelligence, device fingerprints, network signals, and real-time risk signals.

Class Imbalance
The fraud class remains significantly smaller than the normal class.
Class weighting improves the model's attention to fraud but does not completely solve the underlying data imbalance.
False Positives

The current model produces a substantial number of false positives.
This means that the current threshold should not be interpreted as a production decision rule.

Temporal Validation
The current experiment uses a stratified random split rather than a time-based evaluation.
A production system would need to consider temporal drift and evaluate whether a model trained on historical transactions continues to perform on future transactions.

Model Complexity
Logistic Regression provides an interpretable baseline, but more complex models may capture nonlinear relationships that this model cannot.
Future Improvements
Potential next steps include:
- Cross-validation
- Time-based validation
- Hyperparameter search
- Calibration analysis
- Precision-Recall curves
- ROC-AUC comparison
- Additional model baselines
- More robust feature engineering
- Feature importance analysis
- Probability calibration
- Error analysis
- Data drift monitoring
- Model versioning
- CI-based automated testing
- Containerized deployment
- Production-oriented monitoring

These improvements would extend the current baseline rather than replacing the core implementation.
Why This Project?
The main goal of this project is to understand the machine learning pipeline end-to-end.
Instead of relying only on:
model.fit(X, y)

the project implements the core Logistic Regression optimization process directly with NumPy.
This makes it possible to connect the mathematical concepts to an actual working system:

Mathematics
    ↓
NumPy Implementation
    ↓
Model Training
    ↓
Evaluation
    ↓
Testing
    ↓
Interactive Application

Reproducibility
The project uses fixed random seeds for data splitting and follows a consistent preprocessing pipeline.
The intended workflow is:
1. Install dependencies
2. Run tests
3. Run the application
4. Regenerate evaluation results
5. Inspect metrics and plots

Commands:
python3 -m pytest
python3 -m results.generate_results
python3 -m streamlit run app.py
Project Status

Current implementation includes:
- [x] Data preprocessing
- [x] Feature engineering
- [x] Train/validation/test split
- [x] Feature scaling
- [x] Logistic Regression from scratch
- [x] Sigmoid implementation
- [x] Binary cross-entropy loss
- [x] L2 regularization
- [x] Class weighting
- [x] Gradient descent
- [x] Validation threshold selection
- [x] Test-set evaluation
- [x] Scikit-learn comparison
- [x] Streamlit application
- [x] Automated tests
- [x] Evaluation plots
- [x] Reproducible result generation
- [x] Project documentation

License
This project is intended for educational and portfolio purposes.
If the dataset used in this repository has separate licensing or redistribution requirements, those requirements take precedence over this project documentation.
