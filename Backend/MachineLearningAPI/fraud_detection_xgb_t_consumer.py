"""
fraud_detection_xgb_t_consumer.py

This script loads the fine-tuned XGBoost fraud detection model and encoders from 'saved_models/xgboost_tuned',
accepts payment parameters, and predicts fraud. It prints the prediction and explains the result.

Usage example:
    python fraud_detection_xgb_t_consumer.py --amount 2000 --funder Dell --payment_type food --month 3 --student_id S000123 --duplicate_count 1 --activation_month 2
"""

import argparse
from datetime import datetime
import joblib
import numpy as np
import os

from responses.fraud_response import FraudResponse

# Directory where the model and encoders are saved
model_dir = 'saved_models/xgboost_tuned'

# Load model and encoders
model = joblib.load(os.path.join(model_dir, 'fraud_xgboost_finetuned_model.joblib'))
le_funder = joblib.load(os.path.join(model_dir, 'le_funder.joblib'))
le_payment_type = joblib.load(os.path.join(model_dir, 'le_payment_type.joblib'))
le_student_id = joblib.load(os.path.join(model_dir, 'le_student_id.joblib'))

def detect_fraud(amount, funder, payment_type, month, student_id, duplicate_count, activation_month, threshold=0.3):
    """
    Predicts fraud using the fine-tuned XGBoost model.
    Returns a dict with is_fraud (bool) and risk_score (float, probability of fraud).
    threshold: probability cutoff for classifying as fraud (default 0.3 for higher recall)
    """
    funder_enc = le_funder.transform([funder])[0] if funder in le_funder.classes_ else -1
    payment_type_enc = le_payment_type.transform([payment_type])[0] if payment_type in le_payment_type.classes_ else -1
    student_id_enc = le_student_id.transform([student_id])[0] if student_id in le_student_id.classes_ else -1
    features = np.array([[amount, funder_enc, payment_type_enc, month, student_id_enc, duplicate_count, activation_month]])
    proba = model.predict_proba(features)[0][1]
    is_fraud = proba > threshold
    risk_score_percent_rounded = round(proba * 100, 2)
    return FraudResponse(
        is_fraud=is_fraud,
        risk_score=risk_score_percent_rounded,
        message=f"Fraud detected with {proba * 100:.2f}% probability at a 0.3 threshold" if is_fraud else "No fraud detected",
        is_success=True,
        time_stamp= str(datetime.now()),
        error_code="0"
    )

if __name__ == "__main__":
    # Example usage without command-line arguments
    example = True  # Set to False to use command-line arguments
    if example:
        # Example parameters
        amount = 2000
        funder = 'NSFAS'
        payment_type = 'food'
        month = 3
        student_id = 'S000123'
        duplicate_count = 1
        activation_month = 3
        print(f"Running example with: amount={amount}, funder={funder}, payment_type={payment_type}, month={month}, student_id={student_id}, duplicate_count={duplicate_count}, activation_month={activation_month}")
        result = detect_fraud(amount, funder, payment_type, month, student_id, duplicate_count, activation_month, threshold=0.3)
        print(f"Prediction: {result}")
        if result['is_fraud']:
            print(f"This payment is likely FRAUDULENT (risk score: {result['risk_score']:.2f})")
        else:
            print(f"This payment is likely NOT fraud (risk score: {result['risk_score']:.2f})")
    else:
        parser = argparse.ArgumentParser(description="XGBoost Fraud Detection Consumer")
        parser.add_argument('--amount', type=float, required=True, help='Payment amount')
        parser.add_argument('--funder', type=str, required=True, help='Funder name (e.g., NSFAS)')
        parser.add_argument('--payment_type', type=str, required=True, help='Payment type (e.g., food)')
        parser.add_argument('--month', type=int, required=True, help='Month (1-12)')
        parser.add_argument('--student_id', type=str, required=True, help='Student ID (e.g., S000123)')
        parser.add_argument('--duplicate_count', type=int, required=True, help='Number of duplicate payments for this student in the same month')
        parser.add_argument('--activation_month', type=int, required=True, help='Activation month for the student contract')
        args = parser.parse_args()

        result = detect_fraud(args.amount, args.funder, args.payment_type, args.month, args.student_id, args.duplicate_count, args.activation_month, threshold=0.3)
        print(f"Prediction: {result}")
        if result['is_fraud']:
            print(f"This payment is likely FRAUDULENT (risk score: {result['risk_score']:.2f})")
        else:
            print(f"This payment is likely NOT fraud (risk score: {result['risk_score']:.2f})")
