"""
xgboost_fraud_detector.py

This script loads the trained XGBoost fraud detection model and label encoders, and provides a function
to predict fraud given payment details: amount, funder, payment_type, month, and student_id.

Usage example:
    result = detect_fraud(amount=2000, funder='Dell', payment_type='food', month=3, student_id='S000123')
    print(result)
"""

import joblib
import numpy as np

# Load model and encoders
model = joblib.load('fraud_xgboost_model.joblib')
le_funder = joblib.load('le_funder.joblib')
le_payment_type = joblib.load('le_payment_type.joblib')
le_student_id = joblib.load('le_student_id.joblib')

def detect_fraud(amount, funder, payment_type, month, student_id):
    """
    Predicts fraud using the trained XGBoost model.
    Returns a dict with is_fraud (bool) and risk_score (float, probability of fraud).
    """
    # Encode categorical features
    funder_enc = le_funder.transform([funder])[0] if funder in le_funder.classes_ else -1
    payment_type_enc = le_payment_type.transform([payment_type])[0] if payment_type in le_payment_type.classes_ else -1
    student_id_enc = le_student_id.transform([student_id])[0] if student_id in le_student_id.classes_ else -1
    features = np.array([[amount, funder_enc, payment_type_enc, month, student_id_enc]])
    proba = model.predict_proba(features)[0][1]  # Probability of fraud
    is_fraud = proba > 0.5
    return {'is_fraud': bool(is_fraud), 'risk_score': float(proba)}

# Example usage:
if __name__ == "__main__":
    result = detect_fraud(amount=2000, funder='Dell', payment_type='food', month=3, student_id='S000123')
    print("Prediction:", result)
