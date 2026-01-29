# fraud_detector.py

import joblib
import numpy as np

from FinancialAid.Backend.MachineLearningAPI.config import RANDOM_FOREST_CLASSIFIER_FRAUD_MODEL

clf = joblib.load(RANDOM_FOREST_CLASSIFIER_FRAUD_MODEL)

def detect_fraud(payment_data: dict) -> dict:
    # Encode funder
    funder = 0 if payment_data.get("funder") == "NSFAS" else 1
    features = np.array([[payment_data.get("amount", 0), funder, payment_data.get("month", 1)]])
    is_fraud = clf.predict(features)[0]
    risk_score = clf.predict_proba(features)[0][1]
    return {"is_fraud": bool(is_fraud), "risk_score": float(risk_score)}