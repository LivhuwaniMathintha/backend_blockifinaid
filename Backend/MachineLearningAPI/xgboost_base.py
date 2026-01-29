"""
xgboost.py

This script trains an XGBoost model for fraud detection using the same features as train_model.py.
It loads the data, encodes categorical variables, splits into train/test, trains the model, evaluates accuracy, and saves the model.

Requirements:
    pip install xgboost pandas scikit-learn joblib
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv('fraud_data.csv')

# Encode categorical data
le_funder = LabelEncoder()
df['funder'] = le_funder.fit_transform(df['funder'])
le_payment_type = LabelEncoder()
df['payment_type'] = le_payment_type.fit_transform(df['payment_type'])
le_student_id = LabelEncoder()
df['student_id'] = le_student_id.fit_transform(df['student_id'])

# Features and target
X = df[['amount', 'funder', 'payment_type', 'month', 'student_id']]
y = df['is_fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'fraud_xgboost_model.joblib')

# Optional: print accuracy
accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Save encoders for later use in prediction
joblib.dump(le_funder, 'le_funder.joblib')
joblib.dump(le_payment_type, 'le_payment_type.joblib')
joblib.dump(le_student_id, 'le_student_id.joblib')
