import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv('fraud_data.csv')

# Encode categorical data
df['funder'] = df['funder'].astype('category').cat.codes
df['payment_type'] = df['payment_type'].astype('category').cat.codes
df['student_id'] = df['student_id'].astype('category').cat.codes

# Features include student_id and payment_type
X = df[['amount', 'funder', 'payment_type', 'month', 'student_id']]
y = df['is_fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, 'fraud_model.joblib')

# Optional: print accuracy
print("Test accuracy:", clf.score(X_test, y_test))