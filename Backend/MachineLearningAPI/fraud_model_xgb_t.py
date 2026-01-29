import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from services.application_dbcontext import SessionLocal, get_all_training_data

# Try to load data from the database using application_dbcontext
use_db = True
try:
    db = SessionLocal()
    records = get_all_training_data(db)
    db.close()
    if records and len(records) >= 500_000:
        print(f'[*] Loaded {len(records)} records from database.')
        df = pd.DataFrame([{
            'transaction_id': getattr(r, 'transaction_id', None),
            'smart_contract_id': getattr(r, 'smart_contract_id', None),
            'funder': r.funder,
            'payment_type': r.payment_type,
            'student_id': r.student_id,
            'payment_timestamp': getattr(r, 'payment_timestamp', None),
            'month': r.month,
            'day': getattr(r, 'day', None),
            'hour': getattr(r, 'hour', None),
            'minute': getattr(r, 'minute', None),
            'day_of_week': getattr(r, 'day_of_week', None),
            'is_weekend': getattr(r, 'is_weekend', None),
            'contract_version': getattr(r, 'contract_version', None),
            'contract_start_date': getattr(r, 'contract_start_date', None),
            'contract_length_months': getattr(r, 'contract_length_months', None),
            'base_stipend_amount_from_contract': getattr(r, 'base_stipend_amount_from_contract', None),
            'total_contract_value': getattr(r, 'total_contract_value', None),
            'is_first_payment_for_contract': getattr(r, 'is_first_payment_for_contract', None),
            'first_payment_date_for_contract': getattr(r, 'first_payment_date_for_contract', None),
            'months_covered_by_this_payment': getattr(r, 'months_covered_by_this_payment', None),
            'amount_from_contract_read': getattr(r, 'amount_from_contract_read', None),
            'expected_payment_amount': getattr(r, 'expected_payment_amount', None),
            'final_payment_amount': getattr(r, 'final_payment_amount', None),
            'remaining_contract_balance': getattr(r, 'remaining_contract_balance', None),
            'payment_frequency_type': getattr(r, 'payment_frequency_type', None),
            'student_active_start_date': getattr(r, 'student_active_start_date', None),
            'student_active_end_date': getattr(r, 'student_active_end_date', None),
            'is_student_active_at_payment': getattr(r, 'is_student_active_at_payment', None),
            'processing_latency_ms': getattr(r, 'processing_latency_ms', None),
            'transaction_status': getattr(r, 'transaction_status', None),
            'is_anomaly': r.is_anomaly,
            'anomaly_type': getattr(r, 'anomaly_type', None)
        } for r in records])
    else:
        print(f'[*] Not enough records in database ({len(records) if records else 0}), falling back to CSV.')
        use_db = False
except Exception as e:
    print(f'[*] Could not load from database: {e}\nFalling back to CSV.')
    use_db = False

if not use_db:
    df = pd.read_csv('automated_payment_anomalies.csv', parse_dates=[
        'payment_timestamp', 'contract_start_date', 'first_payment_date_for_contract',
        'student_active_start_date', 'student_active_end_date'])
    print(f'[*] Loaded {len(df)} records from CSV.')

# Encode categorical features
print('[*] Encoding categorical features...')
le_funder = LabelEncoder()
df['funder'] = le_funder.fit_transform(df['funder'])
le_payment_type = LabelEncoder()
df['payment_type'] = le_payment_type.fit_transform(df['payment_type'])
le_student_id = LabelEncoder()
df['student_id'] = le_student_id.fit_transform(df['student_id'])
le_contract_version = LabelEncoder()
df['contract_version'] = le_contract_version.fit_transform(df['contract_version'])
le_smart_contract_id = LabelEncoder()
df['smart_contract_id'] = le_smart_contract_id.fit_transform(df['smart_contract_id'])
le_transaction_status = LabelEncoder()
df['transaction_status'] = le_transaction_status.fit_transform(df['transaction_status'])
le_anomaly_type = LabelEncoder()
df['anomaly_type'] = le_anomaly_type.fit_transform(df['anomaly_type'])
le_payment_frequency_type = LabelEncoder()
df['payment_frequency_type'] = le_payment_frequency_type.fit_transform(df['payment_frequency_type'])

print('[*] Creating features and target variable...')
feature_cols = [
    'funder', 'payment_type', 'student_id', 'smart_contract_id', 'contract_version',
    'month', 'day', 'hour', 'minute', 'day_of_week', 'is_weekend',
    'contract_length_months', 'base_stipend_amount_from_contract', 'total_contract_value',
    'is_first_payment_for_contract', 'months_covered_by_this_payment',
    'amount_from_contract_read', 'expected_payment_amount', 'final_payment_amount',
    'remaining_contract_balance', 'payment_frequency_type', 'is_student_active_at_payment',
    'processing_latency_ms', 'transaction_status'
]
X = df[feature_cols]
y = df['is_anomaly']

print('[*] Splitting data into training and test sets...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('[*] Data preparation complete. Starting hyperparameter tuning...')
param_grid = {
    'n_estimators': [200, 400],
    'max_depth': [4, 6],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

print('[*] Parameter grid defined. Starting grid search...')
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid_search = GridSearchCV(xgb_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print('Best parameters found:', best_params)

print('[*] Evaluating the best model on the test set...')
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

def get_model_metrics(threshold, y_test):
    y_pred = (y_pred_proba > threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print(classification_report(y_test, y_pred))

# Default threshold (0.5)
get_model_metrics(0.5, y_test)
# Lower threshold (e.g., 0.3) to catch more anomalies
get_model_metrics(0.3, y_test)

# Save the best model and encoders
model_dir = 'saved_models/xgboost_improved'
os.makedirs(model_dir, exist_ok=True)
joblib.dump(best_model, os.path.join(model_dir, 'fraud_xgboost_improved_model.joblib'))
joblib.dump(le_funder, os.path.join(model_dir, 'le_funder.joblib'))
joblib.dump(le_payment_type, os.path.join(model_dir, 'le_payment_type.joblib'))
joblib.dump(le_student_id, os.path.join(model_dir, 'le_student_id.joblib'))
joblib.dump(le_contract_version, os.path.join(model_dir, 'le_contract_version.joblib'))
joblib.dump(le_smart_contract_id, os.path.join(model_dir, 'le_smart_contract_id.joblib'))
joblib.dump(le_transaction_status, os.path.join(model_dir, 'le_transaction_status.joblib'))
joblib.dump(le_anomaly_type, os.path.join(model_dir, 'le_anomaly_type.joblib'))
joblib.dump(le_payment_frequency_type, os.path.join(model_dir, 'le_payment_frequency_type.joblib'))

print(f"[*] Model and encoders saved to {model_dir}.")
