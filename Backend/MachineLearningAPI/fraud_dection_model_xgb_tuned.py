"""
xgboost_finetuning.py

This script demonstrates how to fine-tune (optimize) the hyperparameters of an XGBoost model for fraud detection.
It uses GridSearchCV to search for the best combination of parameters, retrains the model, and saves the best model.

Hyperparameter tuning is important because the default settings may not be optimal for your data. By searching over a grid
of possible values, you can find a model that performs better (higher accuracy, better generalization).

Requirements:
    pip install xgboost pandas scikit-learn joblib
"""

import json
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
    if records and len(records) > 0:
        print('[*] Loaded', len(records), 'records from database.')
        # Convert SQLAlchemy objects to DataFrame
        df = pd.DataFrame([{
            'amount': r.amount,
            'funder': r.funder,
            'payment_type': r.payment_type,
            'month': r.month,
            'student_id': r.student_id,
            'activation_month': r.activation_month,
            'is_fraud': r.is_fraud
        } for r in records])
    else:
        print('[*] No records found in database, falling back to CSV.')
        use_db = False
except Exception as e:
    print(f'[*] Could not load from database: {e}\nFalling back to CSV.')
    use_db = False

if not use_db:
    df = pd.read_csv('fraud_data.csv')
    print('[*] Loaded', len(df), 'records from CSV.')

# Encode categorical data
print('[*] Encoding categorical features...')
le_funder = LabelEncoder()
df['funder'] = le_funder.fit_transform(df['funder'])
le_payment_type = LabelEncoder()
df['payment_type'] = le_payment_type.fit_transform(df['payment_type'])
le_student_id = LabelEncoder()
df['student_id'] = le_student_id.fit_transform(df['student_id'])

print('[*] Creating additional features...')

# Add duplicate_count feature: how many times a student has received the same payment_type in the same month from same funder
# This is calculated using groupby and cumcount, then +1 to count the current row
# This feature helps the model explicitly learn about duplicate payments
df['duplicate_count'] = df.groupby(['student_id', 'payment_type', 'month', 'funder']).cumcount() + 1

# activation_month is already present in the data, so we just use it as a feature

print('[*] Preparing features and target variable...')
# Features and target
X = df[['amount', 'funder', 'payment_type', 'month', 'student_id', 'duplicate_count', 'activation_month']]
y = df['is_fraud']

print('[*] Splitting data into training and test sets...')
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('[*] Data preparation complete. Starting hyperparameter tuning...')
# Define the parameter grid for tuning
param_grid = {
    'n_estimators': [300, 700],           # Number of trees in the forest
    'max_depth': [4, 6, 8],               # Maximum depth of each tree
    'learning_rate': [0.01, 0.1, 0.2],    # Step size shrinkage
    'subsample': [0.8, 1.0],              # Fraction of samples used for fitting each tree
    'colsample_bytree': [0.8, 1.0],       # Fraction of features used for each tree
}

print('[*] Parameter grid defined. Starting grid search...')
# Create the XGBoost classifier
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# GridSearchCV will try all combinations of the above parameters using cross-validation
# This can take a long time for large datasets or many parameters
# n_jobs=-1 uses all CPU cores for speed
print("[*] Starting grid search for hyperparameter tuning...")
grid_search = GridSearchCV(xgb_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best parameters found:", best_params)

# Evaluate the best model on the test set
print('[*] Evaluating the best model on the test set...')
y_pred_proba = best_model.predict_proba(X_test)[:, 1]



def get_model_metrics(threshold, y_test):
    """
    Calculate and print model metrics at a given threshold.
    Returns a dictionary with accuracy, precision, recall, f1-score, confusion matrix, and classification report.
    """
    y_pred = (y_pred_proba > threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    #print accuracy, precision, recall, f1, confusion matrix and classification report
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)


    report = classification_report(y_test, y_pred)
    print(report)

# Default threshold (0.5)
get_model_metrics(0.5, y_test)

# Lower threshold (e.g., 0.3) to catch more fraud

get_model_metrics(0.3, y_test)


# Optional: To further increase recall, you can set scale_pos_weight in param_grid, e.g.:
# param_grid = {
#     ...existing params...,
#     'scale_pos_weight': [1, 2, 5, 10]  # Try higher values to give more weight to fraud class
# }
# This will make the model more sensitive to fraud cases during training.

"""
Metric explanations:
- Accuracy: Overall, how often is the model correct? (Good if classes are balanced, misleading if not)
- Precision: Of all cases the model predicted as fraud, how many were actually fraud? (High precision = few false positives)
- Recall: Of all actual fraud cases, how many did the model catch? (High recall = few false negatives)
- F1-score: Balance between precision and recall. Good if you want both few false positives and few false negatives.
- Confusion Matrix: Shows counts of true positives, false positives, true negatives, and false negatives.
- Classification Report: Shows all metrics for each class (fraud/not fraud).

How to interpret:
- If fraud is rare, focus on precision and recall for the fraud class.
- High precision, low recall: Model is cautious, only flags obvious frauds (few false alarms, but may miss some frauds).
- Low precision, high recall: Model flags many frauds, but also many false alarms.
- F1-score is a good single metric if you want a balance.
- Use the confusion matrix to see where the model is making mistakes.
"""

# Directory to save models
model_dir = 'saved_models/xgboost_tuned'
os.makedirs(model_dir, exist_ok=True)

# Save the best model and encoders
joblib.dump(best_model, os.path.join(model_dir, 'fraud_xgboost_finetuned_model.joblib'))
joblib.dump(le_funder, os.path.join(model_dir, 'le_funder.joblib'))
joblib.dump(le_payment_type, os.path.join(model_dir, 'le_payment_type.joblib'))
joblib.dump(le_student_id, os.path.join(model_dir, 'le_student_id.joblib'))

"""
Explanation:
- We use GridSearchCV to systematically try different combinations of XGBoost hyperparameters.
- For each combination, the model is trained and evaluated using cross-validation (cv=3 means 3-fold split).
- The best combination is selected based on accuracy.
- The best model is then evaluated on the test set and saved for future use.
- This process helps you find a model that is better tuned to your specific data, potentially improving performance.
"""


def incremental_retrain(new_data_path, model_path=os.path.join(model_dir, 'fraud_xgboost_finetuned_model.joblib')):
    """
    Incrementally retrain the XGBoost model with new data, without starting from scratch.
    Loads the existing model, encodes new data, and continues training (warm start).
    Prints progress and explains each step.
    """
    print('[*] Loading existing model for incremental retraining...')
    model = joblib.load(model_path)
    print('[*] Model loaded.')

    print(f'[*] Loading new data from {new_data_path}...')
    new_df = pd.read_csv(new_data_path)
    print('[*] Encoding categorical features in new data...')
    new_df['funder'] = le_funder.transform(new_df['funder'])
    new_df['payment_type'] = le_payment_type.transform(new_df['payment_type'])
    new_df['student_id'] = le_student_id.transform(new_df['student_id'])
    new_df['duplicate_count'] = new_df.groupby(['student_id', 'payment_type', 'month']).cumcount() + 1
    # Add activation_month as a feature for incremental retraining
    X_new = new_df[['amount', 'funder', 'payment_type', 'month', 'student_id', 'duplicate_count', 'activation_month']]
    y_new = new_df['is_fraud']

    print('[*] Continuing training on new data (warm start)...')
    # XGBoost supports continued training via .fit with xgb_model parameter
    model.fit(X_new, y_new, xgb_model=model)
    print('[*] Model retrained with new data.')

    print('[*] Saving updated model...')
    joblib.dump(model, model_path)
    print(f'[*] Updated model saved to {model_path}.')

    print('[*] Incremental retraining complete. This approach is efficient because it avoids retraining from scratch, saving time and resources.')

# Example usage:
# incremental_retrain('new_fraud_data.csv')
