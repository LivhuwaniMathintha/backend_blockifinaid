import pandas as pd
import numpy as np
from services.application_dbcontext import SessionLocal, add_training_data_bulk

def generate_secondary_fraud_data():
    np.random.seed(42)

    n_rows = 1_000_000
    print('[*] Generating', n_rows, 'rows of secondary fraud data...')
    funders = ['NSFAS', 'Dell', 'Sasol', 'MTN']
    payment_types = ['food', 'books', 'accommodation']
    months = list(range(1, 13))
    students = [f"S{str(i).zfill(6)}" for i in range(1, 20001)]  # 20,000 students

    # Assign a random activation month for each student
    student_activation = {sid: np.random.choice(months) for sid in students}

    # Define normal amounts for each funder/payment_type
    normal_amounts = {
        ('NSFAS', 'food'): 1650,
        ('Dell', 'food'): 2000,
        ('Sasol', 'food'): 1800,
        ('MTN', 'food'): 1700,
        ('NSFAS', 'books'): 5000,
        ('Dell', 'books'): 1200,
        ('Sasol', 'books'): 1100,
        ('MTN', 'books'): 1050,
        ('NSFAS', 'accommodation'): 1500,
        ('Dell', 'accommodation'): 3500,
        ('Sasol', 'accommodation'): 3200,
        ('MTN', 'accommodation'): 3100,
    }

    rows = []
    for _ in range(n_rows):
        student_id = np.random.choice(students)
        funder = np.random.choice(funders)
        payment_type = np.random.choice(payment_types)
        month = np.random.choice(months)
        activation_month = student_activation[student_id]
        base_amount = normal_amounts[(funder, payment_type)]
        # Bulk payment if account activated this month and missed previous months
        if month == activation_month and activation_month > 1:
            missed_months = activation_month
            expected_amount = base_amount * missed_months
            if np.random.rand() < 0.92:
                amount = expected_amount
                is_fraud = 0
            else:
                # Fraud: wrong bulk amount
                if np.random.rand() < 0.5:
                    amount = expected_amount + np.random.uniform(1, 500)
                else:
                    amount = expected_amount - np.random.uniform(1, 500)
                is_fraud = 1
        else:
            if np.random.rand() < 0.92:
                amount = base_amount
                is_fraud = 0
            else:
                # Fraud: any amount not equal to base_amount
                if np.random.rand() < 0.5:
                    amount = base_amount + np.random.uniform(1, 500)
                else:
                    amount = base_amount - np.random.uniform(1, 500)
                is_fraud = 1
        rows.append([amount, funder, payment_type, month, student_id, activation_month, is_fraud])

    df = pd.DataFrame(rows, columns=['amount', 'funder', 'payment_type', 'month', 'student_id', 'activation_month', 'is_fraud'])

    # Flag as fraud if month > activation_month
    df.loc[df['month'] < df['activation_month'], 'is_fraud'] = 1

    # Flag as fraud if a student_id repeats with same payment_type, funder, month, activation_month
    # or if same student_id, payment_type, funder, month but different amount
    df['dup_key'] = df['student_id'].astype(str) + '_' + df['payment_type'] + '_' + df['funder'] + '_' + df['month'].astype(str) + '_' + df['activation_month'].astype(str)
    # Find duplicates (keep first as not fraud, others as fraud)
    dup_mask = df.duplicated(subset=['student_id', 'payment_type', 'funder', 'month', 'activation_month'], keep='first')
    df.loc[dup_mask, 'is_fraud'] = 1

    # Now, for same student_id, payment_type, funder, month, but different amount, flag as fraud
    # Find all groups with more than one unique amount
    amt_group = df.groupby(['student_id', 'payment_type', 'funder', 'month', 'activation_month'])['amount'].transform('nunique')
    diff_amt_mask = amt_group > 1
    df.loc[diff_amt_mask, 'is_fraud'] = 1

    df = df.drop(columns=['dup_key'])
    df.to_csv('fraud_data.csv', index=False)
    print("Generated secondary_fraud_data.csv with", len(df), "rows.")

    # Save to database using bulk insert

    data_dicts = df.to_dict(orient='records')
    db = SessionLocal()
    try:
        add_training_data_bulk(db, data_dicts)
        print(f"Inserted {len(data_dicts)} rows into fraud_training_data table.")
    finally:
        db.close()


def filter_nsfas_food_fraud(csv_path='fraud_data.csv'):
    df = pd.read_csv(csv_path)
    filtered = df[(df['funder'] == 'NSFAS') & (df['payment_type'] == 'food') & (df['is_fraud'] == 0)]
    print(f"Found {len(filtered)} fraudulent NSFAS food payments.")
    bulk_payments = df[(df['funder'] == 'NSFAS') & (df['payment_type'] == 'food') & (df['activation_month'] > 1) & (df['activation_month'] == df['month']) & (df['amount'] > 1650) & (df['is_fraud'] == 0)]
    print(f"Found {len(bulk_payments)} NSFAS food payments that are bulk payments but not flagged as fraud.")
    print(bulk_payments)
 

if __name__ == "__main__":
    generate_secondary_fraud_data()
    data = filter_nsfas_food_fraud()
