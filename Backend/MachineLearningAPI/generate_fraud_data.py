import pandas as pd
import numpy as np

def generate_fraud_data():
    np.random.seed(42)

    n_rows = 500_000
    print('[*] Generating', n_rows, 'rows of fraud data for traing and testing the model...')
    funders = ['NSFAS', 'Dell', 'Sasol', 'MTN']
    payment_types = ['food', 'books', 'accommodation']
    months = list(range(1, 13))
    students = [f"S{str(i).zfill(6)}" for i in range(1, 10001)]  # 10,000 students

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

    # Add duplicate payments for fraud
    for _ in range(int(n_rows * 0.03)):  # 3% duplicates
        row = df.sample(1).iloc[0]
        if row['is_fraud'] == 0:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.at[df.index[-1], 'is_fraud'] = 1  # Mark duplicate as fraud

    df.to_csv('fraud_data.csv', index=False)
    print("Generated fraud_data.csv with", len(df), "rows.")

def filter_nsfas_food_fraud(csv_path='fraud_data.csv'):
    df = pd.read_csv(csv_path)
    filtered = df[(df['funder'] == 'NSFAS') & (df['payment_type'] == 'food') & (df['is_fraud'] == 1)]
    print(f"Found {len(filtered)} fraudulent NSFAS food payments.")
    
    return filtered


generate_fraud_data()
data = filter_nsfas_food_fraud()
print(data)
