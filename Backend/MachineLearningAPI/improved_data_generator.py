import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import random
from services.application_dbcontext import SessionLocal, add_training_data_bulk

# Function to generate a realistic-looking Ethereum-style address
def generate_eth_address():
    return '0x' + ''.join(random.choices('0123456789abcdef', k=40))

def generate_automated_payment_fraud_data(n_rows_target=1_000_000, start_date_str='2024-01-01'):
    np.random.seed(42) # For reproducibility of numpy randomness
    random.seed(42) # For reproducibility of standard random library

    print(f'[*] Generating automated payment anomaly data with chronological contract payments...')

    # --- Configuration Parameters ---
    NUM_STUDENTS = 20_000
    NUM_SMART_CONTRACTS_PER_FUNDER = 100 # Adjust to control total contracts
    
    # Anomaly Rates (will apply per payment event)
    CONTRACT_AMOUNT_ERROR_RATE = 0.005 # Chance a smart contract provides incorrect base monthly stipend
    API_LOGIC_ERROR_RATE = 0.008 # Chance of an API logic error during processing (on final_payment_amount)
    SCHEDULE_DEVIATION_RATE = 0.003 # Chance a payment occurs outside its *intended* schedule (can cause Early/Late flags)
    DUPLICATE_PAYMENT_RATE = 0.002 # Chance of accidental duplicate payment (post-processing)
    HIGH_LATENCY_RATE = 0.05 # Chance of unusually high processing latency
    # NEW: Rate at which a payment *might* occur for an inactive student (despite checks)
    INACTIVE_STUDENT_PAYMENT_ERROR_RATE = 0.001

    # Define categories
    funders = ['NSFAS', 'Dell', 'Sasol', 'MTN', 'Standard Bank']
    payment_types = ['food', 'books', 'accommodation', 'tuition', 'health']
    contract_versions = ['v1.0', 'v1.1', 'v1.2', 'v2.0']
    transaction_statuses = ['SUCCESS', 'FAILED_TEMP', 'FAILED_PERM']

    # Define payment frequency by type
    PAYMENT_FREQUENCY = {
        'tuition': 'ONE_OFF',
        'food': 'MONTHLY',
        'accommodation': 'MONTHLY',
        'books': 'ONE_OFF',
        'health': 'MONTHLY'
    }

    students = [f"S{str(i).zfill(6)}" for i in range(1, NUM_STUDENTS + 1)]
    smart_contracts_map = {} # Map funder -> list of contract addresses
    
    # Pre-generate smart contracts per funder
    for funder in funders:
        smart_contracts_map[funder] = [generate_eth_address() for _ in range(NUM_SMART_CONTRACTS_PER_FUNDER)]

    # Assign base monthly/one-off amounts for each funder and payment type
    base_stipend_amounts = {
        ('NSFAS', 'food'): 1650, ('Dell', 'food'): 2000, ('Sasol', 'food'): 1800, ('MTN', 'food'): 1700, ('Standard Bank', 'food'): 1900,
        ('NSFAS', 'books'): 5000, ('Dell', 'books'): 1200, ('Sasol', 'books'): 1100, ('MTN', 'books'): 1050, ('Standard Bank', 'books'): 1150,
        ('NSFAS', 'accommodation'): 1500, ('Dell', 'accommodation'): 3500, ('Sasol', 'accommodation'): 3200, ('MTN', 'accommodation'): 3100, ('Standard Bank', 'accommodation'): 3300,
        ('NSFAS', 'tuition'): 15000, ('Dell', 'tuition'): 20000, ('Sasol', 'tuition'): 18000, ('MTN', 'tuition'): 17000, ('Standard Bank', 'tuition'): 19000,
        ('NSFAS', 'health'): 500, ('Dell', 'health'): 700, ('Sasol', 'health'): 600, ('MTN', 'health'): 550, ('Standard Bank', 'health'): 650,
    }

    # Simulate some "bad" smart contracts for consistent errors
    all_smart_contracts = [addr for sublist in smart_contracts_map.values() for addr in sublist]
    num_bad_contracts = int(len(all_smart_contracts) * 0.01)
    bad_contracts_set = set(np.random.choice(all_smart_contracts, size=num_bad_contracts, replace=False))
    print(f"[*] Designated {len(bad_contracts_set)} smart contracts as potentially 'bad'.")

    # NEW: Student Activity Periods
    # Students are active for a random period around the simulation year
    student_activity_periods = {}
    sim_start_year = datetime.strptime(start_date_str, '%Y-%m-%d').year
    for student_id in students:
        # Most students active around the sim year
        start_year_offset = np.random.randint(-1, 2) # -1 year, 0 year, +1 year from sim start
        end_year_offset = np.random.randint(1, 4) # Active for 1 to 3 years from their start
        
        active_start_date = datetime(sim_start_year + start_year_offset, np.random.randint(1,13), np.random.randint(1,28))
        active_end_date = active_start_date + timedelta(days=365 * end_year_offset + np.random.randint(0, 365)) # Active for X years + random days
        student_activity_periods[student_id] = {'start': active_start_date, 'end': active_end_date}


    rows = []
    start_sim_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_sim_date = start_sim_date + timedelta(days=364) # Roughly 1 year simulation

    # --- Generate data chronologically for each student-contract pair ---
    generated_rows_count = 0
    contract_iterations = 0
    max_contracts_to_generate = n_rows_target // 15 # Roughly average 15 rows per contract
    
    while generated_rows_count < n_rows_target and contract_iterations < max_contracts_to_generate:
        contract_iterations += 1

        student_id = np.random.choice(students)
        
        funder = np.random.choice(funders)
        payment_type = np.random.choice(payment_types)
        smart_contract_id = np.random.choice(smart_contracts_map[funder])

        contract_start_month = np.random.randint(1, 13)
        contract_start_day = np.random.randint(1, 28)
        contract_start_year = start_sim_date.year
        contract_start_date = datetime(contract_start_year, contract_start_month, contract_start_day)

        contract_length_months = np.random.choice([1, 4, 6, 9, 12])
        base_stipend = base_stipend_amounts[(funder, payment_type)]

        if PAYMENT_FREQUENCY[payment_type] == 'ONE_OFF':
            total_contract_value = base_stipend
            contract_length_months = 1
        else: # MONTHLY
            total_contract_value = base_stipend * contract_length_months

        # State tracking for this contract for this student
        last_payment_date_for_contract = None
        cumulative_months_covered_by_contract = 0
        cumulative_amount_paid_for_contract = 0

        num_payments_made_for_contract = 0
        while cumulative_months_covered_by_contract < contract_length_months and generated_rows_count < n_rows_target:
            
            num_payments_made_for_contract += 1

            is_first_payment_for_contract = False
            first_payment_date_for_contract = None

            if last_payment_date_for_contract is None: # First payment for THIS contract
                is_first_payment_for_contract = True
                
                payment_delay_months = np.random.randint(0, 5) # 0 to 4 months delay for first payment
                target_month = contract_start_date.month + payment_delay_months
                target_year = contract_start_date.year
                while target_month > 12:
                    target_month -= 12
                    target_year += 1

                day_for_first_payment = np.random.randint(1, 28)
                payment_timestamp = datetime(target_year, target_month, day_for_first_payment,
                                             np.random.randint(0,24), np.random.randint(0,60))

                if payment_timestamp < contract_start_date:
                    payment_timestamp = contract_start_date.replace(day=np.random.randint(contract_start_date.day, 28) if contract_start_date.day <= 28 else 1,
                                                                    hour=np.random.randint(0,24), minute=np.random.randint(0,60))
                
                first_payment_date_for_contract = payment_timestamp

                if PAYMENT_FREQUENCY[payment_type] == 'MONTHLY':
                    months_to_cover_this_payment = (payment_timestamp.year - contract_start_date.year) * 12 + \
                                                   (payment_timestamp.month - contract_start_date.month) + 1
                    months_to_cover_this_payment = min(months_to_cover_this_payment, contract_length_months)
                    months_to_cover_this_payment = max(1, months_to_cover_this_payment)
                else: # ONE_OFF
                    months_to_cover_this_payment = 1
                    cumulative_months_covered_by_contract = contract_length_months

            else: # Subsequent payments
                if PAYMENT_FREQUENCY[payment_type] == 'ONE_OFF':
                    break # One-off payments don't have subsequent payments
                
                next_month = last_payment_date_for_contract.month + 1
                next_year = last_payment_date_for_contract.year
                if next_month > 12:
                    next_month = 1
                    next_year += 1
                
                if cumulative_months_covered_by_contract >= contract_length_months:
                    break

                payment_timestamp = datetime(next_year, next_month, np.random.randint(1, 28),
                                             np.random.randint(0, 24), np.random.randint(0, 60))
                months_to_cover_this_payment = 1
                first_payment_date_for_contract = last_payment_date_for_contract

                if payment_timestamp > end_sim_date:
                    break

            if payment_timestamp < start_sim_date or payment_timestamp > end_sim_date:
                continue

            # --- Anomaly introduction logic ---
            is_anomaly = 0
            anomaly_type = "None"
            processing_latency_ms = np.random.randint(50, 500)

            # Check student active status *before* calculating expected amount for base case
            student_active_status_at_payment = (payment_timestamp >= student_activity_periods[student_id]['start'] and
                                                payment_timestamp <= student_activity_periods[student_id]['end'])
            
            # Introduce anomaly for inactive student payment at a certain rate
            # This simulates an error where an inactive student *still* gets paid.
            if not student_active_status_at_payment and np.random.rand() < INACTIVE_STUDENT_PAYMENT_ERROR_RATE:
                is_anomaly = 1
                anomaly_type = "Inactive_Student_Payment"


            # Expected amount based on contract logic (true expected value)
            if PAYMENT_FREQUENCY[payment_type] == 'ONE_OFF':
                expected_payment_amount = base_stipend
            else: # MONTHLY
                expected_payment_amount = base_stipend * months_to_cover_this_payment
            
            if expected_payment_amount > 0:
                expected_payment_amount = expected_payment_amount * (1 + np.random.uniform(-0.005, 0.005))
            expected_payment_amount = max(0, round(expected_payment_amount, 2))

            # What the smart contract *reads* (can have an error here)
            amount_from_contract_read = expected_payment_amount
            if smart_contract_id in bad_contracts_set or np.random.rand() < CONTRACT_AMOUNT_ERROR_RATE:
                deviation_factor = np.random.uniform(0.10, 0.50)
                if np.random.rand() < 0.5:
                    amount_from_contract_read = expected_payment_amount * (1 + deviation_factor)
                else:
                    amount_from_contract_read = expected_payment_amount * (1 - deviation_factor)
                amount_from_contract_read = max(0, round(amount_from_contract_read, 2))

            # Final payment amount as processed by API (can have API logic errors)
            final_payment_amount = amount_from_contract_read
            if np.random.rand() < API_LOGIC_ERROR_RATE:
                deviation_factor = np.random.uniform(0.05, 0.30)
                if np.random.rand() < 0.5:
                    final_payment_amount = amount_from_contract_read * (1 + deviation_factor)
                else:
                    final_payment_amount = amount_from_contract_read * (1 - deviation_factor)
                final_payment_amount = max(0, round(final_payment_amount, 2))
                if is_anomaly == 0: # Only set if not already an anomaly
                    is_anomaly = 1
                    anomaly_type = "API_Logic_Error"
            else:
                final_payment_amount = final_payment_amount * (1 + np.random.uniform(-0.005, 0.005))
                final_payment_amount = max(0, round(final_payment_amount, 2))

            # --- Schedule Deviation Anomalies (applied to final payment_timestamp) ---
            if np.random.rand() < SCHEDULE_DEVIATION_RATE:
                deviation_days = np.random.randint(7, 30)
                if np.random.rand() < 0.5:
                    payment_timestamp += timedelta(days=deviation_days)
                else:
                    payment_timestamp -= timedelta(days=deviation_days)
                if is_anomaly == 0:
                    is_anomaly = 1
                    anomaly_type = "Schedule_Deviation"

            # --- Processing Latency Anomalies ---
            if np.random.rand() < HIGH_LATENCY_RATE:
                 processing_latency_ms = np.random.randint(500, 5000)
                 if is_anomaly == 0 and processing_latency_ms > 2000:
                     is_anomaly = 1
                     anomaly_type = "High_Latency_Anomaly"

            # --- Transaction Status ---
            if is_anomaly == 1:
                transaction_status = np.random.choice(['FAILED_TEMP', 'FAILED_PERM'], p=[0.7, 0.3])
            else:
                transaction_status = 'SUCCESS'

            # Calculate remaining balance (based on actual final_payment_amount)
            cumulative_amount_paid_for_contract += final_payment_amount
            remaining_contract_balance = total_contract_value - cumulative_amount_paid_for_contract

            # Update historical tracking for this student-contract pair
            last_payment_date_for_contract = payment_timestamp
            
            if PAYMENT_FREQUENCY[payment_type] == 'MONTHLY':
                cumulative_months_covered_by_contract += months_to_cover_this_payment
            else: # ONE_OFF
                cumulative_months_covered_by_contract = contract_length_months # It's fully covered


            rows.append([
                f"TRX{str(generated_rows_count).zfill(7)}", smart_contract_id, funder, payment_type, student_id,
                payment_timestamp, payment_timestamp.month, payment_timestamp.day,
                payment_timestamp.hour, payment_timestamp.minute,
                payment_timestamp.weekday(), payment_timestamp.weekday() >= 5,
                contract_versions[0],
                contract_start_date, contract_length_months, base_stipend,
                total_contract_value, is_first_payment_for_contract, first_payment_date_for_contract,
                months_to_cover_this_payment, amount_from_contract_read, expected_payment_amount, final_payment_amount,
                remaining_contract_balance, PAYMENT_FREQUENCY[payment_type], student_activity_periods[student_id]['start'], student_activity_periods[student_id]['end'],
                student_active_status_at_payment, # Actual status at time of payment
                processing_latency_ms, transaction_status, is_anomaly, anomaly_type
            ])
            generated_rows_count += 1
            if generated_rows_count >= n_rows_target:
                break

    # Create DataFrame
    df = pd.DataFrame(rows, columns=[
        'transaction_id', 'smart_contract_id', 'funder', 'payment_type', 'student_id',
        'payment_timestamp', 'month', 'day', 'hour', 'minute', 'day_of_week', 'is_weekend',
        'contract_version', 'contract_start_date', 'contract_length_months', 'base_stipend_amount_from_contract',
        'total_contract_value', 'is_first_payment_for_contract', 'first_payment_date_for_contract',
        'months_covered_by_this_payment', 'amount_from_contract_read', 'expected_payment_amount', 'final_payment_amount',
        'remaining_contract_balance', 'payment_frequency_type', 'student_active_start_date', 'student_active_end_date',
        'is_student_active_at_payment', # New column
        'processing_latency_ms', 'transaction_status', 'is_anomaly', 'anomaly_type'
    ])

    # Convert date columns to datetime objects
    df['contract_start_date'] = pd.to_datetime(df['contract_start_date'])
    df['first_payment_date_for_contract'] = pd.to_datetime(df['first_payment_date_for_contract'])
    df['student_active_start_date'] = pd.to_datetime(df['student_active_start_date'])
    df['student_active_end_date'] = pd.to_datetime(df['student_active_end_date'])


    # --- Post-Generation Anomaly Flagging (API-centric rules) ---

    # Rule: Duplicate Payments
    df['temp_datetime_key'] = df['payment_timestamp'].dt.to_period('H')
    duplicate_payment_check_cols = ['student_id', 'smart_contract_id', 'funder', 'payment_type', 'final_payment_amount', 'temp_datetime_key']
    df['duplicate_count'] = df.groupby(duplicate_payment_check_cols).transform('size')
    df.loc[(df['duplicate_count'] > 1) & (df['is_anomaly'] == 0), 'is_anomaly'] = 1
    df.loc[(df['duplicate_count'] > 1) & (df['anomaly_type'] == "None"), 'anomaly_type'] = "Duplicate_Payment"
    df = df.drop(columns=['temp_datetime_key', 'duplicate_count'])


    # Rule: Final Payment Amount Significantly Different from Expected (Pro-rata aware)
    epsilon = 1e-6 # To prevent division by zero for percentage difference
    df['amount_diff_from_expected'] = abs(df['final_payment_amount'] - df['expected_payment_amount'])
    df['relative_amount_diff_from_expected'] = df['amount_diff_from_expected'] / (df['expected_payment_amount'].replace(0, epsilon))
    amount_tolerance_percent = 0.01 # 1% tolerance for minor noise/rounding

    significant_diff_mask = (df['relative_amount_diff_from_expected'] > amount_tolerance_percent) & (df['is_anomaly'] == 0)
    df.loc[significant_diff_mask, 'is_anomaly'] = 1
    df.loc[significant_diff_mask, 'anomaly_type'] = "Incorrect_Payment_Amount"
    df = df.drop(columns=['amount_diff_from_expected', 'relative_amount_diff_from_expected'])

    # Rule: Payment Made After Contract Balance Reached Zero or Negative
    df.loc[(df['remaining_contract_balance'] < -df['base_stipend_amount_from_contract']) & (df['is_anomaly'] == 0), 'is_anomaly'] = 1
    df.loc[(df['remaining_contract_balance'] < -df['base_stipend_amount_from_contract']) & (df['anomaly_type'] == "None"), 'anomaly_type'] = "Overpaid_Contract_Balance"


    # Rule: Payments with FAILED_TEMP/PERM status but amount is normal
    df.loc[(df['transaction_status'].isin(['FAILED_TEMP', 'FAILED_PERM'])) & (df['is_anomaly'] == 0), 'is_anomaly'] = 1
    df.loc[(df['transaction_status'].isin(['FAILED_TEMP', 'FAILED_PERM'])) & (df['anomaly_type'] == "None"), 'anomaly_type'] = "Failed_Transaction_Unexpected"
    
    # Rule: Payment for 0 expected amount but non-zero final_payment_amount (e.g. after contract fully paid)
    df.loc[(df['expected_payment_amount'] == 0) & (df['final_payment_amount'] > 0) & (df['is_anomaly'] == 0), 'is_anomaly'] = 1
    df.loc[(df['expected_payment_amount'] == 0) & (df['final_payment_amount'] > 0) & (df['anomaly_type'] == "None"), 'anomaly_type'] = "Unexpected_Payment_No_Expectation"

    # NEW: Rule: Payments where is_student_active_at_payment is False but transaction was SUCCESS
    df.loc[(df['is_student_active_at_payment'] == False) & (df['transaction_status'] == 'SUCCESS') & (df['is_anomaly'] == 0), 'is_anomaly'] = 1
    df.loc[(df['is_student_active_at_payment'] == False) & (df['transaction_status'] == 'SUCCESS') & (df['anomaly_type'] == "None"), 'anomaly_type'] = "Payment_to_Inactive_Student"

    print(f"\n[*] Generated {len(df):,} rows of data with {df['is_anomaly'].sum():,} anomalies/potential fraud.")
    print(f"[*] Final anomaly rate: {df['is_anomaly'].mean() * 100:.2f}%")
    print("\nAnomaly Type Distribution (after post-processing rules):")
    print(df['anomaly_type'].value_counts())

    df.to_csv('fraud_data.csv', index=False)
    print("Generated fraud_data.csv.")
    return df

def database_bulk_insert(df):
    """
    Bulk insert the DataFrame rows into the fraud_detection_model_data table using SQLAlchemy.
    Sets modified_by and added_by to 'SYSTEM_DATA_GENERATOR'.
    """
    data_dicts = df.to_dict(orient='records')
    for d in data_dicts:
        d['modified_by'] = 'SYSTEM_DATA_GENERATOR'
        d['added_by'] = 'SYSTEM_DATA_GENERATOR'
    db = SessionLocal()
    try:
        add_training_data_bulk(db, data_dicts)
        print(f"Inserted {len(data_dicts)} rows into fraud_detection_model_data table.")
    finally:
        db.close()

# Filter function updated for the new context
def analyze_automated_payment_anomalies(csv_path='automated_payment_anomalies.csv'):
    df = pd.read_csv(csv_path)

    print(f"\n--- Analysis of Automated Payment Anomalies ({csv_path}) ---")
    print(f"Total records: {len(df)}")
    print(f"Total anomalies detected: {df['is_anomaly'].sum()}")
    print("\nAnomaly Type Distribution:")
    print(df['anomaly_type'].value_counts())

    potential_undetected_contract_issue = df[
        (df['contract_version'] == 'v1.0') &
        (df['anomaly_type'].isin(["Incorrect_Payment_Amount"])) &
        (df['transaction_status'] == 'SUCCESS')
    ]
    print(f"\nFound {len(potential_undetected_contract_issue)} potential undetected contract amount issues (v1.0 contract, SUCCESS status, but amount mismatch).")

    unexplained_high_latency = df[
        (df['processing_latency_ms'] > 4000) &
        (df['is_anomaly'] == 0)
    ]
    print(f"\nFound {len(unexplained_high_latency)} unexplained high latency payments not flagged as an anomaly.")

    low_tuition_success = df[
        (df['payment_type'] == 'tuition') &
        (df['final_payment_amount'] < df['expected_payment_amount'] * 0.5) &
        (df['transaction_status'] == 'SUCCESS') &
        (df['is_anomaly'] == 0)
    ]
    print(f"\nFound {len(low_tuition_success)} tuition payments that are significantly lower than expected and were successful (potential unnoticed underspend).")

    overpaid_contracts = df[
        (df['remaining_contract_balance'] < 0) &
        (df['anomaly_type'] == "None")
    ]
    print(f"\nFound {len(overpaid_contracts)} payments where contract balance went negative unexpectedly.")

    mid_contract_bundle_anomaly = df[
        (df['payment_frequency_type'] == 'MONTHLY') &
        (~df['is_first_payment_for_contract']) &
        (df['months_covered_by_this_payment'] > 1) &
        (df['is_anomaly'] == 0)
    ]
    print(f"\nFound {len(mid_contract_bundle_anomaly)} potential mid-contract monthly bundles (unusual).")

    # NEW: Payments to inactive students that were successful and not flagged
    inactive_successful_payments = df[
        (df['is_student_active_at_payment'] == False) &
        (df['transaction_status'] == 'SUCCESS') &
        (df['is_anomaly'] == 0) # This should be 0 if the new rule catches them correctly
    ]
    print(f"\nFound {len(inactive_successful_payments)} payments to inactive students that were successful and not flagged by the direct rule.")




if __name__ == "__main__":
    generated_df = generate_automated_payment_fraud_data()
    analyze_automated_payment_anomalies()