from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Float
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/financial_aid_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class FraudTrainingData(Base):
    __tablename__ = "training_data_fraud_detection"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, nullable=True)
    smart_contract_id = Column(String, nullable=True)
    funder = Column(String, nullable=False)
    payment_type = Column(String, nullable=False)
    student_id = Column(String, nullable=False)
    payment_timestamp = Column(DateTime, nullable=True)
    month = Column(Integer, nullable=False)
    day = Column(Integer, nullable=True)
    hour = Column(Integer, nullable=True)
    minute = Column(Integer, nullable=True)
    day_of_week = Column(Integer, nullable=True)
    is_weekend = Column(Boolean, nullable=True)
    contract_version = Column(String, nullable=True)
    contract_start_date = Column(DateTime, nullable=True)
    contract_length_months = Column(Integer, nullable=True)
    base_stipend_amount_from_contract = Column(Float, nullable=True)
    total_contract_value = Column(Float, nullable=True)
    is_first_payment_for_contract = Column(Boolean, nullable=True)
    first_payment_date_for_contract = Column(DateTime, nullable=True)
    months_covered_by_this_payment = Column(Integer, nullable=True)
    amount_from_contract_read = Column(Float, nullable=True)
    expected_payment_amount = Column(Float, nullable=True)
    final_payment_amount = Column(Float, nullable=True)
    remaining_contract_balance = Column(Float, nullable=True)
    payment_frequency_type = Column(String, nullable=True)
    student_active_start_date = Column(DateTime, nullable=True)
    student_active_end_date = Column(DateTime, nullable=True)
    is_student_active_at_payment = Column(Boolean, nullable=True)
    processing_latency_ms = Column(Integer, nullable=True)
    transaction_status = Column(String, nullable=True)
    is_anomaly = Column(Boolean, nullable=False)
    anomaly_type = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    modified_by = Column(String, nullable=True)
    is_deleted = Column(Boolean, default=False)
    added_by = Column(String, nullable=True)

def add_training_data(db: Session, data: dict):
    new_data = FraudTrainingData(**data)
    db.add(new_data)
    db.commit()
    db.refresh(new_data)
    return new_data

def add_training_data_bulk(db: Session, data_list: list):
    """Bulk insert a list of dicts representing FraudTrainingData."""
    objs = [FraudTrainingData(**data) for data in data_list]
    db.bulk_save_objects(objs)
    db.commit()
    return objs

def update_training_data(db: Session, data_id: int, update_fields: dict):
    data = db.query(FraudTrainingData).filter(FraudTrainingData.id == data_id, FraudTrainingData.is_deleted == False).first()
    if data:
        for key, value in update_fields.items():
            setattr(data, key, value)
        data.updated_at = datetime.now()
        db.commit()
        db.refresh(data)
    return data

def delete_training_data(db: Session, data_id: int, modified_by: str = None):
    data = db.query(FraudTrainingData).filter(FraudTrainingData.id == data_id, FraudTrainingData.is_deleted == False).first()
    if data:
        data.is_deleted = True
        data.modified_by = modified_by
        data.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(data)
    return data

def get_all_training_data(db: Session):
    return db.query(FraudTrainingData).filter(FraudTrainingData.is_deleted == False).all()

def get_training_data_by_funder(db: Session, funder: str):
    return db.query(FraudTrainingData).filter(FraudTrainingData.funder == funder, FraudTrainingData.is_deleted == False).all()

def get_training_data_by_id(db: Session, data_id: int):
    return db.query(FraudTrainingData).filter(FraudTrainingData.id == data_id, FraudTrainingData.is_deleted == False).first()

# To create the table in the database (run once):
Base.metadata.create_all(bind=engine)
