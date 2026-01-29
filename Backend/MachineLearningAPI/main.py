from fastapi import FastAPI
from data.fraud_request import FraudRequest
from fraud_detection_xgb_t_consumer import detect_fraud
from routers.rabbitmq import consume_from_queue
from contextlib import asynccontextmanager
import threading


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=consume_from_queue, args=("your_queue_name",), daemon=True)
    thread.start()
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {
        "message": "Welcome to the Financial Aid Machine Learning API using python"
    }


@app.post("/ml/payments/fraud_detection")
async def process_payment_info(request: FraudRequest):
    # Process the payment_info as needed
    results = detect_fraud(
        amount=request.amount,
        funder=request.funder,
        payment_type=request.payment_type,
        month=request.month,
        student_id=request.student_id,
        duplicate_count=request.duplicate_count,
        activation_month=request.activation_month
    )

    return results

