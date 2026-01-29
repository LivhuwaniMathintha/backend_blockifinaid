from pydantic import BaseModel


class FraudRequest(BaseModel):
    amount: float
    funder: str
    payment_type: str
    month: int
    student_id: str
    duplicate_count: int
    activation_month: int