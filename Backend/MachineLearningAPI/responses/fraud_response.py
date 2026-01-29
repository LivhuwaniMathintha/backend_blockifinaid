from pydantic import BaseModel

 
class FraudResponse(BaseModel):
    is_fraud: bool
    risk_score: float
    message: str
    is_success: bool
    time_stamp: str
    error_code: str = None