from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100)
    tenure_months: int = Field(..., ge=0, le=240)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    support_calls: int = Field(..., ge=0, le=50)
    contract_type: str
    payment_method: str
    internet_service: str
    paperless_billing: bool
    senior_citizen: bool


class PredictionResponse(BaseModel):
    churn_prediction: bool
    churn_probability: float
    model_name: str
