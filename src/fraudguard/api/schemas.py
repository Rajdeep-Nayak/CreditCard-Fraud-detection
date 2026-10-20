"""Request / response models. Requests are strict: unknown fields, negative amounts and
NaN/inf values are rejected with a 422 before they ever reach the model."""

from typing import List

from pydantic import BaseModel, ConfigDict, Field, create_model

from ..data import V_COLS
from .example import EXAMPLE_TRANSACTION

MAX_BATCH = 1000

_v_fields = {
    name: (float, Field(..., allow_inf_nan=False, description="Anonymised PCA component"))
    for name in V_COLS
}

Transaction = create_model(
    "Transaction",
    __config__=ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_TRANSACTION}),
    Time=(float, Field(..., ge=0, allow_inf_nan=False, description="Seconds since the first transaction")),
    **_v_fields,
    Amount=(float, Field(..., ge=0, allow_inf_nan=False, description="Transaction amount")),
)


class BatchRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    fraud_probability: float = Field(..., description="Model score in [0, 1]")
    is_fraud: bool = Field(..., description="True if the score is at or above the threshold")
    threshold: float = Field(..., description="Threshold used for this decision")


class PredictResponse(Prediction):
    model_version: str


class BatchPredictResponse(BaseModel):
    model_version: str
    threshold: float
    count: int
    flagged: int
    predictions: List[Prediction]
