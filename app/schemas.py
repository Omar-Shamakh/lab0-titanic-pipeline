"""Pydantic schemas for API request and response validation."""

from pydantic import BaseModel, Field
from typing import List, Optional


class Passenger(BaseModel):
    """Single passenger data for prediction."""
    Pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    Sex: str = Field(..., description="'male' or 'female'")
    Age: float = Field(..., ge=0, le=120, description="Age in years")
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    Fare: float = Field(..., ge=0, description="Passenger fare")
    Embarked: str = Field(..., description="Port of embarkation: S, C, or Q")
    Family_size: int = Field(..., ge=0, description="Total family size")
    Alone: int = Field(..., ge=0, le=1, description="1 if travelling alone, 0 otherwise")
    Initial: str = Field(..., description="Title: Mr, Mrs, Miss, Master, Other")
    Fare_cat: int = Field(..., ge=0, le=4, description="Fare category bin (0-4)")
    Age_band: int = Field(..., ge=0, le=4, description="Age band bin (0-4)")

    class Config:
        json_schema_extra = {
            "example": {
                "Pclass": 1, "Sex": "female", "Age": 29.0,
                "SibSp": 0, "Parch": 0, "Fare": 211.3,
                "Embarked": "S", "Family_size": 1, "Alone": 1,
                "Initial": "Mrs", "Fare_cat": 3, "Age_band": 2
            }
        }


class PredictRequest(BaseModel):
    """Batch prediction request — accepts one or more passengers."""
    passengers: List[Passenger] = Field(..., min_length=1)


class PredictionResult(BaseModel):
    """Single prediction result."""
    passenger_index: int
    survived: bool
    survival_probability: float
    prediction_label: str


class PredictResponse(BaseModel):
    """Batch prediction response."""
    model_name: str
    total_passengers: int
    predictions: List[PredictionResult]