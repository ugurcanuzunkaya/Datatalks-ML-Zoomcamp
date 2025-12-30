from pydantic import BaseModel, Field
from typing import Literal


class InsuranceInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Age of the beneficiary")
    sex: Literal["male", "female"] = Field(..., description="Gender")
    bmi: float = Field(..., ge=10.0, le=60.0, description="Body Mass Index")
    children: int = Field(..., ge=0, description="Number of children covered")
    smoker: Literal["yes", "no"] = Field(..., description="Smoking status")
    region: Literal["northeast", "northwest", "southeast", "southwest"] = Field(..., description="Residential region")


class SavingsOpportunity(BaseModel):
    name: str
    description: str
    potential_savings: float
    new_cost: float


class PredictionOutput(BaseModel):
    current_cost: float
    opportunities: list[SavingsOpportunity]
