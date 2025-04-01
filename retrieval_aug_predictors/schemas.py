from pydantic import BaseModel, Field
from typing import List

class PredictionItem(BaseModel):
    """Individual prediction item with entity name and confidence score."""
    entity: str = Field(..., description="Name of the predicted entity")
    score: float = Field(..., description="Confidence score between 0 and 1", ge=0, le=1)

class PredictionResponse(BaseModel):
    """Response model containing a list of entity predictions."""
    predictions: List[PredictionItem] = Field(..., description="List of predicted entities with scores")
