from pydantic import BaseModel
from typing import List

class EmbeddingPayload(BaseModel):
    iteration: int
    embedding_shape: List[int]
    embeddings: List[List[float]]
    labels: List[int]
    filenames: List[str]
    label_types: List[str]

class MetricsPayload(BaseModel):
    iteration: int
    accuracy: float
    loss: float

class AnnotationRequest(BaseModel):
    filenames: List[str]
    indices: List[int]
