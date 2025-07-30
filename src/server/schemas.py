from pydantic import BaseModel
from typing import List

class EmbeddingPayload(BaseModel):
    iteration: int
    embedding_shape: List[int]
    embeddings: List[List[float]]
    actual_labels: List[str]
    predicted_labels: List[str]
    filenames: List[str]

class EmbeddingPayload3D(BaseModel):
    iteration: int
    embeddings: List[List[float]]
    actual_labels: List[str]
    predicted_labels: List[str]
    filenames: List[str]

class MetricsPayload(BaseModel):
    iteration: int
    accuracy: float
    loss: float

class AnnotationRequest(BaseModel):
    filenames: List[str]
    indices: List[int]
class HumanAnnotation(BaseModel):
    indices: List[int]
    labels: List[str]
