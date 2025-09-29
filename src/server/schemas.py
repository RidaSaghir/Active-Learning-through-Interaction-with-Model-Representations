from pydantic import BaseModel, validator
from typing import List

class EmbeddingPayload(BaseModel):
    iteration: int
    embedding_shape: List[int]
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
    @validator("indices")
    def _len_match_indices(cls, v, values):
        fn = values.get("filenames", [])
        if fn and len(v) != len(fn):
            raise ValueError("indices and filenames must have the same length")
        return v

class HumanAnnotation(BaseModel):
    filenames: List[str]
    indices: List[int]
    labels: List[int]

    @validator("labels")
    def _len_match_labels(cls, v, values):
        idx = values.get("indices", [])
        if idx and len(v) != len(idx):
            raise ValueError("labels and indices must have the same length")
        return v
