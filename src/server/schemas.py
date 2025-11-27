from pydantic import BaseModel, validator
from typing import List, Dict, Optional

class EmbeddingPayload(BaseModel):
    iteration: int
    embedding_shape: List[int]
    embeddings: List[List[float]]
    actual_labels: List[str]
    predicted_labels: List[str]
    filenames: List[str]
    indices: List[int]
    is_labeled: List[bool]
    cues: Dict[str, List[float]]
    true_codes: List[int]

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
    user: Optional[str] = "anonymous"

    @validator("labels")
    def _len_match_labels(cls, v, values):
        idx = values.get("indices", [])
        if idx and len(v) != len(idx):
            raise ValueError("labels and indices must have the same length")
        return v
