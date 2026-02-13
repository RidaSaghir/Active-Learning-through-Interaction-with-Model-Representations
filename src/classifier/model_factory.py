from .torch_models import (
    TorchModel, MLPProjector, LinearProjector
)
from .sklearn_models import SklearnModel
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import torch.nn as nn

def build_model(variant, input_dim=1024, num_classes=10):
    if variant == "yamnet_mlp_3d":
        return TorchModel(
            MLPProjector(input_dim, 3),
            nn.Linear(3, num_classes),
            num_classes
        )

    if variant == "yamnet_mlp_2d":
        return TorchModel(
            MLPProjector(input_dim, 2),
            nn.Linear(2, num_classes),
            num_classes
        )

    if variant == "yamnet_linear_2d":
        projector = LinearProjector(input_dim, 2)
        head = nn.Linear(2, num_classes)
        return TorchModel(projector, head, num_classes)

    if variant == "yamnet_linear_3d":
        projector = LinearProjector(input_dim, 3)
        head = nn.Linear(3, num_classes)
        return TorchModel(projector, head, num_classes)

    if variant == "yamnet_svm_2d":
        return SklearnModel(
            classifier=SVC(kernel="linear", probability=True, max_iter=5000),
            projector=LinearProjector(input_dim, 2)
        )

    if variant == "yamnet_logreg_2d":
        return SklearnModel(
            classifier=LogisticRegression(max_iter=10000),
            projector=LinearProjector(input_dim, 2)
        )

    raise ValueError(f"Unknown variant: {variant}")
