from typing import TypeVar
from inferx.models.base_model import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)

_MODELS_REGISTRY: dict[str, type[BaseModel]] = {}


def register_model(model_cls: type[ModelT]) -> type[ModelT]:
    architecture = model_cls.__name__
    if architecture in _MODELS_REGISTRY:
        raise ValueError(f"Model already registered: {architecture}")

    _MODELS_REGISTRY[architecture] = model_cls

    return model_cls


def get_model_class(architectures: list[str] | None) -> type[BaseModel]:
    if not architectures:
        raise ValueError(f"Model config has no architectures")

    for architecture in architectures:
        if architecture in _MODELS_REGISTRY:
            return _MODELS_REGISTRY[architecture]

    raise ValueError(f"Unsupported architectures: {architectures}")
