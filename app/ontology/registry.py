from typing import Type
from sqlmodel import SQLModel

_registry: dict[str, Type[SQLModel]] = {}


def register(type_name: str):
    def decorator(cls: Type[SQLModel]):
        _registry[type_name] = cls
        return cls
    return decorator


def get_type(type_name: str) -> Type[SQLModel]:
    if type_name not in _registry:
        raise KeyError(f"Unknown ontology type: {type_name}")
    return _registry[type_name]


def list_types() -> list[str]:
    return list(_registry.keys())
