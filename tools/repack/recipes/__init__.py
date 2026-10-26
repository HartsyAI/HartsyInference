"""Recipe registry: model name -> Recipe instance.

Add a model by writing recipes/<model>.py exposing a module-level RECIPE, then registering
it here. Only license-permissive models should ever have redistribute=True.
"""

from __future__ import annotations

from recipe import Recipe
from recipes.hurston1 import RECIPE as HURSTON1
from recipes.kokoro import RECIPE as KOKORO

REGISTRY: dict[str, Recipe] = {
    KOKORO.name: KOKORO,
    HURSTON1.name: HURSTON1,
}


def get(name: str) -> Recipe:
    if name not in REGISTRY:
        raise KeyError(f"unknown recipe '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def names() -> list[str]:
    return sorted(REGISTRY)
