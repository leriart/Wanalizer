"""Analyzer modes for wallpaper classification.

Each mode is a thin wrapper around a backend implementation. The
canonical entry point is `get_analyzer(mode, settings)` which the
organize pipeline uses.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseAnalyzer
from .fusion_mode import FusionAnalyzer
from .lowlevel_mode import LowLevelAnalyzer

if TYPE_CHECKING:
    # Imported lazily to avoid pulling torch / requests at import time.
    from ..clip_client import CLIPAnalyzer  # noqa: F401
    from ..ollama_client import OllamaAnalyzer  # noqa: F401


__all__ = [
    "BaseAnalyzer",
    "FusionAnalyzer",
    "LowLevelAnalyzer",
    "get_analyzer",
    "available_modes",
]


AVAILABLE_MODES = {
    "lowlevel": "Classical CV algorithms only (no ML). Fast, deterministic.",
    "clip":     "OpenAI CLIP zero-shot vision-language model.",
    "fusion":   "CLIP + LowLevel CV combined (recommended when CLIP is installed).",
    "ollama":   "Local vision LLM via Ollama (LLaVA, MiniCPM-V, Moondream, ...).",
}


def get_analyzer(mode: str, settings: dict) -> BaseAnalyzer:
    """Factory: return the analyzer implementation for `mode`.

    Raises ValueError on unknown mode so callers fail fast.
    """
    mode = (mode or "lowlevel").lower()
    if mode == "lowlevel":
        return LowLevelAnalyzer(settings)
    if mode == "fusion":
        return FusionAnalyzer(settings)
    if mode == "clip":
        from ..clip_client import CLIPAnalyzer
        return CLIPAnalyzer(settings)
    if mode == "ollama":
        from ..ollama_client import OllamaAnalyzer
        return OllamaAnalyzer(settings)
    raise ValueError(
        f"Unknown analysis mode: {mode!r}. "
        f"Valid modes: {', '.join(AVAILABLE_MODES)}"
    )


def available_modes() -> dict:
    """Return a copy of the {mode: description} map."""
    return dict(AVAILABLE_MODES)