"""Base analyzer interface."""
from abc import ABC, abstractmethod
from typing import Optional

from ..categories import CATEGORIES


class BaseAnalyzer(ABC):
    """Abstract base for all classification analyzers."""

    name: str = "base"

    @abstractmethod
    def analyze(self, image_path: str) -> dict:
        """Analyze image and return a feature profile dict."""

    @abstractmethod
    def classify(self, profile: dict) -> Optional[str]:
        """Classify an image profile into a category. Returns category name or None."""

    def describe(self, image_path: str) -> Optional[str]:
        """Optional image description. Returns text or None."""
        return None

    def nsfw_score(self, image_path: str) -> Optional[float]:
        """Optional NSFW scoring. Returns 0..1 or None."""
        return None

    def get_mode_name(self) -> str:
        return self.name
