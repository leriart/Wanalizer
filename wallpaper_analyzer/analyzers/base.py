"""Base analyzer interface."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

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

    # ------------------------------------------------------------------
    # Content-aware tag detection — used by the AI renamer so the same
    # analyzer pipeline that picks a category also picks descriptive tags.
    # ------------------------------------------------------------------

    def detect_tags(self,
                    image_path: str,
                    max_tags: int = 8,
                    ) -> Tuple[List[str], Optional[str]]:
        """Return (tags, subject) for `image_path` using this analyzer.

        Default implementation runs `analyze()` then delegates to the
        multi-signal `classify_with_confidence` scorer, which produces
        a content-aware tag set (`suggest_tags(profile)`) that covers
        anime / cyberpunk / space / neon / portrait / pixel-art /
        minimalist / etc. — not just colours.

        Subclasses can override to expose additional signals (e.g.
        CLIP's full vocabulary). The returned list is bounded by
        `max_tags`; `subject` is the most informative tag, or None.

        This is the single source of truth for "what tags describe
        this image" — the AI renamer and the category classifier
        share it so the rename output is always consistent with the
        category the analyzer would have assigned.
        """
        try:
            profile = self.analyze(image_path)
        except Exception as e:
            return [], None
        return _profile_tags(profile, max_tags=max_tags)

    def detect_tags_batch(self,
                          files: List[str],
                          max_tags: int = 8,
                          ) -> Dict[str, Tuple[List[str], Optional[str]]]:
        """Batch wrapper around `detect_tags` with per-file error isolation."""
        out: Dict[str, Tuple[List[str], Optional[str]]] = {}
        for f in files:
            try:
                out[f] = self.detect_tags(f, max_tags=max_tags)
            except Exception:
                out[f] = ([], None)
        return out


def _profile_tags(profile: dict, max_tags: int) -> Tuple[List[str], Optional[str]]:
    """Derive a content-aware (tags, subject) tuple from a profile dict.

    Uses the same `classify_with_confidence` pipeline the category
    scorer uses, so the tags returned here describe the SAME content
    signals that drove the category decision. This is the key reason
    the AI renamer now uses this entry point instead of the older
    `suggest_tags_for_category` which was biased toward short colour
    tokens and noisy co-occurrence expansion.
    """
    try:
        from .. import classify as _classify
        info = _classify.classify_with_confidence(profile)
    except Exception:
        info = {}
    raw_tags = info.get("tags") or set()
    # Normalise to a list, preserving insertion order. `suggest_tags`
    # returns a `set` but the source list is built in priority order
    # so we can recover the priority via re-running suggest_tags.
    if not raw_tags:
        # Fallback: run suggest_tags directly so the caller always
        # gets SOMETHING content-aware (not just colours).
        try:
            from ..tag_suggester import suggest_tags
            raw_tags = suggest_tags(profile, max_tags=max_tags)
        except Exception:
            raw_tags = set()
    if not raw_tags:
        return [], None
    tag_list = list(raw_tags)[:max_tags]
    subject = tag_list[0] if tag_list else None
    return tag_list, subject
