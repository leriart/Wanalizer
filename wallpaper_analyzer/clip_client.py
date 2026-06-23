"""CLIP-based zero-shot image classification.

Improvements over the previous implementation:

- **CLIPEngine** singleton with embedding caches (image + text) so
  re-classifying the same image doesn't re-run the encoder. CPU CLIP
  is ~200ms/image for ViT-B/32; the cache cuts repeated calls to
  <1ms.
- **Device auto-detection** (CUDA -> MPS -> CPU) with informative
  load status. The previous code silently fell back to CPU without
  telling the user.
- **Multi-prompt averaging**: each category is scored against 3-5
  diverse prompt templates ("a photo of X wallpaper", "X style art",
  "wallpaper in X aesthetic", etc.) and the resulting text
  embeddings are averaged before cosine similarity. Single-prompt
  CLIP is brittle: "a photo of X" works for some categories and
  fails for others. Averaging across diverse prompts is the standard
  trick that makes CLIP zero-shot usable in practice.
- **Category-type-aware templates**: the template set adapts to
  whether the category is a palette theme (Catppuccin/Dracula/Nord/
  TokyoNight/Gruvbox/Everforest/Neon), a known IP/game (Minecraft/
  Cyberpunk), an art style (Anime/Pixel-Art/Minimalist) or a generic
  noun (Space/Landscape/Portrait). This is what makes CLIP useful on
  the user's curated wallpaper library where many category names are
  uninformative on their own.
- **Softmax temperature normalisation** instead of raw cosine
  similarity, so the per-category scores are comparable to the other
  signals (fingerprint, prompt TF-IDF, etc.).
- **Integration with the main classifier**: `score_image()` returns a
  dict keyed by category that the `classify._clip_signal` function
  picks up from the profile, so CLIP is now a first-class signal in
  `classify_with_confidence` instead of a separate analyzer mode.
"""
from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .analyzers.base import BaseAnalyzer
from .categories import CATEGORIES, get_category_tags
from .profile import get_image_profile


# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------

HAS_TORCH = False
HAS_CLIP = False

try:
    import torch  # type: ignore
    HAS_TORCH = True
except Exception:
    pass

try:
    import clip  # type: ignore
    HAS_CLIP = True
except Exception:
    pass


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Generic templates used for every category. Averaging these gives a much
# more robust text embedding than any single prompt.
_GENERIC_TEMPLATES = [
    "a photo of {c} wallpaper",
    "a {c} style wallpaper",
    "wallpaper in {c} aesthetic",
    "{c} themed artwork",
]

# Extra templates for specific category types, picked automatically from
# the category name and tags below.
_THEME_TEMPLATES = [
    "wallpaper in {c} color palette",
    "{c} theme digital wallpaper",
    "wallpaper with {c} color scheme",
]
_GAME_TEMPLATES = [
    "{c} game screenshot",
    "{c} video game artwork",
    "{c} game graphics",
]
_ART_TEMPLATES = [
    "{c} illustration artwork",
    "{c} digital art piece",
    "{c} drawn artwork",
]
_PHOTO_TEMPLATES = [
    "photograph of {c}",
    "{c} photograph",
]
_ABSTRACT_TEMPLATES = [
    "{c} pattern design",
    "{c} abstract wallpaper",
    "minimal {c} composition",
]

# Map category name (lowercased) -> extra template set. These trigger
# automatically when the category name matches.
_NAME_TEMPLATES: Dict[str, List[str]] = {
    "anime": _ART_TEMPLATES,
    "minecraft": _GAME_TEMPLATES + ["minecraft block world", "voxel game world"],
    "pixel-art": _ART_TEMPLATES + ["8-bit pixel artwork", "retro game graphics"],
    "cyberpunk": _GAME_TEMPLATES + ["cyberpunk neon city", "futuristic tech art"],
    "space": _ABSTRACT_TEMPLATES + ["outer space scene", "cosmic landscape"],
    "landscape": _PHOTO_TEMPLATES + ["natural scenery photograph"],
    "minimalist": _ABSTRACT_TEMPLATES + ["clean minimal design", "simple composition"],
    "monochrome": _ABSTRACT_TEMPLATES + ["black and white photography", "grayscale art"],
    "portrait": _PHOTO_TEMPLATES + ["human portrait photograph", "face photograph"],
}

# Theme/palette category names. These are uninformative on their own
# (CLIP knows the Dracula novel better than the colour theme) so they
# need explicit theme templates.
_THEME_NAMES = {
    "catppuccin", "dracula", "nord", "tokyonight", "gruvbox",
    "everforest", "neon", "monochrome",
}

# NSFW prompt bank used by the explicit NSFW detector.
_NSFW_PROMPTS = [
    "nsfw explicit adult content",
    "suggestive provocative revealing",
    "nudity exposed skin erotic",
    "sexual content adult material",
]
_NSFW_SAFE_PROMPTS = [
    "safe family-friendly photo",
    "sfw everyday photograph",
    "innocent landscape image",
]


def _prompt_set_for(cat: str, cat_tags: Sequence[str] = ()) -> List[str]:
    """Build the list of prompt templates for `cat`.

    Always returns at least `_GENERIC_TEMPLATES` plus category-specific
    extras chosen by name and tag content. Multi-prompt averaging is
    what makes CLIP zero-shot robust against ambiguous category names.
    """
    cat_lower = cat.lower()
    cat_tags_lower = {t.lower() for t in cat_tags}

    prompts = list(_GENERIC_TEMPLATES)
    # Theme categories get explicit theme-palette templates.
    if cat_lower in _THEME_NAMES:
        prompts.extend(_THEME_TEMPLATES)
    # Name-based extras.
    name_extras = _NAME_TEMPLATES.get(cat_lower)
    if name_extras:
        prompts.extend(name_extras)
    # Tag-based extras.
    if {"game", "voxel", "minecraft", "8-bit", "pixel-art"} & cat_tags_lower:
        prompts.extend(_GAME_TEMPLATES)
    if {"art", "illustration", "drawing", "digital-art"} & cat_tags_lower:
        prompts.extend(_ART_TEMPLATES)
    if {"photo", "photograph"} & cat_tags_lower:
        prompts.extend(_PHOTO_TEMPLATES)
    if {"abstract", "pattern", "minimalist", "monochrome"} & cat_tags_lower:
        prompts.extend(_ABSTRACT_TEMPLATES)
    # Substitute the category name into each template.
    return [p.format(c=cat) for p in prompts]


# ---------------------------------------------------------------------------
# Engine singleton (lazy model load + caches)
# ---------------------------------------------------------------------------


class CLIPEngine:
    """Lazy-loaded CLIP model with image + text embedding caches.

    One engine per process. `load()` is thread-safe (uses a lock) so
    the GUI worker thread and the main classifier can both call into
    it without racing the model load.
    """

    def __init__(self, model_name: str = "ViT-B/32"):
        self.model_name = model_name
        self._lock = threading.Lock()
        self._loaded = False
        self._available: Optional[bool] = None
        self._load_error: Optional[str] = None
        self._device: Optional[str] = None
        self.model = None
        self.preprocess = None
        # Caches: image_cache[(abs_path, mtime, size)] -> tensor
        #          text_cache[prompt_text] -> tensor
        self._image_cache: Dict[Tuple[str, int, int], "torch.Tensor"] = {}
        self._text_cache: Dict[str, "torch.Tensor"] = {}

    @property
    def available(self) -> bool:
        """True iff the model loaded successfully. Cached."""
        if self._available is None:
            self.load()
        return self._available

    @property
    def device(self) -> str:
        """The torch device the model is loaded on."""
        if self._device is None:
            self.load()
        return self._device or "cpu"

    @property
    def load_error(self) -> Optional[str]:
        """The exception message if loading failed, for diagnostics."""
        return self._load_error

    def load(self) -> bool:
        """Load the CLIP model if not already loaded. Returns success."""
        if self._loaded:
            return self._available or False
        with self._lock:
            if self._loaded:
                return self._available or False
            if not (HAS_TORCH and HAS_CLIP):
                self._available = False
                self._load_error = "torch or clip not installed"
                self._loaded = True
                return False
            try:
                import clip as _clip  # type: ignore
                import torch as _torch  # type: ignore
                device = self._select_device(_torch)
                self.model, self.preprocess = _clip.load(
                    self.model_name, device=device,
                )
                self.model.eval()
                self._device = device
                self._available = True
                self._loaded = True
                return True
            except Exception as exc:
                self._available = False
                self._load_error = f"{type(exc).__name__}: {exc}"
                self._loaded = True
                return False

    @staticmethod
    def _select_device(torch_mod) -> str:
        """Pick the best available torch device (cuda > mps > cpu)."""
        try:
            if torch_mod.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        try:
            if hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    # Image embedding
    # ------------------------------------------------------------------

    def encode_image(self, image_path: str) -> Optional["torch.Tensor"]:
        """Encode a single image into a normalised 1xD tensor.

        Cached by (abs_path, mtime, file-size) so re-classifying the
        same file is instant. Returns None if CLIP isn't available or
        the image can't be opened.
        """
        if not self.available:
            return None
        try:
            st = os.stat(image_path)
            cache_key = (os.path.abspath(image_path), int(st.st_mtime), int(st.st_size))
        except OSError:
            return None
        cached = self._image_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            img = Image.open(image_path).convert("RGB")
            inp = self.preprocess(img).unsqueeze(0).to(self._device)
            import torch as _torch
            with _torch.no_grad():
                feat = self.model.encode_image(inp)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            self._image_cache[cache_key] = feat.detach().cpu()
            try:
                img.close()
            except Exception:
                pass
            return self._image_cache[cache_key]
        except Exception:
            return None

    def encode_images(self, image_paths: Sequence[str]) -> Dict[str, Optional["torch.Tensor"]]:
        """Encode multiple images in a single batch.

        Much faster than calling `encode_image` N times when CLIP is
        on GPU. On CPU it's roughly the same as N sequential encodes
        but with less Python overhead.
        """
        out: Dict[str, Optional["torch.Tensor"]] = {p: None for p in image_paths}
        if not self.available or not image_paths:
            return out
        # Resolve cache hits first.
        pending: List[str] = []
        for p in image_paths:
            try:
                st = os.stat(p)
                key = (os.path.abspath(p), int(st.st_mtime), int(st.st_size))
            except OSError:
                continue
            cached = self._image_cache.get(key)
            if cached is not None:
                out[p] = cached
            else:
                pending.append(p)
        if not pending:
            return out
        import torch as _torch
        try:
            imgs = []
            valid_paths: List[str] = []
            for p in pending:
                try:
                    imgs.append(self.preprocess(Image.open(p).convert("RGB")))
                    valid_paths.append(p)
                except Exception:
                    continue
            if not imgs:
                return out
            batch = _torch.stack(imgs).to(self._device)
            with _torch.no_grad():
                feats = self.model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_cpu = feats.detach().cpu()
            for p, f in zip(valid_paths, feats_cpu):
                try:
                    st = os.stat(p)
                    key = (os.path.abspath(p), int(st.st_mtime), int(st.st_size))
                except OSError:
                    continue
                self._image_cache[key] = f
                out[p] = f
        except Exception:
            pass
        return out

    # ------------------------------------------------------------------
    # Text embedding
    # ------------------------------------------------------------------

    def encode_text(self, prompt: str) -> Optional["torch.Tensor"]:
        """Encode a single prompt. Cached."""
        if not self.available or not prompt:
            return None
        cached = self._text_cache.get(prompt)
        if cached is not None:
            return cached
        import torch as _torch
        try:
            tok = clip.tokenize([prompt], truncate=True).to(self._device)  # type: ignore
            with _torch.no_grad():
                feat = self.model.encode_text(tok)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            self._text_cache[prompt] = feat.detach().cpu()
            return self._text_cache[prompt]
        except Exception:
            return None

    def encode_texts(self, prompts: Sequence[str]) -> List[Optional["torch.Tensor"]]:
        """Encode multiple prompts in a batch. Cached."""
        out: List[Optional["torch.Tensor"]] = [None] * len(prompts)
        if not self.available or not prompts:
            return out
        pending_idx: List[int] = []
        pending_prompts: List[str] = []
        for i, p in enumerate(prompts):
            cached = self._text_cache.get(p)
            if cached is not None:
                out[i] = cached
            else:
                pending_idx.append(i)
                pending_prompts.append(p)
        if not pending_prompts:
            return out
        import torch as _torch
        try:
            tok = clip.tokenize(pending_prompts, truncate=True).to(self._device)  # type: ignore
            with _torch.no_grad():
                feats = self.model.encode_text(tok)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            feats_cpu = feats.detach().cpu()
            for j, i in enumerate(pending_idx):
                self._text_cache[prompts[i]] = feats_cpu[j]
                out[i] = feats_cpu[j]
        except Exception:
            pass
        return out

    def invalidate_cache(self) -> None:
        """Clear both caches (call after rebuilding category configs)."""
        self._image_cache.clear()
        self._text_cache.clear()


# Module-level engine. One per process.
_ENGINE: Optional[CLIPEngine] = None


def get_engine(model_name: Optional[str] = None) -> CLIPEngine:
    """Return the singleton CLIPEngine for this process.

    `model_name` overrides the default; if None we read
    `clip_model` from the persisted user settings (so the AI Models
    page selection is honoured).
    """
    global _ENGINE
    if model_name is None:
        try:
            from .settings import load_settings
            model_name = load_settings().get("clip_model", "ViT-B/32")
        except Exception:
            model_name = "ViT-B/32"
    if _ENGINE is None or _ENGINE.model_name != model_name:
        _ENGINE = CLIPEngine(model_name)
    return _ENGINE


def available_clip_models() -> List[str]:
    """Return the list of CLIP model names whose weights are already
    downloaded in this environment (via `clip.available_models()`).
    Returns [] if torch/clip aren't installed.
    """
    if not (HAS_TORCH and HAS_CLIP):
        return []
    try:
        import clip as _clip  # type: ignore
        return list(_clip.available_models())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Public scoring API
# ---------------------------------------------------------------------------

def _category_text_embedding(cat: str, cat_tags: Sequence[str] = ()) -> Optional["torch.Tensor"]:
    """Average the text embeddings for the multi-prompt set of `cat`."""
    engine = get_engine()
    prompts = _prompt_set_for(cat, cat_tags)
    feats = engine.encode_texts(prompts)
    valid = [f for f in feats if f is not None]
    if not valid:
        return None
    import torch as _torch
    avg = _torch.stack(valid).mean(dim=0)
    avg = avg / avg.norm()
    return avg


def _build_category_text_matrix() -> Optional[Tuple["torch.Tensor", Tuple[str, ...]]]:
    """Build a (N_categories, D) text-embedding matrix using multi-prompt
    averaging per category. Cached at module level so re-classifications
    don't re-encode every category's prompt set.
    """
    import torch as _torch
    sig_src = tuple(sorted(
        (cat, tuple(sorted(get_category_tags(cat))))
        for cat in CATEGORIES
    ))
    sig = (get_engine().model_name, sig_src)
    cached = getattr(_build_category_text_matrix, "_cache", None)
    if cached and cached[0] == sig:
        return cached[1]
    matrix_rows: List["torch.Tensor"] = []
    valid_cats: List[str] = []
    for cat in CATEGORIES:
        emb = _category_text_embedding(cat, get_category_tags(cat))
        if emb is None:
            return None
        matrix_rows.append(emb)
        valid_cats.append(cat)
    if not matrix_rows:
        return None
    matrix = _torch.stack(matrix_rows)  # (N, D)
    out = (matrix, tuple(valid_cats))
    _build_category_text_matrix._cache = (sig, out)  # type: ignore[attr-defined]
    return out


def score_image(image_path: str, temperature: float = 0.0) -> Optional[Dict[str, float]]:
    """Score `image_path` against every category.

    Returns a dict {category: score in [0, 1]}, or None if CLIP isn't
    available. By default (`temperature=0`) returns the raw cosine
    similarity rescaled to [0, 1] via `(sim + 1) / 2`. This preserves
    the magnitude difference between categories (top match ~0.7, poor
    matches ~0.5) which is what the multi-signal classifier needs to
    distinguish a confident CLIP win from an ambiguous one.

    Pass `temperature > 0` for softmax-temperature normalisation
    instead. Softer when temperature is high (e.g. 0.3), sharper when
    low (e.g. 0.07). Defaults to raw cosine because the multi-signal
    combination in `classify._clip_signal` already weighs CLIP against
    other signals - extra softmax would only compress the dynamic
    range further.
    """
    engine = get_engine()
    if not engine.available:
        return None
    matrix_data = _build_category_text_matrix()
    if matrix_data is None:
        return None
    matrix, valid_cats = matrix_data
    img_feat = engine.encode_image(image_path)
    if img_feat is None:
        return None
    img_feat = img_feat.to(matrix.device)
    sims = (img_feat @ matrix.T).squeeze(0)  # (N,)
    if temperature <= 0:
        # No temperature scaling: return raw cosine in [0, 1].
        return {cat: float((s + 1) / 2) for cat, s in zip(valid_cats, sims.detach().cpu().numpy())}
    probs = (sims / temperature).softmax(dim=-1)
    return {cat: float(p) for cat, p in zip(valid_cats, probs.detach().cpu().numpy())}


def score_image_with_raw_cosine(
    image_path: str,
) -> Optional[Dict[str, float]]:
    """Like `score_image` but returns raw cosine similarity instead of
    softmax probabilities. Useful when the caller wants to combine
    the raw scores with other signals itself.
    """
    engine = get_engine()
    if not engine.available:
        return None
    matrix_data = _build_category_text_matrix()
    if matrix_data is None:
        return None
    matrix, valid_cats = matrix_data
    img_feat = engine.encode_image(image_path)
    if img_feat is None:
        return None
    img_feat = img_feat.to(matrix.device)
    sims = (img_feat @ matrix.T).squeeze(0)  # (N,)
    return {cat: float(s) for cat, s in zip(valid_cats, sims.detach().cpu().numpy())}


# ---------------------------------------------------------------------------
# NSFW detector (uses its own prompt bank)
# ---------------------------------------------------------------------------

def nsfw_score(image_path: str) -> Optional[float]:
    """Return a calibrated NSFW score in [0, 1] for `image_path`, or
    None if CLIP isn't available.

    CLIP is calibrated against a small bank of NSFW / non-NSFW prompts
    and softmax-aggregated. The NSFW mass is normalised against a
    "safe" prompt bank so a normal image (no skin, no NSFW content)
    gets a small score. Amplification factor 2.0 brings the score
    into a more useful range.
    """
    import torch as _torch
    engine = get_engine()
    if not engine.available:
        return None
    img_feat = engine.encode_image(image_path)
    if img_feat is None:
        return None
    feats = engine.encode_texts(list(_NSFW_SAFE_PROMPTS) + list(_NSFW_PROMPTS))
    if any(f is None for f in feats):
        return None
    text_feats = _torch.stack([f for f in feats]).to(engine.device)
    sims = (img_feat.to(text_feats.device) @ text_feats.T).squeeze(0)
    probs = sims.softmax(dim=-1)
    n_safe = len(_NSFW_SAFE_PROMPTS)
    nsfw_mass = float(probs[n_safe:].sum().item())
    return min(1.0, nsfw_mass * 2.0)


# ---------------------------------------------------------------------------
# Backwards-compatible API (legacy module-level function)
# ---------------------------------------------------------------------------

def load_clip_model(model_name: str = "ViT-B/32") -> bool:
    """Legacy entrypoint. Returns True iff CLIP is available and loaded.

    Use `get_engine()` directly in new code.
    """
    return get_engine(model_name).available


# ---------------------------------------------------------------------------
# Analyzer mode (legacy `classify` interface)
# ---------------------------------------------------------------------------

class CLIPAnalyzer(BaseAnalyzer):
    """Analyzer using OpenAI CLIP for zero-shot classification.

    Stores per-category softmax scores in the profile under
    `clip_score_{category}` keys so the main classifier
    (`_clip_signal` in classify.py) can pick them up.
    """

    name = "clip"

    def __init__(self, settings: dict = None):
        self.settings = settings or {}
        self.model_name = self.settings.get("clip_model", "ViT-B/32")
        self.engine = get_engine(self.model_name)

    def _ensure_model(self) -> bool:
        return self.engine.available

    def analyze(self, image_path: str) -> dict:
        profile = get_image_profile(image_path)
        profile["mode"] = "clip"
        if self.engine.available:
            scores = score_image(image_path)
            if scores:
                for cat, p in scores.items():
                    profile[f"clip_score_{cat}"] = p
                raw = score_image_with_raw_cosine(image_path)
                if raw:
                    for cat, s in raw.items():
                        profile[f"clip_cosine_{cat}"] = s
                ns = nsfw_score(image_path)
                if ns is not None:
                    profile["clip_nsfw"] = ns
        profile["mode"] = "clip"
        return profile

    def classify(self, profile: dict) -> Optional[str]:
        scores = {
            cat: profile[f"clip_score_{cat}"]
            for cat in CATEGORIES
            if f"clip_score_{cat}" in profile
        }
        if scores:
            best = max(scores, key=scores.get)
            if scores[best] > 0.10:
                return best
        from .classify import classify_by_tags, derive_tags_from_profile
        img_tags = derive_tags_from_profile(profile)
        if img_tags:
            cat = classify_by_tags(img_tags)
            if cat:
                return cat
        from .classify import palette_fallback
        return palette_fallback(profile)

    def nsfw_score(self, image_path: str) -> Optional[float]:
        return nsfw_score(image_path)
