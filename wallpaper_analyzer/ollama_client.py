"""Ollama analyzer mode: local vision LLMs for classification.

Supports vision-language models via the Ollama HTTP API:
  * LLaVA family (llava:7b, llava:13b, llava-phi3, llava-llama3, bakllava)
  * Llama 3.2 Vision
  * MiniCPM-V
  * Moondream (very fast, small)
  * Granite 3.1 vision (if installed)
  * Qwen2.5-VL (if installed)

Key design choices:
  * Structured JSON output (`format: "json"`) is preferred over free-text
    parsing whenever the model supports it - this avoids the brittle
    "first comma-separated word" parsing that older Ollama clients did.
  * The client auto-detects whether a model supports vision by inspecting
    its `/api/show` metadata (families contains `clip`/`mmproj`).
  * `OllamaAnalyzer.analyze()` now produces a richer profile: NSFW score,
    main subject, primary colour, mood, and structured tags (all
    populated when the model supports them, gracefully skipped otherwise).
  * Classification blends the LLM's per-category scores with the existing
    heuristic multi-signal scorer for higher reliability.
"""
import base64
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image

from .settings import load_settings
from .analyzers.base import BaseAnalyzer
from .categories import CATEGORIES, get_category_tags, get_category_prompt
from .profile import get_image_profile
from .tags import _tags_flat
from .classify import classify_by_tags, classify_with_confidence


HAS_REQUESTS = False
try:
    import requests as _requests_mod
    HAS_REQUESTS = True
except Exception:
    _requests_mod = None


# ---------------------------------------------------------------------------
# Model registry - covers vision-language models commonly available via Ollama
# ---------------------------------------------------------------------------

OLLAMA_VISION_MODELS: Dict[str, Dict] = {
    "llava": {
        "name": "LLaVA (7B)", "description": "LLaVA vision-language model",
        "size_gb": 4.5, "speed": "medium (~2-5s/img)", "accuracy": "high",
        "model_tag": "llava:7b",
        "capabilities": ["describe", "nsfw", "characters", "classify", "tags"],
    },
    "llava13b": {
        "name": "LLaVA (13B)", "description": "LLaVA 13B, more accurate",
        "size_gb": 8.0, "speed": "slow (~5-10s/img)", "accuracy": "very high",
        "model_tag": "llava:13b",
        "capabilities": ["describe", "nsfw", "characters", "classify", "tags"],
    },
    "minicpm_v": {
        "name": "MiniCPM-V (8B)", "description": "Efficient vision-language model",
        "size_gb": 5.0, "speed": "fast (~1-3s/img)", "accuracy": "high",
        "model_tag": "minicpm-v:8b",
        "capabilities": ["describe", "nsfw", "characters", "classify", "tags"],
    },
    "llama3_2_vision": {
        "name": "Llama 3.2 Vision (11B)", "description": "Meta Llama 3.2 with vision",
        "size_gb": 8.0, "speed": "medium (~3-6s/img)", "accuracy": "very high",
        "model_tag": "llama3.2-vision:11b",
        "capabilities": ["describe", "nsfw", "characters", "classify", "tags"],
    },
    "moondream": {
        "name": "Moondream (2B)", "description": "Tiny vision model, fast on CPU",
        "size_gb": 0.9, "speed": "very fast (~0.5-1s/img)", "accuracy": "medium",
        "model_tag": "moondream:latest",
        "capabilities": ["describe", "classify", "tags"],
    },
    "llava_llama3": {
        "name": "LLaVA-LLaMA3 (8B)", "description": "LLaVA built on LLaMA3 - less restrictive",
        "size_gb": 5.5, "speed": "medium (~3-6s/img)", "accuracy": "high",
        "model_tag": "llava-llama3:8b",
        "capabilities": ["describe", "nsfw", "characters", "classify", "tags"],
        "uncensored": True,
        "notes": "LLaMA3 base has lighter safety tuning. Better for NSFW than stock LLaVA.",
    },
    "llava_phi3": {
        "name": "LLaVA-Phi3 (3.8B)", "description": "Phi-3 based LLaVA, smaller and faster",
        "size_gb": 2.9, "speed": "fast (~1-3s/img)", "accuracy": "medium",
        "model_tag": "llava-phi3:3.8b",
        "capabilities": ["describe", "nsfw", "classify", "tags"],
        "uncensored": True,
        "notes": "Phi-3 base has minimal safety tuning. Good for low-VRAM systems.",
    },
    "bakllava": {
        "name": "BakLLaVA (7B)", "description": "BakLLaVA - Mistral-based, less filtered",
        "size_gb": 4.7, "speed": "medium (~3-6s/img)", "accuracy": "high",
        "model_tag": "bakllava:7b",
        "capabilities": ["describe", "nsfw", "characters", "classify", "tags"],
        "uncensored": True,
        "notes": "Mistral-7B base has lighter content filters. Often works for NSFW.",
    },
    "granite3_1_vision": {
        "name": "Granite 3.1 Vision (2B)", "description": "IBM Granite 3.1 with vision adapter",
        "size_gb": 2.5, "speed": "fast (~1-3s/img)", "accuracy": "high",
        "model_tag": "granite3.1-vision:2b",
        "capabilities": ["describe", "classify", "tags"],
    },
    "qwen2_5_vl": {
        "name": "Qwen2.5-VL (3B-7B)", "description": "Alibaba Qwen2.5 Vision-Language",
        "size_gb": 3.5, "speed": "medium (~2-5s/img)", "accuracy": "very high",
        "model_tag": "qwen2.5vl:3b",
        "capabilities": ["describe", "nsfw", "characters", "classify", "tags"],
    },
    "dolphin_llama3_text": {
        "name": "Dolphin-LLaMA3 (8B, text-only)", "description": "Uncensored LLaMA3 - text only, no vision",
        "size_gb": 4.7, "speed": "fast (text only)", "accuracy": "n/a",
        "model_tag": "dolphin-llama3:8b",
        "capabilities": ["classify", "tag"],
        "uncensored": True,
        "vision": False,
        "notes": "For tag generation from descriptions. Use with image-to-text pipeline.",
    },
}


def get_recommended_uncensored_models() -> List[Dict]:
    return [
        {
            "key": k, "name": v["name"], "tag": v["model_tag"],
            "size_gb": v.get("size_gb", 0), "speed": v.get("speed", ""),
            "notes": v.get("notes", ""),
        }
        for k, v in OLLAMA_VISION_MODELS.items() if v.get("uncensored")
    ]


# Curated tag list for "pick tags from list" prompts - ordered roughly by usefulness
# for wallpaper collections. Used by the tag-based classifier.
CURATED_TAGS_FOR_LLM = [
    # Style
    "anime", "illustration", "digital-art", "cartoon", "3d-render",
    "pixel-art", "sketch", "watercolor", "painting", "photograph",
    # Subject (people)
    "person", "portrait", "face", "character", "figure",
    # Subject (other)
    "animal", "cat", "dog", "bird", "dragon", "robot", "car", "vehicle",
    "building", "city", "castle", "temple", "tree", "flower", "mountain",
    # Composition / framing
    "landscape", "portrait-orientation", "square", "wide", "horizontal",
    "vertical", "centered", "closeup", "wide-shot", "aerial",
    # Mood / colour
    "dark", "light", "neon", "pastel", "warm", "cool", "vibrant",
    "monochrome", "black-and-white", "sepia", "vintage", "retro",
    "minimalist", "simple", "detailed", "abstract", "geometric",
    # Themes
    "nature", "forest", "ocean", "desert", "sky", "clouds", "moon",
    "sunset", "sunrise", "night", "day", "space", "stars", "galaxy",
    "fantasy", "sci-fi", "cyberpunk", "futuristic", "medieval", "magic",
    "epic", "majestic", "mysterious", "peaceful", "calm", "dramatic",
    # Effects
    "glow", "silhouette", "reflection", "fire", "ice", "lightning",
    "smoke", "fog", "rain", "snow", "leaves", "wings", "sword",
]


# ---------------------------------------------------------------------------
# Ollama HTTP client
# ---------------------------------------------------------------------------

class OllamaClient:
    """Thin client over the Ollama HTTP API.

    Compared to a raw requests wrapper, this client:
      * Caches the session + availability check
      * Adds structured JSON output support
      * Auto-detects vision-capable models
      * Provides bounded retry/back-off on 503 (model loading)
      * Surfaces a parseable `last_error` for diagnostics
    """

    def __init__(self, base_url="http://localhost:11434", model="llava:7b",
                 timeout=60, max_retries=2):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = None
        self._available: Optional[bool] = None
        self._vision_cache: Dict[str, bool] = {}
        self._last_error = ""

    @property
    def last_error(self) -> str:
        return self._last_error

    # ------------------------------------------------------------------ HTTP

    def _get_session(self):
        if self._session is None and HAS_REQUESTS:
            self._session = _requests_mod.Session()
            adapter = _requests_mod.adapters.HTTPAdapter(
                pool_connections=5, pool_maxsize=10, max_retries=0,
            )
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
        return self._session

    def _post(self, endpoint: str, payload: dict, timeout: Optional[int] = None) -> Optional[dict]:
        if not self.check():
            return None
        url = f"{self.base_url}{endpoint}"
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._get_session().post(
                    url, json=payload, timeout=timeout or self.timeout,
                )
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 503 and attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                self._last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                return None
            except Exception as e:
                self._last_error = str(e)
                if attempt < self.max_retries:
                    time.sleep(1)
                else:
                    return None
        return None

    # ------------------------------------------------------------ Discovery

    def check(self) -> bool:
        if self._available is not None:
            return self._available
        if not HAS_REQUESTS:
            self._available = False
            return False
        try:
            resp = self._get_session().get(f"{self.base_url}/api/tags", timeout=5)
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def health(self) -> dict:
        result = {
            "connected": False, "model_available": False,
            "error": "", "server_version": "", "model_count": 0,
            "vision": False,
        }
        if not HAS_REQUESTS:
            result["error"] = "requests not installed"
            return result
        try:
            resp = self._get_session().get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                result["error"] = f"HTTP {resp.status_code}"
                return result
            data = resp.json()
            result["connected"] = True
            models = data.get("models", [])
            result["model_count"] = len(models)
            names = [m.get("name", "") for m in models]
            result["model_available"] = self.model in names
            # Try to get version from /api/version
            try:
                v = self._get_session().get(f"{self.base_url}/api/version", timeout=2)
                if v.status_code == 200:
                    result["server_version"] = v.json().get("version", "")
            except Exception:
                pass
            result["vision"] = self._supports_vision_via_metadata()
        except Exception as e:
            result["error"] = str(e)
        return result

    def list_models(self) -> List[Dict]:
        """List all installed models with their vision capability."""
        if not self.check():
            return []
        try:
            resp = self._get_session().get(f"{self.base_url}/api/tags", timeout=10)
            raw = resp.json().get("models", [])
        except Exception:
            return []
        out = []
        for m in raw:
            name = m.get("name", "")
            out.append({
                "name": name,
                "vision": self._supports_vision(name),
                "size_gb": round(m.get("size", 0) / (1024 ** 3), 2),
                "family": m.get("details", {}).get("family", ""),
            })
        return out

    # ---------------------------------------------------------- Vision-cap

    def _supports_vision(self, model_name: str) -> bool:
        """Best-effort vision detection for a model.

        Heuristics, in order:
          1. Cached result
          2. /api/show metadata: families contains clip / mmproj / vision
          3. Name contains 'llava', 'vision', 'vl', 'minicpm-v', 'moondream'
        """
        if not model_name:
            return False
        if model_name in self._vision_cache:
            return self._vision_cache[model_name]
        try:
            ok = self._supports_vision_via_metadata(model_name)
        except Exception:
            ok = None
        if ok is None:
            ok = self._supports_vision_via_name(model_name)
        self._vision_cache[model_name] = ok
        return ok

    def _supports_vision_via_metadata(self, model_name: Optional[str] = None) -> Optional[bool]:
        if not HAS_REQUESTS:
            return None
        model_name = model_name or self.model
        try:
            resp = self._get_session().post(
                f"{self.base_url}/api/show",
                json={"name": model_name}, timeout=5,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            families = (data.get("details") or {}).get("families") or []
            capabilities = (data.get("capabilities") or [])
            family = (data.get("details") or {}).get("family", "")
            vision_families = {"clip", "mmproj", "vision", "llava"}
            if any(f in vision_families for f in families):
                return True
            if "vision" in capabilities:
                return True
            if "llava" in family.lower() or "vision" in family.lower():
                return True
            # Text-only models with vision families often still work for text-only
            # tasks; we treat them as non-vision here.
            return False
        except Exception:
            return None

    def _supports_vision_via_name(self, model_name: str) -> bool:
        m = model_name.lower()
        vision_markers = ("llava", "vision", "vl", "minicpm-v",
                          "moondream", "bakllava", "granite3.1-vision",
                          "qwen2.5vl", "qwen-vl", "internvl")
        # Negative markers (text-only)
        if "-text" in m or "embed" in m:
            return False
        return any(marker in m for marker in vision_markers)

    # ---------------------------------------------------------- Generation

    def _img_to_b64(self, path: str, max_size: int = 512) -> Optional[str]:
        """Encode an image file (or extracted video frame) to base64.

        For static images, returns the raw bytes. For videos and animated
        formats, extracts a single frame first to stay within the model's
        context window.
        """
        ext = os.path.splitext(path)[1].lower()
        video_exts = {".mp4", ".m4v", ".webm", ".mkv", ".avi", ".mov",
                      ".flv", ".mpg", ".mpeg", ".mpe", ".mpv", ".ogv",
                      ".wmv", ".asf", ".ts", ".m2ts", ".mts", ".vob",
                      ".3gp", ".3gpp", ".rm", ".rmvb", ".ogm"}
        animated_exts = {".gif", ".apng", ".mng", ".fli", ".flc", ".webp"}

        if ext in video_exts and shutil_which("ffmpeg"):
            frame = _extract_video_frame(path, max_size=max_size)
            if frame is None:
                return None
            buf = io_bytes()
            frame.save(buf, format="PNG", optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        if ext in animated_exts:
            try:
                im = Image.open(path)
                if getattr(im, "is_animated", False) or getattr(im, "n_frames", 1) > 1:
                    im.seek(0)
                im = im.convert("RGB")
                im.thumbnail((max_size, max_size), Image.LANCZOS)
                buf = io_bytes()
                im.save(buf, format="PNG", optimize=True)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                return None

        # For static images, resize first so the base64 payload stays
        # well within the model's context window. Reading the raw file
        # can produce 10-15 MB of base64 for a single 4K PNG, which
        # exceeds the 4096-token context of small vision models
        # (llava-phi3, moondream, etc.).
        try:
            im = Image.open(path)
            im = im.convert("RGB")
            im.thumbnail((max_size, max_size), Image.LANCZOS)
            buf = io_bytes()
            im.save(buf, format="JPEG", quality=85, optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            try:
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                return None

    def generate(self, prompt: str, images: Optional[List[str]] = None,
                 system: str = "", temperature: float = 0.1,
                 max_tokens: int = 200, timeout: Optional[int] = None,
                 format: Optional[str] = None) -> Optional[str]:
        """Free-form generation. Use `format="json"` for structured output."""
        images = images or []
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        if images:
            payload["images"] = images
        if format:
            payload["format"] = format
        data = self._post("/api/generate", payload, timeout=timeout)
        if data is None:
            return None
        text = (data.get("response") or "").strip()
        if not text:
            self._last_error = f"empty response, full data: {json.dumps(data)[:500]}"
        return text

    def chat(self, messages: List[Dict], temperature: float = 0.1,
             max_tokens: int = 200, timeout: Optional[int] = None,
             format: Optional[str] = None) -> Optional[str]:
        """Chat endpoint. `messages` is a list of {role, content, images?}."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if format:
            payload["format"] = format
        data = self._post("/api/chat", payload, timeout=timeout)
        if data is None:
            return None
        return (data.get("message") or {}).get("content", "").strip()

    # ---------------------------------------------------------- Structured

    def generate_structured(
        self,
        prompt: str,
        schema_hint: str,
        images: Optional[List[str]] = None,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 400,
        default: Optional[Dict] = None,
        timeout: Optional[int] = None,
    ) -> Dict:
        """Generate a JSON object matching `schema_hint`.

        Uses Ollama's `format: "json"` mode when the model supports it,
        falling back to plain generation + json.loads on failure. Returns
        `default` (or {}) if no valid JSON is produced.

        Fallback ladder:
          1. format="json" + json.loads (most models)
          2. if that returned an empty/sparse dict (e.g. the model
             echoed back the schema and then truncated), retry without
             format="json" so the model is free to emit prose, then
             parse the first {...} block from that prose.
          3. as a last resort, return `default` (or {}).
        """
        images = images or []
        full_prompt = prompt
        if schema_hint:
            full_prompt = f"{prompt}\n\nReply with JSON only. Schema:\n{schema_hint}"

        def _nonempty(d) -> bool:
            """Treat {} or {'schema_field': ''} as 'empty' so we retry."""
            return isinstance(d, dict) and any(
                v not in (None, "", [], {})
                for v in d.values()
            )

        parsed: Optional[Dict] = None
        # 1) Strict JSON mode
        try:
            raw = self.generate(
                full_prompt, images=images, system=system,
                temperature=temperature, max_tokens=max_tokens,
                format="json", timeout=timeout,
            )
        except Exception:
            raw = None
        if raw:
            try:
                parsed = _safe_json_loads(raw)
            except Exception:
                parsed = None
        # If the strict call produced something meaningful, return it.
        if _nonempty(parsed):
            return parsed  # type: ignore[return-value]

        # 2) Free-form retry - the model emits prose + a JSON block.
        try:
            raw = self.generate(
                full_prompt, images=images, system=system,
                temperature=temperature, max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception:
            raw = None
        if raw:
            try:
                parsed = _safe_json_loads(raw)
            except Exception:
                # Last-resort: extract first {...} block
                m = re.search(r"\{[\s\S]+\}", raw)
                if m:
                    try:
                        parsed = _safe_json_loads(m.group(0))
                    except Exception:
                        parsed = None
        if _nonempty(parsed):
            return parsed  # type: ignore[return-value]

        return default or {}

    # ------------------------------------------------------------ Helpers

    def close(self):
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None


def _safe_json_loads(text: str) -> Dict:
    """Robust JSON parsing: strips code-fences, fixes common LLM issues."""
    if not text:
        return {}
    s = text.strip()
    # Strip ```json ... ``` or ``` ... ```
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"```\s*$", "", s)
    # Sometimes models add trailing commas - try a tolerant decode
    for attempt in (s, s.replace(",\n", "\n").replace(", }", " }").replace(",]", "]")):
        try:
            data = json.loads(attempt)
            return data if isinstance(data, dict) else {"value": data}
        except Exception:
            continue
    return {}


def _extract_video_frame(path: str, max_size: int = 512) -> Optional[Image.Image]:
    """Extract a single thumbnail frame from a video using ffmpeg."""
    import shutil
    import subprocess
    import tempfile
    if not shutil.which("ffmpeg"):
        return None
    try:
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
             "-vframes", "1", "-vf", f"scale='min({max_size},iw)':-1", tmp],
            capture_output=True, timeout=20,
        )
        if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            img = Image.open(tmp).convert("RGB")
            os.remove(tmp)
            return img
        if os.path.exists(tmp):
            os.remove(tmp)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
    return None


def io_bytes():
    import io
    return io.BytesIO()


def shutil_which(name: str) -> bool:
    import shutil
    return shutil.which(name) is not None


# ---------------------------------------------------------------------------
# Structured-analysis helpers
# ---------------------------------------------------------------------------

ANALYSIS_SCHEMA = """{
  "subject":   "<one short noun phrase describing the main subject>",
  "tags":      [<up to 10 short lowercase tags>],
  "colors":    [<up to 5 color names e.g. blue, warm>],
  "mood":      "<one adjective: calm|dramatic|peaceful|epic|mysterious|...>",
  "style":     "<one of: anime|illustration|digital-art|photograph|3d-render|painting|sketch|pixel-art>",
  "nsfw":      <float 0.0-1.0>,
  "prompt":    "<one sentence (max 25 words) describing the aesthetic>"
}"""


CATEGORY_SCORE_SCHEMA = """{
  "scores": { "<category_name>": <float 0-1>, ... },
  "reason":  "<one short sentence>"
}"""


# ---------------------------------------------------------------------------
# OllamaAnalyzer - uses OllamaClient for richer analysis
# ---------------------------------------------------------------------------

class OllamaAnalyzer(BaseAnalyzer):
    """Analyzer using local vision LLMs via Ollama.

    Improvements over the previous version:
      * Structured JSON output -> more reliable tag/colour extraction
      * Hybrid classification: LLM scores blended with CV heuristics
      * Auto-detection of vision capability (skips vision-only calls
        when the loaded model is text-only, falling back to tag-driven
        classification from a textual description)
      * Subject extraction suitable for tag-based renaming
    """

    name = "ollama"

    def __init__(self, settings: dict = None):
        self.settings = settings or {}
        s = self.settings or load_settings()
        self._client = OllamaClient(
            base_url=s.get("ollama_url", "http://localhost:11434"),
            model=s.get("ollama_model", "llava:7b"),
            timeout=s.get("ollama_timeout", 60),
        )
        self._nsfw_enabled = s.get("ollama_nsfw_enabled", True)
        self._describe_enabled = s.get("ollama_describe_enabled", False)
        self._classify_enabled = s.get("ollama_classify_enabled", False)
        self._tags_enabled = s.get("ollama_tags_enabled", True)
        self._nsfw_skip_describe = s.get("nsfw_skip_describe", True)
        self._nsfw_threshold = s.get("nsfw_threshold", 0.5)
        self._nsfw_default_tags = s.get("nsfw_default_tags", [
            "nsfw", "figure", "human", "skin", "portrait", "person", "body",
        ])
        self._classify_method = s.get("ollama_classify_method", "tags")
        self._max_tokens = s.get("ollama_description_max_tokens", 120)
        self._vision_capable: Optional[bool] = None
        self._last_tags: List[str] = []

    # ---------------------------------------------------- Vision detection

    @property
    def vision_capable(self) -> bool:
        if self._vision_capable is None:
            self._vision_capable = self._client._supports_vision(self._client.model)
        return self._vision_capable

    # ---------------------------------------------------- Image encoding

    def _img_to_b64(self, path: str) -> Optional[str]:
        return self._client._img_to_b64(path, max_size=512)

    # ---------------------------------------------------- analyse

    def analyze(self, image_path: str) -> dict:
        profile = get_image_profile(image_path)
        profile["mode"] = "ollama"

        # If the configured model can't see images, skip vision-only calls.
        if not self.vision_capable:
            profile["ollama_vision_skipped"] = True
            return profile

        b64 = self._img_to_b64(image_path)
        if b64 is None:
            return profile

        # 1) Single structured call: subject + tags + colours + mood +
        #    style + nsfw + prompt (when the model supports JSON output).
        #    This is much faster than running 5-6 separate prompts.
        structured = self._structured_analysis(b64)
        if structured:
            self._apply_structured(profile, structured)

        # 2) Optional prose description for the GUI log / prompt-engineering
        if self._describe_enabled and not profile.get("ollama_description"):
            desc = self._client.generate(
                "Describe this image in one concise sentence (max 25 words) "
                "focusing on visual style, colour palette, mood.",
                images=[b64], temperature=0.1, max_tokens=self._max_tokens,
            )
            if desc:
                profile["ollama_description"] = desc.strip()

        return profile

    # ------------------------------------------- Structured analysis

    def _structured_analysis(self, b64: str) -> Dict:
        """Single-call analysis that returns a JSON dict with subject,
        tags, colours, mood, style, nsfw, prompt.

        Falls back to a no-tag result if JSON parsing fails.
        """
        prompt = (
            "You are an expert wallpaper image tagger. Analyse the image and "
            "produce a compact JSON object with the EXACT schema below. "
            "Use lowercase short tags. Limit 'tags' to 10 most informative. "
            "Limit 'colors' to 5. Be concise in 'subject' (<= 4 words) and "
            "'prompt' (<= 25 words)."
        )
        try:
            data = self._client.generate_structured(
                prompt=prompt,
                schema_hint=ANALYSIS_SCHEMA,
                images=[b64],
                temperature=0.2,
                max_tokens=500,
                default={},
                timeout=self._client.timeout,
            )
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _apply_structured(self, profile: dict, data: Dict) -> None:
        """Copy fields from the structured analysis into the profile."""
        # Subject
        subj = (data.get("subject") or "").strip()
        if subj:
            profile["ollama_subject"] = subj
            profile["main_subject"] = subj.lower().replace(" ", "-")

        # Tags
        raw_tags = data.get("tags") or []
        if isinstance(raw_tags, str):
            raw_tags = re.split(r"[,;\n]+", raw_tags)
        tags = []
        for t in raw_tags:
            tt = str(t).strip().lower().rstrip(".,;:!?\"'")
            if tt and 1 < len(tt) < 30 and " " not in tt and tt not in {
                "the", "and", "with", "for", "are", "this", "image"
            }:
                tags.append(tt)
        # Filter to registry when possible
        filtered = [t for t in tags if t in _tags_flat]
        # Keep unknown but well-formed tags as suggestions
        extras = [t for t in tags if t not in _tags_flat][:3]
        profile["ollama_tags"] = filtered
        profile["ollama_tags_extra"] = extras
        profile["ollama_all_tags"] = tags[:12]
        self._last_tags = profile["ollama_all_tags"]

        # Colours
        colours = data.get("colors") or []
        if isinstance(colours, str):
            colours = re.split(r"[,;\n]+", colours)
        profile["ollama_colors"] = [
            str(c).strip().lower() for c in colours if str(c).strip()
        ]

        # Mood / style / prompt
        for key in ("mood", "style", "prompt"):
            v = (data.get(key) or "").strip()
            if v:
                profile[f"ollama_{key}"] = v

        # NSFW
        nsfw_val = data.get("nsfw")
        if nsfw_val is not None:
            try:
                score = float(nsfw_val)
                score = max(0.0, min(1.0, score))
                profile["ollama_nsfw"] = score
            except (TypeError, ValueError):
                pass

    # ------------------------------------------- Classification (LLM side)

    def classify(self, profile: dict) -> Optional[str]:
        """Use the LLM's category scores from the profile if available."""
        if not self._classify_enabled:
            return None

        # Pre-computed LLM scores from the structured analysis
        llm_scores: Dict[str, float] = {}
        for cat in CATEGORIES:
            key = f"ollama_score_{cat}"
            if key in profile:
                llm_scores[cat] = float(profile[key])
        if llm_scores:
            best = max(llm_scores, key=llm_scores.get)
            if llm_scores[best] > 0.05:
                return best

        # If profile has tags but no LLM scores, use the tag-based matcher
        img_tags = set(profile.get("ollama_tags") or [])
        if img_tags:
            cat = classify_by_tags(img_tags)
            if cat:
                return cat

        # Fallback to the multi-signal heuristic scorer
        try:
            info = classify_with_confidence(profile)
            if info.get("category") in CATEGORIES:
                return info["category"]
        except Exception:
            pass
        return None

    def classify_direct(self, image_path: str) -> Optional[str]:
        """Dispatch classification based on `ollama_classify_method`.

        Methods:
          * "prompt"  - numbered-list approach
          * "tags"    - tag-based + category scoring
          * "hybrid"  - tags + multi-signal heuristic, fallback to prompt
        """
        if not self._classify_enabled:
            return None
        if self._classify_method == "prompt":
            return self._classify_by_prompt(image_path)
        if self._classify_method == "hybrid":
            cat = self._classify_by_tags(image_path)
            if cat:
                return cat
            return self._classify_by_prompt(image_path)
        return self._classify_by_tags(image_path)

    def _classify_by_prompt(self, image_path: str) -> Optional[str]:
        """Numbered-list approach: ask the model to pick a category number."""
        if not self._classify_enabled or not self.vision_capable:
            return None
        b64 = self._img_to_b64(image_path)
        if b64 is None:
            return None

        valid_cats = list(CATEGORIES)
        if not self._nsfw_enabled:
            valid_cats = [c for c in valid_cats if c != "NSFW"]
        if not valid_cats:
            return None

        cat_list = []
        for i, cat in enumerate(valid_cats):
            tags = sorted(get_category_tags(cat))
            tag_str = ", ".join(tags[:5]) if tags else ""
            prompt_text = get_category_prompt(cat)
            if prompt_text:
                short = prompt_text[:80] + "..." if len(prompt_text) > 80 else prompt_text
            else:
                short = tag_str[:80]
            cat_list.append(f"{i}. {cat}: {short}")

        prompt = (
            "You are a wallpaper classifier. Choose ONE number from the list below "
            "that best matches the image. Respond with ONLY the number, nothing else.\n\n"
            + "\n".join(cat_list) +
            "\n\nNumber:"
        )
        result = self._client.generate(prompt, images=[b64], temperature=0.0, max_tokens=10)
        if not result:
            return None
        result = result.strip().rstrip(".,;:!?")
        m = re.match(r'^\s*(\d+)', result)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(valid_cats):
                return valid_cats[idx]
        if result in valid_cats:
            return result
        for cat in valid_cats:
            if cat.lower() in result.lower():
                return cat
        return None

    def _classify_by_tags(self, image_path: str) -> Optional[str]:
        """Tag-based + structured JSON classification.

        Single call to get tags + per-category scores, then:
          1. Map per-category scores from the LLM
          2. Fallback to tag-to-category matching if scores absent
          3. Final fallback to multi-signal heuristic
        """
        if not self._classify_enabled:
            return None
        if not self.vision_capable:
            return None
        b64 = self._img_to_b64(image_path)
        if b64 is None:
            return None

        valid_cats = list(CATEGORIES)
        if not valid_cats:
            return None
        if not self._nsfw_enabled:
            valid_cats = [c for c in valid_cats if c != "NSFW"]

        # Build a JSON prompt asking for tags + category scores
        cat_block = "\n".join(f"- {c}" for c in valid_cats)
        curated = ", ".join(CURATED_TAGS_FOR_LLM[:100])
        prompt = (
            "Analyse the wallpaper and respond in JSON:\n"
            "{\n"
            '  "tags": [<5-10 tags from the list below, lowercase>],\n'
            '  "scores": {<one float 0-1 per category below>},\n'
            '  "subject": "<one short noun phrase>"\n'
            "}\n"
            f"Tags list: {curated}\n"
            f"Categories: {cat_block}\n"
            "Be strict: only put a category in 'scores' if the image really fits."
        )
        data = self._client.generate_structured(
            prompt=prompt, schema_hint="", images=[b64],
            temperature=0.2, max_tokens=400, default={},
        )

        # Merge per-category scores
        llm_scores = data.get("scores") or {}
        if isinstance(llm_scores, dict):
            best_cat = max(llm_scores, key=lambda k: float(llm_scores.get(k, 0) or 0)) \
                if llm_scores else None
            if best_cat and float(llm_scores.get(best_cat, 0) or 0) > 0.2 \
                    and best_cat in valid_cats:
                return best_cat

        # Fallback: parse tags and match against category tag sets
        raw_tags = data.get("tags") or []
        if isinstance(raw_tags, str):
            raw_tags = re.split(r"[,;\n]+", raw_tags)
        img_tags = set()
        for t in raw_tags:
            tt = str(t).strip().lower().rstrip(".,;:!?\"'")
            if tt and 1 < len(tt) < 30 and tt in _tags_flat:
                img_tags.add(tt)
        if img_tags:
            cat = classify_by_tags(img_tags)
            if cat:
                return cat

        # Final fallback: use heuristic multi-signal scorer from profile
        try:
            profile = get_image_profile(image_path)
            info = classify_with_confidence(profile)
            if info.get("category") in valid_cats:
                return info["category"]
        except Exception:
            pass

        return None

    # ------------------------------------------- Convenience methods

    def describe(self, image_path: str) -> Optional[str]:
        if not self.vision_capable:
            return None
        b64 = self._img_to_b64(image_path)
        if b64 is None:
            return None
        return self._client.generate(
            "Describe this image in one concise sentence.",
            images=[b64], temperature=0.1,
            max_tokens=self.settings.get("ollama_description_max_tokens", 80),
        )

    def nsfw_score(self, image_path: str) -> Optional[float]:
        """Cheap NSFW probe via structured analysis."""
        if not self.vision_capable:
            return None
        b64 = self._img_to_b64(image_path)
        if b64 is None:
            return None
        data = self._client.generate_structured(
            prompt="Rate the NSFW content 0.0..1.0.",
            schema_hint='{"nsfw": <float 0-1>}',
            images=[b64], temperature=0.0, max_tokens=20, default={},
        )
        try:
            return max(0.0, min(1.0, float(data.get("nsfw", 0.0))))
        except (TypeError, ValueError):
            return None

    def detect_tags(self, image_path: str, max_tags: int = 8) -> List[str]:
        """Return up to `max_tags` descriptive tags for the image.

        Used by the rename pipeline to build tag-based filenames.
        """
        if not self.vision_capable:
            return []
        b64 = self._img_to_b64(image_path)
        if b64 is None:
            return []
        curated = ", ".join(CURATED_TAGS_FOR_LLM[:120])
        prompt = (
            f"From this tag list, pick the {max_tags} tags that best describe "
            f"the image. Reply as JSON: {{\"tags\": [...]}}\n"
            f"Tags: {curated}"
        )
        data = self._client.generate_structured(
            prompt=prompt, schema_hint="", images=[b64],
            temperature=0.1, max_tokens=120, default={},
        )
        raw = data.get("tags") or []
        if isinstance(raw, str):
            raw = re.split(r"[,;\n]+", raw)
        out = []
        for t in raw:
            tt = str(t).strip().lower().rstrip(".,;:!?\"'")
            if tt and 1 < len(tt) < 30 and tt not in out:
                out.append(tt)
            if len(out) >= max_tags:
                break
        return out

    def detect_main_subject(self, image_path: str) -> Optional[str]:
        """Return a 1-3 word description of the image's main subject."""
        if not self.vision_capable:
            return None
        b64 = self._img_to_b64(image_path)
        if b64 is None:
            return None
        data = self._client.generate_structured(
            prompt="Describe the main subject of this image in 1-3 words.",
            schema_hint='{"subject": "<1-3 words>"}',
            images=[b64], temperature=0.1, max_tokens=30, default={},
        )
        s = (data.get("subject") or "").strip()
        return s.lower().replace(" ", "-") if s else None

    def close(self):
        try:
            self._client.close()
        except Exception:
            pass
