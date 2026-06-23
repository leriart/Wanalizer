"""Background worker threads for the GUI."""
import os, sys, threading, subprocess, tempfile, shutil, traceback
from typing import Set
from PySide6.QtCore import QThread, Signal
from .. import settings as settings_mod
from .. import formats as formats_mod
from .. import categories as cats_mod
from .. import organize as org_mod


def _has_ffmpeg():
    return shutil.which("ffmpeg") is not None


def _video_frame(path, max_size=512):
    """Extract a single thumbnail frame from a video file using ffmpeg.
    Returns a PIL.Image or None on failure.
    """
    if not _has_ffmpeg():
        return None
    from PIL import Image as PILImage
    try:
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vframes", "1",
             "-vf", f"scale='min({max_size},iw)':-1", tmp],
            capture_output=True, timeout=15,
        )
        if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            img = PILImage.open(tmp).convert("RGB")
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


class OrganizeWorker(QThread):
    progress = Signal(str, int, int, str, str)
    finished_ok = Signal(dict)
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, mode="lowlevel", dry=False, dedupe=True, parallel=1,
                 quality_min=0.0, source_dir=None, cats_dir=None,
                 rename_strategy="none", rename_category_prefix=True,
                 rename_max_tags: int = 3):
        super().__init__()
        self.mode = mode
        self.dry = dry
        self.dedupe = dedupe
        self.parallel = parallel
        self.quality_min = quality_min
        self.source_dir = source_dir
        self.cats_dir = cats_dir
        self.rename_strategy = rename_strategy
        self.rename_category_prefix = rename_category_prefix
        self.rename_max_tags = rename_max_tags
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            if self.source_dir:
                formats_mod.WALLPAPERS_DIR = self.source_dir
            if self.cats_dir:
                cats_mod.CATEGORIES_DIR = self.cats_dir
                cats_mod.discover_categories(self.cats_dir)

            r, w = os.pipe()
            orig = sys.stdout
            sys.stdout = os.fdopen(w, "w", buffering=1)

            def pump():
                try:
                    with os.fdopen(r, "r") as fp:
                        for line in fp:
                            line = line.rstrip()
                            if line:
                                self.log.emit(line)
                except Exception:
                    pass

            t = threading.Thread(target=pump, daemon=True)
            t.start()
            try:
                org_mod.organize(
                    mode=self.mode,
                    dry_run=self.dry,
                    dedupe=self.dedupe,
                    parallel=self.parallel,
                    quality_min=self.quality_min,
                    progress_callback=self._emit,
                    should_cancel=lambda: self._cancel,
                    rename_strategy=self.rename_strategy,
                    rename_category_prefix=self.rename_category_prefix,
                    rename_max_tags=self.rename_max_tags,
                )
                self.finished_ok.emit({})
            finally:
                sys.stdout = orig
                try:
                    os.close(w)
                except Exception:
                    pass
        except Exception as e:
            self.failed.emit(f"{e}\n{traceback.format_exc()}")

    def _emit(self, stage, cur, total, fname, info):
        self.progress.emit(stage, cur, total, fname, info)


class HealthCheckWorker(QThread):
    result_ready = Signal(dict)

    def __init__(self, url, model):
        super().__init__()
        self.url = url
        self.model = model

    def run(self):
        try:
            from ..ollama_client import OllamaClient
            c = OllamaClient(base_url=self.url, model=self.model, timeout=5)
            result = c.health()
            if result is None:
                result = {
                    "connected": False, "model_available": False,
                    "error": "No response", "server_version": "", "model_count": 0,
                }
        except Exception as e:
            result = {
                "connected": False, "model_available": False,
                "error": str(e), "server_version": "", "model_count": 0,
            }
        self.result_ready.emit(result)


class OllamaPullWorker(QThread):
    progress = Signal("qlonglong", "qlonglong", str)
    finished_ok = Signal()
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, url, model, timeout=600):
        super().__init__()
        self.url = url
        self.model = model
        self.timeout = timeout
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        import requests as req
        import json as _j

        try:
            self.log.emit(f"Pulling {self.model}...")
            resp = req.post(
                f"{self.url.rstrip('/')}/api/pull",
                json={"name": self.model, "stream": True},
                stream=True,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                self.failed.emit(f"HTTP {resp.status_code}")
                return
            for line in resp.iter_lines(decode_unicode=True):
                if self._cancel:
                    self.log.emit("[CANCELLED]")
                    return
                if not line:
                    continue
                try:
                    d = _j.loads(line)
                    st = d.get("status", "")
                    t = d.get("total", 0)
                    c = d.get("completed", 0)
                    err = d.get("error", "")
                    if err:
                        # Ollama returns 200 with {"error": "..."} for bad tags
                        self.failed.emit(err)
                        return
                    self.progress.emit(int(c), int(t), st)
                    if st == "success":
                        self.finished_ok.emit()
                        return
                except Exception:
                    continue
            # Stream ended without explicit success or error - check final state
            self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(str(e))


class DepWorker(QThread):
    log_line = Signal(str)
    module_done = Signal(str, bool)
    all_done = Signal()

    def __init__(self, modules):
        super().__init__()
        self.modules = modules
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        vp = os.path.join(settings_mod.PROJECT_DIR, ".venv", "bin", "python")
        if not os.path.exists(vp):
            vp = sys.executable

        pip_index = ["--index-url", "https://download.pytorch.org/whl/cpu"]

        pkg_map = {
            "cv2": "opencv-python-headless",
            "imagehash": "imagehash",
            "sklearn": "scikit-learn",
            "clip": "git+https://github.com/openai/CLIP.git",
            "torch": "torch",
            "torchvision": "torchvision",
            "PySide6": "PySide6",
            "ftfy": "ftfy",
            "regex": "regex",
        }

        for mod in self.modules:
            if self._cancel:
                self.log_line.emit("[CANCELLED]")
                break

            pkg = pkg_map.get(mod, mod)

            if mod == "clip":
                if shutil.which("git") is None:
                    self.log_line.emit(
                        "[ERR] Git is not installed. CLIP install requires git."
                    )
                    self.module_done.emit(mod, False)
                    continue
                self.log_line.emit(
                    "[pip] Installing clip (this may take several minutes)..."
                )
            else:
                self.log_line.emit(f"[pip] Installing {pkg}...")

            idx = pip_index if mod in ("torch", "torchvision") else None
            try:
                cmd = [vp, "-m", "pip", "install", "--disable-pip-version-check", pkg]
                if idx:
                    cmd.extend(idx)
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                ok = True
                for line in iter(proc.stdout.readline, ""):
                    if self._cancel:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except Exception:
                            proc.kill()
                        break
                    line = line.rstrip()
                    if line:
                        self.log_line.emit(line)
                if not self._cancel:
                    try:
                        proc.wait(timeout=900)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        self.log_line.emit("[ERR] Installation timed out")
                        ok = False
                if proc.returncode != 0:
                    ok = False
            except Exception as exc:
                ok = False
                self.log_line.emit(f"[ERR] {exc}")

            self.module_done.emit(mod, ok)

        self.all_done.emit()


class GenerateTagsWorker(QThread):
    """Worker for AI-powered tag and prompt generation from sample images.

    Runs in a background thread to keep the UI responsive.
    Supports: ollama, clip, lowlevel (heuristic-only).
    """
    log = Signal(str)
    progress = Signal(int, int, str)  # current, total, current_image
    finished = Signal(str, dict)  # category_name, {"tags": [...], "prompt": "..."}
    failed = Signal(str, str)  # category_name, error

    def __init__(self, category_name, mode="ollama", max_samples=5, model="llava:7b",
                 base_url="http://localhost:11434"):
        super().__init__()
        self.category_name = category_name
        self.mode = mode
        self.max_samples = max_samples
        self.model = model
        self.base_url = base_url
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _img_to_b64(self, path):
        """Read a media file as base64. For videos, extract a single
        thumbnail frame first (otherwise we send raw video bytes and
        overflow the model's context window).
        """
        import base64
        ext = os.path.splitext(path)[1].lower()
        # Video files: extract a single frame first
        if ext in {".mp4", ".m4v", ".webm", ".mkv", ".avi", ".mov", ".flv",
                   ".mpg", ".mpeg", ".mpe", ".mpv", ".ogv", ".wmv", ".asf",
                   ".ts", ".m2ts", ".mts", ".vob", ".3gp", ".3gpp", ".rm",
                   ".rmvb", ".ogm"}:
            frame = _video_frame(path, max_size=512)
            if frame is None:
                return None
            try:
                from io import BytesIO
                buf = BytesIO()
                frame.save(buf, format="PNG", optimize=True)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                return None
        # Animated formats: take first frame
        if ext in {".gif", ".apng", ".mng", ".fli", ".flc"} or ext == ".webp":
            try:
                from PIL import Image as PILImage
                im = PILImage.open(path)
                if getattr(im, "is_animated", False) or getattr(im, "n_frames", 1) > 1:
                    im.seek(0)
                im = im.convert("RGB")
                im.thumbnail((512, 512), PILImage.LANCZOS)
                from io import BytesIO
                buf = BytesIO()
                im.save(buf, format="PNG", optimize=True)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                return None
        # Static image: read bytes directly
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return None

    def _get_tags_lowlevel(self, img_path):
        """Heuristic tag derivation from low-level CV profile.

        Uses the new tag_suggester for richer, content-aware tags and
        optionally combines with Ollama descriptions when available.
        """
        from ..profile import get_image_profile
        from ..tag_suggester import (
            suggest_tags, suggest_tags_for_category,
            expand_tags_with_cooccurrence,
        )
        try:
            prof = get_image_profile(img_path)
            cat_name = self.category_name
            tags = suggest_tags_for_category(cat_name, prof, max_tags=18)
            return sorted(tags)
        except Exception as e:
            self.log.emit(f"  [WARN] lowlevel profile error: {e}")
            return []

    def _get_prompt_lowlevel(self, imgs, category_name=""):
        """Build a heuristic style description from aggregated profiles.

        Uses the new prompt_generator to combine palette, style, content and
        composition signals into a single natural-language description.
        """
        from ..profile import get_image_profile
        from ..prompt_generator import generate_category_prompt
        profiles = []
        for path in imgs[: self.max_samples]:
            try:
                profiles.append(get_image_profile(path))
            except Exception:
                pass
        if not profiles:
            return "wallpaper with mixed visual elements"
        return generate_category_prompt(
            profiles, style="detailed", category_name=category_name,
        )

    def _get_palette_weights_lowlevel(self, imgs):
        """Aggregate per-image palette weights into a single dict for the
        category's `palette_weights`. Uses the new prompt_generator helper."""
        from ..profile import get_image_profile
        from ..prompt_generator import suggest_palette_weights
        agg = {}
        for path in imgs[: self.max_samples]:
            try:
                prof = get_image_profile(path)
                pw = suggest_palette_weights(prof)
                for color, w in pw.items():
                    agg[color] = agg.get(color, 0.0) + w
            except Exception:
                pass
        # Normalise to 0..1 range
        if not agg:
            return {}
        m = max(agg.values()) or 1
        return {c: round(v / m, 4) for c, v in agg.items() if v > 0}

    def _get_tags_ollama(self, client, b64, tag_list, nsfw_mode=False, min_tags=5):
        """Generate tags for an image using a curated, anti-pollution list.

        Anti-pollution steps (so the registry doesn't end up with the
        same generic tags in every category):
          1. Use a focused curated list (~50 tags) instead of the full
             registry, so the LLM has discriminating choices.
          2. Strip already-present tags so the LLM must suggest new ones.
          3. Parse the response and validate against the curated list
             (tags not in the list are dropped).
          4. Cap the returned set at LLM_TAG_PICK_MAX to keep categories
             compact.
          5. filter_polluted_tags() removes GENERIC_OVERUSED tags that
             would otherwise pollute every category.
        """
        from ..tag_policies import (
            CURATED_NSFW_TAGS, CURATED_SAFE_TAGS, LLM_TAG_PICK_MAX,
            build_focused_curated_list, cap_category_tags,
            filter_polluted_tags, parse_llm_tag_response,
        )

        # Use a focused curated list (not the user's full registry) so
        # the LLM has discriminating choices instead of 150 generic
        # candidates. Then intersect with what already exists in this
        # category so the LLM must suggest something *new*.
        existing = c_mod.get_category_config(self.category_name).get("tags") or []
        available = build_focused_curated_list(
            full_registry=tag_list,
            existing=existing,
            nsfw=nsfw_mode,
        )
        if len(available) < 8:
            # Fall back to the curated list as-is (don't drop more)
            base = CURATED_NSFW_TAGS if nsfw_mode else CURATED_SAFE_TAGS
            available = list(base)

        # Strategies - identical pattern to before but with the focused list
        if nsfw_mode:
            prompt_strategies = [
                (
                    f"List 5-10 matching visual style tags from this list for the image:\n"
                    f"{', '.join(available)}\n"
                    "Focus on: art style, color palette, lighting, composition, mood. "
                    "Reply with ONLY the tags separated by commas. No body description.",
                    "",
                    100,
                    "generate",
                ),
                (
                    "Visual style tags (comma-separated):",
                    "You are a visual style classifier. Reply with only style tags.",
                    100,
                    "generate",
                ),
                (
                    f"Tags: {', '.join(available[:30])}",
                    "",
                    80,
                    "chat",
                ),
            ]
        else:
            prompt_strategies = [
                (
                    f"Select matching tags from this list for the image:\n"
                    f"{', '.join(available)}\n"
                    "Reply with ONLY the matching tags separated by commas. "
                    "Choose 5-10 tags. No other text.",
                    "",
                    120,
                    "generate",
                ),
                (
                    "What are the best tags for this image? Choose from the list. "
                    "Reply with just the tags separated by commas.",
                    "You are a helpful image tagging assistant. Reply concisely with only the requested tags.",
                    120,
                    "generate",
                ),
                (
                    "Tags (comma-separated, lowercase):",
                    "",
                    100,
                    "generate",
                ),
                (
                    f"Look at the image. Select matching tags from this list: {', '.join(available)}. "
                    "Reply with only the matching tags separated by commas.",
                    "You are a helpful image tagging assistant.",
                    100,
                    "chat",
                ),
            ]

        last_response = ""
        accumulated: set = set()
        for attempt, (prompt, system, max_tok, mode) in enumerate(prompt_strategies):
            if self._cancel:
                break
            try:
                if mode == "chat":
                    r = client.chat(
                        prompt, images=[b64], system=system,
                        temperature=0.1, max_tokens=max_tok
                    )
                else:
                    r = client.generate(
                        prompt, images=[b64], system=system,
                        temperature=0.1, max_tokens=max_tok
                    )
            except Exception as e:
                if attempt == 0:
                    self.log.emit(f"  [WARN] Tag generation error: {e}")
                continue

            if r and r.strip():
                last_response = r
                parsed = parse_llm_tag_response(
                    r, available, max_tags=LLM_TAG_PICK_MAX,
                )
                if parsed:
                    accumulated.update(parsed)
                    if attempt > 0:
                        self.log.emit(f"  [OK] Recovered with strategy {attempt + 1}: +{len(parsed)}")
                    if len(accumulated) >= min_tags:
                        return accumulated

        if accumulated:
            # Final post-filter: drop generic tags and cap the size
            # one more time in case the LLM went wild.
            tag_registry = {t.lower() for t in tag_list}
            kept = [t for t in accumulated if t in tag_registry]
            kept = filter_polluted_tags(kept)
            return cap_category_tags(kept, max_tags=LLM_TAG_PICK_MAX)

        # All strategies failed - log what the model returned for debugging
        err = getattr(client, 'last_error', '')
        if err:
            self.log.emit(f"  [DEBUG] Client error: {err[:200]}")
            err_lower = err.lower()
            if "unknown model architecture" in err_lower or "mllama" in err_lower:
                self.log.emit(
                    f"  [ERROR] Your Ollama installation cannot load model '{self.model}'.\n"
                    f"     This is an Ollama/server issue, not a client problem.\n"
                    f"     The model architecture is not supported by your Ollama build.\n"
                    f"     Try a different model (e.g. llava:7b) or update Ollama."
                )
            elif "model not found" in err_lower or "404" in err:
                self.log.emit(
                    f"  [ERROR] Model '{self.model}' not found. "
                    f"Pull it first: ollama pull {self.model}"
                )
            elif "out of memory" in err_lower or "oom" in err_lower:
                self.log.emit(
                    f"  [ERROR] Ollama ran out of memory loading '{self.model}'. "
                    f"Try a smaller model or close other applications."
                )
            elif "connection" in err_lower or "timeout" in err_lower:
                self.log.emit(
                    f"  [ERROR] Connection issue with Ollama. "
                    f"Check that Ollama is running at {self.base_url}"
                )
        if last_response:
            self.log.emit(f"  [DEBUG] Last response: '{last_response[:120]}'")
        else:
            self.log.emit("  [DEBUG] Model returned empty/None for all strategies")
        return []


    def _parse_tags_response(self, r):
        """Backward-compat wrapper - prefer tag_policies.parse_llm_tag_response."""
        from ..tag_policies import CURATED_SAFE_TAGS, parse_llm_tag_response
        return set(parse_llm_tag_response(r, CURATED_SAFE_TAGS, max_tags=12))

    def _extract_tags_from_text(self, descriptions, valid_tags):
        """Extract valid tags by matching words from descriptions.

        Filters out GENERIC_OVERUSED tags and caps the result so we
        don't bloat the category registry with the same vague words
        we already get from every other category.
        """
        from ..tag_policies import (
            GENERIC_OVERUSED,
            TAG_BUDGET_PER_CATEGORY,
            cap_category_tags,
            filter_polluted_tags,
        )
        text = " ".join(descriptions).lower()
        import re
        words = set()
        for w in re.findall(r"[a-z][a-z\-]{1,20}", text):
            words.add(w)
        found = set()
        for tag in valid_tags:
            tag_lower = tag.lower()
            if tag_lower in GENERIC_OVERUSED:
                continue
            if tag_lower in words:
                found.add(tag)
            elif " " in tag_lower:
                parts = tag_lower.split()
                if all(p in text for p in parts):
                    found.add(tag)
            else:
                for w in words:
                    if (tag_lower in w or w in tag_lower) and len(w) >= 3 and len(tag_lower) >= 3:
                        found.add(tag)
                        break
        kept = filter_polluted_tags(found)
        return set(cap_category_tags(kept, max_tags=TAG_BUDGET_PER_CATEGORY))

    def _get_desc_ollama(self, client, b64):
        prompt = (
            "Describe this image in 1-2 sentences. Focus on the visual style, "
            "color palette, mood, and composition. Be concise."
        )
        return client.generate(prompt, images=[b64], temperature=0.3, max_tokens=120)

    def _merge_prompts_ollama(self, client, descriptions):
        if not descriptions:
            return ""
        if len(descriptions) == 1:
            return descriptions[0]

        # Use a shorter timeout for the merge to avoid hanging
        import copy
        merge_client = copy.copy(client)
        merge_client.timeout = 30  # 30 second timeout for merge

        prompt = (
            "Combine these style descriptions into one concise paragraph "
            "describing the unified aesthetic. Keep it under 80 words:\n\n"
            + "\n".join(f"- {d[:200]}" for d in descriptions[:3])
        )
        try:
            result = merge_client.generate(prompt, images=[], temperature=0.3, max_tokens=120)
            if result and len(result.strip()) > 10:
                return result.strip()
        except Exception as e:
            self.log.emit(f"  [WARN] Merge failed: {e}, using fallback")

        # Fallback: just pick the longest description (usually most detailed)
        best = max(descriptions, key=len)
        return best.strip()

    def run(self):
        from .. import settings as s_mod
        from .. import tags as t_mod
        from .. import categories as c_mod

        name = self.category_name
        dest = s_mod.resolve_dest_dir(s_mod.load_settings())
        fld = os.path.join(dest, name)

        if not os.path.isdir(fld):
            self.failed.emit(name, "Category folder not found")
            return

        exts = formats_mod.STATIC_EXTENSIONS | formats_mod.ANIMATED_EXTENSIONS
        imgs = sorted([
            fn for fn in os.listdir(fld)
            if os.path.isfile(os.path.join(fld, fn))
            and not fn.startswith(".")
            and os.path.splitext(fn)[1].lower() in exts
        ])
        if not imgs:
            self.failed.emit(name, "No sample images")
            return

        samples = imgs[: min(self.max_samples, len(imgs))]
        self.log.emit(f"Generating for: {name}  ({len(samples)} of {len(imgs)} images)")
        self.log.emit(f"Mode: {self.mode}" + (f"  Model: {self.model}" if self.mode == "ollama" else ""))

        all_tags = set()
        descriptions = []

        # Prepare client
        client = None
        if self.mode == "ollama":
            from ..ollama_client import OllamaClient
            client = OllamaClient(base_url=self.base_url, model=self.model, timeout=60)
            if not client.check():
                self.failed.emit(name, f"Ollama not available at {self.base_url}")
                return

        for i, img in enumerate(samples):
            if self._cancel:
                self.log.emit("[CANCELLED]")
                return
            self.progress.emit(i + 1, len(samples), img)

            if self.mode == "ollama":
                full_path = os.path.join(fld, img)
                b64 = self._img_to_b64(full_path)
                if not b64:
                    self.log.emit(f"  [{i+1}/{len(samples)}] Skipped {img} (read error)")
                    continue
                self.log.emit(f"  [{i+1}/{len(samples)}] Analyzing {img}...")

                # NSFW detection via CV skin_fraction (fast, no API call)
                is_nsfw = False
                try:
                    from ..profile import get_image_profile
                    prof = get_image_profile(full_path)
                    skin = prof.get("skin_fraction", 0.0)
                    is_nsfw = skin >= 0.10
                except Exception:
                    pass

                # Also detect NSFW category from name
                if name.lower() in {"nsfw", "nsfw_characters", "18+", "explicit", "ecchi"}:
                    is_nsfw = True

                tag_list = sorted(t_mod.get_all_tags())

                if is_nsfw:
                    # NSFW path: get more tags (8-10), no description
                    tags = self._get_tags_ollama(client, b64, tag_list,
                                                  nsfw_mode=True, min_tags=8)
                    # Always add NSFW default tags
                    nsfw_defaults = s_mod.SETTINGS_DEFAULTS.get(
                        "nsfw_default_tags", [])
                    tags.update(t.lower() for t in nsfw_defaults)
                    all_tags.update(tags)
                    self.log.emit(f"    tags: {len(tags)} (NSFW, no description)")
                else:
                    tags = self._get_tags_ollama(client, b64, tag_list,
                                                  nsfw_mode=False, min_tags=5)
                    all_tags.update(tags)
                    self.log.emit(f"    tags: {len(tags)}")

                    desc = self._get_desc_ollama(client, b64)
                    if desc and desc not in descriptions:
                        descriptions.append(desc)
                        self.log.emit(f"    desc: {len(desc)} chars")
            elif self.mode == "lowlevel":
                self.log.emit(f"  [{i+1}/{len(samples)}] Analyzing {img}...")
                tags = self._get_tags_lowlevel(os.path.join(fld, img))
                all_tags.update(tags)
            else:
                self.failed.emit(name, f"Mode '{self.mode}' not supported")
                return

        # Fallback: if no tags were generated but we have descriptions,
        # extract tags by matching words in the descriptions against valid tags
        if not all_tags and descriptions and self.mode == "ollama":
            self.log.emit("  [INFO] No tags generated, extracting from descriptions...")
            valid = set(t_mod.get_all_tags())
            extracted = self._extract_tags_from_text(descriptions, valid)
            all_tags.update(extracted)
            if extracted:
                self.log.emit(f"  [OK] Extracted {len(extracted)} tags from descriptions")
                return

        # Generate merged prompt
        style_prompt = ""
        if self.mode == "ollama" and descriptions:
            self.progress.emit(len(samples), len(samples) + 1, "Merging descriptions...")
            if not self._cancel:
                style_prompt = self._merge_prompts_ollama(client, descriptions)
        elif self.mode == "lowlevel" and samples:
            self.progress.emit(len(samples), len(samples) + 1, "Building prompt...")
            if not self._cancel:
                style_prompt = self._get_prompt_lowlevel(
                    [os.path.join(fld, i) for i in samples],
                    category_name=name,
                )

        # Build palette weights for lowlevel mode (cheap + useful for matching)
        palette_weights = {}
        if self.mode == "lowlevel" and samples:
            palette_weights = self._get_palette_weights_lowlevel(
                [os.path.join(fld, i) for i in samples]
            )

        # Save results
        # Cap the tag list at TAG_BUDGET_PER_CATEGORY so even a very
        # generous LLM can't bloat the registry. We also run the
        # generic-tag filter one more time in case tags from different
        # images (across multiple LLM strategies) were merged together.
        from ..tag_policies import (
            TAG_BUDGET_PER_CATEGORY,
            cap_category_tags,
            filter_polluted_tags,
        )
        # Lower-case + dedupe before the cap
        normalised: Set[str] = set()
        for tag in all_tags:
            t = str(tag).strip().lower()
            if t:
                normalised.add(t)
        # Filter out GENERIC_OVERUSED and registry-wide stop words
        # before capping so the most informative tags survive.
        normalised = set(filter_polluted_tags(normalised))
        # Cap at budget
        capped = cap_category_tags(normalised, max_tags=TAG_BUDGET_PER_CATEGORY)
        valid_tags = capped
        # Keep only tags that are actually in the user's registry so we
        # don't pollute it with made-up words the LLM hallucinated.
        all_valid = t_mod.get_all_tags()
        valid_tags = {t for t in valid_tags if t in all_valid}

        cfg = c_mod.get_category_config(name)
        if valid_tags:
            cfg["tags"] = sorted(valid_tags)
        if style_prompt:
            cfg["prompt"] = style_prompt.strip()
        if palette_weights:
            cfg["palette_weights"] = palette_weights
        c_mod.write_category_config(name, cfg)

        result = {
            "tags": sorted(valid_tags),
            "prompt": style_prompt.strip(),
            "samples_used": len(samples),
        }
        self.finished.emit(name, result)


# ---------------------------------------------------------------------------
# RegenerateCategoryWorker
# ---------------------------------------------------------------------------

REGENERATE_VISION_PROMPT = """You are tagging a wallpaper image for a personal library.

Look at the image and return STRICT JSON in this exact shape:
{{
  "subject": "<1-3 word description of the main subject, lowercase>",
  "style":  "<visual style: e.g. anime, photograph, pixel-art, 3d-render, watercolor, illustration, digital-art, sketch>",
  "tags":   [<up to {max_tags} discriminating tags from the list below, lowercase, single words or hyphenated>],
  "palette": "<1-3 word colour description, lowercase>",
  "mood":   "<1-2 word mood, lowercase>"
}}

Rules for `tags`:
  * ONLY pick tags from the list below.
  * SKIP generic words: do not pick `illustration`, `digital-art`, `pastel`,
    `neon`, `vintage`, `retro`, `monochrome`, `minimalist`, `simple`,
    `detailed`, `sky`, `landscape`, `building`, `city`, `urban`, `nature`,
    `abstract`, `vibrant`, `colorful`, `warm`, `cool`, `dark`, `light`,
    `night`, `day` - they pollute the registry.
  * PREFER specific subjects (e.g. `dragon`, `samurai`, `forest`, `temple`)
    and specific styles (e.g. `pixel-art`, `cyberpunk`, `watercolor`).

Allowed tags (pick from this list):
{tag_list}

Respond with JSON only."""


REGENERATE_PROMPT_SYNTHESIS = """You are writing a unified-aesthetic prompt that captures
the visual identity of a wallpaper CATEGORY.

You will be given:
  * the category name
  * 3-6 short visual descriptions of individual wallpapers
  * the most common style, palette and mood signatures observed

Write a 2-3 sentence PROMPT that:
  * describes what the wallpapers look like as a unified set
  * mentions the dominant palette (concrete adjective, not just "vibrant")
  * mentions the dominant subject/scene type if one is obvious
  * mentions the dominant artistic style
  * does NOT start with "The image features..." or "A unified aesthetic..."
  * is written as a stable description that fits new wallpapers added later

Category: {category}
Sample descriptions:
{descriptions}
Dominant style: {style}
Dominant palette: {palette}
Dominant mood: {mood}

Respond with ONLY the prompt (2-3 sentences), no preamble, no bullets."""


class RegenerateCategoryWorker(QThread):
    """Vision + Text pipeline to (re)generate tags + prompt for a category.

    Pipeline per category:
      1. Pick N representative sample images (smallest + largest + spread).
      2. For each sample, call the *vision* model to extract structured
         JSON (subject, style, tags, palette, mood).
      3. Aggregate across samples: union of tags, majority vote on
         style / palette / mood, accumulate short descriptions.
      4. Call the *text* model once to write a 2-3 sentence unified
         aesthetic prompt from the aggregated descriptions.
      5. Run every tag through `tag_policies.filter_polluted_tags` and
         cap at `TAG_BUDGET_PER_CATEGORY`.
      6. Save `.category.json` and emit `finished(name, result)`.

    If the vision model errors out on a sample, a cheap heuristic
    palette signature is used so we never lose the whole category.

    Compared to the legacy GenerateTagsWorker, this worker uses TWO
    different Ollama models - the user can pick a vision-capable
    model for image analysis (e.g. llava-phi3) and a faster text
    model for prompt synthesis (e.g. qwen2.5) in the UI.
    """

    log = Signal(str)
    progress = Signal(int, int, str)
    finished = Signal(str, dict)
    failed = Signal(str, str)

    def __init__(self, category_name, vision_model="llava-phi3:3.8b",
                 text_model="qwen2.5:3b", base_url="http://localhost:11434",
                 max_samples=3, vision_timeout=240, text_timeout=120,
                 skip_videos=False):
        super().__init__()
        self.category_name = category_name
        self.vision_model = vision_model
        self.text_model = text_model
        self.base_url = base_url
        self.max_samples = max_samples
        self._vision_timeout = vision_timeout
        self._text_timeout = text_timeout
        self._skip_videos = skip_videos
        self._cancel = False

    def cancel(self):
        self._cancel = True

    @staticmethod
    def _pick_samples(files, n):
        if len(files) <= n:
            return files
        files_sorted = sorted(files, key=lambda p: os.path.getsize(p))
        pick = [files_sorted[0], files_sorted[-1]]
        if n >= 3:
            step = max(1, len(files_sorted) // (n - 1))
            for i in range(step, len(files_sorted) - 1, step):
                pick.append(files_sorted[i])
        return list(dict.fromkeys(pick))[:n]

    @staticmethod
    def _img_to_b64(path):
        """Read a file as a JPEG-encoded base64 string for the vision model.

        Handles:
          * Static images (PNG / JPG / WEBP)            -> PIL resize
          * Animated images (multi-frame GIF / WEBP)    -> first frame
          * Videos (MP4 / MKV / WEBM / MOV / AVI / ...) -> extract a
            single representative frame via ffmpeg. Without ffmpeg
            (or if extraction fails) we return None so the caller
            can fall back to the heuristic signature.

        The returned base64 is always JPEG @ <=512px so the payload
        fits comfortably in any small vision model's context window.
        """
        from io import BytesIO
        from PIL import Image
        import base64
        ext = os.path.splitext(path)[1].lower()
        animated_exts = {".gif", ".apng", ".mng", ".fli", ".flc", ".webp"}
        video_exts = frozenset({
            ".mp4", ".m4v", ".webm", ".mkv", ".avi", ".mov", ".flv",
            ".mpg", ".mpeg", ".mpe", ".mpv", ".ogv", ".wmv", ".asf",
            ".ts", ".m2ts", ".mts", ".vob", ".3gp", ".3gpp",
            ".rm", ".rmvb", ".ogm",
        })

        # Video path: extract one frame via ffmpeg
        if ext in video_exts:
            try:
                import shutil as _sh
                import subprocess
                import tempfile
                ffmpeg = os.environ.get("FFMPEG_BIN") or "ffmpeg"
                if not _sh.which(ffmpeg):
                    return None
                # Use a known-good time (3% of duration) to avoid the
                # typical intro / title-card frame at t=0. Fall back to
                # t=0.5s if we can't probe the duration.
                probe_t = 0.5
                try:
                    ffprobe = _sh.which("ffprobe") or "ffprobe"
                    out = subprocess.run(
                        [ffprobe, "-v", "error",
                         "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", path],
                        capture_output=True, text=True, timeout=10,
                    ).stdout.strip()
                    if out:
                        duration = float(out)
                        # Skip dark intro (3% beats most title cards).
                        probe_t = max(0.5, duration * 0.03)
                except Exception:
                    pass
                with tempfile.TemporaryDirectory(prefix="wa_") as td:
                    dst = os.path.join(td, "frame.jpg")
                    r = subprocess.run(
                        [ffmpeg, "-y", "-loglevel", "error",
                         "-ss", f"{probe_t:.3f}", "-i", path,
                         "-vframes", "1",
                         "-vf", "scale='min(512,iw)':-2",
                         "-q:v", "2", dst],
                        capture_output=True, timeout=30,
                    )
                    if r.returncode != 0 or not os.path.exists(dst):
                        return None
                    with open(dst, "rb") as f:
                        return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                return None

        # Image path
        try:
            im = Image.open(path)
            if ext in animated_exts and (
                getattr(im, "is_animated", False)
                or getattr(im, "n_frames", 1) > 1
            ):
                im.seek(0)
            im = im.convert("RGB")
            im.thumbnail((512, 512), Image.LANCZOS)
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=85, optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            return None

    @staticmethod
    def _heuristic_signature(img_path):
        from collections import Counter
        from PIL import Image
        try:
            im = Image.open(img_path).convert("RGB")
            im.thumbnail((128, 128))
            pixels = list(im.getdata())
        except Exception:
            return {}
        if not pixels:
            return {}
        r = sum(p[0] for p in pixels) // len(pixels)
        g = sum(p[1] for p in pixels) // len(pixels)
        b = sum(p[2] for p in pixels) // len(pixels)
        palette_bits = []
        if r + g + b < 200:
            palette_bits.append("dark")
        elif r + g + b > 600:
            palette_bits.append("light")
        if abs(r - g) < 20 and abs(g - b) < 20:
            palette_bits.append("monochrome")
        if b > r and b > g:
            palette_bits.append("cool")
        elif r > g and r > b:
            palette_bits.append("warm")
        return {
            "subject": "",
            "style": "heuristic",
            "tags": [],
            "palette": " ".join(palette_bits) or "neutral",
            "mood": "unknown",
        }

    @staticmethod
    def _clean_tags(raw_tags, focused_set):
        from ..tag_policies import GENERIC_OVERUSED
        if not raw_tags:
            return []
        if isinstance(raw_tags, str):
            from ..tag_policies import parse_llm_tag_response
            raw_tags = parse_llm_tag_response(
                raw_tags, focused_set, max_tags=8,
            )
        out = []
        seen = set()
        for t in raw_tags:
            tt = str(t).strip().lower().rstrip(".,;:!?\"'")
            if not tt or tt in seen:
                continue
            if tt in GENERIC_OVERUSED:
                continue
            if tt not in focused_set:
                continue
            out.append(tt)
            seen.add(tt)
            if len(out) >= 8:
                break
        return out

    def _majority(self, lst):
        if not lst:
            return ""
        counts: dict = {}
        for x in lst:
            counts[x] = counts.get(x, 0) + 1
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return best[0][0] if best else ""

    def _analyse_image(self, vision_client, img_path, focused_tags):
        tag_list = ", ".join(focused_tags)
        prompt = REGENERATE_VISION_PROMPT.format(
            tag_list=tag_list, max_tags=8,
        )
        b64 = self._img_to_b64(img_path)
        if not b64:
            return self._heuristic_signature(img_path)
        try:
            data = vision_client.generate_structured(
                prompt=prompt,
                schema_hint=(
                    '{"subject": "", "style": "", "tags": [], '
                    '"palette": "", "mood": ""}'
                ),
                images=[b64],
                temperature=0.05,
                max_tokens=200,
                default={},
                timeout=self._vision_timeout,
            )
        except Exception as e:
            self.log.emit(f"    [WARN] vision error: {e}")
            return self._heuristic_signature(img_path)
        if not data or not isinstance(data, dict):
            return self._heuristic_signature(img_path)
        return data

    def _synthesise_prompt(self, text_client, category, descriptions,
                            style, palette, mood):
        desc_block = "\n".join(f"  - {d}" for d in descriptions if d)
        if not desc_block:
            return None
        prompt = REGENERATE_PROMPT_SYNTHESIS.format(
            category=category,
            descriptions=desc_block,
            style=style or "unspecified",
            palette=palette or "unspecified",
            mood=mood or "unspecified",
        )
        try:
            out = text_client.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=220,
                timeout=self._text_timeout,
            )
        except Exception as e:
            self.log.emit(f"    [WARN] prompt synthesis error: {e}")
            return None
        if not out:
            return None
        out = out.strip().strip('"').strip("'")
        bad_prefixes = (
            "the image features", "a unified aesthetic can be achieved",
            "a unified aesthetic combines", "the unified aesthetic",
            "this category", "category:",
        )
        lower = out.lower()
        for bp in bad_prefixes:
            if lower.startswith(bp):
                for sep in (".", "\n"):
                    idx = out.find(sep)
                    if idx != -1 and idx < 200:
                        out = out[idx + 1:].strip()
                        break
                break
        if len(out) > 600:
            out = out[:600].rsplit(".", 1)[0] + "."
        return out or None

    def run(self):
        from .. import settings as s_mod
        from .. import tags as t_mod
        from .. import categories as c_mod
        from ..ollama_client import OllamaClient

        name = self.category_name
        dest = s_mod.resolve_dest_dir(s_mod.load_settings())
        fld = os.path.join(dest, name)

        if not os.path.isdir(fld):
            self.failed.emit(name, "Category folder not found")
            return

        # Note: formats_mod.ANIMATED_EXTENSIONS includes video formats
        # alongside GIF / WEBP / APNG. So when the user asks to "skip
        # videos" we must explicitly strip the video extensions and
        # keep only the animated-image ones.
        VIDEO_ONLY_EXTS = {
            ".mp4", ".m4v", ".webm", ".mkv", ".avi", ".mov", ".flv",
            ".mpg", ".mpeg", ".mpe", ".mpv", ".ogv", ".wmv", ".asf",
            ".ts", ".m2ts", ".mts", ".vob", ".3gp", ".3gpp",
            ".rm", ".rmvb", ".ogm", ".m2v",
        }
        if self._skip_videos:
            exts = formats_mod.STATIC_EXTENSIONS | (
                formats_mod.ANIMATED_EXTENSIONS - VIDEO_ONLY_EXTS
            )
        else:
            exts = formats_mod.STATIC_EXTENSIONS | formats_mod.ANIMATED_EXTENSIONS
        imgs = sorted([
            fn for fn in os.listdir(fld)
            if os.path.isfile(os.path.join(fld, fn))
            and not fn.startswith(".")
            and os.path.splitext(fn)[1].lower() in exts
        ])
        if not imgs:
            if self._skip_videos:
                self.failed.emit(name, "No static/animated images (videos skipped)")
            else:
                self.failed.emit(name, "No sample images")
            return

        full_paths = [os.path.join(fld, fn) for fn in imgs]
        samples = self._pick_samples(full_paths, self.max_samples)

        if self._skip_videos:
            skipped = sum(
                1 for fn in os.listdir(fld)
                if os.path.isfile(os.path.join(fld, fn))
                and not fn.startswith(".")
                and os.path.splitext(fn)[1].lower() in VIDEO_ONLY_EXTS
            )
            if skipped:
                self.log.emit(f"  (skipped {skipped} video files)")

        self.log.emit(
            f"Regenerating '{name}' with {len(samples)} samples\n"
            f"  Vision model: {self.vision_model}\n"
            f"  Text   model: {self.text_model}"
        )

        try:
            vision_client = OllamaClient(
                base_url=self.base_url,
                model=self.vision_model,
                timeout=self._vision_timeout,
            )
            text_client = OllamaClient(
                base_url=self.base_url,
                model=self.text_model,
                timeout=self._text_timeout,
            )
        except Exception as e:
            self.failed.emit(name, f"Could not init Ollama: {e}")
            return

        # Wait for Ollama to come back. Under memory pressure the
        # server sometimes goes down briefly (HTTP 503/connection
        # refused) while it swaps models. We retry for up to ~2 min
        # before giving up.
        import time as _wait_t
        wait_deadline = _wait_t.time() + 120  # 2 min
        backoff = 2.0
        while True:
            if vision_client.check():
                break
            if self._cancel:
                self.log.emit("[CANCELLED]")
                return
            remaining = wait_deadline - _wait_t.time()
            if remaining <= 0:
                self.failed.emit(
                    name, "Ollama is not reachable (2 min timeout)",
                )
                return
            self.log.emit(
                f"  Ollama unreachable, retrying in {min(backoff, remaining):.0f}s..."
            )
            _wait_t.sleep(min(backoff, remaining))
            backoff = min(backoff * 1.5, 15.0)

        if not vision_client._supports_vision(self.vision_model):
            self.failed.emit(
                name,
                f"Vision model '{self.vision_model}' does not support images",
            )
            return

        nsfw = name.lower() in {
            "nsfw", "nsfw_characters", "18+", "explicit", "ecchi",
        }
        from ..tag_policies import (
            CURATED_NSFW_TAGS, CURATED_SAFE_TAGS, PROTECTED_TAGS,
            TAG_BUDGET_PER_CATEGORY, build_focused_curated_list,
            cap_category_tags, dedupe_against_existing,
            filter_polluted_tags,
        )

        existing = c_mod.get_category_config(name)
        existing_tags = list(existing.get("tags") or []) + list(PROTECTED_TAGS)
        base_curated = CURATED_NSFW_TAGS if nsfw else CURATED_SAFE_TAGS
        focused = build_focused_curated_list(
            full_registry=t_mod.get_all_tags(),
            existing=existing_tags,
            nsfw=nsfw,
        )
        # Intersect with curated so we don't pass non-curated tags to the LLM
        focused = [t for t in focused if t in set(base_curated)][:60]
        focused_set = set(focused)

        all_tags: set = set()
        descriptions: list = []
        styles: list = []
        palettes: list = []
        moods: list = []
        subjects: list = []

        total_steps = len(samples) + 1
        for i, img_path in enumerate(samples, 1):
            if self._cancel:
                self.log.emit("[CANCELLED]")
                return
            self.progress.emit(i, total_steps, os.path.basename(img_path))
            self.log.emit(f"  [{i}/{len(samples)}] vision -> {os.path.basename(img_path)}")

            data = self._analyse_image(vision_client, img_path, focused)
            clean_tags = self._clean_tags(data.get("tags", []), focused_set)
            all_tags.update(clean_tags)

            sub = (data.get("subject") or "").strip().lower()
            sty = (data.get("style") or "").strip().lower()
            pal = (data.get("palette") or "").strip().lower()
            m = (data.get("mood") or "").strip().lower()
            if sub and sub != "unspecified":
                subjects.append(sub)
            if sty and sty not in ("unspecified", "heuristic"):
                styles.append(sty)
            if pal and pal != "unspecified":
                palettes.append(pal)
            if m and m != "unspecified":
                moods.append(m)

            bits = []
            if sub:
                bits.append(sub)
            if sty and sty != "heuristic":
                bits.append(f"in {sty} style")
            if pal:
                bits.append(f"{pal} palette")
            if m:
                bits.append(f"{m} mood")
            desc = ", ".join(bits) if bits else ""
            if desc:
                descriptions.append(desc)
            self.log.emit(
                f"    -> {len(clean_tags)} tags, '{desc or '(no desc)'}'"
            )

        if not all_tags and not descriptions:
            self.failed.emit(name, "Vision model produced no usable output")
            return

        all_tags.add(name.lower())
        filtered = set(filter_polluted_tags(all_tags))
        deduped = dedupe_against_existing(filtered, existing.get("tags", []))
        capped = cap_category_tags(deduped, max_tags=TAG_BUDGET_PER_CATEGORY)
        valid_tags = sorted(capped)

        dom_style = self._majority(styles)
        dom_palette = self._majority(palettes)
        dom_mood = self._majority(moods)

        self.progress.emit(total_steps, total_steps, "Synthesising prompt...")
        self.log.emit(f"  prompt -> {self.text_model}")
        new_prompt = self._synthesise_prompt(
            text_client, name, descriptions,
            dom_style, dom_palette, dom_mood,
        )
        if not new_prompt:
            new_prompt = (
                f"{name} wallpapers: {dom_style or 'distinctive'} style, "
                f"{dom_palette or 'characteristic'} palette, "
                f"{dom_mood or 'consistent'} mood."
            )
        new_prompt = new_prompt.strip()

        import time as _t
        cfg = dict(existing)
        cfg["name"] = name
        cfg["tags"] = valid_tags
        cfg["prompt"] = new_prompt
        cfg["style"] = dom_style
        cfg["palette"] = dom_palette
        cfg["mood"] = dom_mood
        cfg["samples_analysed"] = len(samples)
        cfg["regenerated_at"] = _t.strftime("%Y-%m-%dT%H:%M:%S")
        cfg["regenerated_with"] = {
            "vision_model": self.vision_model,
            "text_model": self.text_model,
            "tag_policies": True,
        }
        c_mod.write_category_config(name, cfg)

        try:
            vision_client.close()
            text_client.close()
        except Exception:
            pass

        self.log.emit(f"  final tags ({len(valid_tags)}): {valid_tags}")
        self.log.emit(f"  prompt: {new_prompt[:120]!r}")

        self.finished.emit(name, {
            "tags": valid_tags,
            "prompt": new_prompt,
            "samples_used": len(samples),
            "style": dom_style,
            "palette": dom_palette,
            "mood": dom_mood,
        })
