"""AI-powered rename dialog.

Lets the user pick an AI backend (CLIP / Ollama / Heuristic / Auto),
a rename strategy, and apply the rename to all currently
visible/selected files.

UX principles
--------------
* No auto-preview on open — the dialog opens with a clear "Run preview"
  button. The user configures backend + strategy first, then triggers
  preview when ready.
* No auto-refresh on settings change — adjusting the backend or
  strategy does NOT immediately re-run tag detection. The preview
  reflects the *latest* result; the user clicks "Refresh preview" to
  re-run with the new settings.
* Cancel/Stop button while preview is running so the user can abort
  a slow run.
* Memory-conscious: the CLIP vocabulary is capped (see ``rename._clip_tag_vocab``)
  and tensors are freed after each file.
* Limit input: a "Preview only the first N files" spinner keeps the
  dialog responsive on huge selections.

The dialog is backend-aware:
  * Auto   — tries Ollama → CLIP → heuristic, picks the first that
    responds.
  * CLIP   — uses the local OpenAI CLIP model + curated tag vocabulary.
  * Ollama — uses the configured local vision LLM.
  * Heuristic — pure CV pipeline (no AI), still useful as a fallback.
"""
import os
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QGroupBox, QFormLayout, QSpinBox, QCheckBox, QMessageBox, QProgressBar,
)

from ..rename import (
    RENAME_STRATEGIES, TAG_BASED_STRATEGIES, AI_TAG_BACKENDS, AI_TAG_MODES,
    ai_compute_renames, apply_renames,
)


# Canonical list of CLIP models for the dialog dropdown. Mirrors the
# table in the AI Models page but flattened for direct iteration.
_CLIP_MODELS = [
    "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px",
    "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64",
]

# Curated Ollama vision models for the dialog dropdown. The actual
# server-side list is queried dynamically when the backend changes.
_OLLAMA_DEFAULT_MODELS = [
    "llava:7b", "llava:13b", "llama3.2-vision:11b",
    "minicpm-v:8b", "moondream:latest",
]


def _list_clip_models() -> List[str]:
    """Return CLIP model names: curated list + locally available."""
    out = list(_CLIP_MODELS)
    try:
        from ..clip_client import available_clip_models
        for m in available_clip_models():
            if m not in out:
                out.append(m)
    except Exception:
        pass
    return out


def _list_ollama_models() -> List[str]:
    """Return Ollama model names: configured + locally available on server."""
    try:
        from .. import settings as _s
        from ..ollama_client import OllamaClient
        cfg = _s.load_settings()
        url = cfg.get("ollama_url", "http://localhost:11434")
        client = OllamaClient(base_url=url, timeout=3)
        try:
            installed = client.list_models() or []
        finally:
            client.close()
    except Exception:
        installed = []
    out: List[str] = []
    # Put the active model first so it's the default selection.
    try:
        active = cfg.get("ollama_model")
    except Exception:
        active = None
    if active and active not in out:
        out.append(active)
    for m in installed:
        if isinstance(m, dict):
            name = m.get("name")
        else:
            name = m
        if name and name not in out:
            out.append(name)
    # Pad with curated defaults so the dropdown isn't empty when the
    # server is unreachable.
    for m in _OLLAMA_DEFAULT_MODELS:
        if m not in out:
            out.append(m)
    return out


class _PreviewJob(QThread):
    """Compute preview pairs in the background so the dialog stays responsive.

    Uses ``AIRenamer`` directly so the Ollama HTTP client and per-file
    tag cache are reused across the batch (and survive across preview
    refreshes within the same dialog session). Honors an internal
    ``cancel`` flag so the user can stop a slow preview without
    closing the dialog.
    """
    progress = Signal(int, int, str)         # (cur, total, msg)
    finished_ok = Signal(list, list)         # (pairs, log_lines)
    failed = Signal(str)

    def __init__(self, files, strategy, backend, category, max_tags,
                 model=None, force_reprocess=False, mode="auto"):
        super().__init__()
        self.files = list(files)
        self.strategy = strategy
        self.backend = backend
        self.category = category
        self.max_tags = max_tags
        self.model = model
        self.force_reprocess = force_reprocess
        self.mode = mode
        self._cancel = False
        # Filled in by run(); used by the slot to surface per-file stats.
        self.renamer: Optional[object] = None

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            from ..rename import AIRenamer, TAG_BASED_STRATEGIES, build_renames

            log: List[str] = []
            ren = AIRenamer(
                backend=self.backend,
                mode=self.mode,
                model=self.model,
                max_tags=self.max_tags,
                force_reprocess=self.force_reprocess,
            )
            self.renamer = ren

            # Emit batch-level progress (count of files).
            total = len(self.files)
            self.progress.emit(
                0, total,
                f"Processing {total} files with {self.backend}"
                f" (mode={self.mode})…",
            )

            # Iterate per-file so we can emit granular progress events.
            # detect_tags() never raises — failures are isolated and
            # logged. We build the rename pair inline so the user
            # sees results as they come in.
            tags_by_file = {}
            subject_by_file = {}
            for i, p in enumerate(self.files, 1):
                if self._cancel:
                    raise InterruptedError("cancelled")
                tags, subject = ren.detect_tags(p, category=self.category)
                tags_by_file[p] = tags
                subject_by_file[p] = subject
                self.progress.emit(
                    i, total,
                    f"{i}/{total}: {os.path.basename(p)} → {tags[:3]}",
                )

            # Build rename pairs only for tag-based strategies; for
            # other strategies the per-file tags aren't needed.
            if self.strategy in TAG_BASED_STRATEGIES:
                pairs = build_renames(
                    self.files, strategy=self.strategy,
                    category=self.category or "",
                    tags_by_file=tags_by_file,
                    subject_by_file=subject_by_file,
                    max_tags=self.max_tags,
                )
            else:
                pairs = build_renames(
                    self.files, strategy=self.strategy,
                    category=self.category or "",
                )

            # Drain renamer log to the dialog log too.
            log.extend(ren.log_lines)
            self.finished_ok.emit(pairs, log)
        except InterruptedError:
            self.failed.emit("Cancelled")
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class AIRenameDialog(QDialog):
    """Dialog: AI tag detection + rename preview + apply.

    The dialog never starts tag detection on its own. The user picks a
    backend, a strategy, and an optional preview cap, then explicitly
    clicks "Run preview" to compute the rename plan.
    """

    def __init__(self, files: List[str], category: str = "",
                 default_backend: str = "auto",
                 default_strategy: str = "category_tags",
                 max_tags: int = 3,
                 preview_limit: int = 50,
                 default_model: Optional[str] = None,
                 parent=None):
        super().__init__(parent)
        self.files = list(files)
        self.category = category
        self._default_model = default_model
        self._pairs: List[Tuple[str, str]] = []
        self._preview_job: Optional[_PreviewJob] = None
        self._has_preview = False
        # Default to force-reprocess so every dialog open runs the AI
        # over the full file list — the user's explicit ask.
        self._force_reprocess = True
        self.setWindowTitle(f"AI Rename — {len(self.files)} files")
        self.setMinimumSize(840, 720)
        self._build(default_backend, default_strategy, max_tags, preview_limit)
        # NOTE: we intentionally do NOT auto-run preview here. The user
        # explicitly clicks "Run preview" once the settings are right.

    # -------- BUILD --------

    def _build(self, default_backend, default_strategy, max_tags, preview_limit):
        l = QVBoxLayout(self)
        l.setContentsMargins(16, 16, 16, 16)
        l.setSpacing(10)

        title = QLabel(f"AI Rename — {len(self.files)} files")
        title.setObjectName("title")
        l.addWidget(title)
        if self.category:
            sub = QLabel(f"Category: {self.category}")
            sub.setObjectName("subtitle")
            l.addWidget(sub)

        # ---- Backend selector ----
        bg = QGroupBox("AI backend")
        bg_layout = QFormLayout(bg)
        self._backend = QComboBox()
        labels = {
            "auto":      "Auto (CLIP → Ollama → Analyzer — recommended)",
            "heuristic": "Analyzer (same as category assignment, no AI)",
            "clip":      "CLIP (local OpenAI model — semantic match against registry)",
            "ollama":    "Ollama (local vision LLM)",
        }
        for k in AI_TAG_BACKENDS:
            self._backend.addItem(labels.get(k, k), k)
        # Default to "auto" so the user gets AI-driven semantic tag
        # detection against the full registry (CLIP if installed,
        # Ollama otherwise). This matches the user's explicit request:
        # "use AI to use the available tags and classify the images".
        idx = self._backend.findData(default_backend)
        if idx < 0:
            idx = self._backend.findData("auto")
        if idx >= 0:
            self._backend.setCurrentIndex(idx)
        self._backend.currentIndexChanged.connect(self._on_backend_changed)
        bg_layout.addRow("Backend:", self._backend)

        # ---- Backend status (CLIP / Ollama availability) ----
        # The user needs to see which AI backends are actually wired up
        # so they know why the auto cascade picks the path it does.
        self._backend_status = QLabel("")
        self._backend_status.setObjectName("statSmall")
        self._backend_status.setStyleSheet("color: #888;")
        self._backend_status.setWordWrap(True)
        bg_layout.addRow("Status:", self._backend_status)
        self._refresh_backend_status()

        # ---- Analyzer mode ----
        # When the cascade falls back to the analyzer (last resort) or
        # when "heuristic" is picked explicitly, this dropdown picks
        # WHICH analyzer runs. Default "auto" uses the configured
        # `organize_mode` from settings.
        self._mode = QComboBox()
        mode_labels = {
            "auto":     "Auto (use configured organize mode)",
            "lowlevel": "LowLevel CV (no AI, pure heuristics)",
            "fusion":   "Fusion (CLIP + LowLevel — best quality)",
            "clip":     "CLIP zero-shot (vision-language)",
            "ollama":   "Ollama vision LLM (most expensive)",
        }
        for k in AI_TAG_MODES:
            self._mode.addItem(mode_labels.get(k, k), k)
        self._mode.setCurrentIndex(0)  # auto
        self._mode.setToolTip(
            "Analyzer the rename backend uses as the last-resort fallback.\n"
            "Only used when no AI backend (CLIP/Ollama) is available.\n"
            "  Auto   - same analyzer configured for organising\n"
            "  Fusion - CLIP + LowLevel (recommended when CLIP is loaded)\n"
            "  CLIP   - CLIP zero-shot only (no CV heuristics)\n"
            "  LowLevel - classical CV only, no AI model\n"
            "  Ollama - local vision LLM"
        )
        bg_layout.addRow("Analyzer fallback:", self._mode)

        # Model selector — visible only for backends that need it.
        # When "auto" is picked, the model field is disabled (we use
        # whatever is configured for each backend as we cascade).
        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.setMinimumWidth(180)
        self._model_combo.setToolTip(
            "Pick the specific model the selected backend should use. "
            "Leave on '(configured)' to use the active model from settings."
        )
        bg_layout.addRow("Model:", self._model_combo)

        self._model_refresh_btn = QPushButton("Refresh list")
        self._model_refresh_btn.setObjectName("ghost")
        self._model_refresh_btn.clicked.connect(self._refresh_model_list)
        bg_layout.addRow("", self._model_refresh_btn)

        l.addWidget(bg)

        # ---- Strategy + options ----
        sg = QGroupBox("Rename strategy")
        sg_layout = QFormLayout(sg)
        self._strat = QComboBox()
        for key, label, desc in RENAME_STRATEGIES:
            self._strat.addItem(f"{label}  —  {desc}", key)
        idx = self._strat.findData(default_strategy)
        if idx >= 0:
            self._strat.setCurrentIndex(idx)
        # Note: changing the strategy does NOT auto-refresh preview;
        # user clicks "Run preview" to apply the new setting.
        sg_layout.addRow("Strategy:", self._strat)

        self._max_tags = QSpinBox()
        self._max_tags.setRange(1, 8)
        self._max_tags.setValue(max_tags)
        sg_layout.addRow("Max tags in filename:", self._max_tags)

        # Preview cap to keep the dialog responsive on huge selections.
        self._preview_limit = QSpinBox()
        self._preview_limit.setRange(1, max(1, len(self.files)))
        self._preview_limit.setValue(min(preview_limit, len(self.files)))
        self._preview_limit.setToolTip(
            "Only compute preview names for the first N files. "
            "Set to the total to preview everything."
        )
        sg_layout.addRow("Preview limit:", self._preview_limit)

        self._force_reprocess_chk = QCheckBox(
            "Reprocess every image (bypass AI tag cache)"
        )
        self._force_reprocess_chk.setChecked(self._force_reprocess)
        self._force_reprocess_chk.setToolTip(
            "When checked, every image is re-tagged from scratch even if "
            "it was already processed earlier in this session. Slower but "
            "guarantees the latest model / settings are used for every file."
        )
        self._force_reprocess_chk.toggled.connect(
            lambda v: setattr(self, "_force_reprocess", v)
        )
        sg_layout.addRow(self._force_reprocess_chk)

        self._dry = QCheckBox("Dry run (preview only)")
        self._dry.setChecked(False)
        sg_layout.addRow(self._dry)
        l.addWidget(sg)

        # ---- Warning + progress ----
        self._warning = QLabel("")
        self._warning.setObjectName("statSmall")
        self._warning.setWordWrap(True)
        self._warning.setVisible(False)
        l.addWidget(self._warning)

        self._prog = QProgressBar()
        self._prog.setVisible(False)
        self._prog.setTextVisible(True)
        l.addWidget(self._prog)

        # ---- Preview ----
        pt = QLabel("Preview (right-click rows to inspect)")
        pt.setObjectName("sectionLabel")
        l.addWidget(pt)
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Original", "AI tags", "New name"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        l.addWidget(self._table, 1)

        self._stats = QLabel("")
        self._stats.setObjectName("statSmall")
        l.addWidget(self._stats)

        # ---- Buttons ----
        bb = QHBoxLayout()
        self._run_btn = QPushButton("Run preview")
        self._run_btn.setObjectName("primary")
        self._run_btn.clicked.connect(self._refresh_preview)
        bb.addWidget(self._run_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("danger")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_preview)
        bb.addWidget(self._stop_btn)

        bb.addStretch()
        b_cancel = QPushButton("Cancel")
        b_cancel.setObjectName("ghost")
        b_cancel.clicked.connect(self.reject)
        bb.addWidget(b_cancel)

        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setObjectName("success")
        self._apply_btn.setEnabled(False)  # disabled until preview is ready
        self._apply_btn.clicked.connect(self._on_apply)
        bb.addWidget(self._apply_btn)
        l.addLayout(bb)

        self._on_strategy_changed()
        self._on_backend_changed()
        self._update_warning()

    # -------- warning / size check --------

    def _update_warning(self):
        """Show a warning when the file count is large enough to be slow."""
        n = len(self.files)
        backend = self._backend.currentData()
        warn_text = ""
        if backend in ("ollama", "auto") and n > 10:
            est = n * 5  # rough: 5s per image via Ollama
            warn_text = (
                f"⚠ {n} files via Ollama may take a few minutes "
                f"(~{est}s). Consider Preview limit + Dry run."
            )
        elif backend == "clip" and n > 100:
            warn_text = (
                f"⚠ {n} files via CLIP — preview limit recommended to "
                f"keep memory in check."
            )
        elif n > 500:
            warn_text = (
                f"⚠ {n} files — preview limit strongly recommended."
            )
        if warn_text:
            self._warning.setText(warn_text)
            self._warning.setStyleSheet("color: #e0a040;")
            self._warning.setVisible(True)
        else:
            self._warning.setVisible(False)

    # -------- behaviour --------

    def _on_backend_changed(self):
        b = self._backend.currentData()
        # Populate the model dropdown with backend-appropriate entries.
        self._populate_model_combo(b)
        if b == "ollama":
            self._model_combo.setEnabled(True)
            self._model_refresh_btn.setEnabled(True)
            self._backend_status.setText(
                "Ollama runs locally. Pick a vision model below — every "
                "file will be tagged with that specific model."
            )
        elif b == "clip":
            self._model_combo.setEnabled(True)
            self._model_refresh_btn.setEnabled(True)
            self._backend_status.setText(
                "CLIP encodes the image and scores it against the full "
                "tag registry. Pick the specific CLIP variant below."
            )
        elif b == "heuristic":
            self._model_combo.setEnabled(False)
            self._model_refresh_btn.setEnabled(False)
            self._backend_status.setText(
                "Analyzer pipeline (same as category assignment). "
                "No AI model — tags come from CV heuristics + content "
                "detectors (anime/cyberpunk/portrait/...) when their "
                "signals fire. Best when no CLIP/Ollama is available."
            )
        else:  # auto
            self._model_combo.setEnabled(False)
            self._model_refresh_btn.setEnabled(False)
            self._refresh_backend_status()
        self._update_warning()
        self._apply_btn.setEnabled(self._has_preview)

    def _refresh_backend_status(self):
        """Probe CLIP and Ollama availability for the auto cascade."""
        if self._backend.currentData() != "auto":
            return
        # CLIP availability probe (no model load — just check the engine).
        clip_ok = False
        clip_msg = ""
        try:
            from ..clip_client import get_engine
            engine = get_engine()
            clip_ok = bool(engine.available)
        except Exception as e:
            clip_msg = f" ({e})"
        # Ollama availability probe.
        ollama_ok = False
        try:
            from .. import settings as _s
            from ..ollama_client import OllamaClient
            cfg = _s.load_settings()
            client = OllamaClient(
                base_url=cfg.get("ollama_url", "http://localhost:11434"),
                timeout=2,
            )
            try:
                ollama_ok = bool(client.list_models())
            finally:
                client.close()
        except Exception:
            ollama_ok = False

        cascade = []
        if clip_ok:
            cascade.append("CLIP")
        if ollama_ok:
            cascade.append("Ollama")
        cascade.append("Analyzer")
        chain = " → ".join(cascade)
        if clip_ok:
            self._backend_status.setText(
                f"Auto cascade: {chain}. "
                "CLIP will score each image against the full registry "
                "and pick the top-K most relevant tags.{oll}".format(
                    oll=("" if ollama_ok else " Ollama server not reachable.")
                )
            )
        elif ollama_ok:
            self._backend_status.setText(
                f"Auto cascade: {chain}. CLIP not loaded — Ollama "
                "will be used (slower, vision LLM)."
            )
        else:
            self._backend_status.setText(
                f"Auto cascade: {chain}. No AI backend available — "
                "will fall back to analyzer content heuristics. "
                "Install CLIP (`pip install git+https://github.com/openai/CLIP.git`) "
                "or start Ollama for AI-driven semantic tagging."
            )

    def _on_strategy_changed(self):
        strat = self._strat.currentData()
        is_tag = strat in TAG_BASED_STRATEGIES
        self._max_tags.setEnabled(is_tag)

    # -------- model dropdown --------

    def _populate_model_combo(self, backend: str):
        """Fill the model combo with entries appropriate for `backend`."""
        prev = self._model_combo.currentText() if self._model_combo.count() else ""
        self._model_combo.blockSignals(True)
        try:
            self._model_combo.clear()
            if backend == "clip":
                models = _list_clip_models()
                # Use the configured active CLIP model by default.
                if not self._default_model:
                    try:
                        from .. import settings as _s
                        self._default_model = _s.load_settings().get("clip_model")
                    except Exception:
                        pass
                self._model_combo.addItem("(configured)", "")
                self._model_combo.addItems(models)
                # Pre-select default_model (or "ViT-B/32" fallback).
                target = self._default_model or "ViT-B/32"
                idx = self._model_combo.findText(target)
                if idx < 0 and models:
                    # Not in the catalog — add it manually so the user
                    # can still type it.
                    self._model_combo.addItem(target)
                    idx = self._model_combo.findText(target)
                if idx >= 0:
                    self._model_combo.setCurrentIndex(idx)
            elif backend == "ollama":
                models = _list_ollama_models()
                if not self._default_model:
                    try:
                        from .. import settings as _s
                        self._default_model = _s.load_settings().get("ollama_model")
                    except Exception:
                        pass
                self._model_combo.addItem("(configured)", "")
                self._model_combo.addItems(models)
                target = self._default_model or (models[0] if models else "")
                idx = self._model_combo.findText(target)
                if idx < 0 and target:
                    self._model_combo.addItem(target)
                    idx = self._model_combo.findText(target)
                if idx >= 0:
                    self._model_combo.setCurrentIndex(idx)
            else:
                self._model_combo.addItem("(configured)", "")
                self._model_combo.setCurrentIndex(0)
        finally:
            self._model_combo.blockSignals(False)
        # If user previously picked something, keep it when possible.
        if prev and self._model_combo.findText(prev) >= 0:
            self._model_combo.setCurrentIndex(self._model_combo.findText(prev))

    def _refresh_model_list(self):
        """Re-query CLIP/Ollama and rebuild the dropdown."""
        backend = self._backend.currentData()
        self._populate_model_combo(backend)
        # Briefly tell the user we updated the list.
        prev = self._backend_status.text()
        self._backend_status.setText(prev + "  •  model list refreshed")

    def _selected_model(self) -> Optional[str]:
        """Return the explicit model name, or None for '(configured)'."""
        text = self._model_combo.currentText().strip()
        if not text or text == "(configured)":
            return None
        return text

    def _refresh_preview(self):
        if not self.files:
            self._stats.setText("No files to rename.")
            return
        if self._preview_job and self._preview_job.isRunning():
            # Already running — stop it before starting a new run.
            self._preview_job.cancel()
            self._preview_job.wait(500)
        strat = self._strat.currentData()
        backend = self._backend.currentData()
        cap = int(self._preview_limit.value())
        files_to_process = self.files[:cap]
        # Pull the explicit model selection (None means "use whatever's
        # configured") so the user can pick a specific model per-rename.
        model = self._selected_model()
        if model:
            backend_label = f"{backend} ({model})"
        else:
            backend_label = backend
        self._prog.setVisible(True)
        self._prog.setRange(0, len(files_to_process))
        self._prog.setValue(0)
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._apply_btn.setEnabled(False)
        self._backend_status.setText(
            f"Running preview with backend={backend_label} on {len(files_to_process)} files…"
        )
        self._preview_job = _PreviewJob(
            files=files_to_process,
            strategy=strat,
            backend=backend,
            category=self.category,
            max_tags=self._max_tags.value(),
            model=model,
            force_reprocess=self._force_reprocess,
            mode=self._mode.currentData() or "auto",
        )
        self._preview_job.progress.connect(self._on_preview_progress)
        self._preview_job.finished_ok.connect(self._on_preview_done)
        self._preview_job.failed.connect(self._on_preview_failed)
        self._preview_job.start()

    def _stop_preview(self):
        if self._preview_job and self._preview_job.isRunning():
            self._preview_job.cancel()
            self._backend_status.setText("Stopping preview…")

    def _on_preview_progress(self, cur, total, msg):
        self._prog.setValue(cur)
        self._backend_status.setText(msg)

    def _on_preview_done(self, pairs, log_lines):
        self._prog.setVisible(False)
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._pairs = pairs
        self._has_preview = True
        self._apply_btn.setEnabled(bool(self._pairs))
        self._refresh_table()
        backend = self._backend.currentData()
        model = self._selected_model()
        label = backend + (f" / {model}" if model else " / (configured)")
        # Surface AIRenamer stats (processed / cached / failed) so the
        # user knows how many files were retagged vs served from cache
        # vs failed outright.
        ren = getattr(self._preview_job, "renamer", None)
        stats_extra = ""
        if ren is not None:
            stats_extra = (
                f"  •  processed {ren.processed}"
                f"  •  cache hits {ren.cached_hits}"
                f"  •  failed {ren.failed}"
            )
        status = f"Preview ready — {label} • {len(pairs)} pair(s){stats_extra}"
        if log_lines:
            status += f"  ({len(log_lines)} log lines)"
        self._backend_status.setText(status)

    def _on_preview_failed(self, msg):
        self._prog.setVisible(False)
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._apply_btn.setEnabled(False)
        self._has_preview = False
        self._backend_status.setText(f"Preview failed: {msg}")

    def _refresh_table(self):
        visible = self._pairs[:200]
        self._table.setRowCount(len(visible))
        for i, (old, new) in enumerate(visible):
            old_name = os.path.basename(old)
            new_name = os.path.basename(new)
            old_item = QTableWidgetItem(old_name)
            new_item = QTableWidgetItem(new_name)
            tag_item = QTableWidgetItem(
                ", ".join(_extract_tag_hints(old, new, self._max_tags.value())) or "—"
            )
            if old == new:
                old_item.setForeground(Qt.GlobalColor.gray)
                new_item.setForeground(Qt.GlobalColor.gray)
            else:
                new_item.setForeground(Qt.GlobalColor.green)
            self._table.setItem(i, 0, old_item)
            self._table.setItem(i, 1, tag_item)
            self._table.setItem(i, 2, new_item)
        self._table.resizeRowsToContents()
        changed = sum(1 for o, n in self._pairs if o != n)
        unchanged = len(self._pairs) - changed
        collisions = sum(1 for o, n in self._pairs
                        if n in [p[1] for p in self._pairs if p[0] != o])
        cap = self._preview_limit.value()
        suffix = (f"  (preview limited to {cap}/{len(self.files)} files)"
                  if cap < len(self.files) else "")
        self._stats.setText(
            f"{changed} will be renamed, {unchanged} unchanged{suffix}"
            + (f"  (showing first {len(visible)} of {len(self._pairs)})"
               if len(self._pairs) > len(visible) else "")
            + (f"  |  {collisions} potential collisions" if collisions else "")
        )

    def _on_apply(self):
        if not self._pairs:
            QMessageBox.information(self, "Nothing to apply",
                                    "No preview pairs available.")
            return
        n_changed = sum(1 for o, n in self._pairs if o != n)
        if n_changed == 0:
            QMessageBox.information(self, "No changes",
                                    "Nothing to rename.")
            return
        msg = f"Rename {n_changed} files using AI-detected tags?"
        if self._dry.isChecked():
            msg += "\n\n(Dry run — no files will be modified)"
        r = QMessageBox.question(self, "Confirm AI rename", msg,
                                 QMessageBox.Yes | QMessageBox.No)
        if r != QMessageBox.Yes:
            return
        self._apply_btn.setEnabled(False)
        self._run_btn.setEnabled(False)
        self._backend_status.setText("Applying rename…")
        stats = apply_renames(self._pairs, dry_run=self._dry.isChecked())
        parts = [f"Renamed: {stats['renamed']}"]
        if stats["skipped"]:
            parts.append(f"Skipped: {stats['skipped']}")
        if stats["errors"]:
            parts.append(f"Errors: {stats['errors']}")
            if stats["error_list"]:
                parts.append("\nFirst errors:\n" +
                             "\n".join(stats["error_list"][:5]))
        QMessageBox.information(self, "Result", "\n".join(parts))
        if not self._dry.isChecked():
            self.accept()

    def reject(self):
        if self._preview_job and self._preview_job.isRunning():
            self._preview_job.cancel()
        super().reject()


def _extract_tag_hints(old: str, new: str, max_tags: int) -> List[str]:
    """Best-effort: extract candidate tag tokens from a new filename.

    The dialog doesn't get the per-file tag list (it lives only inside
    ai_compute_renames), but for tag-based strategies the new filename
    *is* built from the tags. We split on `_` and strip the category /
    date prefix to recover a rough preview of what was detected.
    """
    try:
        from . import categories as cats
        cat = cats.NSFW_FOLDER
    except Exception:
        cat = "NSFW"
    base = os.path.basename(new)
    name, _ext = os.path.splitext(base)
    parts = [p for p in name.split("_") if p and p != cat]
    # Drop leading category/date prefixes.
    cleaned: List[str] = []
    for p in parts:
        # Drop purely-numeric parts (sequential, date), short ones,
        # and any obvious category names.
        if p.isdigit():
            continue
        if len(p) < 2:
            continue
        cleaned.append(p)
        if len(cleaned) >= max_tags:
            break
    return cleaned