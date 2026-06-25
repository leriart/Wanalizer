"""AI-powered rename dialog.

Mirrors the Organize page AI workflow:
  * Pick an analysis mode (Low-Level CV / CLIP / CLIP + LowLevel / Ollama).
  * Toggle AI options (NSFW detection, description → tags,
    direct classification, classify by tags/prompt).
  * Pick a model when mode is CLIP or Ollama.
  * Run preview, inspect, apply.

UX principles
--------------
* No auto-preview on open — the dialog opens with a clear "Run preview"
  button. The user configures mode + options first, then triggers
  preview when ready.
* No auto-refresh on settings change — adjusting the mode or
  options does NOT immediately re-run tag detection. The preview
  reflects the *latest* result; the user clicks "Refresh preview" to
  re-run with the new settings.
* Cancel/Stop button while preview is running so the user can abort
  a slow run without closing the dialog.
* Memory-conscious: the CLIP vocabulary is capped (see ``rename._clip_tag_vocab``)
  and tensors are freed after each file.
* Limit input: a "Preview only the first N files" spinner keeps the
  dialog responsive on huge selections.
"""
import os
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QGroupBox, QFormLayout, QSpinBox, QCheckBox, QMessageBox, QProgressBar,
)

from .. import settings as s
from ..rename import (
    RENAME_STRATEGIES, TAG_BASED_STRATEGIES,
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

    When ``use_organize_pipeline`` is True (the default), the renamer
    runs the exact same analyzer pipeline as the Organize page:
    ``get_analyzer(mode, settings)`` + ``_detect_tags_for_file``.
    """
    progress = Signal(int, int, str)         # (cur, total, msg)
    finished_ok = Signal(list, list)         # (pairs, log_lines)
    failed = Signal(str)

    def __init__(self, files, strategy, mode, category, max_tags,
                 model=None, force_reprocess=False,
                 organize_options=None):
        super().__init__()
        self.files = list(files)
        self.strategy = strategy
        self.mode = mode
        self.category = category
        self.max_tags = max_tags
        self.model = model
        self.force_reprocess = force_reprocess
        self.organize_options = organize_options or {}
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
                backend="organize",
                mode=self.mode,
                model=self.model,
                max_tags=self.max_tags,
                force_reprocess=self.force_reprocess,
                organize_options=self.organize_options,
            )
            self.renamer = ren

            # Emit batch-level progress (count of files).
            total = len(self.files)
            self.progress.emit(
                0, total,
                f"Processing {total} files with mode={self.mode}",
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

    Mirrors the Organize page AI workflow:
      * Pick an analysis mode (Low-Level CV / CLIP / Fusion / Ollama).
      * Toggle AI options (NSFW detection, description → tags,
        direct classification, classify by tags/prompt).
      * Pick a model when mode is CLIP or Ollama.
      * Click "Run preview" to compute the rename plan.

    The dialog never starts tag detection on its own. The user picks a
    mode and options, then explicitly clicks "Run preview".
    """

    # Legacy backend -> Organize mode mapping.
    _LEGACY_BACKEND_MAP = {
        "auto": "auto",
        "heuristic": "lowlevel",
        "clip": "clip",
        "ollama": "ollama",
    }

    def __init__(self, files: List[str], category: str = "",
                 default_backend: str = "auto",
                 default_strategy: str = "category_tags",
                 max_tags: int = 3,
                 preview_limit: int = 50,
                 default_model: Optional[str] = None,
                 nsfw: bool = True,
                 describe: bool = False,
                 classify: bool = False,
                 classify_method: str = "tags",
                 parent=None):
        super().__init__(parent)
        self.files = list(files)
        self.category = category
        self._default_model = default_model
        self._nsfw = nsfw
        self._describe = describe
        self._classify = classify
        self._classify_method = classify_method
        self._mode = self._resolve_mode(default_backend)
        self._pairs: List[Tuple[str, str]] = []
        self._preview_job: Optional[_PreviewJob] = None
        self._has_preview = False
        # Default to force-reprocess so every dialog open runs the AI
        # over the full file list — the user's explicit ask.
        self._force_reprocess = True
        self.setWindowTitle(f"AI Rename — {len(self.files)} files")
        self.setMinimumSize(840, 720)
        self._build(default_strategy, max_tags, preview_limit)
        # NOTE: we intentionally do NOT auto-run preview here. The user
        # explicitly clicks "Run preview" once the settings are right.

    def _resolve_mode(self, backend: str) -> str:
        """Map a legacy backend string to an Organize analysis mode."""
        cfg = s.load_settings()
        mode = self._LEGACY_BACKEND_MAP.get(backend, backend)
        if mode == "auto":
            mode = cfg.get("organize_mode", "lowlevel")
        if mode not in ("lowlevel", "clip", "fusion", "ollama"):
            mode = "lowlevel"
        return mode

    # -------- BUILD --------

    def _build(self, default_strategy, max_tags, preview_limit):
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

        # ---- Analysis mode selector (same as Organize page) ----
        bg = QGroupBox("Analysis Mode")
        bg_layout = QFormLayout(bg)
        self._mode_combo = QComboBox()
        mode_items = [
            ("lowlevel", "Low-Level CV (edges, textures, shapes, HOG)"),
            ("clip", "CLIP (vision-language model)"),
            ("fusion", "CLIP + LowLevel (fusion, recommended)"),
            ("ollama", "Ollama (local vision LLM)"),
        ]
        for key, label in mode_items:
            self._mode_combo.addItem(label, key)
        idx = self._mode_combo.findData(self._mode)
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)
        self._mode_desc = QLabel("")
        self._mode_desc.setObjectName("statSmall")
        bg_layout.addRow("Mode:", self._mode_combo)
        bg_layout.addRow("", self._mode_desc)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        # Model selector — visible only for CLIP / Ollama modes.
        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.setMinimumWidth(180)
        self._model_combo.setToolTip(
            "Pick the specific model the selected mode should use. "
            "Leave on '(configured)' to use the active model from settings."
        )
        bg_layout.addRow("Model:", self._model_combo)

        self._model_refresh_btn = QPushButton("Refresh list")
        self._model_refresh_btn.setObjectName("ghost")
        self._model_refresh_btn.clicked.connect(self._refresh_model_list)
        bg_layout.addRow("", self._model_refresh_btn)

        # Status label showing whether the selected mode is available.
        self._backend_status = QLabel("")
        self._backend_status.setObjectName("statSmall")
        self._backend_status.setStyleSheet("color: #888;")
        self._backend_status.setWordWrap(True)
        bg_layout.addRow("Status:", self._backend_status)

        l.addWidget(bg)

        # ---- AI Options (same as Organize page) ----
        self._ai_opts = QGroupBox("AI Options")
        ai_l = QVBoxLayout(self._ai_opts)
        self._cb_nsfw = QCheckBox("AI NSFW detection")
        self._cb_nsfw.setChecked(self._nsfw)
        ai_l.addWidget(self._cb_nsfw)
        ar = QHBoxLayout()
        self._cb_describe = QCheckBox("AI description -> tags")
        self._cb_describe.setChecked(self._describe)
        self._cb_classify = QCheckBox("AI direct classification")
        self._cb_classify.setChecked(self._classify)
        ar.addWidget(self._cb_describe)
        ar.addWidget(self._cb_classify)
        ar.addStretch()
        ai_l.addLayout(ar)
        # Classification method toggle (tags vs prompt)
        meth_l = QHBoxLayout()
        meth_l.addWidget(QLabel("Classify by:"))
        self._classify_method_combo = QComboBox()
        self._classify_method_combo.addItem("Tags (IA sugiere tags, sistema elige categoria)", "tags")
        self._classify_method_combo.addItem("Prompt (IA elige categoria directamente)", "prompt")
        idx = self._classify_method_combo.findData(self._classify_method)
        if idx >= 0:
            self._classify_method_combo.setCurrentIndex(idx)
        meth_l.addWidget(self._classify_method_combo, 1)
        meth_l.addStretch()
        ai_l.addLayout(meth_l)
        self._ai_opts.setVisible(self._mode in ("clip", "fusion", "ollama"))
        l.addWidget(self._ai_opts)

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
        self._on_mode_changed()
        self._update_warning()

    # -------- warning / size check --------

    def _update_warning(self):
        """Show a warning when the file count is large enough to be slow."""
        n = len(self.files)
        mode = self._mode_combo.currentData()
        warn_text = ""
        if mode == "ollama" and n > 10:
            est = n * 5  # rough: 5s per image via Ollama
            warn_text = (
                f"⚠ {n} files via Ollama may take a few minutes "
                f"(~{est}s). Consider Preview limit + Dry run."
            )
        elif mode in ("clip", "fusion") and n > 100:
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

    def _on_mode_changed(self):
        mode = self._mode_combo.currentData()
        is_ai = mode in ("clip", "fusion", "ollama")
        self._ai_opts.setVisible(is_ai)
        # Populate the model dropdown with mode-appropriate entries.
        self._populate_model_combo(mode)
        if mode == "ollama":
            self._model_combo.setEnabled(True)
            self._model_refresh_btn.setEnabled(True)
            self._refresh_backend_status()
        elif mode == "clip":
            self._model_combo.setEnabled(True)
            self._model_refresh_btn.setEnabled(True)
            self._refresh_backend_status()
        elif mode == "fusion":
            self._model_combo.setEnabled(True)
            self._model_refresh_btn.setEnabled(True)
            self._backend_status.setText(
                "Fusion runs LowLevel CV + CLIP. Pick the CLIP model below."
            )
        else:  # lowlevel
            self._model_combo.setEnabled(False)
            self._model_refresh_btn.setEnabled(False)
            self._backend_status.setText(
                "Low-Level CV analyzer (no AI model). Tags come from "
                "classical computer-vision heuristics + content detectors."
            )
        self._update_mode_desc()
        self._update_warning()
        self._apply_btn.setEnabled(self._has_preview)

    def _update_mode_desc(self):
        descs = {
            "lowlevel": "Classical CV: edge detection, texture, silhouette, HOG, Fourier",
            "clip": "OpenAI CLIP zero-shot: semantic understanding",
            "fusion": "CLIP semantic + LowLevel CV statistics - best of both",
            "ollama": "Local vision LLM via Ollama",
        }
        self._mode_desc.setText(descs.get(self._mode_combo.currentData(), ""))

    def _refresh_backend_status(self):
        """Probe CLIP or Ollama availability for the selected mode."""
        mode = self._mode_combo.currentData()
        if mode == "clip":
            try:
                from ..clip_client import get_engine
                engine = get_engine()
                clip_ok = bool(engine.available)
            except Exception as e:
                clip_ok = False
            if clip_ok:
                self._backend_status.setText(
                    "✓ CLIP engine available. "
                    "Will score each image against the full tag registry."
                )
                self._backend_status.setStyleSheet("color: #6fbf73;")
            else:
                self._backend_status.setText(
                    "✗ CLIP not loaded. Install CLIP or pick a different mode."
                )
                self._backend_status.setStyleSheet("color: #c95;")
        elif mode == "ollama":
            try:
                from ..ollama_client import OllamaClient
                cfg = s.load_settings()
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
            if ollama_ok:
                self._backend_status.setText(
                    "✓ Ollama server reachable. Pick a vision model below."
                )
                self._backend_status.setStyleSheet("color: #6fbf73;")
            else:
                self._backend_status.setText(
                    "✗ Ollama server not reachable. Start Ollama or pick a different mode."
                )
                self._backend_status.setStyleSheet("color: #c95;")
        elif mode == "fusion":
            try:
                from ..clip_client import get_engine
                engine = get_engine()
                clip_ok = bool(engine.available)
            except Exception:
                clip_ok = False
            if clip_ok:
                self._backend_status.setText(
                    "✓ CLIP available — Fusion will use CLIP + LowLevel CV."
                )
                self._backend_status.setStyleSheet("color: #6fbf73;")
            else:
                self._backend_status.setText(
                    "CLIP not loaded — Fusion will fall back to LowLevel CV only."
                )
                self._backend_status.setStyleSheet("color: #c95;")

    def _on_strategy_changed(self):
        strat = self._strat.currentData()
        is_tag = strat in TAG_BASED_STRATEGIES
        self._max_tags.setEnabled(is_tag)

    # -------- model dropdown --------

    def _populate_model_combo(self, mode: str):
        """Fill the model combo with entries appropriate for `mode`."""
        prev = self._model_combo.currentText() if self._model_combo.count() else ""
        self._model_combo.blockSignals(True)
        try:
            self._model_combo.clear()
            if mode == "clip" or mode == "fusion":
                models = _list_clip_models()
                # Use the configured active CLIP model by default.
                if not self._default_model:
                    try:
                        self._default_model = s.load_settings().get("clip_model")
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
            elif mode == "ollama":
                models = _list_ollama_models()
                if not self._default_model:
                    try:
                        self._default_model = s.load_settings().get("ollama_model")
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
        mode = self._mode_combo.currentData()
        self._populate_model_combo(mode)
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
        mode = self._mode_combo.currentData()
        cap = int(self._preview_limit.value())
        files_to_process = self.files[:cap]
        # Pull the explicit model selection (None means "use whatever's
        # configured") so the user can pick a specific model per-rename.
        model = self._selected_model()
        # Build Organize-style AI option overrides.
        organize_options = {}
        if mode in ("clip", "fusion", "ollama"):
            organize_options["ollama_nsfw_enabled"] = self._cb_nsfw.isChecked()
            organize_options["ollama_describe_enabled"] = self._cb_describe.isChecked()
            organize_options["ollama_classify_enabled"] = self._cb_classify.isChecked()
            organize_options["ollama_classify_method"] = (
                self._classify_method_combo.currentData() or "tags"
            )
        mode_label = mode + (f" ({model})" if model else " / (configured)")
        self._prog.setVisible(True)
        self._prog.setRange(0, len(files_to_process))
        self._prog.setValue(0)
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._apply_btn.setEnabled(False)
        self._backend_status.setText(
            f"Running preview with mode={mode_label} on {len(files_to_process)} files…"
        )
        self._preview_job = _PreviewJob(
            files=files_to_process,
            strategy=strat,
            mode=mode,
            category=self.category,
            max_tags=self._max_tags.value(),
            model=model,
            force_reprocess=self._force_reprocess,
            organize_options=organize_options,
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
        mode = self._mode_combo.currentData()
        model = self._selected_model()
        label = mode + (f" / {model}" if model else " / (configured)")
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