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
    RENAME_STRATEGIES, TAG_BASED_STRATEGIES, AI_TAG_BACKENDS,
    ai_compute_renames, apply_renames,
)


class _PreviewJob(QThread):
    """Compute preview pairs in the background so the dialog stays responsive.

    Honors an internal ``cancel`` flag so the user can stop a slow
    preview without closing the dialog.
    """
    progress = Signal(int, int, str)
    finished_ok = Signal(list, list)  # pairs, log_lines
    failed = Signal(str)

    def __init__(self, files, strategy, backend, category, max_tags):
        super().__init__()
        self.files = list(files)
        self.strategy = strategy
        self.backend = backend
        self.category = category
        self.max_tags = max_tags
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            log: List[str] = []
            def _cb(stage, *args):
                if self._cancel:
                    raise InterruptedError("cancelled")
                if stage == "progress":
                    cur, total = args
                    self.progress.emit(cur, total, f"Tagged {cur}/{total}")
                else:
                    msg = args[0] if args else stage
                    log.append(f"[{stage}] {msg}")
            pairs = ai_compute_renames(
                self.files,
                strategy=self.strategy,
                backend=self.backend,
                category=self.category,
                max_tags=self.max_tags,
                progress_cb=_cb,
            )
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
                 default_backend: str = "heuristic",
                 default_strategy: str = "category_tags",
                 max_tags: int = 3,
                 preview_limit: int = 50,
                 parent=None):
        super().__init__(parent)
        self.files = list(files)
        self.category = category
        self._pairs: List[Tuple[str, str]] = []
        self._preview_job: Optional[_PreviewJob] = None
        self._has_preview = False
        self.setWindowTitle(f"AI Rename — {len(self.files)} files")
        self.setMinimumSize(840, 640)
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
            "auto":      "Auto (Ollama → CLIP → Heuristic)",
            "heuristic": "Heuristic (no AI, fast CV only)",
            "clip":      "CLIP (local OpenAI model)",
            "ollama":    "Ollama (local vision LLM)",
        }
        for k in AI_TAG_BACKENDS:
            self._backend.addItem(labels.get(k, k), k)
        # Default to "heuristic" — it's the safest option and lets the
        # user upgrade to a heavier backend when they're ready.
        idx = self._backend.findData(default_backend)
        if idx < 0:
            idx = self._backend.findData("heuristic")
        if idx >= 0:
            self._backend.setCurrentIndex(idx)
        self._backend.currentIndexChanged.connect(self._on_backend_changed)
        bg_layout.addRow("Backend:", self._backend)

        self._backend_status = QLabel("")
        self._backend_status.setObjectName("statSmall")
        self._backend_status.setStyleSheet("color: #888;")
        self._backend_status.setWordWrap(True)
        bg_layout.addRow("Status:", self._backend_status)
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
        if b == "ollama":
            self._backend_status.setText(
                "Ollama runs locally. The configured model "
                "(see AI Models page) will be queried for each file."
            )
        elif b == "clip":
            self._backend_status.setText(
                "CLIP encodes the image and scores it against a curated "
                "tag vocabulary. Best balance of speed and accuracy."
            )
        elif b == "heuristic":
            self._backend_status.setText(
                "Pure CV heuristics — no AI. Tags come from colour / "
                "edge / palette analysis. Fastest option."
            )
        else:
            self._backend_status.setText(
                "Auto: tries Ollama first, falls back to CLIP, then "
                "heuristics. Each file uses whichever backend responds."
            )
        self._update_warning()
        self._apply_btn.setEnabled(self._has_preview)

    def _on_strategy_changed(self):
        strat = self._strat.currentData()
        is_tag = strat in TAG_BASED_STRATEGIES
        self._max_tags.setEnabled(is_tag)

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
        self._prog.setVisible(True)
        self._prog.setRange(0, len(files_to_process))
        self._prog.setValue(0)
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._apply_btn.setEnabled(False)
        self._backend_status.setText(
            f"Running preview with backend={backend} on {len(files_to_process)} files…"
        )
        self._preview_job = _PreviewJob(
            files=files_to_process,
            strategy=strat,
            backend=backend,
            category=self.category,
            max_tags=self._max_tags.value(),
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
        status = f"Preview ready — backend={backend} • {len(pairs)} pair(s)"
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