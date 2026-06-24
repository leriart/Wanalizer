"""Organize page: run classification pipeline."""
import os
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox,
    QProgressBar, QPlainTextEdit, QGroupBox, QFrame, QGridLayout,
    QWidget, QMessageBox,
)
from PySide6.QtCore import Qt, QTimer
from ... import settings as s
from ... import formats as f
from ... import categories as c
from ...rename import RENAME_STRATEGIES
from ...ollama_client import OLLAMA_VISION_MODELS
from ..workers import OrganizeWorker


class OrganizePage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.worker = None
        self._build()

    def _build(self):
        l = QVBoxLayout(self)
        l.setContentsMargins(24, 24, 24, 24)

        t = QLabel("Organize Wallpapers")
        t.setObjectName("title")
        l.addWidget(t)
        st = QLabel("Classify and sort images into category folders.")
        st.setObjectName("subtitle")
        l.addWidget(st)

        # Source folder
        h = QHBoxLayout()
        _init_cfg = s.load_settings()
        _init_src = _init_cfg.get("source_dir") or s.PROJECT_DIR
        self._dir_edit = QLineEdit(_init_src)
        self._dir_edit.setMinimumWidth(200)
        b = QPushButton("Browse...")
        b.clicked.connect(lambda: self._browse("source"))
        h.addWidget(QLabel("Source:"))
        h.addWidget(self._dir_edit, 1)
        h.addWidget(b)
        l.addLayout(h)

        # Destination folder
        h2 = QHBoxLayout()
        self._dest_edit = QLineEdit(s.resolve_dest_dir(s.load_settings()))
        self._dest_edit.setMinimumWidth(200)
        b2 = QPushButton("Browse...")
        b2.clicked.connect(lambda: self._browse("dest"))
        h2.addWidget(QLabel("Destination:"))
        h2.addWidget(self._dest_edit, 1)
        h2.addWidget(b2)
        l.addLayout(h2)

        # Mode
        mg = QGroupBox("Analysis Mode")
        ml = QHBoxLayout(mg)
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Low-Level CV (edges, textures, shapes, HOG)", "lowlevel")
        self._mode_combo.addItem("CLIP (vision-language model)", "clip")
        self._mode_combo.addItem("CLIP + LowLevel (fusion, recommended)", "fusion")
        self._mode_combo.addItem("Ollama (local vision LLM)", "ollama")
        self._mode_desc = QLabel("")
        self._mode_desc.setObjectName("statSmall")
        ml.addWidget(QLabel("Mode:"))
        ml.addWidget(self._mode_combo, 1)
        ml.addWidget(self._mode_desc, 2)
        self._mode_combo.currentIndexChanged.connect(self._update_desc)
        l.addWidget(mg)

        # Options
        g = QGroupBox("Options")
        gl = QGridLayout(g)
        self.cb_dry = QCheckBox("Dry run (preview only)")
        self.cb_dedupe = QCheckBox("Find and remove duplicates")
        self.cb_dedupe.setChecked(True)
        self.cb_full = QCheckBox("Full reset (flatten all)")
        cfg = s.load_settings()
        self.spin_parallel = QSpinBox()
        self.spin_parallel.setRange(1, 16)
        self.spin_parallel.setValue(int(cfg.get("parallel", 4) or 4))
        self.spin_parallel.setPrefix("Workers: ")
        self.spin_quality = QDoubleSpinBox()
        self.spin_quality.setRange(0, 1)
        self.spin_quality.setSingleStep(0.05)
        self.spin_quality.setValue(float(cfg.get("quality_min", 0.0) or 0.0))
        self.spin_quality.setPrefix("Min quality: ")
        gl.addWidget(self.cb_dry, 0, 0)
        gl.addWidget(self.cb_dedupe, 0, 1)
        gl.addWidget(self.cb_full, 0, 2)
        gl.addWidget(self.spin_parallel, 1, 0)
        gl.addWidget(self.spin_quality, 1, 1)
        l.addWidget(g)

        # AI Options
        self._ai_opts = QGroupBox("AI Options")
        ai_l = QVBoxLayout(self._ai_opts)
        self.cb_nsfw = QCheckBox("AI NSFW detection")
        self.cb_nsfw.setChecked(True)
        ai_l.addWidget(self.cb_nsfw)
        ar = QHBoxLayout()
        self.cb_describe = QCheckBox("AI description -> tags")
        self.cb_classify = QCheckBox("AI direct classification")
        ar.addWidget(self.cb_describe)
        ar.addWidget(self.cb_classify)
        ar.addStretch()
        ai_l.addLayout(ar)
        # Classification method toggle (tags vs prompt)
        meth_l = QHBoxLayout()
        meth_l.addWidget(QLabel("Classify by:"))
        self._classify_method = QComboBox()
        self._classify_method.addItem("Tags (IA sugiere tags, sistema elige categoria)", "tags")
        self._classify_method.addItem("Prompt (IA elige categoria directamente)", "prompt")
        meth_l.addWidget(self._classify_method, 1)
        meth_l.addStretch()
        ai_l.addLayout(meth_l)
        # Ollama model selector (only visible when mode=Ollama)
        self._ollama_row = QWidget()
        ollama_l = QHBoxLayout(self._ollama_row)
        ollama_l.setContentsMargins(0, 0, 0, 0)
        ollama_l.addWidget(QLabel("Ollama model:"))
        self._ollama_model = QComboBox()
        self._ollama_model.setEditable(True)
        self._populate_ollama_models()
        ollama_l.addWidget(self._ollama_model, 1)
        b_refresh = QPushButton("Refresh")
        b_refresh.setObjectName("ghost")
        b_refresh.clicked.connect(self._populate_ollama_models)
        ollama_l.addWidget(b_refresh)
        ai_l.addWidget(self._ollama_row)
        self._ai_opts.setVisible(False)
        self._mode_combo.currentTextChanged.connect(self._on_mode_change)
        l.addWidget(self._ai_opts)

        # Rename Options
        self._rename_opts = QGroupBox("Rename Options (when moving files)")
        rn_l = QGridLayout(self._rename_opts)
        rn_l.addWidget(QLabel("Strategy:"), 0, 0)
        self._rename_strat = QComboBox()
        self._rename_strat.addItem("No rename (keep original names)", "none")
        for key, label, desc in RENAME_STRATEGIES:
            self._rename_strat.addItem(f"{label}  —  {desc}", key)
        rn_l.addWidget(self._rename_strat, 0, 1)
        self.cb_rename_prefix = QCheckBox("Use category as prefix (for category strategy)")
        self.cb_rename_prefix.setChecked(True)
        rn_l.addWidget(self.cb_rename_prefix, 1, 0, 1, 2)
        rn_l.addWidget(QLabel("Max tags in filename:"), 2, 0)
        self._max_tags = QSpinBox()
        self._max_tags.setRange(1, 8)
        self._max_tags.setValue(3)
        self._max_tags.setToolTip("Only applies to tag-based rename strategies")
        rn_l.addWidget(self._max_tags, 2, 1)
        l.addWidget(self._rename_opts)

        # Buttons
        hb = QHBoxLayout()
        self.btn_run = QPushButton("Organize Now")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(lambda: self._start(False))
        self.btn_dry = QPushButton("Dry Run")
        self.btn_dry.setObjectName("ghost")
        self.btn_dry.clicked.connect(lambda: self._start(True))
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        hb.addWidget(self.btn_run)
        hb.addWidget(self.btn_dry)
        hb.addWidget(self.btn_stop)
        hb.addStretch()
        l.addLayout(hb)

        # Progress
        self.prog = QProgressBar()
        self.prog_label = QLabel("Ready")
        self.prog_label.setObjectName("statSmall")
        l.addWidget(self.prog)
        l.addWidget(self.prog_label)

        # Log
        l.addWidget(QLabel("Output"))
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(1000)
        l.addWidget(self._log, 1)

        self._update_desc()
        self._load_settings()

    def _load_settings(self):
        """Load saved settings into the UI controls."""
        cfg = s.load_settings()
        method = cfg.get("ollama_classify_method", "tags")
        idx = self._classify_method.findData(method)
        if idx >= 0:
            self._classify_method.setCurrentIndex(idx)
        self.cb_classify.setChecked(cfg.get("ollama_classify_enabled", False))
        self.cb_describe.setChecked(cfg.get("ollama_describe_enabled", False))
        self.cb_nsfw.setChecked(cfg.get("ollama_nsfw_enabled", True))

    def _browse(self, which):
        d = QFileDialog.getExistingDirectory(
            self,
            "Choose folder",
            self._dir_edit.text() if which == "source" else self._dest_edit.text(),
        )
        if d:
            if which == "source":
                self._dir_edit.setText(d)
            else:
                self._dest_edit.setText(d)

    def _update_desc(self):
        descs = {
            "lowlevel": "Classical CV: edge detection, texture, silhouette, HOG, Fourier",
            "clip": "OpenAI CLIP zero-shot: semantic understanding",
            "fusion": "CLIP semantic + LowLevel CV statistics - best of both",
            "ollama": "Local vision LLM via Ollama",
        }
        self._mode_desc.setText(descs.get(self._mode_combo.currentData(), ""))

    def _on_mode_change(self, text):
        """Show AI options and Ollama model selector when relevant."""
        mode = self._mode_combo.currentData()
        is_ai = mode in ("clip", "ollama")
        self._ai_opts.setVisible(is_ai)
        self._ollama_row.setVisible(mode == "ollama")
        # Load current classify method
        if is_ai:
            cfg = s.load_settings()
            method = cfg.get("ollama_classify_method", "tags")
            idx = self._classify_method.findData(method)
            if idx >= 0:
                self._classify_method.setCurrentIndex(idx)

    def _populate_ollama_models(self):
        """Populate the Ollama model combo: known models + locally available."""
        cfg = s.load_settings()
        current = self._ollama_model.currentText() or cfg.get("ollama_model", "llava:7b")
        self._ollama_model.clear()
        # Add all known models (curated list)
        for key, meta in OLLAMA_VISION_MODELS.items():
            self._ollama_model.addItem(f"{meta['name']}  ({meta['model_tag']})", meta['model_tag'])
        # Add locally available models
        try:
            from ...ollama_client import OllamaClient
            url = cfg.get("ollama_url", "http://localhost:11434")
            oc = OllamaClient(base_url=url, timeout=3)
            local = oc.list_models()
            for m in local:
                # Avoid duplicates
                found = False
                for i in range(self._ollama_model.count()):
                    if self._ollama_model.itemData(i) == m:
                        found = True
                        break
                if not found:
                    self._ollama_model.addItem(f"(local) {m}", m)
        except Exception:
            pass
        # Restore current selection
        idx = self._ollama_model.findData(current)
        if idx >= 0:
            self._ollama_model.setCurrentIndex(idx)
        else:
            self._ollama_model.setEditText(current)

    def _start(self, dry):
        self._set_running(True)
        src = self._dir_edit.text().strip()
        dest = self._dest_edit.text().strip()
        mode = self._mode_combo.currentData()

        if not os.path.isdir(src):
            QMessageBox.warning(self, "Error", "Source folder not found")
            self._set_running(False)
            return

        os.makedirs(dest, exist_ok=True)
        cfg = s.load_settings()
        cfg["organize_mode"] = mode
        cfg["dest_dir"] = dest
        cfg["source_dir"] = src
        cfg["parallel"] = self.spin_parallel.value()
        cfg["quality_min"] = self.spin_quality.value()
        cfg["ollama_nsfw_enabled"] = self.cb_nsfw.isChecked()
        cfg["ollama_describe_enabled"] = self.cb_describe.isChecked()
        cfg["ollama_classify_enabled"] = self.cb_classify.isChecked()
        cfg["ollama_classify_method"] = self._classify_method.currentData() or "tags"
        # Save selected Ollama model (only if mode=ollama)
        if mode == "ollama":
            sel_model = self._ollama_model.currentData() or self._ollama_model.currentText()
            if sel_model:
                cfg["ollama_model"] = sel_model
        s.save_settings(cfg)

        c.CATEGORIES_DIR = dest
        c.discover_categories(dest)

        self.prog.setValue(0)
        self.prog_label.setText("Starting...")
        self._log.clear()
        self.main.append_log(f"Mode: {mode}  Source: {src}  Destination: {dest}")

        if self.cb_full.isChecked():
            self._log.appendPlainText("Full reset will run before classification.")

        self.worker = OrganizeWorker(
            mode=mode,
            dry=dry or self.cb_dry.isChecked(),
            dedupe=self.cb_dedupe.isChecked(),
            parallel=self.spin_parallel.value(),
            quality_min=self.spin_quality.value(),
            source_dir=src,
            cats_dir=dest,
            rename_strategy=self._rename_strat.currentData() or "none",
            rename_category_prefix=self.cb_rename_prefix.isChecked(),
            rename_max_tags=self._max_tags.value(),
            full_reset=self.cb_full.isChecked(),
        )
        self.worker.progress.connect(self._prog)
        self.worker.log.connect(self._log.appendPlainText)
        self.worker.finished_ok.connect(lambda _: self._done())
        self.worker.failed.connect(self._fail)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.cancel()

    def _set_running(self, r):
        self.btn_run.setEnabled(not r)
        self.btn_dry.setEnabled(not r)
        self.btn_stop.setEnabled(r)

    def _prog(self, stage, cur, total, fname, info):
        if stage == "start":
            self.prog.setMaximum(total or 1)
            self.prog.setValue(0)
            if info:
                self._log.appendPlainText(f"[start] {info}")
        elif stage == "progress":
            self.prog.setValue(cur)
            # Show progress in label AND log so user sees activity
            label_text = f"{fname} -> {info}" if info else fname
            self.prog_label.setText(label_text)
            if info == "analyzing":
                # Only log "analyzing" once per file
                self._log.appendPlainText(f"  [{cur}/{total}] {fname}...")
            elif info and info not in ("analyzing",):
                # Final classification result
                self._log.appendPlainText(f"  [{cur}/{total}] {fname} -> {info}")
        elif stage == "done":
            self.prog_label.setText(f"Complete: {cur} files")
            self._log.appendPlainText(f"[done] {cur} files processed")

    def _done(self):
        self._set_running(False)
        QMessageBox.information(self, "Done", "Organization complete.")

    def _fail(self, msg):
        self._set_running(False)
        self._log.appendPlainText(f"[ERR] {msg}")
        QMessageBox.critical(self, "Error", msg[:200])
