"""AI Models page: manage CLIP and Ollama backends."""
import shutil
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit,
    QGroupBox, QFormLayout, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QProgressBar, QPlainTextEdit, QWidget,
    QMessageBox,
)
from PySide6.QtCore import Qt, QTimer
from ..widgets import BTN_TEXT, BTN_KIND, BTN_DATA, setup_table_buttons, min_button_column_width, refresh_action_columns
from ... import settings as s
from ..workers import HealthCheckWorker, OllamaPullWorker

HAS_CLIP = False
HAS_TORCH = False
try:
    import torch as _
    HAS_TORCH = True
except Exception:
    pass
try:
    import clip as _
    HAS_CLIP = True
except Exception:
    pass


class AIModelsPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self._health_worker = None
        self._pull_worker = None
        self._build()
        self._load()

    def _build(self):
        l = QVBoxLayout(self)
        l.setContentsMargins(24, 24, 24, 24)

        t = QLabel("AI Models")
        t.setObjectName("title")
        l.addWidget(t)
        st = QLabel("Manage AI backends and models for classification.")
        st.setObjectName("subtitle")
        l.addWidget(st)

        bg = QGroupBox("Active Backend")
        bl = QFormLayout(bg)
        self._backend = QComboBox()
        self._backend.addItem("Low-Level CV (no ML needed)", "lowlevel")
        self._backend.addItem("CLIP / PyTorch", "clip")
        self._backend.addItem("Ollama (local vision LLM)", "ollama")
        self._backend.currentIndexChanged.connect(self._save_backend)
        bl.addRow("Primary mode:", self._backend)
        l.addWidget(bg)

        self._tabs = QTabWidget()
        self._build_clip_tab()
        self._build_ollama_tab()
        self._build_uncensored_tab()
        self._tabs.addTab(self._clip_tab, "CLIP / PyTorch")
        self._tabs.addTab(self._ollama_tab, "Ollama")
        self._tabs.addTab(self._uncensored_tab, "Uncensored (NSFW)")
        l.addWidget(self._tabs, 1)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(200)
        self._log.setFixedHeight(80)
        l.addWidget(self._log)

    def _build_clip_tab(self):
        self._clip_tab = QWidget()
        vl = QVBoxLayout(self._clip_tab)
        vl.addWidget(QLabel("CLIP Models"))
        vl.addWidget(QLabel("OpenAI CLIP for zero-shot classification.", objectName="statSmall"))

        self._clip_table = QTableWidget(0, 5)
        self._clip_table.setHorizontalHeaderLabels(["Model", "Size", "Status", "", ""])
        self._clip_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._clip_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._clip_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._clip_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._clip_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._clip_table.horizontalHeader().setMinimumSectionSize(100)
        self._clip_table.setAlternatingRowColors(True)
        vl.addWidget(self._clip_table)

        hb = QHBoxLayout()
        b = QPushButton("Install CLIP + PyTorch")
        b.setObjectName("ghost")
        b.clicked.connect(self._clip_install)
        hb.addWidget(b)
        b2 = QPushButton("Refresh")
        b2.setObjectName("ghost")
        b2.clicked.connect(self._clip_populate)
        hb.addWidget(b2)
        hb.addStretch()
        vl.addLayout(hb)

        def _clip_install_cb(r, c, d):
            self._clip_install()

        def _clip_use_cb(r, c, d):
            self._clip_activate(d)

        setup_table_buttons(self._clip_table, {"action": _clip_install_cb, "use": _clip_use_cb}, [3, 4])
        self._clip_populate()

    def _build_ollama_tab(self):
        self._ollama_tab = QWidget()
        vl = QVBoxLayout(self._ollama_tab)

        hu = QHBoxLayout()
        self._oll_url = QLineEdit("http://localhost:11434")
        b = QPushButton("Test Connection")
        b.setObjectName("primary")
        b.clicked.connect(self._oll_test)
        hu.addWidget(QLabel("Server:"))
        hu.addWidget(self._oll_url, 1)
        hu.addWidget(b)
        vl.addLayout(hu)

        self._oll_status = QLabel("Not tested")
        self._oll_status.setObjectName("statSmall")
        vl.addWidget(self._oll_status)

        self._oll_bar = QProgressBar()
        self._oll_bar.setVisible(False)
        self._oll_bar.setFixedHeight(6)
        self._oll_bar.setTextVisible(False)
        vl.addWidget(self._oll_bar)

        self._oll_table = QTableWidget(0, 4)
        self._oll_table.setHorizontalHeaderLabels(["Model", "Size", "Status", ""])
        self._oll_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._oll_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._oll_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._oll_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._oll_table.horizontalHeader().setMinimumSectionSize(100)
        self._oll_table.setAlternatingRowColors(True)
        vl.addWidget(self._oll_table)

        hp = QHBoxLayout()
        self._oll_combo = QComboBox()
        self._oll_combo.setEditable(True)
        self._oll_combo.addItems([
            "llava:7b", "llava:13b", "minicpm-v:8b",
            "llama3.2-vision:11b", "moondream:latest",
        ])
        br = QPushButton("Refresh List")
        br.setObjectName("ghost")
        br.clicked.connect(self._oll_refresh)
        bp = QPushButton("Pull Model")
        bp.setObjectName("success")
        bp.clicked.connect(self._oll_pull)
        bs = QPushButton("Stop")
        bs.setObjectName("danger")
        bs.setEnabled(False)
        bs.clicked.connect(self._oll_stop)
        self._oll_stop_btn = bs
        self._oll_pull_btn = bp
        hp.addWidget(self._oll_combo, 1)
        hp.addWidget(br)
        hp.addWidget(bp)
        hp.addWidget(bs)
        vl.addLayout(hp)

        self._oll_pull_prog = QProgressBar()
        self._oll_pull_prog.setVisible(False)
        self._oll_pull_status = QLabel("")
        self._oll_pull_status.setObjectName("statSmall")
        vl.addWidget(self._oll_pull_prog)
        vl.addWidget(self._oll_pull_status)

        def _oll_act_cb(r, c, tag):
            available = False
            try:
                from ...ollama_client import OllamaClient
                oc = OllamaClient(base_url=self._oll_url.text().strip(), model=tag, timeout=3)
                installed = oc.list_models()
                # list_models returns List[Dict] in the current client; older
                # versions returned List[str]. Accept both.
                installed_names = {
                    m.get("name") if isinstance(m, dict) else m
                    for m in installed
                }
                available = tag in installed_names
            except Exception:
                pass
            if available:
                self._oll_activate(tag)
            else:
                self._oll_pull_model(tag)

        setup_table_buttons(self._oll_table, {"action": _oll_act_cb}, [3])
        self._oll_populate()
        QTimer.singleShot(500, self._oll_test)

    def _build_uncensored_tab(self):
        self._uncensored_tab = QWidget()
        vl = QVBoxLayout(self._uncensored_tab)

        vl.addWidget(QLabel("Uncensored Vision Models for NSFW Content"))
        vl.addWidget(QLabel(
            "These models are recommended for describing/analysing NSFW content. "
            "Stock models (LLaVA, Llama 3.2 Vision) refuse or sanitize NSFW content. "
            "The pipeline now auto-skips AI description when NSFW is detected, but "
            "an uncensored model will still give you better categorization.",
            objectName="statSmall"))

        # NSFW behavior settings
        bg = QGroupBox("NSFW Behavior")
        bl = QFormLayout(bg)
        from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox
        self._nsfw_skip = QCheckBox("Skip AI description when NSFW detected (use CV-only tags)")
        bl.addRow(self._nsfw_skip)
        self._nsfw_thresh = QDoubleSpinBox()
        self._nsfw_thresh.setRange(0.0, 1.0)
        self._nsfw_thresh.setSingleStep(0.05)
        bl.addRow("NSFW detection threshold:", self._nsfw_thresh)
        vl.addWidget(bg)

        try:
            from ...ollama_client import get_recommended_uncensored_models
            uncensored_models = get_recommended_uncensored_models()
        except Exception as e:
            uncensored_models = []
            vl.addWidget(QLabel(f"Could not load recommendations: {e}"))

        if uncensored_models:
            vl.addWidget(QLabel("Recommended Uncensored Vision Models:", objectName="sectionLabel"))
            self._unc_table = QTableWidget(0, 5)
            self._unc_table.setHorizontalHeaderLabels(
                ["Model", "Size", "Speed", "Notes", ""])
            self._unc_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            self._unc_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            self._unc_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            self._unc_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
            self._unc_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
            self._unc_table.horizontalHeader().setMinimumSectionSize(100)
            self._unc_table.setAlternatingRowColors(True)
            self._unc_table.setRowCount(len(uncensored_models))
            for i, m in enumerate(uncensored_models):
                self._unc_table.setItem(i, 0, QTableWidgetItem(m["name"]))
                self._unc_table.setItem(i, 1, QTableWidgetItem(f"{m['size_gb']:.1f} GB"))
                self._unc_table.setItem(i, 2, QTableWidgetItem(m["speed"]))
                self._unc_table.setItem(i, 3, QTableWidgetItem(m["notes"]))
                act = QTableWidgetItem()
                act.setData(BTN_TEXT, "Pull")
                act.setData(BTN_KIND, "action")
                act.setData(BTN_DATA, m["tag"])
                self._unc_table.setItem(i, 4, act)
            refresh_action_columns(self._unc_table)
            vl.addWidget(self._unc_table)

            def _unc_act_cb(r, c, tag):
                self._oll_combo.setEditText(tag)
                self._tabs.setCurrentWidget(self._ollama_tab)
                self._oll_pull()

            setup_table_buttons(self._unc_table, {"action": _unc_act_cb}, [4])
        else:
            vl.addWidget(QLabel("No uncensored models available in this build."))

        vl.addStretch()

        # Save behavior settings
        def _save_nsfw_settings():
            cfg = s.load_settings()
            cfg["nsfw_skip_describe"] = self._nsfw_skip.isChecked()
            cfg["nsfw_threshold"] = self._nsfw_thresh.value()
            s.save_settings(cfg)
        self._nsfw_skip.stateChanged.connect(lambda _: _save_nsfw_settings())
        self._nsfw_thresh.valueChanged.connect(lambda _: _save_nsfw_settings())

        # Load current values
        cfg = s.load_settings()
        self._nsfw_skip.setChecked(cfg.get("nsfw_skip_describe", True))
        self._nsfw_thresh.setValue(cfg.get("nsfw_threshold", 0.5))

    def _save_backend(self):
        cfg = s.load_settings()
        cfg["organize_mode"] = self._backend.currentData()
        s.save_settings(cfg)

    def _clip_populate(self):
        """Detect installed CLIP models and populate the table.

        Reads `clip.available_models()` so models whose weights are
        already downloaded appear as `ready` automatically. The
        "Use" button activates the chosen model (writes to settings).
        """
        self._clip_table.setRowCount(0)
        from ...clip_client import available_clip_models
        installed = set(available_clip_models())
        # Canonical model list (newest first; user wants ViT-L/14).
        catalog = [
            ("ViT-B/32",      "~150 MB", "Fast, lowest accuracy"),
            ("ViT-B/16",      "~150 MB", "Slightly better than B/32"),
            ("ViT-L/14",      "~890 MB", "Recommended: highest accuracy"),
            ("ViT-L/14@336px","~890 MB", "L/14 at 336px resolution"),
            ("RN50",          "~150 MB", "ResNet 50"),
            ("RN101",         "~150 MB", "ResNet 101"),
            ("RN50x4",        "~600 MB", "ResNet 50x4"),
            ("RN50x16",       "~1.2 GB", "ResNet 50x16"),
            ("RN50x64",       "~2.0 GB", "ResNet 50x64"),
        ]
        cfg = s.load_settings()
        active = cfg.get("clip_model", "ViT-B/32")
        deps_ok = HAS_CLIP and HAS_TORCH

        rows = []
        if deps_ok and installed:
            for nm in catalog:
                if nm[0] in installed:
                    rows.append((nm[0], nm[1], "ready", True))
        elif deps_ok:
            # No models installed yet - show B/32 as the "needs install" starter.
            rows.append(("ViT-B/32", "~150 MB", "not installed", False))
        else:
            rows.append(("ViT-B/32", "~150 MB", "missing torch/clip", False))

        self._clip_table.setRowCount(len(rows))
        for i, (nm, sz, st, ready) in enumerate(rows):
            self._clip_table.setItem(i, 0, QTableWidgetItem(nm))
            self._clip_table.setItem(i, 1, QTableWidgetItem(sz))
            self._clip_table.setItem(i, 2, QTableWidgetItem(st))
            # Column 3: install action
            install = QTableWidgetItem()
            install.setData(BTN_TEXT, "Installed" if ready else "Install deps")
            install.setData(BTN_KIND, "action")
            if not ready:
                install.setData(BTN_DATA, "clip")
            self._clip_table.setItem(i, 3, install)
            if ready:
                install.setFlags(install.flags() & ~Qt.ItemIsEnabled)
            # Column 4: use this model
            use = QTableWidgetItem()
            is_active = nm == active
            use.setData(BTN_TEXT, "Active" if is_active else "Use")
            use.setData(BTN_KIND, "use")
            use.setData(BTN_DATA, nm)
            self._clip_table.setItem(i, 4, use)
            if not ready or is_active:
                use.setFlags(use.flags() & ~Qt.ItemIsEnabled)
        refresh_action_columns(self._clip_table)

    def _clip_activate(self, model_name: str):
        """Set `model_name` as the active CLIP model in settings."""
        if not model_name:
            return
        # Warn the user if they picked a heavyweight model.
        heavy = {
            "RN50x16": "~1.2 GB - slow on CPU",
            "RN50x64": "~2.0 GB - very slow on CPU",
            "ViT-L/14@336px": "~890 MB - higher res, slower than ViT-L/14",
        }
        cfg = s.load_settings()
        if cfg.get("clip_model") == model_name:
            return
        if model_name in heavy:
            from PySide6.QtWidgets import QMessageBox
            r = QMessageBox.question(
                self,
                "Heavy CLIP model selected",
                f"{model_name} is {heavy[model_name]}.\n\n"
                "Each image will take 5-15 seconds to classify on CPU.\n"
                "Consider ViT-B/32 (fast) or ViT-L/14 (recommended) instead.\n\n"
                "Continue anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if r != QMessageBox.Yes:
                return
        cfg["clip_model"] = model_name
        s.save_settings(cfg)
        self._log.appendPlainText(
            f"[CLIP] Active model set to {model_name}. "
            f"Next organize pass will use it."
        )
        self._clip_populate()

    def _clip_install(self):
        from ..workers import DepWorker
        if hasattr(self, "_clip_worker") and self._clip_worker and self._clip_worker.isRunning():
            self._log.appendPlainText("[pip] Already installing, please wait...")
            return
        if shutil.which("git") is None:
            self._log.appendPlainText("[ERR] Git is required to install CLIP. Install git first.")
            return
        self._log.appendPlainText("[pip] Installing CLIP + PyTorch (this may take several minutes)...")
        self._clip_worker = DepWorker(["torch", "torchvision", "ftfy", "regex", "clip"])
        self._clip_worker.log_line.connect(self._log.appendPlainText)
        self._clip_worker.module_done.connect(lambda mod, ok: self._log.appendPlainText(
            f"  [{'OK' if ok else 'FAILED'}] {mod}"
        ))
        self._clip_worker.all_done.connect(self._clip_populate)
        self._clip_worker.start()

    def _oll_populate(self):
        self._oll_table.setRowCount(0)
        models_data = [
            ("llava:7b", "4.5 GB", "LLaVA 7B"),
            ("llava:13b", "8.0 GB", "LLaVA 13B"),
            ("minicpm-v:8b", "5.0 GB", "MiniCPM-V 8B"),
            ("llama3.2-vision:11b", "8.0 GB", "Llama 3.2 Vision"),
            ("moondream:latest", "0.9 GB", "Moondream 2B"),
        ]
        cfg = s.load_settings()
        active = cfg.get("ollama_model", "llava:7b")
        self._oll_table.setRowCount(len(models_data))
        for i, (tag, sz, nm) in enumerate(models_data):
            self._oll_table.setItem(i, 0, QTableWidgetItem(f"{nm} ({tag})"))
            self._oll_table.setItem(i, 1, QTableWidgetItem(sz))
            available = False
            try:
                from ...ollama_client import OllamaClient
                oc = OllamaClient(base_url=self._oll_url.text().strip(), model=tag, timeout=3)
                installed = oc.list_models()
                installed_names = {
                    m.get("name") if isinstance(m, dict) else m
                    for m in installed
                }
                available = tag in installed_names
            except Exception:
                pass
            is_active = tag == active
            st = "active" if is_active else ("available" if available else "not pulled")
            self._oll_table.setItem(i, 2, QTableWidgetItem(st))
            act = QTableWidgetItem()
            act.setData(BTN_TEXT, "Active" if is_active else ("Activate" if available else "Pull"))
            act.setData(BTN_KIND, "action")
            act.setData(BTN_DATA, tag)
            self._oll_table.setItem(i, 3, act)
            if is_active:
                act.setFlags(act.flags() & ~Qt.ItemIsEnabled)
        refresh_action_columns(self._oll_table)

    def _oll_activate(self, tag):
        cfg = s.load_settings()
        cfg["ollama_model"] = tag
        s.save_settings(cfg)
        self._oll_populate()
        self._log.appendPlainText(f"[OK] Active model: {tag}")

    def _oll_pull_model(self, tag):
        self._oll_combo.setEditText(tag)
        self._oll_pull()

    def _oll_refresh(self):
        import requests as req
        try:
            resp = req.get(f"{self._oll_url.text().strip().rstrip('/')}/api/tags", timeout=10)
            if resp.status_code == 200:
                cur = self._oll_combo.currentText()
                self._oll_combo.clear()
                for m in resp.json().get("models", []):
                    self._oll_combo.addItem(m.get("name", ""))
                idx = self._oll_combo.findText(cur)
                if idx >= 0:
                    self._oll_combo.setCurrentIndex(idx)
                self._log.appendPlainText(f"[OK] {len(resp.json().get('models', []))} models found")
                self._oll_populate()
        except Exception as e:
            self._log.appendPlainText(f"[ERR] {e}")

    def _oll_test(self):
        self._oll_status.setText("Testing...")
        self._oll_bar.setVisible(True)
        self._oll_bar.setValue(0)
        if self._health_worker and self._health_worker.isRunning():
            try:
                self._health_worker.wait(500)
            except Exception:
                pass
        self._health_worker = HealthCheckWorker(
            self._oll_url.text().strip(),
            self._oll_combo.currentText().strip(),
        )
        self._health_worker.result_ready.connect(self._oll_on_health)
        self._health_worker.start()

    def _oll_on_health(self, result):
        if result.get("connected"):
            if result.get("model_available"):
                self._oll_status.setText(f"Connected ({result['model_count']} models)")
                self._oll_status.setObjectName("statusOk")
                self._oll_bar.setValue(100)
            else:
                self._oll_status.setText("Connected (model missing)")
                self._oll_status.setObjectName("statusWarn")
                self._oll_bar.setValue(50)
        else:
            self._oll_status.setText(f"Failed: {result.get('error', '')}")
            self._oll_status.setObjectName("statusErr")
            self._oll_bar.setValue(0)

    def _oll_pull(self):
        # Don't start a second pull if one is already running
        if self._pull_worker and self._pull_worker.isRunning():
            self._log.appendPlainText(f"[pull] Already downloading, please wait...")
            return
        model = self._oll_combo.currentText().strip()
        if not model:
            return
        self._log.appendPlainText(f"[pull] Downloading {model}...")
        self._oll_pull_btn.setEnabled(False)
        self._oll_stop_btn.setEnabled(True)
        self._oll_pull_prog.setVisible(True)
        self._oll_pull_prog.setValue(0)
        self._oll_pull_status.setText("Starting...")

        self._pull_worker = OllamaPullWorker(self._oll_url.text().strip(), model, 600)
        self._pull_worker.progress.connect(lambda c, t, st: (
            self._oll_pull_prog.setMaximum(max(t, 1)),
            self._oll_pull_prog.setValue(c),
            self._oll_pull_status.setText(f"{st} ({c}/{t} bytes)" if t else st),
        ))
        self._pull_worker.finished_ok.connect(self._oll_pull_done)
        self._pull_worker.failed.connect(self._oll_pull_fail)
        self._pull_worker.log.connect(self._log.appendPlainText)
        self._pull_worker.start()

    def _oll_pull_done(self):
        self._oll_pull_btn.setEnabled(True)
        self._oll_stop_btn.setEnabled(False)
        self._oll_pull_prog.setVisible(False)
        self._oll_pull_status.setText("")
        self._log.appendPlainText("[OK] Pulled")
        self._oll_refresh()

    def _oll_pull_fail(self, msg):
        self._oll_pull_btn.setEnabled(True)
        self._oll_stop_btn.setEnabled(False)
        self._oll_pull_prog.setVisible(False)
        self._oll_pull_status.setText("")
        self._log.appendPlainText(f"[ERR] {msg}")

    def _oll_stop(self):
        if self._pull_worker:
            self._pull_worker.cancel()

    def _load(self):
        cfg = s.load_settings()
        mode = cfg.get("organize_mode", "lowlevel")
        for i in range(self._backend.count()):
            if self._backend.itemData(i) == mode:
                self._backend.setCurrentIndex(i)
                break
        self._oll_url.setText(cfg.get("ollama_url", "http://localhost:11434"))
        model = cfg.get("ollama_model", "llava:7b")
        idx = self._oll_combo.findText(model)
        if idx >= 0:
            self._oll_combo.setCurrentIndex(idx)
        else:
            self._oll_combo.setEditText(model)
