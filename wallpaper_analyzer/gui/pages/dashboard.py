"""Dashboard page: overview, mode selector, AI health."""
import os
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QGroupBox, QProgressBar, QPlainTextEdit, QWidget,
)
from PySide6.QtCore import Qt, QTimer
from ... import settings as s
from ... import categories as c
from ..workers import HealthCheckWorker


class DashboardPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self._health_worker = None
        self._build()
        QTimer.singleShot(200, self.refresh)

    def _build(self):
        l = QVBoxLayout(self)
        l.setContentsMargins(24, 24, 24, 24)

        t = QLabel("Dashboard")
        t.setObjectName("title")
        l.addWidget(t)
        st = QLabel("Overview of your wallpaper collection")
        st.setObjectName("subtitle")
        l.addWidget(st)

        sc = QHBoxLayout()
        self._cards = {}
        for key, label in [
            ("total", "Total Files"),
            ("cats", "Categories"),
            ("dupes", "Duplicates"),
            ("nsfw", "NSFW Flagged"),
        ]:
            card = QFrame()
            card.setObjectName("card")
            vl = QVBoxLayout(card)
            vl.setSpacing(4)
            v = QLabel("--")
            v.setObjectName("statBig")
            lb = QLabel(label)
            lb.setObjectName("statSmall")
            vl.addWidget(v)
            vl.addWidget(lb)
            sc.addWidget(card, 1)
            self._cards[key] = v
        l.addLayout(sc)

        g = QGroupBox("Analysis Mode")
        gl = QVBoxLayout(g)
        self._mode_group = []
        for val, desc in [
            ("lowlevel", "Low-Level CV  --  Classical algorithms (edges, textures, HOG, Fourier)"),
            ("clip", "CLIP  --  OpenAI vision-language model (requires PyTorch)"),
            ("ollama", "Ollama  --  Local vision LLMs (LLaVA, MiniCPM-V, Moondream)"),
        ]:
            from PySide6.QtWidgets import QRadioButton
            rb = QRadioButton(desc)
            rb.setObjectName(val)
            gl.addWidget(rb)
            self._mode_group.append(rb)
        gl.addStretch()
        l.addWidget(g)

        ai_g = QGroupBox("AI Services")
        ai_l = QVBoxLayout(ai_g)
        ai_row = QHBoxLayout()
        self._clip_status = QLabel("CLIP: checking...")
        self._clip_status.setObjectName("statSmall")
        self._ollama_status = QLabel("Ollama: checking...")
        self._ollama_status.setObjectName("statSmall")
        self._ollama_url = QLabel("")
        self._ollama_url.setObjectName("statSmall")
        ai_row.addWidget(self._clip_status)
        ai_row.addStretch()
        ai_row.addWidget(self._ollama_status)
        ai_row.addWidget(self._ollama_url)
        ai_l.addLayout(ai_row)
        self._ollama_bar = QProgressBar()
        self._ollama_bar.setVisible(False)
        self._ollama_bar.setFixedHeight(6)
        self._ollama_bar.setTextVisible(False)
        ai_l.addWidget(self._ollama_bar)
        l.addWidget(ai_g)

        act = QHBoxLayout()
        btn_org = QPushButton("Open Organizer")
        btn_org.setObjectName("primary")
        def go_organize():
            for rb in self._mode_group:
                if rb.isChecked():
                    self.main.switch_page(1)
                    return
            self.main.switch_page(1)
        btn_org.clicked.connect(go_organize)
        btn_ai = QPushButton("AI Models")
        btn_ai.setObjectName("ghost")
        btn_ai.clicked.connect(lambda: self.main.switch_page(3))
        btn_ref = QPushButton("Refresh")
        btn_ref.setObjectName("ghost")
        btn_ref.clicked.connect(self.refresh)
        act.addWidget(btn_org)
        act.addWidget(btn_ai)
        act.addWidget(btn_ref)
        act.addStretch()
        l.addLayout(act)

        l.addWidget(QLabel("Recent Activity"))
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(100)
        self._log.setFixedHeight(120)
        l.addWidget(self._log)
        l.addStretch()

    def refresh(self):
        cfg = s.load_settings()
        mode = cfg.get("organize_mode", "lowlevel")
        for rb in self._mode_group:
            rb.setChecked(rb.objectName() == mode)

        dest = s.resolve_dest_dir(cfg)
        c.CATEGORIES_DIR = dest
        c.discover_categories(dest)

        n_cats = n_files = dupes = nsfw = 0
        if os.path.isdir(dest):
            for e in os.listdir(dest):
                p = os.path.join(dest, e)
                if not os.path.isdir(p) or e.startswith("."):
                    continue
                # NSFW is now a category too (so it appears in the
                # Reorganize sidebar), but it's auto-managed by the
                # classifier so we count it separately in the dashboard.
                if e == c.NSFW_FOLDER:
                    nsfw += sum(1 for f in os.listdir(p)
                                if os.path.isfile(os.path.join(p, f))
                                and not f.startswith("."))
                    continue
                is_cat = (e in c.CATEGORIES
                          or os.path.isfile(os.path.join(p, ".category.json")))
                if is_cat:
                    n_cats += 1
                    n_files += sum(
                        1 for f in os.listdir(p)
                        if os.path.isfile(os.path.join(p, f)) and not f.startswith(".")
                    )
                elif e == c.DUPLICATES_FOLDER:
                    dupes += sum(1 for f in os.listdir(p) if os.path.isfile(os.path.join(p, f)))
        self._cards["total"].setText(str(n_files))
        self._cards["cats"].setText(str(n_cats))
        self._cards["dupes"].setText(str(dupes))
        self._cards["nsfw"].setText(str(nsfw))
        self._check_health()

    def _check_health(self):
        cfg = s.load_settings()
        try:
            import torch
            torch_ok = True
        except Exception:
            torch_ok = False
        try:
            import clip
            clip_ok = True
        except Exception:
            clip_ok = False
        if torch_ok and clip_ok:
            self._clip_status.setText("CLIP: Ready")
            self._clip_status.setObjectName("statusOk")
        elif torch_ok:
            self._clip_status.setText("CLIP: Missing (torch OK)")
            self._clip_status.setObjectName("statusWarn")
        elif clip_ok:
            self._clip_status.setText("CLIP: Missing (clip OK)")
            self._clip_status.setObjectName("statusWarn")
        else:
            self._clip_status.setText("CLIP: Not installed")
            self._clip_status.setObjectName("statusErr")
        self._clip_status.style().unpolish(self._clip_status)
        self._clip_status.style().polish(self._clip_status)

        url = cfg.get("ollama_url", "http://localhost:11434")
        model = cfg.get("ollama_model", "llava:7b")
        self._ollama_url.setText(url)
        self._ollama_bar.setVisible(True)
        self._ollama_bar.setValue(0)
        self._ollama_status.setText("Checking...")
        self._ollama_status.setObjectName("statSmall")

        if self._health_worker and self._health_worker.isRunning():
            try:
                self._health_worker.wait(500)
            except Exception:
                pass
        self._health_worker = HealthCheckWorker(url, model)
        self._health_worker.result_ready.connect(self._on_health)
        self._health_worker.start()

    def _on_health(self, result):
        if result.get("connected"):
            if result.get("model_available"):
                self._ollama_status.setText(f"Connected ({result['model_count']} models)")
                self._ollama_status.setObjectName("statusOk")
                self._ollama_bar.setValue(100)
            else:
                self._ollama_status.setText("Connected (model missing)")
                self._ollama_status.setObjectName("statusWarn")
                self._ollama_bar.setValue(50)
        else:
            self._ollama_status.setText(f"Offline: {result.get('error', '')}")
            self._ollama_status.setObjectName("statusErr")
            self._ollama_bar.setValue(0)
        self._ollama_status.style().unpolish(self._ollama_status)
        self._ollama_status.style().polish(self._ollama_status)
        QTimer.singleShot(1500, lambda: self._ollama_bar.setVisible(False))

    def append_log(self, msg):
        self._log.appendPlainText(msg)
