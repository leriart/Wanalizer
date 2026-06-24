"""AI Models page: manage CLIP and Ollama backends.

Redesigned with:
* Backend cards at the top (Low-Level CV / CLIP / Ollama) for one-click switching.
* Active-model hero card showing what's currently selected + key params.
* Per-backend panels with model tables, install/pull actions, and "Test" button.
* Hardware-aware model recommendations (CPU vs GPU; small vs accurate).
* Hardware-friendly color palette, generous spacing, real status indicators.
"""
import shutil
from typing import List, Tuple

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit,
    QGroupBox, QFormLayout, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QProgressBar, QPlainTextEdit, QWidget,
    QMessageBox, QFrame, QSizePolicy, QCheckBox, QDoubleSpinBox, QSpinBox,
)
from PySide6.QtCore import Qt, QTimer, QSize, Signal
from PySide6.QtGui import QFont, QColor, QPainter, QPixmap, QIcon

from ..widgets import (
    BTN_TEXT, BTN_KIND, BTN_DATA,
    setup_table_buttons, min_button_column_width, refresh_action_columns,
)
from ... import settings as s
from ..workers import HealthCheckWorker, OllamaPullWorker, DepWorker


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


# ---------------------------------------------------------------------------
# Static knowledge bases
# ---------------------------------------------------------------------------

CLIP_CATALOG = [
    # (name, size, status_hint, vram_mb, speed_note, accuracy_note, recommended)
    ("ViT-B/32",      "~150 MB", "fast",     600,  "fastest",        "baseline",         False),
    ("ViT-B/16",      "~150 MB", "balanced", 700,  "fast",           "slightly better",  False),
    ("ViT-L/14",      "~890 MB", "balanced", 1500, "medium",         "best accuracy",    True),
    ("ViT-L/14@336px","~890 MB", "accurate", 1700, "slow",           "highest accuracy", False),
    ("RN50",          "~150 MB", "fast",     600,  "fast",           "older arch",       False),
    ("RN101",         "~150 MB", "fast",     700,  "fast",           "older arch",       False),
    ("RN50x4",        "~600 MB", "balanced", 1100, "medium",         "good",             False),
    ("RN50x16",       "~1.2 GB", "heavy",    3500, "slow",           "high",             False),
    ("RN50x64",       "~2.0 GB", "heavy",    10000, "very slow",     "highest",          False),
]

OLLAMA_CATALOG = [
    # (tag, name, size, speed, quality, nsfw_ok, vram_mb)
    ("llava:7b",              "LLaVA 7B",            "4.5 GB", "fast",   "good",      False, 6000),
    ("llava:13b",             "LLaVA 13B",           "8.0 GB", "medium", "better",    False, 10000),
    ("llama3.2-vision:11b",   "Llama 3.2 Vision",    "8.0 GB", "medium", "good",      False, 10000),
    ("minicpm-v:8b",          "MiniCPM-V 8B",        "5.0 GB", "fast",   "good",      False, 7000),
    ("moondream:latest",      "Moondream 2B",        "0.9 GB", "very fast", "basic",  False, 2000),
    ("llava-phi3:3.8b",       "LLaVA-Phi3 (small)",  "2.5 GB", "very fast", "ok",     False, 4000),
]


# ---------------------------------------------------------------------------
# Small visual primitives
# ---------------------------------------------------------------------------

class StatusDot(QWidget):
    """Tiny colored dot — used in cards/labels as an at-a-glance indicator."""

    def __init__(self, color: str = "#666666", size: int = 10, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._size = size
        self.setFixedSize(size + 4, size + 4)

    def set_color(self, color: str):
        self._color = QColor(color)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setBrush(self._color)
        p.setPen(Qt.NoPen)
        p.drawEllipse(2, 2, self._size, self._size)
        p.end()


def _hrule() -> QFrame:
    """Horizontal divider line."""
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet("color: #222; background: #222; max-height: 1px;")
    return f


def _vrule() -> QFrame:
    """Vertical divider line."""
    f = QFrame()
    f.setFrameShape(QFrame.VLine)
    f.setStyleSheet("color: #222; background: #222; max-width: 1px;")
    return f


# ---------------------------------------------------------------------------
# Backend picker cards
# ---------------------------------------------------------------------------

BACKEND_DEFS = [
    {
        "key": "lowlevel",
        "title": "Low-Level CV",
        "subtitle": "No ML dependencies",
        "summary": "Edges, textures, silhouettes, color heuristics. Fast and deterministic.",
        "pros": ["Instant (CPU only)", "No models to download", "Reproducible results"],
        "cons": ["No semantic understanding", "Struggles with abstract art"],
        "color": "#3b8eff",
    },
    {
        "key": "clip",
        "title": "CLIP",
        "subtitle": "OpenAI vision-language model",
        "summary": "Zero-shot semantic classification. Best balance of speed and accuracy.",
        "pros": ["Semantic understanding", "Tag-based reasoning", "GPU-accelerated"],
        "cons": ["~150-900 MB download", "Needs torch + clip installed"],
        "color": "#10b981",
    },
    {
        "key": "ollama",
        "title": "Ollama",
        "subtitle": "Local vision LLM",
        "summary": "Full language model with vision. Best detail (descriptions, NSFW, characters).",
        "pros": ["Best detail", "Generates descriptions", "NSFW-capable models exist"],
        "cons": ["Large (4-10 GB)", "Slower than CLIP", "Needs Ollama server running"],
        "color": "#a855f7",
    },
]


class BackendCard(QFrame):
    """Selectable card for picking the active backend."""

    clicked = Signal(str)

    def __init__(self, defn: dict, active: bool = False, parent=None):
        super().__init__(parent)
        self.key = defn["key"]
        self._active = active
        self.setObjectName("backendCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumWidth(220)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self._build(defn)
        self._apply_style()

    def _build(self, d: dict):
        l = QVBoxLayout(self)
        l.setContentsMargins(16, 14, 16, 14)
        l.setSpacing(6)

        head = QHBoxLayout()
        head.setSpacing(8)
        self._dot = StatusDot(d["color"], 10)
        head.addWidget(self._dot)
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        t = QLabel(d["title"])
        t.setObjectName("cardTitle")
        f = QFont()
        f.setBold(True)
        f.setPointSize(11)
        t.setFont(f)
        title_box.addWidget(t)
        sub = QLabel(d["subtitle"])
        sub.setObjectName("statSmall")
        sub.setStyleSheet("color: #888;")
        title_box.addWidget(sub)
        head.addLayout(title_box, 1)

        self._active_lbl = QLabel("")
        self._active_lbl.setStyleSheet(
            "color: #10b981; font-weight: 700; font-size: 9pt;"
            " background: #0e1f18; padding: 2px 6px; border-radius: 4px;"
        )
        self._active_lbl.setVisible(False)
        head.addWidget(self._active_lbl)
        l.addLayout(head)

        l.addWidget(_hrule())

        summary = QLabel(d["summary"])
        summary.setWordWrap(True)
        summary.setStyleSheet("color: #c0c0c0;")
        l.addWidget(summary)

        pros = QLabel("\n".join(f"+ {p}" for p in d["pros"]))
        pros.setStyleSheet("color: #5fbf80; font-size: 9pt;")
        l.addWidget(pros)

        cons = QLabel("\n".join(f"- {c}" for c in d["cons"]))
        cons.setStyleSheet("color: #c08080; font-size: 9pt;")
        l.addWidget(cons)

        l.addStretch()

    def set_active(self, active: bool):
        self._active = active
        self._active_lbl.setVisible(active)
        self._active_lbl.setText("ACTIVE")
        self._apply_style()

    def _apply_style(self):
        if self._active:
            border = "#10b981"
            bg = "#0c1612"
        else:
            border = "#222"
            bg = "#0a0a0a"
        self.setStyleSheet(
            f"QFrame#backendCard {{ background: {bg}; border: 1px solid {border};"
            " border-radius: 8px; }"
        )

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.clicked.emit(self.key)
        super().mousePressEvent(ev)


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

class AIModelsPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self._health_worker = None
        self._pull_worker = None
        self._clip_worker = None
        self._backend_cards: List[BackendCard] = []
        self._build()
        self._load()
        QTimer.singleShot(300, self._refresh_all)

    # ---------- BUILD ----------

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(12)

        title = QLabel("AI Models")
        title.setObjectName("title")
        outer.addWidget(title)
        sub = QLabel(
            "Pick a backend, install what you need, and tune thresholds. "
            "Changes are saved automatically."
        )
        sub.setObjectName("subtitle")
        outer.addWidget(sub)

        # ---- Active-model hero ----
        self._build_hero(outer)

        # ---- Backend picker cards ----
        outer.addWidget(_hrule())
        cards_row = QHBoxLayout()
        cards_row.setSpacing(10)
        cards_row.setContentsMargins(0, 4, 0, 4)
        for d in BACKEND_DEFS:
            card = BackendCard(d)
            card.clicked.connect(self._on_backend_picked)
            self._backend_cards.append(card)
            cards_row.addWidget(card, 1)
        outer.addLayout(cards_row)

        # ---- Backend panels (tabs) ----
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._build_clip_tab()
        self._build_ollama_tab()
        self._build_uncensored_tab()
        self._tabs.addTab(self._clip_tab, "CLIP / PyTorch")
        self._tabs.addTab(self._ollama_tab, "Ollama")
        self._tabs.addTab(self._uncensored_tab, "Vision LLMs")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        outer.addWidget(self._tabs, 1)

        # ---- Log ----
        log_box = QGroupBox("Activity log")
        ll = QVBoxLayout(log_box)
        ll.setContentsMargins(8, 8, 8, 8)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(400)
        self._log.setMinimumHeight(110)
        ll.addWidget(self._log)
        log_row = QHBoxLayout()
        log_row.addStretch()
        clear = QPushButton("Clear log")
        clear.setObjectName("ghost")
        clear.clicked.connect(lambda: self._log.clear())
        log_row.addWidget(clear)
        ll.addLayout(log_row)
        outer.addWidget(log_box)

    def _build_hero(self, parent_layout):
        """Top hero card: shows the currently active backend + key params."""
        hero = QFrame()
        hero.setObjectName("card")
        hero.setStyleSheet(
            "QFrame#card { background: #0c0c0c; border: 1px solid #1a1a1a;"
            " border-radius: 8px; }"
        )
        hl = QHBoxLayout(hero)
        hl.setContentsMargins(16, 12, 16, 12)
        hl.setSpacing(18)

        left = QVBoxLayout()
        left.setSpacing(2)
        lbl = QLabel("ACTIVE SETUP")
        lbl.setStyleSheet("color: #888; font-size: 8pt; font-weight: 700; letter-spacing: 1px;")
        left.addWidget(lbl)
        self._hero_mode = QLabel("—")
        self._hero_mode.setStyleSheet("font-size: 18pt; font-weight: 700; color: #ffffff;")
        left.addWidget(self._hero_mode)
        self._hero_model = QLabel("No model selected")
        self._hero_model.setStyleSheet("color: #c0c0c0;")
        left.addWidget(self._hero_model)
        hl.addLayout(left, 1)

        hl.addWidget(_vrule())

        # Status indicators
        status_box = QVBoxLayout()
        status_box.setSpacing(6)
        s1 = QHBoxLayout()
        s1.setSpacing(6)
        self._hero_clip_dot = StatusDot("#666")
        s1.addWidget(self._hero_clip_dot)
        self._hero_clip = QLabel("CLIP: unknown")
        self._hero_clip.setStyleSheet("color: #c0c0c0;")
        s1.addWidget(self._hero_clip)
        s1.addStretch()
        status_box.addLayout(s1)

        s2 = QHBoxLayout()
        s2.setSpacing(6)
        self._hero_oll_dot = StatusDot("#666")
        s2.addWidget(self._hero_oll_dot)
        self._hero_oll = QLabel("Ollama: not tested")
        self._hero_oll.setStyleSheet("color: #c0c0c0;")
        s2.addWidget(self._hero_oll)
        s2.addStretch()
        status_box.addLayout(s2)

        hl.addLayout(status_box, 1)
        parent_layout.addWidget(hero)

    def _build_clip_tab(self):
        self._clip_tab = QWidget()
        vl = QVBoxLayout(self._clip_tab)
        vl.setContentsMargins(8, 12, 8, 8)
        vl.setSpacing(8)

        # Intro
        intro = QLabel(
            "OpenAI CLIP classifies images via zero-shot text prompts (semantic). "
            "It needs `torch` and `clip` Python packages — the button below installs them."
        )
        intro.setWordWrap(True)
        intro.setObjectName("statSmall")
        intro.setStyleSheet("color: #aaa;")
        vl.addWidget(intro)

        # Hardware presets
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        preset_lbl = QLabel("Presets:")
        preset_lbl.setStyleSheet("color: #888;")
        preset_row.addWidget(preset_lbl)
        for label, model in [
            ("CPU only",   "ViT-B/32"),
            ("Balanced",   "ViT-L/14"),
            ("Best quality", "ViT-L/14"),
            ("Smallest",   "ViT-B/32"),
        ]:
            b = QPushButton(label)
            b.setObjectName("ghost")
            b.setToolTip(f"Switch active CLIP model to {model}")
            b.clicked.connect(lambda _checked=False, m=model: self._clip_activate(m))
            preset_row.addWidget(b)
        preset_row.addStretch()
        vl.addLayout(preset_row)

        # Install / refresh row
        top = QHBoxLayout()
        self._clip_install_btn = QPushButton("Install CLIP + PyTorch")
        self._clip_install_btn.setObjectName("primary")
        self._clip_install_btn.clicked.connect(self._clip_install)
        top.addWidget(self._clip_install_btn)
        self._clip_status_lbl = QLabel("")
        self._clip_status_lbl.setObjectName("statSmall")
        top.addWidget(self._clip_status_lbl, 1)
        self._clip_refresh_btn = QPushButton("Refresh")
        self._clip_refresh_btn.setObjectName("ghost")
        self._clip_refresh_btn.clicked.connect(self._clip_populate)
        top.addWidget(self._clip_refresh_btn)
        vl.addLayout(top)

        # Model table
        self._clip_table = QTableWidget(0, 6)
        self._clip_table.setHorizontalHeaderLabels(
            ["Model", "Size", "VRAM", "Speed/Accuracy", "Status", "Action"]
        )
        self._clip_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3, 4):
            self._clip_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self._clip_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self._clip_table.horizontalHeader().setMinimumSectionSize(min_button_column_width(2))
        self._clip_table.setAlternatingRowColors(True)
        self._clip_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._clip_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        vl.addWidget(self._clip_table, 1)

        def _use_cb(_r, _c, d):
            self._clip_activate(d)

        setup_table_buttons(self._clip_table, {"use": _use_cb}, [5])
        self._clip_populate()

    def _build_ollama_tab(self):
        self._ollama_tab = QWidget()
        vl = QVBoxLayout(self._ollama_tab)
        vl.setContentsMargins(8, 12, 8, 8)
        vl.setSpacing(10)

        # Server bar
        server_box = QGroupBox("Ollama server")
        sl = QFormLayout(server_box)
        url_row = QHBoxLayout()
        self._oll_url = QLineEdit("http://localhost:11434")
        self._oll_url.setPlaceholderText("http://localhost:11434")
        url_row.addWidget(self._oll_url, 1)
        self._oll_test_btn = QPushButton("Test connection")
        self._oll_test_btn.setObjectName("primary")
        self._oll_test_btn.clicked.connect(self._oll_test)
        url_row.addWidget(self._oll_test_btn)
        url_widget = QWidget()
        url_widget.setLayout(url_row)
        sl.addRow("URL:", url_widget)

        self._oll_status = QLabel("Not tested")
        self._oll_status.setObjectName("statSmall")
        sl.addRow("Status:", self._oll_status)

        self._oll_meta = QLabel("")
        self._oll_meta.setObjectName("statSmall")
        self._oll_meta.setStyleSheet("color: #888;")
        sl.addRow("Server:", self._oll_meta)
        vl.addWidget(server_box)

        # Pull bar
        pull_box = QGroupBox("Pull a model")
        pl = QVBoxLayout(pull_box)
        prow = QHBoxLayout()
        self._oll_combo = QComboBox()
        self._oll_combo.setEditable(True)
        self._oll_combo.addItems([
            "llava:7b", "llava:13b", "llama3.2-vision:11b",
            "minicpm-v:8b", "moondream:latest",
        ])
        prow.addWidget(self._oll_combo, 1)
        self._oll_refresh_btn = QPushButton("Refresh list")
        self._oll_refresh_btn.setObjectName("ghost")
        self._oll_refresh_btn.clicked.connect(self._oll_refresh)
        prow.addWidget(self._oll_refresh_btn)
        self._oll_pull_btn = QPushButton("Pull")
        self._oll_pull_btn.setObjectName("success")
        self._oll_pull_btn.clicked.connect(self._oll_pull)
        prow.addWidget(self._oll_pull_btn)
        self._oll_stop_btn = QPushButton("Stop")
        self._oll_stop_btn.setObjectName("danger")
        self._oll_stop_btn.setEnabled(False)
        self._oll_stop_btn.clicked.connect(self._oll_stop)
        prow.addWidget(self._oll_stop_btn)
        pl.addLayout(prow)

        self._oll_pull_prog = QProgressBar()
        self._oll_pull_prog.setVisible(False)
        pl.addWidget(self._oll_pull_prog)
        self._oll_pull_status = QLabel("")
        self._oll_pull_status.setObjectName("statSmall")
        pl.addWidget(self._oll_pull_status)
        vl.addWidget(pull_box)

        # Installed models table
        installed_box = QGroupBox("Available models (click a row to activate / pull)")
        il = QVBoxLayout(installed_box)
        self._oll_table = QTableWidget(0, 6)
        self._oll_table.setHorizontalHeaderLabels(
            ["Tag", "Display name", "Size", "VRAM", "Speed/Quality", "Action"]
        )
        self._oll_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._oll_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for c in (2, 3, 4):
            self._oll_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self._oll_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self._oll_table.horizontalHeader().setMinimumSectionSize(min_button_column_width(1))
        self._oll_table.setAlternatingRowColors(True)
        self._oll_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._oll_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        il.addWidget(self._oll_table)

        def _oll_act_cb(_r, _c, tag):
            self._on_oll_row_action(tag)
        setup_table_buttons(self._oll_table, {"action": _oll_act_cb}, [5])
        vl.addWidget(installed_box, 1)
        self._oll_populate()
        QTimer.singleShot(400, self._oll_test)

    def _build_uncensored_tab(self):
        """Vision LLM recommendations + NSFW behavior settings."""
        self._uncensored_tab = QWidget()
        vl = QVBoxLayout(self._uncensored_tab)
        vl.setContentsMargins(8, 12, 8, 8)
        vl.setSpacing(10)

        intro = QLabel(
            "Curated list of vision LLMs suited for wallpaper analysis. "
            "Stock models (LLaVA, Llama 3.2 Vision) sanitize or refuse NSFW content; "
            "uncensored variants handle the full spectrum. The pipeline auto-skips "
            "AI description on NSFW by default, but uncensored models still improve "
            "categorization for that content."
        )
        intro.setWordWrap(True)
        intro.setObjectName("statSmall")
        intro.setStyleSheet("color: #aaa;")
        vl.addWidget(intro)

        # NSFW behavior
        nsfw_box = QGroupBox("NSFW behavior")
        nl = QFormLayout(nsfw_box)
        self._nsfw_skip = QCheckBox(
            "Skip AI description when NSFW detected (use CV-only tags)"
        )
        nl.addRow(self._nsfw_skip)
        self._nsfw_thresh = QDoubleSpinBox()
        self._nsfw_thresh.setRange(0.0, 1.0)
        self._nsfw_thresh.setSingleStep(0.05)
        self._nsfw_thresh.setDecimals(2)
        nl.addRow("CV NSFW threshold:", self._nsfw_thresh)
        self._oll_nsfw_thresh = QDoubleSpinBox()
        self._oll_nsfw_thresh.setRange(0.5, 1.0)
        self._oll_nsfw_thresh.setSingleStep(0.05)
        self._oll_nsfw_thresh.setDecimals(2)
        self._oll_nsfw_thresh.setToolTip(
            "Minimum Ollama NSFW score to flag the image. Scores in the "
            "'uncertain zone' (0.40-0.65) are ignored."
        )
        nl.addRow("Ollama NSFW threshold:", self._oll_nsfw_thresh)
        self._nsfw_uncertain_floor = QDoubleSpinBox()
        self._nsfw_uncertain_floor.setRange(0.0, 1.0)
        self._nsfw_uncertain_floor.setSingleStep(0.05)
        self._nsfw_uncertain_floor.setDecimals(2)
        nl.addRow("Uncertain-zone floor:", self._nsfw_uncertain_floor)
        self._nsfw_uncertain_ceiling = QDoubleSpinBox()
        self._nsfw_uncertain_ceiling.setRange(0.0, 1.0)
        self._nsfw_uncertain_ceiling.setSingleStep(0.05)
        self._nsfw_uncertain_ceiling.setDecimals(2)
        nl.addRow("Uncertain-zone ceiling:", self._nsfw_uncertain_ceiling)

        self._nsfw_skip.stateChanged.connect(self._save_nsfw_settings)
        for w in (self._nsfw_thresh, self._oll_nsfw_thresh,
                  self._nsfw_uncertain_floor, self._nsfw_uncertain_ceiling):
            w.valueChanged.connect(self._save_nsfw_settings)
        vl.addWidget(nsfw_box)

        # Recommendations table
        rec_box = QGroupBox("Recommended vision LLMs")
        rl = QVBoxLayout(rec_box)
        try:
            from ...ollama_client import get_recommended_uncensored_models
            uncensored = list(get_recommended_uncensored_models())
        except Exception:
            uncensored = []

        if uncensored:
            self._unc_table = QTableWidget(0, 5)
            self._unc_table.setHorizontalHeaderLabels(
                ["Model", "Size", "Speed", "Notes", "Action"]
            )
            self._unc_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            self._unc_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            self._unc_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            self._unc_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
            self._unc_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
            self._unc_table.horizontalHeader().setMinimumSectionSize(min_button_column_width(1))
            self._unc_table.setAlternatingRowColors(True)
            self._unc_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self._unc_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self._unc_table.setRowCount(len(uncensored))
            for i, m in enumerate(uncensored):
                self._unc_table.setItem(i, 0, QTableWidgetItem(m.get("name", "?")))
                self._unc_table.setItem(i, 1, QTableWidgetItem(f"{m.get('size_gb', 0):.1f} GB"))
                self._unc_table.setItem(i, 2, QTableWidgetItem(m.get("speed", "?")))
                self._unc_table.setItem(i, 3, QTableWidgetItem(m.get("notes", "")))
                act = QTableWidgetItem()
                act.setData(BTN_TEXT, "Pull")
                act.setData(BTN_KIND, "action")
                act.setData(BTN_DATA, m.get("tag", ""))
                self._unc_table.setItem(i, 4, act)
            refresh_action_columns(self._unc_table)

            def _unc_act_cb(_r, _c, tag):
                self._oll_combo.setEditText(tag)
                self._tabs.setCurrentWidget(self._ollama_tab)
                self._oll_pull()

            setup_table_buttons(self._unc_table, {"action": _unc_act_cb}, [4])
            rl.addWidget(self._unc_table)
        else:
            rl.addWidget(QLabel("No curated recommendations available."))
        vl.addWidget(rec_box, 1)

    # ---------- LOAD / REFRESH ----------

    def _load(self):
        cfg = s.load_settings()
        mode = cfg.get("organize_mode", "lowlevel")
        self._set_active_backend(mode)

        self._oll_url.setText(cfg.get("ollama_url", "http://localhost:11434"))
        model = cfg.get("ollama_model", "llava:7b")
        idx = self._oll_combo.findText(model)
        if idx >= 0:
            self._oll_combo.setCurrentIndex(idx)
        else:
            self._oll_combo.setEditText(model)

        self._nsfw_skip.setChecked(cfg.get("nsfw_skip_describe", True))
        self._nsfw_thresh.setValue(cfg.get("nsfw_threshold", 0.5))
        self._oll_nsfw_thresh.setValue(cfg.get("ollama_nsfw_threshold", 0.70))
        self._nsfw_uncertain_floor.setValue(cfg.get("ollama_nsfw_uncertain_floor", 0.40))
        self._nsfw_uncertain_ceiling.setValue(cfg.get("ollama_nsfw_uncertain_ceiling", 0.65))

    def _refresh_all(self):
        """Re-query everything (CLIP detect + Ollama health)."""
        self._clip_populate()
        self._oll_test()

    def _set_active_backend(self, mode: str):
        for card in self._backend_cards:
            card.set_active(card.key == mode)
        nice = {"lowlevel": "Low-Level CV",
                "clip": "CLIP",
                "ollama": "Ollama"}.get(mode, mode)
        self._hero_mode.setText(nice)
        cfg = s.load_settings()
        if mode == "clip":
            m = cfg.get("clip_model", "ViT-B/32")
            self._hero_model.setText(f"Model: {m}")
        elif mode == "ollama":
            m = cfg.get("ollama_model", "llava:7b")
            self._hero_model.setText(f"Model: {m}")
        else:
            self._hero_model.setText("No model — heuristic-only classification")

    # ---------- BACKEND PICKER ----------

    def _on_backend_picked(self, key: str):
        if key not in ("lowlevel", "clip", "ollama"):
            return
        cfg = s.load_settings()
        if cfg.get("organize_mode") == key:
            return
        cfg["organize_mode"] = key
        s.save_settings(cfg)
        self._set_active_backend(key)
        self._log.appendPlainText(f"[setup] Active backend set to {key}")
        # Jump to the right tab so the user sees what's relevant.
        idx = {"clip": 0, "ollama": 1, "lowlevel": -1}.get(key, -1)
        if idx >= 0:
            self._tabs.setCurrentIndex(idx)

    def _on_tab_changed(self, idx: int):
        # Refresh content for the tab being shown.
        if idx == 0:
            self._clip_populate()
        elif idx == 1:
            self._oll_test()
            self._oll_populate()

    # ---------- CLIP TAB ----------

    def _clip_populate(self):
        from ...clip_client import available_clip_models
        try:
            installed = set(available_clip_models())
        except Exception:
            installed = set()

        cfg = s.load_settings()
        active = cfg.get("clip_model", "ViT-B/32")
        deps_ok = HAS_CLIP and HAS_TORCH

        # Update hero status
        if deps_ok:
            if installed:
                self._hero_clip_dot.set_color("#10b981")
                self._hero_clip.setText(f"CLIP: ready ({len(installed)} model(s) available)")
            else:
                self._hero_clip_dot.set_color("#cc8800")
                self._hero_clip.setText("CLIP: deps installed, no models yet")
        else:
            self._hero_clip_dot.set_color("#cc4040")
            self._hero_clip.setText("CLIP: torch/clip not installed")

        self._clip_status_lbl.setText(
            f"  ({len(installed)} of {len(CLIP_CATALOG)} models installed)"
            if deps_ok else "  Install dependencies to see available models"
        )

        self._clip_table.setRowCount(len(CLIP_CATALOG))
        for i, (nm, sz, _hint, vram, speed_note, acc_note, recommended) in enumerate(CLIP_CATALOG):
            is_installed = nm in installed
            is_active = nm == active
            self._clip_table.setItem(i, 0, QTableWidgetItem(nm + ("  ★" if recommended else "")))
            sz_item = QTableWidgetItem(sz)
            sz_item.setToolTip(f"VRAM ≈ {vram} MB" if vram else "")
            self._clip_table.setItem(i, 1, sz_item)
            self._clip_table.setItem(i, 2, QTableWidgetItem(f"{vram} MB" if vram else "—"))
            self._clip_table.setItem(i, 3, QTableWidgetItem(f"{speed_note} / {acc_note}"))
            if is_active:
                st = QTableWidgetItem("Active")
                st.setForeground(QColor("#10b981"))
            elif is_installed:
                st = QTableWidgetItem("Ready")
            else:
                st = QTableWidgetItem("Not installed")
                st.setForeground(QColor("#888"))
            self._clip_table.setItem(i, 4, st)

            act = QTableWidgetItem()
            if not deps_ok:
                act.setData(BTN_TEXT, "Install")
                act.setData(BTN_KIND, "use")
                act.setData(BTN_DATA, "INSTALL")
            elif is_active:
                act.setData(BTN_TEXT, "Active")
                act.setData(BTN_KIND, "use")
                act.setData(BTN_DATA, nm)
            elif is_installed:
                act.setData(BTN_TEXT, "Use")
                act.setData(BTN_KIND, "use")
                act.setData(BTN_DATA, nm)
            else:
                act.setData(BTN_TEXT, "Use (downloads)")
                act.setData(BTN_KIND, "use")
                act.setData(BTN_DATA, nm)
            self._clip_table.setItem(i, 5, act)
        refresh_action_columns(self._clip_table)

    def _clip_activate(self, model_name: str):
        if model_name == "INSTALL":
            self._clip_install()
            return
        if not model_name:
            return
        heavy = {"RN50x16": "very heavy (1.2 GB)",
                 "RN50x64": "very heavy (2.0 GB)"}
        cfg = s.load_settings()
        if cfg.get("clip_model") == model_name:
            return
        if model_name in heavy:
            r = QMessageBox.question(
                self, "Heavy CLIP model",
                f"{model_name} is {heavy[model_name]}.\n\n"
                "Each image will take 5-15 s to classify on CPU.\n"
                "Continue anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if r != QMessageBox.Yes:
                return
        cfg["clip_model"] = model_name
        s.save_settings(cfg)
        self._log.appendPlainText(f"[CLIP] active model set to {model_name}")
        self._set_active_backend("clip")
        self._clip_populate()

    def _clip_install(self):
        if self._clip_worker and self._clip_worker.isRunning():
            self._log.appendPlainText("[pip] Already installing, please wait...")
            return
        if shutil.which("git") is None:
            QMessageBox.warning(
                self, "Missing git",
                "Git is required to install CLIP (OpenAI's CLIP repo).\n"
                "Install git first (apt install git / brew install git).",
            )
            return
        self._log.appendPlainText("[pip] Installing CLIP + PyTorch (may take several minutes)...")
        self._clip_install_btn.setEnabled(False)
        self._clip_install_btn.setText("Installing…")
        self._clip_worker = DepWorker(["torch", "torchvision", "ftfy", "regex", "clip"])
        self._clip_worker.log_line.connect(self._log.appendPlainText)
        self._clip_worker.module_done.connect(
            lambda mod, ok: self._log.appendPlainText(f"  [{'OK' if ok else 'FAIL'}] {mod}")
        )
        self._clip_worker.all_done.connect(self._clip_install_done)
        self._clip_worker.start()

    def _clip_install_done(self):
        self._clip_install_btn.setEnabled(True)
        self._clip_install_btn.setText("Install CLIP + PyTorch")
        self._log.appendPlainText("[pip] Done. Reloading model list…")
        self._clip_populate()

    # ---------- OLLAMA TAB ----------

    def _oll_populate(self):
        from ...ollama_client import OllamaClient
        cfg = s.load_settings()
        active = cfg.get("ollama_model", "llava:7b")
        url = self._oll_url.text().strip()

        installed = set()
        try:
            oc = OllamaClient(base_url=url, timeout=3)
            res = oc.list_models()
            installed = {m.get("name") if isinstance(m, dict) else m for m in res}
        except Exception:
            pass

        self._oll_table.setRowCount(len(OLLAMA_CATALOG))
        for i, (tag, name, sz, speed, quality, _nsfw, vram) in enumerate(OLLAMA_CATALOG):
            self._oll_table.setItem(i, 0, QTableWidgetItem(tag))
            self._oll_table.setItem(i, 1, QTableWidgetItem(name))
            self._oll_table.setItem(i, 2, QTableWidgetItem(sz))
            self._oll_table.setItem(i, 3, QTableWidgetItem(f"{vram} MB"))
            self._oll_table.setItem(i, 4, QTableWidgetItem(f"{speed} / {quality}"))

            is_active = tag == active
            is_avail = tag in installed
            act = QTableWidgetItem()
            if is_active:
                act.setData(BTN_TEXT, "Active")
                act.setData(BTN_KIND, "action")
                act.setData(BTN_DATA, tag)
            elif is_avail:
                act.setData(BTN_TEXT, "Activate")
                act.setData(BTN_KIND, "action")
                act.setData(BTN_DATA, tag)
            else:
                act.setData(BTN_TEXT, "Pull")
                act.setData(BTN_KIND, "action")
                act.setData(BTN_DATA, tag)
            self._oll_table.setItem(i, 5, act)
        refresh_action_columns(self._oll_table)

    def _on_oll_row_action(self, tag: str):
        from ...ollama_client import OllamaClient
        try:
            oc = OllamaClient(base_url=self._oll_url.text().strip(), model=tag, timeout=3)
            installed = oc.list_models()
            names = {m.get("name") if isinstance(m, dict) else m for m in installed}
        except Exception:
            names = set()
        if tag in names:
            self._oll_activate(tag)
        else:
            self._oll_combo.setEditText(tag)
            self._oll_pull()

    def _oll_activate(self, tag):
        cfg = s.load_settings()
        cfg["ollama_model"] = tag
        cfg["ollama_url"] = self._oll_url.text().strip()
        s.save_settings(cfg)
        self._log.appendPlainText(f"[Ollama] active model set to {tag}")
        self._set_active_backend("ollama")
        self._oll_populate()

    def _oll_refresh(self):
        import requests as req
        try:
            r = req.get(f"{self._oll_url.text().strip().rstrip('/')}/api/tags", timeout=10)
            if r.status_code == 200:
                cur = self._oll_combo.currentText()
                self._oll_combo.clear()
                for m in r.json().get("models", []):
                    self._oll_combo.addItem(m.get("name", ""))
                idx = self._oll_combo.findText(cur)
                if idx >= 0:
                    self._oll_combo.setCurrentIndex(idx)
                n = len(r.json().get("models", []))
                self._log.appendPlainText(f"[Ollama] {n} model(s) found on server")
                self._oll_populate()
        except Exception as e:
            self._log.appendPlainText(f"[Ollama] refresh failed: {e}")

    def _oll_test(self):
        self._oll_status.setText("Testing…")
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
            self._hero_oll_dot.set_color("#10b981")
            if result.get("model_available"):
                self._oll_status.setText(f"Connected — {result['model_count']} model(s) available")
                self._oll_status.setStyleSheet("color: #10b981;")
                self._hero_oll.setText(
                    f"Ollama: connected ({result['model_count']} models)"
                )
            else:
                self._oll_status.setText("Connected — selected model not pulled")
                self._oll_status.setStyleSheet("color: #cc8800;")
                self._hero_oll.setText("Ollama: connected, model missing")
            ver = result.get("server_version", "")
            self._oll_meta.setText(
                f"v{ver} • {result.get('model_count', 0)} model(s) • "
                f"{self._oll_url.text().strip()}"
            )
        else:
            self._hero_oll_dot.set_color("#cc4040")
            err = result.get("error", "offline")
            self._oll_status.setText(f"Offline: {err[:60]}")
            self._oll_status.setStyleSheet("color: #cc4040;")
            self._hero_oll.setText(f"Ollama: offline ({err[:40]})")
            self._oll_meta.setText("Start Ollama and click 'Test connection'")
        self._oll_populate()

    def _oll_pull(self):
        if self._pull_worker and self._pull_worker.isRunning():
            self._log.appendPlainText("[pull] Already downloading, please wait…")
            return
        model = self._oll_combo.currentText().strip()
        if not model:
            return
        cfg = s.load_settings()
        cfg["ollama_url"] = self._oll_url.text().strip()
        s.save_settings(cfg)
        self._log.appendPlainText(f"[pull] Downloading {model}…")
        self._oll_pull_btn.setEnabled(False)
        self._oll_stop_btn.setEnabled(True)
        self._oll_pull_prog.setVisible(True)
        self._oll_pull_prog.setValue(0)
        self._oll_pull_status.setText("Starting…")
        self._pull_worker = OllamaPullWorker(self._oll_url.text().strip(), model, 600)
        self._pull_worker.progress.connect(self._on_pull_progress)
        self._pull_worker.finished_ok.connect(self._oll_pull_done)
        self._pull_worker.failed.connect(self._oll_pull_fail)
        self._pull_worker.log.connect(self._log.appendPlainText)
        self._pull_worker.start()

    def _on_pull_progress(self, c, t, st):
        self._oll_pull_prog.setMaximum(max(t, 1))
        self._oll_pull_prog.setValue(c)
        if t:
            self._oll_pull_status.setText(f"{st}  •  {_human_bytes(c)} / {_human_bytes(t)}")
        else:
            self._oll_pull_status.setText(st)

    def _oll_pull_done(self):
        self._oll_pull_btn.setEnabled(True)
        self._oll_stop_btn.setEnabled(False)
        self._oll_pull_prog.setVisible(False)
        self._oll_pull_status.setText("Done")
        self._log.appendPlainText("[pull] complete")
        # Activate the freshly pulled model
        model = self._oll_combo.currentText().strip()
        if model:
            self._oll_activate(model)
        self._oll_refresh()

    def _oll_pull_fail(self, msg):
        self._oll_pull_btn.setEnabled(True)
        self._oll_stop_btn.setEnabled(False)
        self._oll_pull_prog.setVisible(False)
        self._oll_pull_status.setText("")
        self._log.appendPlainText(f"[pull] failed: {msg}")

    def _oll_stop(self):
        if self._pull_worker:
            self._pull_worker.cancel()

    # ---------- NSFW ----------

    def _save_nsfw_settings(self):
        cfg = s.load_settings()
        cfg["nsfw_skip_describe"] = self._nsfw_skip.isChecked()
        cfg["nsfw_threshold"] = self._nsfw_thresh.value()
        cfg["ollama_nsfw_threshold"] = self._oll_nsfw_thresh.value()
        cfg["ollama_nsfw_uncertain_floor"] = self._nsfw_uncertain_floor.value()
        cfg["ollama_nsfw_uncertain_ceiling"] = self._nsfw_uncertain_ceiling.value()
        s.save_settings(cfg)


def _human_bytes(n: int) -> str:
    """Compact byte size for the pull-progress label."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.0f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024 ** 3:.2f} GB"