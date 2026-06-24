"""Wanalizer v3 - Qt6 Graphical Interface.

This module provides the full 9-page GUI for wallpaper organization.
It is a distributable component of the wanalizer package.

Usage:
    python -m wallpaper_analyzer.gui
    wanalizer-gui
"""
import os, sys, traceback
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QButtonGroup, QFrame, QStackedWidget,
    QStatusBar, QMessageBox,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QFontDatabase

from .. import __version__, __project__
from .. import settings as s
from ..parallel import cpu_count, is_free_threaded
from ..formats import STATIC_EXTENSIONS  # noqa: F401  (raises PIL decompression-bomb limit on import)
from .theme import apply_theme
from .pages import (
    DashboardPage, OrganizePage, ReorganizePage, AIModelsPage,
    CategoriesPage, TagsPage, DuplicatesPage, DependenciesPage, SettingsPage,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{__project__} v{__version__}")
        self.resize(1200, 800)
        self.setMinimumSize(800, 560)

        central = QWidget()
        hl = QHBoxLayout(central)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(180)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(8, 16, 8, 16)

        logo = QLabel(__project__)
        logo.setStyleSheet("font-size: 13pt; font-weight: 700; color: #e01020; padding: 6px 14px;")
        sl.addWidget(logo)

        sp = QFrame()
        sp.setFrameShape(QFrame.HLine)
        sp.setStyleSheet("background: #333333; max-height: 1px; margin: 8px 10px;")
        sl.addWidget(sp)

        nav_items = [
            ("Dashboard", 0),
            ("Organize", 1),
            ("Reorganize", 2),
            ("AI Models", 3),
            ("Categories", 4),
            ("Tags", 5),
            ("Duplicates", 6),
            ("Dependencies", 7),
            ("Settings", 8),
        ]
        self._nav_btns = []
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)

        for label, idx in nav_items:
            btn = QPushButton(f"  {label}")
            btn.setObjectName("navBtn")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, i=idx: self.switch_page(i))
            self._nav_group.addButton(btn, idx)
            sl.addWidget(btn)
            self._nav_btns.append(btn)

        sl.addStretch()
        ver = QLabel(f"v{__version__}")
        ver.setObjectName("statSmall")
        sl.addWidget(ver)
        hl.addWidget(sidebar)

        self.stack = QStackedWidget()
        self.pages = [
            DashboardPage(self),
            OrganizePage(self),
            ReorganizePage(self),
            AIModelsPage(self),
            CategoriesPage(self),
            TagsPage(self),
            DuplicatesPage(self),
            DependenciesPage(self),
            SettingsPage(self),
        ]
        for p in self.pages:
            self.stack.addWidget(p)
        hl.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.sb = QStatusBar()
        self.setStatusBar(self.sb)
        self._sb_label = QLabel("Ready")
        self.sb.addPermanentWidget(self._sb_label, 1)

        # Right-aligned runtime indicator (Python build + core count)
        py_label = QLabel(self._runtime_label())
        py_label.setObjectName("statSmall")
        self.sb.addPermanentWidget(py_label)

        self.switch_page(0)

    def _runtime_label(self) -> str:
        """Short status bar label showing the active Python build."""
        ft_tag = "free-threaded" if is_free_threaded() else "with GIL"
        return f"Python {sys.version_info.major}.{sys.version_info.minor} ({ft_tag}, {cpu_count()} cores)"

    def switch_page(self, index):
        if 0 <= index < len(self.pages):
            self.stack.setCurrentIndex(index)
            for btn in self._nav_btns:
                btn.setChecked(False)
            if 0 <= index < len(self._nav_btns):
                self._nav_btns[index].setChecked(True)
            if index == 0:
                try:
                    self.pages[0].refresh()
                except Exception:
                    pass

    def append_log(self, msg):
        self._sb_label.setText(msg)
        try:
            self.pages[0].append_log(msg)
        except Exception:
            pass


def _detect_font():
    candidates = [
        "Inter", "SF Pro Display", "Segoe UI", "Ubuntu",
        "DejaVu Sans", "Noto Sans", "Sans", "sans-serif",
    ]
    db = QFontDatabase()
    for family in candidates:
        if family in db.families():
            return family
    return "Sans"


def main():
    """Launch the Wanalizer GUI.

    Returns exit code (0 on success).
    """
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        app = QApplication(sys.argv)
        app.setFont(QFont(_detect_font(), 10))
        from .. import settings as _s
        _cfg = _s.load_settings()
        apply_theme(app, mode=_cfg.get("theme", "dark"))
        w = MainWindow()
        w.show()
        return app.exec()
    except Exception as e:
        print(f"[GUI] Fatal: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1
