"""Category configuration dialog.

Two paths:
  1. `CategoryConfigDialog` - Q&A wizard with checkable buttons for
     each question (aspect ratios, file kinds, palettes, keywords,
     min resolution). The user answers each question and the dialog
     builds the `expected` dict.
  2. `AIConfigDialog` - shows the AI-generated proposal (from
     `category_config.generate_config_from_samples`) and lets the
     user review/edit before saving.

Both dialogs write to `.category.json` via `category_config.write_expected`.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QSpinBox, QWidget,
    QGridLayout, QScrollArea, QMessageBox,
)

from ..category_config import (
    ALL_ASPECT_RATIOS, ALL_FILE_KINDS, ALL_PALETTES, ALL_STYLE_KEYWORDS,
    DEFAULT_EXPECTED, generate_config_from_samples, get_expected, write_expected,
)


# ---------------------------------------------------------------------------
# Multi-select chip widget
# ---------------------------------------------------------------------------

class _ChipGroup(QWidget):
    """A row of toggleable chips for multi-select choices.

    Each chip is a checkable QPushButton. The user can select any
    combination. `value()` returns the list of selected option keys.
    """

    def __init__(self, options: List[str], selected: Optional[List[str]] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._buttons: Dict[str, QPushButton] = {}
        sel = set(selected or [])
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        # Lay chips out in rows of 4
        cols = 4
        for i, opt in enumerate(options):
            btn = QPushButton(opt)
            btn.setCheckable(True)
            btn.setChecked(opt in sel)
            btn.setObjectName("ghost")
            btn.setMaximumHeight(28)
            self._buttons[opt] = btn
            grid.addWidget(btn, i // cols, i % cols)

    def value(self) -> List[str]:
        return [k for k, b in self._buttons.items() if b.isChecked()]

    def set_value(self, selected: List[str]) -> None:
        sel = set(selected)
        for k, b in self._buttons.items():
            b.setChecked(k in sel)


# ---------------------------------------------------------------------------
# Questionnaire dialog
# ---------------------------------------------------------------------------

class CategoryConfigDialog(QDialog):
    """Interactive Q&A wizard to build a category's `expected` config.

    The dialog is a single scrollable page with one group per question:
      * What file kinds? (image / video / animated)
      * What aspect ratios? (horizontal / vertical / square / any)
      * Minimum resolution?
      * Color palette? (dark / warm / cool / neon / pastel / vibrant / muted / monochrome)
      * Style keywords? (multi-select with the registry vocabulary)
      * Exclude keywords? (anti-pattern keywords)
    """

    def __init__(self, category: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.category = category
        self.setWindowTitle(f"Configure Category: {category}")
        self.setMinimumSize(640, 720)
        self._build()
        self._load_current()

    def _build(self):
        l = QVBoxLayout(self)
        l.setContentsMargins(20, 20, 20, 20)

        title = QLabel(f"Configure Category: {self.category}")
        title.setObjectName("title")
        l.addWidget(title)
        sub = QLabel(
            "Tell the classifier what kinds of images belong in this "
            "category. Leave a section empty to accept any."
        )
        sub.setObjectName("subtitle")
        sub.setWordWrap(True)
        l.addWidget(sub)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        scroll.setWidget(body)
        bl = QVBoxLayout(body)
        bl.setContentsMargins(8, 8, 8, 8)

        # Q1: File kinds
        gb1 = QGroupBox("File kinds (what does this category accept?)")
        gl1 = QVBoxLayout(gb1)
        self._file_kinds = _ChipGroup(list(ALL_FILE_KINDS))
        gl1.addWidget(self._file_kinds)
        bl.addWidget(gb1)

        # Q2: Aspect ratios
        gb2 = QGroupBox("Aspect ratios")
        gl2 = QVBoxLayout(gb2)
        self._aspect_ratios = _ChipGroup(
            [a.title() if a != "any" else "Any" for a in ALL_ASPECT_RATIOS],
            selected=["Any"],
        )
        # Map display labels back to internal keys
        self._ar_label_to_key = {
            ("Horizontal" if a != "any" else "Any"): a for a in ALL_ASPECT_RATIOS
        }
        gl2.addWidget(self._aspect_ratios)
        bl.addWidget(gb2)

        # Q3: Minimum resolution
        gb3 = QGroupBox("Minimum resolution (0 = no minimum)")
        gl3 = QFormLayout(gb3)
        self._min_w = QSpinBox()
        self._min_w.setRange(0, 10000)
        self._min_w.setSingleStep(100)
        self._min_w.setSuffix(" px")
        gl3.addRow("Min width:", self._min_w)
        self._min_h = QSpinBox()
        self._min_h.setRange(0, 10000)
        self._min_h.setSingleStep(100)
        self._min_h.setSuffix(" px")
        gl3.addRow("Min height:", self._min_h)
        bl.addWidget(gb3)

        # Q4: Color palette
        gb4 = QGroupBox("Color palette (multi-select)")
        gl4 = QVBoxLayout(gb4)
        self._palette = _ChipGroup([p.title() for p in ALL_PALETTES])
        gl4.addWidget(self._palette)
        bl.addWidget(gb4)

        # Q5: Style keywords
        gb5 = QGroupBox("Style keywords (what describes this category)")
        gl5 = QVBoxLayout(gb5)
        self._keywords = _ChipGroup(list(ALL_STYLE_KEYWORDS))
        gl5.addWidget(self._keywords)
        bl.addWidget(gb5)

        # Q6: Exclude keywords
        gb6 = QGroupBox("Exclude keywords (anti-patterns)")
        gl6 = QVBoxLayout(gb6)
        self._exclude = _ChipGroup(list(ALL_STYLE_KEYWORDS))
        gl6.addWidget(self._exclude)
        bl.addWidget(gb6)

        l.addWidget(scroll, 1)

        # Buttons
        btns = QHBoxLayout()
        self._b_ai = QPushButton("Suggest from Samples (AI)")
        self._b_ai.setObjectName("ghost")
        self._b_ai.clicked.connect(self._suggest_from_ai)
        btns.addWidget(self._b_ai)
        btns.addStretch()
        b_cancel = QPushButton("Cancel")
        b_cancel.setObjectName("ghost")
        b_cancel.clicked.connect(self.reject)
        btns.addWidget(b_cancel)
        self._b_save = QPushButton("Save")
        self._b_save.setObjectName("primary")
        self._b_save.clicked.connect(self._save)
        btns.addWidget(self._b_save)
        l.addLayout(btns)

    def _load_current(self):
        """Pre-fill the widgets from the existing `expected` block."""
        cfg = get_expected(self.category)
        self._file_kinds.set_value(cfg.get("file_kinds") or [])
        ars = cfg.get("aspect_ratios") or []
        if "any" in ars:
            ars = ["Any"] + [a for a in ars if a != "any"]
        self._aspect_ratios.set_value(
            [a.title() if a != "any" else "Any" for a in ars]
        )
        self._palette.set_value([p.title() for p in cfg.get("color_palette") or []])
        self._keywords.set_value(cfg.get("style_keywords") or [])
        self._exclude.set_value(cfg.get("exclude_keywords") or [])
        min_res = cfg.get("min_resolution") or [0, 0]
        self._min_w.setValue(int(min_res[0]) if len(min_res) >= 1 else 0)
        self._min_h.setValue(int(min_res[1]) if len(min_res) >= 2 else 0)

    def _collect(self) -> Dict:
        """Build the `expected` dict from the current widget state."""
        ars = self._aspect_ratios.value()
        # Map display labels back to internal keys
        ar_keys = []
        for label in ars:
            for display, key in self._ar_label_to_key.items():
                if display == label:
                    ar_keys.append(key)
                    break
        return {
            "aspect_ratios": ar_keys,
            "file_kinds": self._file_kinds.value(),
            "min_resolution": [self._min_w.value(), self._min_h.value()],
            "color_palette": [p.lower() for p in self._palette.value()],
            "style_keywords": self._keywords.value(),
            "exclude_keywords": self._exclude.value(),
            "source": "user",
        }

    def _save(self):
        cfg = self._collect()
        if not cfg["aspect_ratios"]:
            cfg["aspect_ratios"] = ["any"]
        if not cfg["file_kinds"]:
            cfg["file_kinds"] = ["image"]
        write_expected(self.category, cfg)
        self.accept()

    def _suggest_from_ai(self):
        """Run the AI generator and pre-fill the dialog with its output."""
        try:
            cfg = generate_config_from_samples(self.category, max_samples=20)
        except Exception as e:
            QMessageBox.warning(self, "AI Suggest Failed", str(e))
            return
        # Populate widgets with the AI proposal
        self._file_kinds.set_value(cfg.get("file_kinds") or [])
        ars = cfg.get("aspect_ratios") or ["any"]
        self._aspect_ratios.set_value(
            [a.title() if a != "any" else "Any" for a in ars]
        )
        self._palette.set_value([p.title() for p in cfg.get("color_palette") or []])
        self._keywords.set_value(cfg.get("style_keywords") or [])
        self._exclude.set_value(cfg.get("exclude_keywords") or [])
        min_res = cfg.get("min_resolution") or [0, 0]
        self._min_w.setValue(int(min_res[0]) if len(min_res) >= 1 else 0)
        self._min_h.setValue(int(min_res[1]) if len(min_res) >= 2 else 0)
        QMessageBox.information(
            self, "AI Suggest",
            "AI suggested config based on samples. Review and tweak, "
            "then click Save."
        )


# ---------------------------------------------------------------------------
# AI-only dialog (simpler, just shows the proposal and lets you save)
# ---------------------------------------------------------------------------

class AIConfigDialog(QDialog):
    """Show the AI-generated `expected` proposal for a category.

    Read-only summary; user clicks Save to write it, Cancel to abort.
    """

    def __init__(self, category: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.category = category
        self.setWindowTitle(f"AI Suggest: {category}")
        self.setMinimumSize(480, 480)
        self._build()

    def _build(self):
        l = QVBoxLayout(self)
        l.setContentsMargins(20, 20, 20, 20)
        title = QLabel(f"AI Suggestion for {self.category}")
        title.setObjectName("title")
        l.addWidget(title)

        # Compute AI suggestion.
        try:
            self._cfg = generate_config_from_samples(self.category, max_samples=25)
        except Exception as e:
            l.addWidget(QLabel(f"AI generation failed: {e}"))
            self._cfg = dict(DEFAULT_EXPECTED)

        info = QLabel(
            "The AI analysed sample images in this category and suggests "
            "the configuration below. Click Save to apply, Cancel to discard."
        )
        info.setWordWrap(True)
        info.setObjectName("subtitle")
        l.addWidget(info)

        summary = QLabel(_format_config(self._cfg))
        summary.setWordWrap(True)
        summary.setStyleSheet(
            "QLabel { background: #0a0a0a; border: 1px solid #2a2a2a; "
            "border-radius: 8px; padding: 12px; font-family: monospace; }"
        )
        l.addWidget(summary, 1)

        btns = QHBoxLayout()
        btns.addStretch()
        b_cancel = QPushButton("Cancel")
        b_cancel.setObjectName("ghost")
        b_cancel.clicked.connect(self.reject)
        btns.addWidget(b_cancel)
        b_save = QPushButton("Save")
        b_save.setObjectName("primary")
        b_save.clicked.connect(self._save)
        btns.addWidget(b_save)
        l.addLayout(btns)

    def _save(self):
        write_expected(self.category, self._cfg)
        self.accept()


def _format_config(cfg: Dict) -> str:
    """Render an `expected` dict as a multi-line summary string."""
    lines = []
    lines.append(f"  Aspect ratios:   {', '.join(cfg.get('aspect_ratios') or ['any'])}")
    lines.append(f"  File kinds:      {', '.join(cfg.get('file_kinds') or ['image'])}")
    mr = cfg.get("min_resolution") or [0, 0]
    if mr and (mr[0] or mr[1]):
        lines.append(f"  Min resolution:  {mr[0]}x{mr[1]}")
    else:
        lines.append("  Min resolution:  (no minimum)")
    pal = cfg.get("color_palette") or []
    lines.append(f"  Color palette:   {', '.join(p.title() for p in pal) or '(none)'}")
    kw = cfg.get("style_keywords") or []
    lines.append(f"  Style keywords:  {', '.join(kw) or '(none)'}")
    ex = cfg.get("exclude_keywords") or []
    lines.append(f"  Exclude words:   {', '.join(ex) or '(none)'}")
    lines.append(f"  Source:          {cfg.get('source', 'user')}")
    return "\n".join(lines)
