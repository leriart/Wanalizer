"""Settings page: theme, destination, thresholds, data management."""
import os
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QFormLayout, QGroupBox, QCheckBox, QDoubleSpinBox, QLineEdit,
    QTextEdit, QFileDialog, QMessageBox, QWidget,
)
from ... import settings as s


class SettingsPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self._build()
        self._load()

    def _build(self):
        l = QVBoxLayout(self)
        l.setContentsMargins(24, 24, 24, 24)
        tl = QLabel("Settings")
        tl.setObjectName("title")
        l.addWidget(tl)
        st = QLabel("Configure application preferences.")
        st.setObjectName("subtitle")
        l.addWidget(st)

        # Destination Folder
        g0 = QGroupBox("Destination Folder")
        g0l = QHBoxLayout(g0)
        self._dest_edit = QLineEdit(s.resolve_dest_dir(s.load_settings()))
        self._dest_edit.setMinimumWidth(200)

        def _browse():
            d = QFileDialog.getExistingDirectory(self, "Choose destination", self._dest_edit.text())
            if d:
                self._dest_edit.setText(d)

        b = QPushButton("Browse...")
        b.clicked.connect(_browse)
        g0l.addWidget(self._dest_edit, 1)
        g0l.addWidget(b)
        l.addWidget(g0)

        # Appearance
        g1 = QGroupBox("Appearance")
        gl1 = QHBoxLayout(g1)
        self._cmb_theme = QComboBox()
        self._cmb_theme.addItems(["Dark", "Light"])
        gl1.addWidget(QLabel("Theme:"))
        gl1.addWidget(self._cmb_theme)
        gl1.addStretch()
        l.addWidget(g1)

        # Resolution
        g2 = QGroupBox("Resolution Organization")
        g2l = QVBoxLayout(g2)
        self._cb_res = QCheckBox("Enable resolution-based organization")
        g2l.addWidget(self._cb_res)
        self._bins_edit = QTextEdit()
        self._bins_edit.setFixedHeight(100)
        g2l.addWidget(self._bins_edit)
        l.addWidget(g2)

        # Quality Thresholds
        g3 = QGroupBox("Quality Thresholds")
        gf = QFormLayout(g3)
        self._spin_quality = QDoubleSpinBox()
        self._spin_quality.setRange(0, 1)
        self._spin_quality.setSingleStep(0.05)
        gf.addRow("Min aesthetic score:", self._spin_quality)
        self._spin_lap = QDoubleSpinBox()
        self._spin_lap.setRange(10, 500)
        self._spin_lap.setValue(80)
        self._spin_lap.setSingleStep(10)
        gf.addRow("Min sharpness (Laplacian):", self._spin_lap)
        l.addWidget(g3)

        # NSFW
        g4 = QGroupBox("NSFW Detection")
        gf2 = QFormLayout(g4)
        self._spin_nsfw = QDoubleSpinBox()
        self._spin_nsfw.setRange(0, 1)
        self._spin_nsfw.setSingleStep(0.05)
        self._spin_nsfw.setValue(0.5)
        gf2.addRow("NSFW threshold (CV):", self._spin_nsfw)
        self._cb_ollama_nsfw = QCheckBox("Trust Ollama NSFW scores")
        self._cb_ollama_nsfw.setToolTip(
            "If unchecked, Ollama's NSFW score is ignored entirely.\n"
            "Vision-language models often output ~0.5 as a neutral default\n"
            "for safe content, which produces false positives when used as\n"
            "a hard threshold."
        )
        gf2.addRow(self._cb_ollama_nsfw)
        self._spin_ollama_nsfw = QDoubleSpinBox()
        self._spin_ollama_nsfw.setRange(0.5, 1.0)
        self._spin_ollama_nsfw.setSingleStep(0.05)
        self._spin_ollama_nsfw.setValue(0.70)
        self._spin_ollama_nsfw.setToolTip(
            "Minimum Ollama NSFW score to flag the image.\n"
            "Scores in the 'uncertain zone' (0.40-0.65) are ignored."
        )
        gf2.addRow("Ollama NSFW threshold:", self._spin_ollama_nsfw)
        l.addWidget(g4)

        # Data
        g5 = QGroupBox("Data")
        g5l = QHBoxLayout(g5)
        b_cc = QPushButton("Clear Hash Cache")
        b_cc.setObjectName("ghost")
        b_cc.clicked.connect(self._clear_cache)
        b_reset = QPushButton("Reset All Settings")
        b_reset.setObjectName("danger")
        b_reset.clicked.connect(self._reset)
        g5l.addWidget(b_cc)
        g5l.addWidget(b_reset)
        g5l.addStretch()
        l.addWidget(g5)

        hb = QHBoxLayout()
        btn_save = QPushButton("Save Settings")
        btn_save.setObjectName("primary")
        btn_save.clicked.connect(self._save)
        hb.addWidget(btn_save)
        hb.addStretch()
        l.addLayout(hb)
        l.addStretch()

    def _load(self):
        cfg = s.load_settings()
        self._cmb_theme.setCurrentIndex(0 if cfg.get("theme", "dark") == "dark" else 1)
        self._cb_res.setChecked(cfg.get("by_resolution", False))
        bins = cfg.get("resolution_bins", [])
        self._bins_edit.setPlainText("\n".join(f'{b["name"]} = {b["max_pixels"]}' for b in bins))
        self._spin_quality.setValue(cfg.get("quality_min", 0.0))
        self._spin_lap.setValue(cfg.get("laplacian_min", 80))
        self._spin_nsfw.setValue(cfg.get("nsfw_threshold", 0.5))
        self._cb_ollama_nsfw.setChecked(cfg.get("ollama_nsfw_use", True))
        self._spin_ollama_nsfw.setValue(cfg.get("ollama_nsfw_threshold", 0.70))
        self._dest_edit.setText(cfg.get("dest_dir", "WP"))

    def _save(self):
        cfg = s.load_settings()
        cfg["theme"] = "dark" if self._cmb_theme.currentIndex() == 0 else "light"
        cfg["by_resolution"] = self._cb_res.isChecked()
        cfg["dest_dir"] = self._dest_edit.text().strip()

        bins = []
        for line in self._bins_edit.toPlainText().strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                nm, val = line.rsplit("=", 1)
                px = int(val.strip())
                if px > 0:
                    bins.append({"name": nm.strip(), "max_pixels": px})
            except Exception:
                pass
        if bins:
            cfg["resolution_bins"] = bins

        cfg["quality_min"] = self._spin_quality.value()
        cfg["laplacian_min"] = self._spin_lap.value()
        cfg["nsfw_threshold"] = self._spin_nsfw.value()
        cfg["ollama_nsfw_use"] = self._cb_ollama_nsfw.isChecked()
        cfg["ollama_nsfw_threshold"] = self._spin_ollama_nsfw.value()
        s.save_settings(cfg)
        QMessageBox.information(self, "Saved", "Settings saved.")

    def _clear_cache(self):
        path = s.resolve_hash_cache_path(s.load_settings())
        if os.path.exists(path):
            os.remove(path)
            QMessageBox.information(self, "Done", "Cache cleared.")
        else:
            QMessageBox.information(self, "Info", "No cache file found.")

    def _reset(self):
        r = QMessageBox.question(self, "Reset", "Reset all settings to defaults?", QMessageBox.Yes | QMessageBox.No)
        if r == QMessageBox.Yes:
            if os.path.exists(s.CONFIG_PATH):
                os.remove(s.CONFIG_PATH)
            self._load()
            QMessageBox.information(self, "Done", "Settings reset.")
