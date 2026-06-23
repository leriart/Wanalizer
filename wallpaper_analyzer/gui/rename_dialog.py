"""Rename dialog: select strategy, preview changes, apply to files."""
import os
from typing import List, Tuple
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QGroupBox, QFormLayout, QSpinBox, QCheckBox, QMessageBox,
)
from PySide6.QtCore import Qt
from ..rename import build_renames, apply_renames, RENAME_STRATEGIES


class RenameDialog(QDialog):
    """Dialog to rename a list of files using various strategies.

    Shows a preview table with before/after, lets the user pick
    a strategy, and applies on confirmation.
    """

    def __init__(self, files: List[str], category: str = "", parent=None):
        super().__init__(parent)
        self.files = files
        self.category = category
        self.pairs: List[Tuple[str, str]] = []
        self.setWindowTitle(f"Rename {len(files)} Files")
        self.setMinimumSize(720, 540)
        self._build()
        self._apply_strategy()

    def _build(self):
        l = QVBoxLayout(self)
        l.setContentsMargins(16, 16, 16, 16)

        title = QLabel(f"Rename {len(self.files)} files")
        title.setObjectName("title")
        l.addWidget(title)
        if self.category:
            sub = QLabel(f"Category: {self.category}")
            sub.setObjectName("subtitle")
            l.addWidget(sub)

        # Strategy selector
        sg = QGroupBox("Strategy")
        sgl = QFormLayout(sg)
        self._strat = QComboBox()
        for key, label, desc in RENAME_STRATEGIES:
            self._strat.addItem(f"{label}  -  {desc}", key)
        self._strat.currentIndexChanged.connect(self._apply_strategy)
        sgl.addRow("Type:", self._strat)

        # Options (per strategy)
        self._opt_pad = QSpinBox()
        self._opt_pad.setRange(2, 8)
        self._opt_pad.setValue(3)
        self._opt_pad.valueChanged.connect(self._apply_strategy)
        sgl.addRow("Zero-pad width:", self._opt_pad)

        self._opt_start = QSpinBox()
        self._opt_start.setRange(0, 9999)
        self._opt_start.setValue(1)
        self._opt_start.valueChanged.connect(self._apply_strategy)
        sgl.addRow("Start number:", self._opt_start)

        self._opt_truncate = QSpinBox()
        self._opt_truncate.setRange(8, 200)
        self._opt_truncate.setValue(32)
        self._opt_truncate.valueChanged.connect(self._apply_strategy)
        sgl.addRow("Max length (truncate):", self._opt_truncate)

        self._dry = QCheckBox("Dry run (preview only, don't rename)")
        self._dry.setChecked(False)
        sgl.addRow(self._dry)
        l.addWidget(sg)

        # Preview table
        pt = QLabel("Preview (first 200 files):")
        pt.setObjectName("sectionLabel")
        l.addWidget(pt)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Original", "New name"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        l.addWidget(self._table, 1)

        # Stats
        self._stats = QLabel("")
        self._stats.setObjectName("statSmall")
        l.addWidget(self._stats)

        # Buttons
        bb = QHBoxLayout()
        bb.addStretch()
        b_cancel = QPushButton("Cancel")
        b_cancel.setObjectName("ghost")
        b_cancel.clicked.connect(self.reject)
        bb.addWidget(b_cancel)
        self._b_apply = QPushButton("Apply")
        self._b_apply.setObjectName("primary")
        self._b_apply.clicked.connect(self._on_apply)
        bb.addWidget(self._b_apply)
        l.addLayout(bb)

    def _apply_strategy(self):
        """Rebuild preview for the selected strategy."""
        strategy = self._strat.currentData()
        if not strategy:
            return
        try:
            self.pairs = build_renames(
                self.files,
                strategy=strategy,
                category=self.category,
                start=self._opt_start.value(),
                pad=self._opt_pad.value(),
                truncate_len=self._opt_truncate.value(),
            )
        except Exception as e:
            self._stats.setText(f"Error building renames: {e}")
            return
        self._refresh_table()

    def _refresh_table(self):
        """Update the preview table."""
        # Show up to 200 entries
        visible = self.pairs[:200]
        self._table.setRowCount(len(visible))
        changed = sum(1 for o, n in self.pairs if o != n)
        unchanged = len(self.pairs) - changed
        collisions = sum(1 for o, n in self.pairs if n in [p[1] for p in self.pairs if p[0] != o])
        self._stats.setText(
            f"{changed} will be renamed, {unchanged} unchanged"
            + (f" (showing first 200 of {len(self.pairs)})" if len(self.pairs) > 200 else "")
            + (f"  |  {collisions} potential collisions" if collisions else "")
        )
        for i, (old, new) in enumerate(visible):
            old_name = os.path.basename(old)
            new_name = os.path.basename(new)
            old_item = QTableWidgetItem(old_name)
            new_item = QTableWidgetItem(new_name)
            if old == new:
                # Unchanged
                old_item.setForeground(Qt.GlobalColor.gray)
                new_item.setForeground(Qt.GlobalColor.gray)
            else:
                # Will be renamed
                new_item.setForeground(Qt.GlobalColor.green)
            self._table.setItem(i, 0, old_item)
            self._table.setItem(i, 1, new_item)
        self._table.resizeRowsToContents()

    def _on_apply(self):
        """Apply the rename operation."""
        if not self.pairs:
            return
        n_changed = sum(1 for o, n in self.pairs if o != n)
        if n_changed == 0:
            QMessageBox.information(self, "No changes", "Nothing to rename.")
            return
        msg = f"Rename {n_changed} files?"
        if self._dry.isChecked():
            msg += "\n\n(Dry run - no files will be modified)"
        r = QMessageBox.question(self, "Confirm rename", msg,
                                 QMessageBox.Yes | QMessageBox.No)
        if r != QMessageBox.Yes:
            return
        stats = apply_renames(self.pairs, dry_run=self._dry.isChecked())
        msg_parts = [f"Renamed: {stats['renamed']}"]
        if stats["skipped"]:
            msg_parts.append(f"Skipped: {stats['skipped']}")
        if stats["errors"]:
            msg_parts.append(f"Errors: {stats['errors']}")
            if stats["error_list"]:
                msg_parts.append("\nFirst errors:\n" + "\n".join(stats["error_list"][:5]))
        QMessageBox.information(self, "Result", "\n".join(msg_parts))
        if not self._dry.isChecked():
            self.accept()
