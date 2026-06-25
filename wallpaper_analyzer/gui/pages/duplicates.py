"""Duplicates page: scan and remove duplicate wallpapers across the entire dest dir.

Supports:
  - MD5 (byte-identical) duplicate detection
  - Perceptual (image-content-aware) detection: catches the same image
    re-encoded as JPG vs PNG, re-compressed, resized, or saved under
    a different container. Uses dHash + pHash + colour histogram,
    bucketed via LSH on a 16-bit coarse dHash for fast candidate
    filtering.
  - Tiers: exact (MD5), reencode (same content, different bytes),
    resize (different resolution), similar (visually close).
  - Cross-category detection
  - Per-group decision: keep one, move/delete the rest
  - Filter: same-category only
  - Live stats broken down by file kind and tier
  - Background worker with cancellation
"""
import os
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWidget, QGroupBox, QFormLayout, QCheckBox, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QPlainTextEdit, QMessageBox, QProgressBar,
)
from PySide6.QtCore import QTimer, QThread, Signal
from ... import settings as s
from ... import categories as c
from ...duplicates import (
    scan_and_hash, scan_and_hash_perceptual,
    find_duplicate_groups, get_duplicate_stats,
    move_to_duplicates, delete_files, save_hash_cache, load_hash_cache,
    TIER_LABELS, TIER_EXACT, TIER_REENCODE, TIER_RESIZE, TIER_SIMILAR,
)
from ..widgets import BTN_TEXT, BTN_KIND, BTN_DATA, setup_table_buttons, refresh_action_columns


TIER_ORDER = [TIER_EXACT, TIER_REENCODE, TIER_RESIZE, TIER_SIMILAR]


def _collect_all_files(dest_dir: str) -> list:
    """Collect all media files under dest_dir (recursively)."""
    from ...formats import STATIC_EXTENSIONS, ANIMATED_EXTENSIONS
    exts = STATIC_EXTENSIONS | ANIMATED_EXTENSIONS
    out = []
    for d, _, fns in os.walk(dest_dir):
        rel = os.path.relpath(d, dest_dir)
        parts = rel.split(os.sep)
        if any(p.startswith(".") or p in c.SPECIAL_FOLDERS for p in parts):
            continue
        for fn in fns:
            if fn.startswith("."):
                continue
            if os.path.splitext(fn)[1].lower() in exts:
                out.append(os.path.join(d, fn))
    return out


class _ScanCancelled(Exception):
    """Raised internally to abort a running scan_and_hash."""


class DupeScanWorker(QThread):
    """Background thread for scanning + hashing + finding duplicates."""
    log = Signal(str)
    progress = Signal(int, int, str, str)
    stage = Signal(str, str, int, int)
    finished_ok = Signal(list, dict)
    failed = Signal(str)

    def __init__(self, files, same_category_only, perceptual, min_tier, cache):
        super().__init__()
        self.files = files
        self.same_category_only = same_category_only
        self.perceptual = perceptual
        self.min_tier = min_tier
        self.cache = cache
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self.log.emit(f"Scanning {len(self.files)} files...")
            self.stage.emit("MD5", "md5", 0, len(self.files))

            def cb(cur, total, fn, st):
                self.progress.emit(cur, total, os.path.basename(fn), st)
                if self._cancel:
                    raise _ScanCancelled()
            try:
                scan_and_hash(self.files, self.cache, progress_callback=cb, parallel=4)
            except _ScanCancelled:
                self.log.emit("[CANCELLED]")
                return
            if self._cancel:
                return

            if self.perceptual:
                self.log.emit("Computing perceptual hashes...")
                self.stage.emit("Perceptual", "perceptual", 0, len(self.files))
                try:
                    scan_and_hash_perceptual(
                        self.files, self.cache, progress_callback=cb, parallel=4,
                    )
                except _ScanCancelled:
                    self.log.emit("[CANCELLED]")
                    return
                if self._cancel:
                    return

            self.log.emit(f"Finding duplicates (mode={'soft' if self.perceptual else 'hard'}, "
                          f"min_tier={self.min_tier})...")
            self.stage.emit("Grouping", "group", 0, 1)
            groups = find_duplicate_groups(
                self.files, self.cache,
                mode='soft' if self.perceptual else 'hard',
                min_tier=self.min_tier,
                same_category_only=self.same_category_only,
            )
            stats = get_duplicate_stats(groups)
            self.finished_ok.emit(groups, stats)
        except Exception as e:
            self.failed.emit(str(e))


class DuplicatesPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self._groups: list = []
        self._cache: dict = {}
        self._worker: DupeScanWorker = None
        self._build()
        QTimer.singleShot(300, self._load_cache)

    def _build(self):
        l = QVBoxLayout(self)
        l.setContentsMargins(24, 24, 24, 24)

        t = QLabel("Duplicate Detection")
        t.setObjectName("title")
        l.addWidget(t)
        st = QLabel(
            "Find and remove duplicate wallpapers across all categories. "
            "MD5 catches byte-identical copies; perceptual detection "
            "catches the same image re-encoded (JPG\u2194PNG, re-compressed, "
            "resized, different container)."
        )
        st.setObjectName("subtitle")
        st.setWordWrap(True)
        l.addWidget(st)

        opts = QGroupBox("Options")
        ol = QFormLayout(opts)
        self._same_cat = QCheckBox("Restrict to within same category")
        self._same_cat.setChecked(False)
        ol.addRow(self._same_cat)

        self._perceptual = QCheckBox(
            "Perceptual detection (catches re-encoded / resized copies)"
        )
        self._perceptual.setChecked(True)
        ol.addRow(self._perceptual)

        self._min_tier = QComboBox()
        for tier in TIER_ORDER:
            label = TIER_LABELS.get(tier, tier)
            self._min_tier.addItem(label, tier)
        self._min_tier.setCurrentIndex(0)  # exact
        self._min_tier.setToolTip(
            "Strictest tier to report in the perceptual pass.\n"
            "  exact    - byte-identical files (MD5)\n"
            "  reencode - same image, different format / quality\n"
            "  resize   - same image, different resolution\n"
            "  similar  - visually close (broader)"
        )
        self._perceptual.toggled.connect(self._min_tier.setEnabled)
        ol.addRow("Min tier (perceptual):", self._min_tier)
        l.addWidget(opts)

        act = QHBoxLayout()
        self._b_scan = QPushButton("Scan for Duplicates")
        self._b_scan.setObjectName("primary")
        self._b_scan.clicked.connect(self._scan)
        act.addWidget(self._b_scan)
        self._b_cancel = QPushButton("Cancel")
        self._b_cancel.setObjectName("ghost")
        self._b_cancel.setEnabled(False)
        self._b_cancel.clicked.connect(self._cancel)
        act.addWidget(self._b_cancel)
        self._b_clear = QPushButton("Clear Cache")
        self._b_clear.setObjectName("ghost")
        self._b_clear.clicked.connect(self._clear_cache)
        act.addWidget(self._b_clear)
        act.addStretch()
        l.addLayout(act)

        self._progress = QProgressBar()
        self._progress.setFixedHeight(6)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        l.addWidget(self._progress)

        self._stats_lbl = QLabel("No scan yet")
        self._stats_lbl.setObjectName("sectionLabel")
        self._stats_lbl.setWordWrap(True)
        l.addWidget(self._stats_lbl)

        # Columns: # | Files in group (with keep marker) | Tier | Signature | Same size | Action
        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["#", "Files (\u2713 = kept)", "Tier", "Signature",
             "Same size", "Action"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        l.addWidget(self._table, 1)

        act2 = QHBoxLayout()
        self._b_move_all = QPushButton("Move ALL duplicates to Duplicates/")
        self._b_move_all.setObjectName("success")
        self._b_move_all.setEnabled(False)
        self._b_move_all.clicked.connect(self._move_all)
        act2.addWidget(self._b_move_all)
        self._b_delete_all = QPushButton("Delete ALL duplicates (permanent)")
        self._b_delete_all.setObjectName("danger")
        self._b_delete_all.setEnabled(False)
        self._b_delete_all.clicked.connect(self._delete_all)
        act2.addWidget(self._b_delete_all)
        act2.addStretch()
        l.addLayout(act2)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(300)
        self._log.setFixedHeight(120)
        l.addWidget(self._log)

    def _load_cache(self):
        self._cache = load_hash_cache()
        n = len(self._cache)
        if n:
            self._log.appendPlainText(f"Loaded {n} cached hashes from previous scans")

    def _clear_cache(self):
        r = QMessageBox.question(
            self, "Clear cache",
            "Clear the hash cache? Next scan will re-hash all files.",
            QMessageBox.Yes | QMessageBox.No)
        if r == QMessageBox.Yes:
            try:
                cache_path = s.resolve_hash_cache_path(s.load_settings())
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                self._cache = {}
                self._log.appendPlainText("Cache cleared")
            except Exception as e:
                self._log.appendPlainText(f"[ERR] {e}")

    def _scan(self):
        if self._worker and self._worker.isRunning():
            return
        dest = s.resolve_dest_dir(s.load_settings())
        if not os.path.isdir(dest):
            QMessageBox.warning(self, "No destination",
                                f"Destination directory not found:\n{dest}")
            return
        files = _collect_all_files(dest)
        if not files:
            QMessageBox.information(self, "No files", "No image files found in destination.")
            return
        self._table.setRowCount(0)
        self._stats_lbl.setText("Scanning...")
        self._b_scan.setEnabled(False)
        self._b_cancel.setEnabled(True)
        self._progress.setVisible(True)
        self._progress.setMaximum(len(files))
        self._progress.setValue(0)
        perceptual = self._perceptual.isChecked()
        min_tier = self._min_tier.currentData() or TIER_EXACT
        mode_label = "MD5 + perceptual" if perceptual else "MD5 only"
        self._log.appendPlainText(
            f"Starting {mode_label} scan of {len(files)} files "
            f"(min_tier={min_tier})..."
        )
        self._worker = DupeScanWorker(
            files,
            self._same_cat.isChecked(),
            perceptual,
            min_tier,
            self._cache,
        )
        self._worker.log.connect(self._log.appendPlainText)
        self._worker.progress.connect(self._on_progress)
        self._worker.stage.connect(self._on_stage)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._log.appendPlainText("[cancel] Requesting cancel...")

    def _on_progress(self, cur, total, fn, st):
        if total > 0:
            self._progress.setMaximum(total)
            self._progress.setValue(cur)
        if cur % 25 == 0 or cur == total:
            self._log.appendPlainText(f"  [{cur}/{total}] {fn}: {st}")

    def _on_stage(self, label, name, cur, total):
        # Stage labels are emitted by the worker (MD5 / Perceptual /
        # Grouping). We surface them as section headers in the log so
        # the user knows which phase is running.
        if cur == 0:
            self._log.appendPlainText(f"[stage] {label}")

    def _on_done(self, groups, stats):
        self._groups = groups
        save_hash_cache(self._cache)
        self._progress.setVisible(False)
        self._b_scan.setEnabled(True)
        self._b_cancel.setEnabled(False)
        self._b_move_all.setEnabled(len(groups) > 0)
        self._b_delete_all.setEnabled(len(groups) > 0)
        self._stats_lbl.setText(_format_stats(stats))
        self._log.appendPlainText(f"[OK] {_format_stats(stats)}")
        self._refresh_table(groups)

    def _on_fail(self, msg):
        self._log.appendPlainText(f"[ERR] {msg}")
        self._progress.setVisible(False)
        self._b_scan.setEnabled(True)
        self._b_cancel.setEnabled(False)

    def _refresh_table(self, groups):
        dest = s.resolve_dest_dir(s.load_settings())
        self._table.setRowCount(len(groups))
        for i, g in enumerate(groups):
            num = QTableWidgetItem(str(i + 1))
            self._table.setItem(i, 0, num)

            files_lines = []
            for j, f in enumerate(g["files"]):
                rel = os.path.relpath(f, dest)
                marker = " \u2713" if j == 0 else ""
                files_lines.append(f"{rel}{marker}")
            files_item = QTableWidgetItem("\n".join(files_lines))
            files_item.setToolTip("\n".join(files_lines))
            self._table.setItem(i, 1, files_item)

            tier = g.get("tier", TIER_EXACT)
            tier_item = QTableWidgetItem(TIER_LABELS.get(tier, tier))
            tier_item.setToolTip(tier)
            self._table.setItem(i, 2, tier_item)

            md5_text = g.get("md5") or ""
            if md5_text:
                sig_text = md5_text[:12] + ("..." if len(md5_text) > 12 else "")
                sig_tooltip = "MD5: " + md5_text
            elif g.get("score") is not None:
                sig_text = f"{g['score']:.2f}"
                sig_tooltip = "Perceptual similarity score"
            else:
                sig_text = "\u2014"
                sig_tooltip = "(no signature)"
            sig_item = QTableWidgetItem(sig_text)
            sig_item.setToolTip(sig_tooltip)
            self._table.setItem(i, 3, sig_item)

            sm = QTableWidgetItem("Yes" if g.get("size_match") else "No")
            self._table.setItem(i, 4, sm)

            act = QTableWidgetItem()
            act.setData(BTN_TEXT, "Move|Del")
            act.setData(BTN_KIND, "action|del")
            act.setData(BTN_DATA, str(i))
            self._table.setItem(i, 5, act)

        refresh_action_columns(self._table)
        self._table.resizeRowsToContents()

        def _on_action(r, c, idx_str):
            idx = int(idx_str)
            if c == 0:
                self._move_group(idx)
            elif c == 1:
                self._delete_group(idx)

        setup_table_buttons(self._table, {"action": _on_action, "del": _on_action}, [5])

    def _move_group(self, idx):
        if idx >= len(self._groups):
            return
        g = self._groups[idx]
        to_move = g["files"][1:]
        if not to_move:
            QMessageBox.information(self, "No duplicates",
                                    "This group has only one file (no duplicates to move).")
            return
        kept = os.path.basename(g["files"][0])
        r = QMessageBox.question(
            self, "Move duplicates",
            f"Move {len(to_move)} duplicate(s) to Duplicates/ ?\n"
            f"Kept (highest resolution): {kept}",
            QMessageBox.Yes | QMessageBox.No)
        if r != QMessageBox.Yes:
            return
        n = move_to_duplicates(to_move)
        self._log.appendPlainText(f"[move] Group {idx + 1}: moved {n} duplicate(s) to Duplicates/")
        del self._groups[idx]
        self._refresh_table(self._groups)
        self._stats_lbl.setText(_format_stats(get_duplicate_stats(self._groups)))

    def _delete_group(self, idx):
        if idx >= len(self._groups):
            return
        g = self._groups[idx]
        to_del = g["files"][1:]
        if not to_del:
            return
        r = QMessageBox.question(
            self, "Delete duplicates",
            f"Permanently delete {len(to_del)} duplicate file(s)?\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No)
        if r != QMessageBox.Yes:
            return
        n = delete_files(to_del)
        self._log.appendPlainText(f"[delete] Group {idx + 1}: deleted {n} file(s)")
        del self._groups[idx]
        self._refresh_table(self._groups)
        self._stats_lbl.setText(_format_stats(get_duplicate_stats(self._groups)))

    def _move_all(self):
        if not self._groups:
            return
        all_dupes = []
        for g in self._groups:
            all_dupes.extend(g["files"][1:])
        r = QMessageBox.question(
            self, "Move all",
            f"Move {len(all_dupes)} duplicate file(s) to Duplicates/ ?",
            QMessageBox.Yes | QMessageBox.No)
        if r != QMessageBox.Yes:
            return
        n = move_to_duplicates(all_dupes)
        self._log.appendPlainText(f"[move all] Moved {n} duplicate(s) to Duplicates/")
        self._groups = []
        self._refresh_table([])
        self._stats_lbl.setText("All duplicates moved")

    def _delete_all(self):
        if not self._groups:
            return
        all_dupes = []
        for g in self._groups:
            all_dupes.extend(g["files"][1:])
        r = QMessageBox.warning(
            self, "Delete ALL",
            f"Permanently delete {len(all_dupes)} duplicate file(s)?\n"
            "This CANNOT be undone!",
            QMessageBox.Yes | QMessageBox.No)
        if r != QMessageBox.Yes:
            return
        n = delete_files(all_dupes)
        self._log.appendPlainText(f"[delete all] Deleted {n} duplicate(s)")
        self._groups = []
        self._refresh_table([])
        self._stats_lbl.setText("All duplicates deleted")


def _format_stats(stats: dict) -> str:
    by_kind = stats.get("by_kind", {})
    kind_parts = []
    for ext in sorted(by_kind):
        kind_parts.append(f"{by_kind[ext]} {ext.lstrip('.')}")
    kind_breakdown = " [" + ", ".join(kind_parts) + "]" if kind_parts else ""
    by_tier = stats.get("by_tier", {})
    tier_parts = []
    for tier in TIER_ORDER:
        if tier in by_tier:
            tier_parts.append(f"{TIER_LABELS.get(tier, tier)}: {by_tier[tier]}")
    tier_breakdown = " (" + ", ".join(tier_parts) + ")" if tier_parts else ""
    wasted_mb = stats.get("wasted_mb", 0)
    return (
        f"{stats['total_groups']} duplicate groups, "
        f"{stats['total_duplicates']} duplicate files"
        f"{tier_breakdown}{kind_breakdown}, "
        f"~{wasted_mb:.1f} MB wasted"
    )
