"""Reorganize page: visual file browser with thumbnails for all media types.

Supports: JPEG, PNG, GIF, APNG, WebP, BMP, TIFF, AVIF, HEIC, RAW (CR2/NEF),
PSD, SVG, PDF, MP4, WebM, MKV, AVI, MOV, and more.

Performance notes
-----------------
Thumbnails are generated in a background QThread so the UI never freezes,
even on libraries with thousands of files or when ffmpeg is extracting
video frames. After a move or delete, only the affected items are removed
from the grid (and the sidebar counts are patched) - the rest of the
library stays loaded and is not re-scanned.

Features
--------
* Browse a category or "All files" with a thumbnail grid
* Search files by name (debounced)
* Filter by type: All / Images / Videos / Animated
* Sort by name, date, size, or type
* Multi-select with bulk move / delete via the context menu
* Drag files from the grid onto a category in the sidebar to move them
* Undo the last move(s) - one click reverses the most recent move
"""
import os, shutil, subprocess, tempfile, queue

from PIL import Image, ImageOps
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QSplitter, QListWidget, QListWidgetItem, QMenu, QMessageBox,
    QSlider, QWidget, QCheckBox, QDialog,
    QLineEdit, QComboBox, QProgressBar, QAbstractItemView,
    QSpinBox, QGroupBox, QFormLayout,
)
from PySide6.QtCore import Qt, QTimer, QSize, QThread, Signal, QMimeData
from PySide6.QtGui import QPixmap, QImage, QIcon, QPainter, QColor, QFont, QPen

from ... import settings as s
from ... import formats as f
from ... import categories as c
from ...category_config import aspect_ratio_class, get_expected
from ...rename import RENAME_STRATEGIES, TAG_BASED_STRATEGIES, compute_renames, apply_renames

VIDEO_EXTS = {".mp4", ".m4v", ".webm", ".mkv", ".avi", ".mov", ".flv",
              ".mpg", ".mpeg", ".mpe", ".mpv", ".ogv", ".wmv", ".asf",
              ".ts", ".m2ts", ".mts", ".vob", ".3gp", ".3gpp", ".rm",
              ".rmvb", ".ogm"}
ANIM_EXTS = {".gif", ".apng", ".mng", ".fli", ".flc"}
ANIMATED_WEBP = ".webp"


def _has_ffmpeg():
    return shutil.which("ffmpeg") is not None


def _file_type(path):
    """Return 'video', 'animated', 'image', or 'unknown'."""
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in ANIM_EXTS:
        return "animated"
    if ext in (f.STATIC_EXTENSIONS | f.PLUGIN_EXTENSIONS) or ext == ANIMATED_WEBP:
        return "image"
    return "unknown"


def _video_thumb(src, size):
    """Extract a thumbnail frame from a video file using ffmpeg."""
    if not _has_ffmpeg():
        return None
    try:
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vframes", "1",
             "-s", f"{size}x{size}", tmp],
            capture_output=True, timeout=10,
        )
        if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            img = Image.open(tmp).convert("RGB")
            os.remove(tmp)
            return img
        else:
            if os.path.exists(tmp):
                os.remove(tmp)
            return None
    except Exception:
        return None


def _make_pixmap(source, size):
    """Generate a thumbnail QPixmap from any media file path.

    Always returns a *square* QPixmap of exactly ``size x size`` so the
    grid renders with consistent proportions regardless of source aspect
    ratio. Non-square sources are letterboxed on a neutral dark background
    rather than stretched or scaled non-uniformly.

    Handles images, animated GIFs (first frame), videos (ffmpeg frame),
    and formats requiring plugins (RAW, PSD, AVIF, HEIC, etc.).
    Returns (QPixmap, format_label) or (None, error_msg).
    """
    ftype = _file_type(source)
    img = None
    err = None

    if ftype == "video":
        img = _video_thumb(source, size)
        if img is None and not _has_ffmpeg():
            err = "no ffmpeg"
        elif img is None:
            err = "ffmpeg failed"

    if img is None:
        try:
            pil_img = Image.open(source)
            if getattr(pil_img, "is_animated", False) or getattr(pil_img, "n_frames", 1) > 1:
                pil_img.seek(0)
            pil_img = ImageOps.exif_transpose(pil_img)
            pil_img = pil_img.convert("RGB")
            img = pil_img
        except Exception as exc:
            err = err or "format error"

    if img is None:
        # Always return a square placeholder of the requested size so the
        # grid stays aligned. Caller decides how to render the error
        # overlay (see ``_error_icon``).
        return _error_pixmap(size, err or "error"), ftype

    try:
        # Downscale preserving aspect ratio (never upscale).
        img.thumbnail((size, size), Image.LANCZOS)
        iw, ih = img.size

        # Compose onto a square canvas so the QPixmap is exactly size×size.
        canvas = Image.new("RGB", (size, size), (10, 10, 10))
        ox = (size - iw) // 2
        oy = (size - ih) // 2
        canvas.paste(img, (ox, oy))

        data = canvas.tobytes("raw", "RGB")
        qi = QImage(data, size, size, size * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qi)
        return pix, ftype
    except Exception as e:
        return _error_pixmap(size, str(e)), ftype


def _error_pixmap(size, msg):
    """Square placeholder used when thumbnail generation fails.

    Renders a darker red-tinted square with an exclamation glyph so the
    user can spot broken files at a glance instead of guessing whether a
    dark square is 'loading' or 'broken'.
    """
    pix = QPixmap(size, size)
    pix.fill(QColor("#1a0808"))
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    # Inset border
    p.setPen(QColor("#7a1f1f"))
    p.drawRect(0, 0, size - 1, size - 1)
    # Diagonal hatch lines so a black video doesn't camouflage
    pen = QPen(QColor("#3a1010"))
    pen.setWidth(1)
    p.setPen(pen)
    for x in range(-size, size * 2, 8):
        p.drawLine(x, 0, x + size, size)
    # Centered error glyph
    font = QFont()
    font.setBold(True)
    font.setPixelSize(max(14, size // 4))
    p.setFont(font)
    p.setPen(QColor("#e04040"))
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "!")
    # Tag below glyph (small, optional)
    if msg and size >= 80:
        small = QFont()
        small.setPixelSize(max(8, size // 10))
        p.setFont(small)
        p.setPen(QColor("#7a1f1f"))
        p.drawText(
            pix.rect().adjusted(0, size // 3, 0, 0),
            Qt.AlignmentFlag.AlignCenter,
            (msg or "error")[:8],
        )
    p.end()
    return pix


def _draw_badge(pix, label, color):
    """Draw a small badge text on the bottom-right of a pixmap."""
    if label == "image":
        return pix
    w, h = pix.width(), pix.height()
    result = QPixmap(w, h)
    result.fill(QColor(0, 0, 0, 0))
    p = QPainter(result)
    p.drawPixmap(0, 0, pix)

    font = QFont()
    font.setBold(True)
    font.setPixelSize(max(9, w // 12))
    p.setFont(font)
    p.setPen(QColor(color))
    text = {"video": "VID", "animated": "GIF"}.get(label, label.upper()[:4])
    fm = p.fontMetrics()
    tw = fm.horizontalAdvance(text) + 6
    th = fm.height() + 4
    bx = w - tw - 2
    by = h - th - 2
    p.fillRect(bx, by, tw, th, QColor(0, 0, 0, 200))
    p.setPen(QColor(color))
    p.drawText(bx + 3, by + fm.ascent() + 1, text)
    p.end()
    return result


def _placeholder_icon(label, size):
    """Return a subtle placeholder icon shown while a thumbnail is loading."""
    pix = QPixmap(size, size)
    pix.fill(QColor("#0a0a0a"))
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setPen(QColor("#333333"))
    f = QFont()
    f.setBold(True)
    f.setPixelSize(max(9, size // 6))
    p.setFont(f)
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, (label or "?").upper()[:4])
    p.setPen(QColor("#1a1a1a"))
    p.drawRect(0, 0, size - 1, size - 1)
    p.end()
    return QIcon(pix)


def _human_size(n):
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


# ---------------------------------------------------------------------------
# Background thumbnail worker
# ---------------------------------------------------------------------------

class ThumbWorker(QThread):
    """Generates thumbnails off the GUI thread.

    The page calls `request(path, size, token)` for every file it wants
    shown. The worker decodes the file with PIL (and ffmpeg for videos),
    composes the badge, and emits `icon_ready(path, size, token, icon)`.
    The page ignores signals whose token no longer matches its current
    generation, so a category switch or zoom change cleanly supersedes
    any in-flight work without having to track every pending request.

    Each request carries a generation token. When the page calls
    `drain()` (e.g. after a category switch), the worker empties its
    pending queue so old requests never get processed - this prevents
    a long backlog of stale requests (each of which may take seconds
    for a video via ffmpeg) from blocking the new view's thumbnails.
    """
    icon_ready = Signal(str, int, str, object)  # path, size, token, QIcon|None

    def __init__(self):
        super().__init__()
        self._queue: "queue.Queue" = queue.Queue()
        self._stop = False

    def request(self, path: str, size: int, token: str):
        self._queue.put((path, size, token))

    def drain(self):
        """Empty the pending queue. In-flight work continues but new
        pending requests are dropped. The next `get()` will see an
        empty queue and the worker idle-loop will keep spinning on
        `_queue.get(timeout=0.25)` until the next real request comes in.
        """
        dropped = 0
        try:
            while True:
                self._queue.get_nowait()
                dropped += 1
        except queue.Empty:
            pass
        return dropped

    def stop(self):
        self._stop = True
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def run(self):
        while not self._stop:
            try:
                path, size, token = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                pix, ftype = _make_pixmap(path, size)
                # _make_pixmap always returns a pixmap (error pixmap on failure)
                # but we still guard against None in case of catastrophic
                # failure (out-of-memory, etc.).
                if pix is None:
                    self.icon_ready.emit(path, size, token, None)
                    continue
                if ftype != "image":
                    pix = _draw_badge(
                        pix, ftype,
                        "#ff0000" if ftype == "video" else "#cc6600",
                    )
                self.icon_ready.emit(path, size, token, QIcon(pix))
            except Exception:
                self.icon_ready.emit(path, size, token, None)


# ---------------------------------------------------------------------------
# Drag-and-drop helpers
# ---------------------------------------------------------------------------


class _RenameJob(QThread):
    """Background worker: compute rename pairs and apply them.

    For tag-based strategies, tags are detected per file (cached). For move
    jobs, files are first copied to the target directory then renamed to
    keep the destination's existing index pattern intact.

    Signals:
        progress(int, int)  - (done, total)
        finished_ok(dict)   - {'pairs': [...], 'renamed': n, 'errors': n, ...}
        failed(str)         - error message
    """
    progress = Signal(int, int)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, paths, target_dir, strategy, category, max_tags):
        super().__init__()
        self.paths = list(paths)
        self.target_dir = target_dir  # None for in-place rename
        self.strategy = strategy
        self.category = category or ""
        self.max_tags = max_tags

    def run(self):
        try:
            from ... import rename as _rename
            # Build pairs in chunks so we can emit progress for big batches.
            tags_by_file = {}
            subject_by_file = {}
            if self.strategy in _rename.TAG_BASED_STRATEGIES:
                total = len(self.paths)
                for i, p in enumerate(self.paths, 1):
                    tags, subject = _rename.get_tags_for_file(
                        p, category=self.category, max_tags=self.max_tags,
                    )
                    tags_by_file[p] = tags
                    subject_by_file[p] = subject
                    if i % 5 == 0 or i == total:
                        self.progress.emit(i, total)

            pairs = _rename.build_renames(
                self.paths,
                strategy=self.strategy,
                category=self.category,
                max_tags=self.max_tags,
                tags_by_file=tags_by_file,
                subject_by_file=subject_by_file,
            )

            moved = 0
            renamed = 0
            errors: List[str] = []
            # If target_dir is given, the new filename is computed against the
            # SOURCE directory by build_renames; relocate it to target_dir.
            final_pairs = []
            for old, new in pairs:
                if self.target_dir:
                    new = os.path.join(self.target_dir, os.path.basename(new))
                final_pairs.append((old, new))

            for i, (old, new) in enumerate(final_pairs, 1):
                try:
                    if old == new:
                        continue
                    os.makedirs(os.path.dirname(new), exist_ok=True)
                    if os.path.exists(new):
                        base, ext = os.path.splitext(new)
                        counter = 1
                        candidate = f"{base}_{counter}{ext}"
                        while os.path.exists(candidate):
                            counter += 1
                            candidate = f"{base}_{counter}{ext}"
                        new = candidate
                    if os.path.dirname(old) != os.path.dirname(new):
                        shutil.move(old, new)
                        moved += 1
                    else:
                        os.rename(old, new)
                    renamed += 1
                    if i % 10 == 0 or i == len(final_pairs):
                        self.progress.emit(i, len(final_pairs))
                except Exception as e:
                    errors.append(f"{os.path.basename(old)}: {e}")
            self.finished_ok.emit({
                "pairs": final_pairs,
                "renamed": renamed,
                "moved": moved,
                "errors": len(errors),
                "error_list": errors[:10],
                "strategy": self.strategy,
            })
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class DragGridList(QListWidget):
    """Grid that exports dragged items as a custom MIME with file paths."""
    PATHS_MIME = "application/x-wanalizer-paths"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

    def mimeTypes(self):
        return [self.PATHS_MIME]

    def mimeData(self, items):
        md = QMimeData()
        paths = []
        seen = set()
        for it in items:
            d = it.data(Qt.ItemDataRole.UserRole)
            if d and d not in seen:
                seen.add(d)
                paths.append(str(d))
        md.setData(self.PATHS_MIME, "\n".join(paths).encode("utf-8"))
        return md


class DropCatList(QListWidget):
    """Sidebar list that accepts drops of file paths from DragGridList."""
    files_dropped = Signal(str, list)  # target_category, list_of_paths

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._hover_row = -1

    def dragEnterEvent(self, event):
        md = event.mimeData()
        if md.hasFormat(DragGridList.PATHS_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(DragGridList.PATHS_MIME):
            event.acceptProposedAction()
            row = self.row(self.itemAt(event.position().toPoint()))
            if row != self._hover_row:
                self._hover_row = row
                self.viewport().update()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._hover_row = -1
        self.viewport().update()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        md = event.mimeData()
        if not md.hasFormat(DragGridList.PATHS_MIME):
            event.ignore()
            return
        item = self.itemAt(event.position().toPoint())
        if not item:
            event.ignore()
            self._hover_row = -1
            self.viewport().update()
            return
        cat = item.data(Qt.ItemDataRole.UserRole) or ""
        if not cat:
            # Dropped on the "All files" row - reject.
            event.ignore()
            self._hover_row = -1
            self.viewport().update()
            return
        try:
            data = bytes(md.data(DragGridList.PATHS_MIME)).decode("utf-8")
        except Exception:
            data = ""
        paths = [p for p in data.split("\n") if p]
        if not paths:
            event.ignore()
            return
        self.files_dropped.emit(cat, paths)
        event.acceptProposedAction()
        self._hover_row = -1
        self.viewport().update()


# ---------------------------------------------------------------------------
# ReorganizePage
# ---------------------------------------------------------------------------

class ReorganizePage(QWidget):
    SORT_MODES = [
        ("name_asc",  "Name (A to Z)"),
        ("name_desc", "Name (Z to A)"),
        ("date_desc", "Newest first"),
        ("date_asc",  "Oldest first"),
        ("size_desc", "Largest first"),
        ("size_asc",  "Smallest first"),
        ("type",      "Type"),
    ]

    TYPE_FILTERS = [
        ("all",      "All"),
        ("image",    "Images"),
        ("video",    "Videos"),
        ("animated", "Animated"),
    ]

    ASPECT_FILTERS = [
        ("any",       "Any aspect"),
        ("horizontal","Horizontal"),
        ("vertical",  "Vertical"),
        ("square",    "Square"),
    ]

    def __init__(self, main):
        super().__init__()
        self.main = main
        self._cur_cat = ""
        self._thumb_size = 140
        self._thumb_cache: dict = {}

        self._all_files: list = []      # raw scan: list of (fname, fp, cat, mtime, size, ftype)
        self._visible: list = []        # after filter+sort
        self._items: dict = {}          # path -> QListWidgetItem
        self._generation = 0            # bumped on every refresh; stale thumbs dropped
        self._pending_thumbs = 0
        self._undo_stack: list = []     # list of (src_path, dst_path)
        self._search_text = ""
        self._type_filter = "all"
        self._aspect_filter = "any"
        self._sort_mode = "name_asc"

        self._build()

        self._thumb_worker = ThumbWorker()
        self._thumb_worker.icon_ready.connect(self._on_thumb_ready)
        self._thumb_worker.start()

        try:
            from PySide6.QtWidgets import QApplication
            QApplication.instance().aboutToQuit.connect(self._thumb_worker.stop)
        except Exception:
            pass

        QTimer.singleShot(300, self._load_cats)

    # ------------------------ UI BUILD ------------------------

    def _build(self):
        l = QVBoxLayout(self)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(0)

        # ----- Header -----
        hdr = QFrame()
        hdr.setObjectName("card")
        hdr.setStyleSheet(
            "QFrame#card { background: #0a0a0a; border: none;"
            " border-bottom: 1px solid #1a1a1a; border-radius: 0;"
            " padding: 0px; }"
        )
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(24, 16, 24, 12)
        hl.setSpacing(10)

        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        t = QLabel("Reorganize")
        t.setObjectName("title")
        title_box.addWidget(t)
        self._dest_label = QLabel("Destination: -")
        self._dest_label.setObjectName("statSmall")
        self._dest_label.setStyleSheet("color: #666666;")
        title_box.addWidget(self._dest_label)
        hl.addLayout(title_box)
        hl.addStretch()

        self._undo_btn = QPushButton("Undo")
        self._undo_btn.setObjectName("ghost")
        self._undo_btn.setToolTip("Reverse the most recent move")
        self._undo_btn.clicked.connect(self._on_undo)
        self._undo_btn.setEnabled(False)
        hl.addWidget(self._undo_btn)

        self.cb_rename = QCheckBox("Rename on move")
        self.cb_rename.setChecked(True)
        self.cb_rename.setToolTip(
            "When moving files, also rename them according to the "
            "Strategy and Max tags settings below."
        )
        hl.addWidget(self.cb_rename)

        self.btn_rename = QPushButton("Rename dialog...")
        self.btn_rename.setObjectName("primary")
        self.btn_rename.setToolTip(
            "Open the full Rename dialog (preview, dry-run, batch options)."
        )
        self.btn_rename.clicked.connect(self._on_rename)
        self.btn_rename.setEnabled(False)
        hl.addWidget(self.btn_rename)

        self.btn_rename_only = QPushButton("Rename only")
        self.btn_rename_only.setObjectName("success")
        self.btn_rename_only.setToolTip(
            "Rename currently visible/selected files in place — "
            "without moving them — using the Strategy below."
        )
        self.btn_rename_only.clicked.connect(self._on_rename_only)
        self.btn_rename_only.setEnabled(False)
        hl.addWidget(self.btn_rename_only)

        self.btn_ai_rename = QPushButton("AI Rename...")
        self.btn_ai_rename.setObjectName("success")
        self.btn_ai_rename.setToolTip(
            "AI-powered rename: pick a backend (Ollama / CLIP / "
            "Heuristic / Auto) and rename the currently visible "
            "files using AI-detected content tags. Supports preview "
            "and dry-run."
        )
        self.btn_ai_rename.clicked.connect(self._on_ai_rename)
        self.btn_ai_rename.setEnabled(False)
        hl.addWidget(self.btn_ai_rename)

        self.btn_ai_rename_cat = QPushButton("AI Rename category...")
        self.btn_ai_rename_cat.setObjectName("primary")
        self.btn_ai_rename_cat.setToolTip(
            "Pick a category in the sidebar, then click this to rename "
            "EVERY file in that category at once using AI-detected tags. "
            "Opens the AI Rename dialog with the full category pre-loaded."
        )
        self.btn_ai_rename_cat.clicked.connect(self._on_ai_rename_category)
        self.btn_ai_rename_cat.setEnabled(False)
        hl.addWidget(self.btn_ai_rename_cat)

        self.btn_dupes = QPushButton("Find Duplicates")
        self.btn_dupes.setObjectName("ghost")
        self.btn_dupes.clicked.connect(self._on_find_duplicates)
        hl.addWidget(self.btn_dupes)

        self.btn_ref = QPushButton("Refresh")
        self.btn_ref.setObjectName("ghost")
        self.btn_ref.clicked.connect(self._refresh)
        hl.addWidget(self.btn_ref)

        self.btn_open_dest = QPushButton("Open folder")
        self.btn_open_dest.setObjectName("ghost")
        self.btn_open_dest.setToolTip("Open the destination folder in the system file manager.")
        self.btn_open_dest.clicked.connect(self._open_destination)
        hl.addWidget(self.btn_open_dest)

        l.addWidget(hdr)

        # ----- Rename options row -----
        rn = QFrame()
        rn.setObjectName("card")
        rn.setStyleSheet(
            "QFrame#card { background: #080808; border: none;"
            " border-bottom: 1px solid #1a1a1a; border-radius: 0;"
            " padding: 0px; }"
        )
        rnl = QHBoxLayout(rn)
        rnl.setContentsMargins(24, 4, 24, 8)
        rnl.setSpacing(10)

        rnl.addWidget(QLabel("Strategy:"))
        self._rename_strat = QComboBox()
        self._rename_strat.setMinimumWidth(220)
        for key, label, desc in RENAME_STRATEGIES:
            self._rename_strat.addItem(f"{label}  —  {desc}", key)
        self._rename_strat.setCurrentIndex(1)  # default to "By category"
        self._rename_strat.setToolTip(
            "How files are named when 'Rename on move' is enabled "
            "or when using the 'Rename only' button. Tag-based strategies "
            "auto-detect content using the heuristic CV pipeline."
        )
        self._rename_strat.currentIndexChanged.connect(self._on_rename_strat_changed)
        rnl.addWidget(self._rename_strat)

        self._cb_rename_prefix = QCheckBox("Use category as prefix")
        self._cb_rename_prefix.setChecked(True)
        self._cb_rename_prefix.setToolTip(
            "For the 'category' strategy: prefix the category name to the "
            "new filename (e.g. Anime_001.jpg)."
        )
        rnl.addWidget(self._cb_rename_prefix)

        rnl.addWidget(QLabel("Max tags:"))
        self._max_tags = QSpinBox()
        self._max_tags.setRange(1, 8)
        self._max_tags.setValue(3)
        self._max_tags.setToolTip(
            "How many content tags to embed in the filename for tag-based "
            "strategies. More tags = longer filenames."
        )
        self._max_tags.valueChanged.connect(lambda _v: self._refresh_rename_buttons())
        rnl.addWidget(self._max_tags)

        self._rename_strategy_hint = QLabel("")
        self._rename_strategy_hint.setObjectName("statSmall")
        self._rename_strategy_hint.setStyleSheet("color: #777;")
        rnl.addWidget(self._rename_strategy_hint, 1)

        self._on_rename_strat_changed()
        l.addWidget(rn)

        l.addWidget(hdr)

        # ----- Body splitter -----
        sp = QSplitter(Qt.Horizontal)
        sp.setHandleWidth(3)
        sp.setChildrenCollapsible(False)

        # Left: categories
        left = QFrame()
        left.setMinimumWidth(180)
        left.setMaximumWidth(280)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 12, 4, 8)
        ll.setSpacing(6)
        lh = QHBoxLayout()
        cat_title = QLabel("Categories")
        cat_title.setStyleSheet("font-weight: 700; color: #999999; font-size: 9pt;"
                                " text-transform: uppercase; letter-spacing: 1px;")
        lh.addWidget(cat_title)
        lh.addStretch()
        self._cat_count_lbl = QLabel("")
        self._cat_count_lbl.setObjectName("statSmall")
        lh.addWidget(self._cat_count_lbl)
        ll.addLayout(lh)

        self._cat_list = DropCatList()
        self._cat_list.currentItemChanged.connect(self._on_cat_sel)
        self._cat_list.files_dropped.connect(self._on_files_dropped)
        ll.addWidget(self._cat_list, 1)

        tip = QLabel("Tip: drag files onto a category to move them.")
        tip.setObjectName("statSmall")
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #555555; padding: 4px 2px;")
        ll.addWidget(tip)
        sp.addWidget(left)

        # Center: toolbar + grid + footer
        center = QFrame()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(8, 8, 8, 4)
        cl.setSpacing(8)

        # Toolbar (search / filter / sort / zoom)
        tb = QHBoxLayout()
        tb.setSpacing(8)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search by file name...")
        self._search.setClearButtonEnabled(True)
        self._search.setFixedWidth(220)
        self._search.textChanged.connect(self._on_search_changed)
        tb.addWidget(self._search)

        self._type_combo = self._make_combo(self.TYPE_FILTERS, "all", self._on_type_changed)
        tb.addWidget(self._labeled("Type:", self._type_combo))
        self._aspect_combo = self._make_combo(self.ASPECT_FILTERS, "any", self._on_aspect_changed)
        tb.addWidget(self._labeled("Aspect:", self._aspect_combo))
        self._sort_combo = self._make_combo(self.SORT_MODES, "name_asc", self._on_sort_changed)
        tb.addWidget(self._labeled("Sort:", self._sort_combo))

        tb.addStretch()

        zoom_lbl = QLabel("Size:")
        zoom_lbl.setObjectName("statSmall")
        tb.addWidget(zoom_lbl)
        self._zoom = QSlider(Qt.Horizontal)
        self._zoom.setRange(60, 280)
        self._zoom.setValue(140)
        self._zoom.setFixedWidth(120)
        self._zoom.valueChanged.connect(self._on_zoom)
        tb.addWidget(self._zoom)

        cl.addLayout(tb)

        # Sub-toolbar with current view info
        sb = QHBoxLayout()
        self._grid_title = QLabel("Select a category")
        self._grid_title.setStyleSheet("font-weight: 700; font-size: 11pt;")
        sb.addWidget(self._grid_title)
        sb.addStretch()
        self._sel_count = QLabel("")
        self._sel_count.setObjectName("statSmall")
        self._sel_count.setStyleSheet("color: #cc6600; font-weight: 600;")
        sb.addWidget(self._sel_count)
        cl.addLayout(sb)

        # Grid
        self._grid = DragGridList()
        self._grid.setViewMode(QListWidget.ViewMode.IconMode)
        self._grid.setIconSize(QSize(140, 140))
        self._grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._grid.setMovement(QListWidget.Movement.Static)
        self._grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._grid.customContextMenuRequested.connect(self._on_context)
        self._grid.itemClicked.connect(self._on_file_click)
        self._grid.itemSelectionChanged.connect(self._update_sel_count)
        self._grid.setSpacing(6)
        self._grid.setUniformItemSizes(True)
        cl.addWidget(self._grid, 1)

        # Footer: progress + stats
        ft = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        self._progress.setFixedHeight(8)
        self._progress.setTextVisible(False)
        ft.addWidget(self._progress, 1)

        self._stats = QLabel("")
        self._stats.setObjectName("statSmall")
        self._stats.setStyleSheet("color: #888888; padding-left: 8px;")
        ft.addWidget(self._stats)
        cl.addLayout(ft)

        sp.addWidget(center)

        # Right: preview
        right = QFrame()
        right.setMinimumWidth(260)
        right.setMaximumWidth(420)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 12, 12, 8)
        rl.setSpacing(8)

        prev_title = QLabel("Preview")
        prev_title.setStyleSheet("font-weight: 700; color: #999999; font-size: 9pt;"
                                 " text-transform: uppercase; letter-spacing: 1px;")
        rl.addWidget(prev_title)

        self._prev_frame = QFrame()
        self._prev_frame.setObjectName("card")
        self._prev_frame.setMinimumHeight(280)
        pf = QVBoxLayout(self._prev_frame)
        pf.setContentsMargins(12, 12, 12, 12)
        self._prev_label = QLabel("No file selected")
        self._prev_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._prev_label.setMinimumHeight(220)
        self._prev_label.setStyleSheet("color: #666666; font-size: 11pt;")
        pf.addWidget(self._prev_label)
        rl.addWidget(self._prev_frame)

        self._prev_info = QLabel("")
        self._prev_info.setObjectName("statSmall")
        self._prev_info.setWordWrap(True)
        self._prev_info.setStyleSheet("color: #cccccc; font-size: 9pt; padding: 4px;")
        self._prev_info.setMinimumHeight(80)
        rl.addWidget(self._prev_info)

        rl.addStretch()
        sp.addWidget(right)

        sp.setStretchFactor(0, 0)
        sp.setStretchFactor(1, 1)
        sp.setStretchFactor(2, 0)
        sp.setSizes([200, 560, 320])
        l.addWidget(sp, 1)

        # Status bar
        sb_row = QHBoxLayout()
        sb_row.setContentsMargins(24, 4, 24, 8)
        self._status = QLabel("Ready")
        self._status.setObjectName("statSmall")
        sb_row.addWidget(self._status)
        sb_row.addStretch()
        l.addLayout(sb_row)

    @staticmethod
    def _labeled(text, widget):
        wrap = QWidget()
        hl = QHBoxLayout(wrap)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(4)
        lbl = QLabel(text)
        lbl.setObjectName("statSmall")
        hl.addWidget(lbl)
        hl.addWidget(widget)
        return wrap

    @staticmethod
    def _make_combo(options, current, slot):
        cb = QComboBox()
        for key, label in options:
            cb.addItem(label, key)
        idx = next((i for i, (k, _) in enumerate(options) if k == current), 0)
        cb.setCurrentIndex(idx)
        cb.currentIndexChanged.connect(slot)
        return cb

    # ------------------------ CATEGORY SCAN ------------------------

    def _all_exts(self):
        return f.STATIC_EXTENSIONS | f.ANIMATED_EXTENSIONS | f.PLUGIN_EXTENSIONS

    def _load_cats(self):
        self._cat_list.blockSignals(True)
        self._cat_list.clear()
        self._cat_list.addItem("-- All files --")
        dest = s.resolve_dest_dir(s.load_settings())
        self._dest_label.setText(f"Destination: {dest}")
        c.CATEGORIES_DIR = dest
        c.discover_categories(dest)
        if not os.path.isdir(dest):
            self._cat_list.blockSignals(False)
            self._cat_count_lbl.setText("0 categories")
            return
        exts = self._all_exts()
        total = 0
        for e in sorted(os.listdir(dest)):
            p = os.path.join(dest, e)
            if not os.path.isdir(p) or e.startswith(".") or e in c.SPECIAL_FOLDERS:
                continue
            cnt = sum(
                1 for fn in os.listdir(p)
                if os.path.isfile(os.path.join(p, fn))
                and not fn.startswith(".")
                and os.path.splitext(fn)[1].lower() in exts
            )
            total += cnt
            item = QListWidgetItem(f"  {e}  ({cnt})")
            item.setData(Qt.ItemDataRole.UserRole, e)
            tip = f"{e}\n{cnt} files"
            # Show the per-category expected config in the tooltip so the
            # user can see at a glance what each folder is "for" without
            # opening the questionnaire.
            try:
                exp = get_expected(e)
                if exp.get("source") and exp.get("source") != "default":
                    bits = []
                    if exp.get("aspect_ratios"):
                        bits.append("aspect=" + "|".join(exp["aspect_ratios"]))
                    if exp.get("file_kinds"):
                        bits.append("kind=" + "|".join(exp["file_kinds"]))
                    if exp.get("color_palette"):
                        bits.append("palette=" + "|".join(exp["color_palette"]))
                    if exp.get("style_keywords"):
                        bits.append("kw=" + ",".join(exp["style_keywords"][:5]))
                    if bits:
                        tip += "\n\nExpected (" + exp.get("source", "") + "):\n  " + "\n  ".join(bits)
            except Exception:
                pass
            item.setToolTip(tip)
            self._cat_list.addItem(item)
        if self._cat_list.count() > 0:
            self._cat_list.item(0).setText(f"-- All files ({total}) --")
        self._cat_list.blockSignals(False)
        self._cat_count_lbl.setText(f"{self._cat_list.count() - 1} cats - {total} files")
        # If we had a previously selected category, restore it.
        if self._cur_cat:
            for i in range(self._cat_list.count()):
                it = self._cat_list.item(i)
                if it and it.data(Qt.ItemDataRole.UserRole) == self._cur_cat:
                    self._cat_list.setCurrentItem(it)
                    return
        # Otherwise default to "All files".
        if self._cat_list.count() > 0:
            self._cat_list.setCurrentRow(0)

    def _patch_count(self, item, new_count, *, prefix, suffix=""):
        """Rewrite an item's text keeping its original prefix/suffix intact."""
        import re
        if item is None:
            return
        num_re = re.compile(r"\((\d+)\)")
        txt = item.text()
        if not num_re.search(txt):
            item.setText(f"{prefix}{new_count}{suffix}")
        else:
            item.setText(num_re.sub(f"({new_count})", txt))
        item.setToolTip(f"{item.text()}\n{new_count} files")

    def _update_cat_count(self, cat: str, delta: int, *, total_delta=None):
        """Patch the count badge in the sidebar without re-scanning.

        `delta` is applied to the matching category. `total_delta` is
        applied to the "All files" total (defaults to `delta` if not given,
        which is right for single-cat operations like delete).
        """
        import re
        if total_delta is None:
            total_delta = delta
        num_re = re.compile(r"\((\d+)\)")

        def _current(item):
            if item is None:
                return None
            m = num_re.search(item.text() or "")
            if not m:
                return None
            try:
                return int(m.group(1))
            except Exception:
                return None

        # Patch the matching category row.
        for i in range(self._cat_list.count()):
            it = self._cat_list.item(i)
            if it and it.data(Qt.ItemDataRole.UserRole) == cat:
                cur = _current(it)
                if cur is not None:
                    self._patch_count(it, max(0, cur + delta),
                                       prefix=f"  {cat}  (", suffix=")")
                break

        # Patch the "All files" total (separate, with total_delta).
        if self._cat_list.count() > 0:
            head = self._cat_list.item(0)
            cur = _current(head)
            if cur is not None:
                new = max(0, cur + total_delta)
                head.setText(f"-- All files ({new}) --")

    # ------------------------ VIEW LOGIC ------------------------

    def _on_cat_sel(self, cur, prev):
        if not cur:
            return
        nm = cur.data(Qt.ItemDataRole.UserRole)
        self._cur_cat = nm or ""
        self._grid_title.setText(
            f"Category: {nm}" if nm else "All files"
        )
        self._populate()

    def _populate(self):
        """Scan disk, apply filters, rebuild the grid.

        Note: `_thumb_cache` is intentionally NOT cleared here, so a category
        the user has already viewed comes back instantly with cached thumbs
        rather than a wall of placeholder icons while the worker re-encodes.
        The cache is only invalidated on zoom changes (see `_on_zoom`).
        """
        self._generation += 1
        # Drain any pending thumbnail requests from the previous view -
        # they're going to be dropped anyway (token mismatch) but each
        # one can take seconds to process (ffmpeg for videos), so we
        # don't want them sitting in front of the new view's requests.
        try:
            self._thumb_worker.drain()
        except Exception:
            pass
        self._pending_thumbs = 0
        self._progress.setVisible(False)

        dest = s.resolve_dest_dir(s.load_settings())
        exts = self._all_exts()
        files: list = []

        if not os.path.isdir(dest):
            self._all_files = []
            self._visible = []
            self._refresh_grid()
            self._update_stats()
            self._reset_preview()
            return

        if self._cur_cat:
            p = os.path.join(dest, self._cur_cat)
            if os.path.isdir(p):
                for fn in sorted(os.listdir(p)):
                    fp = os.path.join(p, fn)
                    if not os.path.isfile(fp):
                        continue
                    if fn.startswith("."):
                        continue
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in exts:
                        continue
                    try:
                        st = os.stat(fp)
                        mtime = st.st_mtime
                        size = st.st_size
                    except OSError:
                        mtime, size = 0.0, 0
                    files.append((fn, fp, self._cur_cat, mtime, size, _file_type(fp)))
        else:
            for e in sorted(os.listdir(dest)):
                ep = os.path.join(dest, e)
                if not os.path.isdir(ep) or e.startswith(".") or e in c.SPECIAL_FOLDERS:
                    continue
                for fn in sorted(os.listdir(ep)):
                    fp = os.path.join(ep, fn)
                    if not os.path.isfile(fp):
                        continue
                    if fn.startswith("."):
                        continue
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in exts:
                        continue
                    try:
                        st = os.stat(fp)
                        mtime = st.st_mtime
                        size = st.st_size
                    except OSError:
                        mtime, size = 0.0, 0
                    files.append((fn, fp, e, mtime, size, _file_type(fp)))

        self._all_files = files
        self._apply_filter_sort()
        self._update_stats()
        self._refresh_rename_buttons()
        self._status.setText(
            f"Loaded {len(self._all_files)} files"
            + (f"  -  {len(self._visible)} shown" if len(self._visible) != len(self._all_files) else "")
        )
        # Auto-select the first item so the preview pane updates on
        # category switch / initial load (matches file-manager UX).
        self._auto_select_first()

    def _apply_filter_sort(self):
        """Refilter and re-sort `_all_files` into `_visible` and rebuild the grid."""
        text = self._search_text.lower().strip()
        tfilter = self._type_filter
        afilter = self._aspect_filter
        smode = self._sort_mode

        out = []
        for rec in self._all_files:
            fname, fp, cat, mtime, size, ftype = rec
            if text and text not in fname.lower():
                continue
            if tfilter != "all" and ftype != tfilter:
                continue
            if afilter != "any":
                # Read dimensions from PIL (cached by the OS).
                try:
                    from PIL import Image as _PILImage
                    with _PILImage.open(fp) as _im:
                        w, h = _im.size
                except Exception:
                    w, h = 0, 0
                if aspect_ratio_class(w, h) != afilter:
                    continue
            out.append(rec)

        if smode == "name_asc":
            out.sort(key=lambda r: r[0].lower())
        elif smode == "name_desc":
            out.sort(key=lambda r: r[0].lower(), reverse=True)
        elif smode == "date_desc":
            out.sort(key=lambda r: r[3], reverse=True)
        elif smode == "date_asc":
            out.sort(key=lambda r: r[3])
        elif smode == "size_desc":
            out.sort(key=lambda r: r[4], reverse=True)
        elif smode == "size_asc":
            out.sort(key=lambda r: r[4])
        elif smode == "type":
            out.sort(key=lambda r: (r[5], r[0].lower()))

        self._visible = out
        self._refresh_grid()

    def _refresh_grid(self):
        """Rebuild the grid from `_visible` and queue thumbnails.

        Does NOT bump `_generation` itself - the caller (`_populate`)
        is responsible for that and the worker drain. This avoids the
        double-increment that used to discard the freshly-queued thumbs
        on the very next refresh.
        """
        gen = str(self._generation)

        self._grid.setUpdatesEnabled(False)
        self._grid.blockSignals(True)
        try:
            self._grid.clear()
            self._items.clear()
            if not self._visible:
                return
            size = self._thumb_size
            for rec in self._visible:
                fname, fp, cat, _mtime, _size, ftype = rec
                item = QListWidgetItem(fname)
                item.setData(Qt.ItemDataRole.UserRole, fp)
                item.setData(Qt.ItemDataRole.UserRole + 1, cat)
                item.setData(Qt.ItemDataRole.UserRole + 2, ftype)
                item.setIcon(_placeholder_icon(os.path.splitext(fname)[1], size))
                tip = f"{fname}\nCategory: {cat}\nType: {ftype.upper()}\n{os.path.dirname(fp)}"
                item.setToolTip(tip)
                self._grid.addItem(item)
                self._items[fp] = item
                # Serve from cache if we have it
                key = f"{fp}:{size}"
                if key in self._thumb_cache:
                    item.setIcon(self._thumb_cache[key])
                else:
                    self._thumb_worker.request(fp, size, gen)
                    self._pending_thumbs += 1
        finally:
            self._grid.blockSignals(False)
            self._grid.setUpdatesEnabled(True)

        if self._pending_thumbs > 0:
            self._progress.setVisible(True)
            self._progress.setRange(0, 0)  # busy indicator
            self._status.setText(f"Loading thumbnails... 0/{self._pending_thumbs}")

    def _on_thumb_ready(self, path, size, token, icon):
        """Called by the worker when a thumbnail is ready (GUI thread)."""
        if token != str(self._generation):
            return  # stale
        item = self._items.get(path)
        if item is not None and icon is not None and size == self._thumb_size:
            item.setIcon(icon)
            key = f"{path}:{size}"
            self._thumb_cache[key] = icon
        self._pending_thumbs = max(0, self._pending_thumbs - 1)
        if self._pending_thumbs <= 0:
            self._progress.setVisible(False)
            self._status.setText(f"Ready  -  {len(self._visible)} files")
        else:
            done = len(self._visible) - self._pending_thumbs
            self._status.setText(f"Loading thumbnails... {done}/{len(self._visible)}")

    def _update_stats(self):
        n = len(self._all_files)
        if n == 0:
            self._stats.setText("No files")
            return
        total = sum(r[4] for r in self._all_files)
        by_type: dict = {}
        for r in self._all_files:
            by_type[r[5]] = by_type.get(r[5], 0) + 1
        parts = []
        for key in ("image", "animated", "video"):
            if by_type.get(key):
                parts.append(f"{by_type[key]} {key}{'s' if by_type[key] != 1 else ''}")
        breakdown = "  |  ".join(parts) if parts else ""
        self._stats.setText(
            f"{n} files  -  {_human_size(total)}"
            + (f"  ({breakdown})" if breakdown else "")
        )

    def _update_sel_count(self):
        n = len(self._grid.selectedItems())
        if n == 0:
            self._sel_count.setText("")
        elif n == 1:
            self._sel_count.setText("1 selected")
        else:
            self._sel_count.setText(f"{n} selected")

    def _reset_preview(self, message="No file selected"):
        """Clear the right-hand preview pane back to its empty state."""
        self._prev_label.clear()
        self._prev_label.setText(message)
        self._prev_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._prev_label.setMinimumHeight(220)
        self._prev_info.clear()

    def _auto_select_first(self):
        """Select + preview the first item in the grid (if any).

        Called after a category change / initial load so the preview pane
        is always populated for the current view. Items in the grid are
        ordered by the active sort, so this respects "newest first" /
        "largest first" / etc.
        """
        if self._grid.count() == 0:
            msg = (
                f"No files in '{self._cur_cat}'"
                if self._cur_cat
                else "No files yet  -  drop some into the destination folder"
            )
            self._reset_preview(msg)
            return
        first = self._grid.item(0)
        self._grid.setCurrentItem(first)
        path = first.data(Qt.ItemDataRole.UserRole)
        cat = first.data(Qt.ItemDataRole.UserRole + 1) or ""
        fname = first.text() or os.path.basename(path or "")
        if path and os.path.isfile(path):
            self._show_preview(path, fname, cat)
        else:
            self._reset_preview()

    # ------------------------ FILTERS / SORT ------------------------

    def _on_search_changed(self, text):
        self._search_text = text
        if not hasattr(self, "_search_timer"):
            self._search_timer = QTimer(self)
            self._search_timer.setSingleShot(True)
            self._search_timer.setInterval(180)
            self._search_timer.timeout.connect(self._apply_filter_sort)
        self._search_timer.start()

    def _on_type_changed(self, idx):
        self._type_filter = self._type_combo.itemData(idx) or "all"
        self._apply_filter_sort()

    def _on_aspect_changed(self, idx):
        self._aspect_filter = self._aspect_combo.itemData(idx) or "any"
        self._apply_filter_sort()

    def _on_sort_changed(self, idx):
        self._sort_mode = self._sort_combo.itemData(idx) or "name_asc"
        self._apply_filter_sort()

    def _on_zoom(self, val):
        self._thumb_size = val
        self._grid.setIconSize(QSize(val, val))
        # Drop cached thumbs at the OLD size, then rebuild so the worker
        # re-encodes at the new resolution.
        old_keys = [k for k in self._thumb_cache if not k.endswith(f":{val}")]
        for k in old_keys:
            self._thumb_cache.pop(k, None)
        self._refresh_grid()

    # ------------------------ INTERACTIONS ------------------------

    def _on_file_click(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        cat = item.data(Qt.ItemDataRole.UserRole + 1)
        fname = item.text() or os.path.basename(path or "")
        if path and os.path.isfile(path):
            self._show_preview(path, fname, cat)

    def _show_preview(self, path, fname, cat):
        """Render the right-hand preview pane.

        Uses the same square-canvas composer as the grid so the preview
        matches the cell aspect ratio (no stretching or scrollbars).
        """
        ftype = _file_type(path)
        fsz = os.path.getsize(path)
        ext = os.path.splitext(fname)[1].lower()

        # Render at 480 px so the preview pane has a clean aspect ratio
        # without overflowing typical layouts.
        pix, _ = _make_pixmap(path, 480)
        if pix is None:
            self._prev_label.setText("Cannot preview")
            self._prev_info.setText(
                f"Name: {fname}\nType: {ftype.upper()} ({ext})\nCategory: {cat}"
            )
            self._status.setText(f"Selected: {fname}")
            return

        # For videos we already know it failed; show the inline error icon.
        animated = (ftype == "animated") or (ext in ANIM_EXTS)
        info_lines = [
            f"Name: {fname}",
            f"Type: {('ANIMATED' if animated else ftype.upper())} ({ext})",
        ]
        # Try to recover real dimensions for the info panel.
        try:
            with Image.open(path) as im:
                real_w, real_h = im.size
        except Exception:
            real_w = real_h = 0
        if real_w and real_h:
            info_lines.append(f"Dimensions: {real_w}x{real_h}")
        info_lines.append(f"Size: {_human_size(fsz)}")
        info_lines.append(f"Category: {cat or '—'}")

        self._prev_label.setPixmap(pix)
        self._prev_info.setText("\n".join(info_lines))
        if real_w and real_h:
            self._status.setText(f"Selected: {fname}  [{real_w}x{real_h}]")
        else:
            self._status.setText(f"Selected: {fname}")

    # ------------------------ CONTEXT MENU / MOVE / DELETE ------------------------

    def _selected_items_or_under_cursor(self, pos):
        """Return the items that should be affected by the context menu.

        File-manager convention: if the right-clicked item is already part
        of the current selection, operate on the whole selection; otherwise
        make the right-clicked item the sole selection first.
        """
        item = self._grid.itemAt(pos)
        sel = self._grid.selectedItems()
        if item and item not in sel:
            self._grid.clearSelection()
            item.setSelected(True)
            sel = [item]
        return sel

    def _on_context(self, pos):
        items = self._selected_items_or_under_cursor(pos)
        if not items:
            return

        menu = QMenu(self)
        if len(items) == 1:
            path = items[0].data(Qt.ItemDataRole.UserRole)
            fname = items[0].text() or os.path.basename(path or "")
            ftype = _file_type(path) if path else "image"
            ext = os.path.splitext(fname)[1].lower()
            menu.addAction(f"File: {fname}  [{ftype.upper()} {ext}]").setEnabled(False)
        else:
            menu.addAction(f"{len(items)} files selected").setEnabled(False)

        menu.addSeparator()
        move_header = menu.addAction("Move to:")
        move_header.setEnabled(False)
        dest = s.resolve_dest_dir(s.load_settings())
        c.CATEGORIES_DIR = dest
        c.discover_categories(dest)

        targets = [cat for cat in sorted(c.CATEGORIES)
                   if len(items) > 1 or (items[0].data(Qt.ItemDataRole.UserRole + 1) != cat)]

        # If multi-select, allow all categories (any item could move).
        if len(items) > 1:
            targets = sorted(c.CATEGORIES)
        if not targets:
            menu.addAction("  (no other categories)").setEnabled(False)
        else:
            for cat in targets:
                a = menu.addAction(f"  -> {cat}")
                a.setData(("move", cat))
        menu.addSeparator()
        a = menu.addAction("Delete" + (" selected" if len(items) > 1 else " file"))
        a.setData(("delete", None))

        action = menu.exec(self._grid.mapToGlobal(pos))
        if not action:
            return
        data = action.data()
        if not data:
            return
        kind, target = data
        if kind == "move" and target:
            self._move_items([it.data(Qt.ItemDataRole.UserRole) for it in items], target)
        elif kind == "delete":
            self._delete_items([it.data(Qt.ItemDataRole.UserRole) for it in items])

    def _on_files_dropped(self, target_cat: str, paths: list):
        """Called when files are dragged onto a category in the sidebar."""
        if not target_cat or not paths:
            return
        self._move_items(paths, target_cat)

    def _move_items(self, paths: list, target_cat: str):
        if not paths or not target_cat:
            return
        dest = s.resolve_dest_dir(s.load_settings())
        dst_dir = os.path.join(dest, target_cat)
        os.makedirs(dst_dir, exist_ok=True)
        rename_on_move = self.cb_rename.isChecked()
        strategy = self._rename_strat.currentData() or "sequential"

        # If rename-on-move is disabled, do the move with collision-safe names.
        if not rename_on_move or strategy == "none":
            self._move_keep_names(paths, target_cat, dst_dir)
            return

        # Otherwise: build pairs (move + rename) using the chosen strategy.
        # Category hint for tag boosting:
        cat_hint = target_cat if self._cb_rename_prefix.isEnabled() else ""
        # Filter out files already in the target category.
        filtered = [p for p in paths if p and os.path.isfile(p)
                    and self._category_for(p) != target_cat]
        if not filtered:
            self._status.setText("Nothing to move (already in target category)")
            return

        self._set_busy(True, f"Moving + renaming {len(filtered)} files -> {target_cat}...")
        self._run_rename_job(
            paths=filtered,
            target_dir=dst_dir,
            strategy=strategy,
            category=cat_hint or target_cat,
            max_tags=self._max_tags.value(),
            mode="move",
        )
        # Stash the destination category so _on_rename_done can adjust counts.
        self._pending_move_target = target_cat
        self._pending_move_src_cats = {p: self._category_for(p) for p in filtered}

    def _move_keep_names(self, paths, target_cat, dst_dir):
        """Move files to dst_dir keeping their original names (collision-safe)."""
        moved = 0
        skipped = 0
        for src in paths:
            if not src or not os.path.isfile(src):
                skipped += 1
                continue
            cur_cat = self._category_for(src)
            if cur_cat == target_cat:
                skipped += 1
                continue
            fname = os.path.basename(src)
            ext = os.path.splitext(fname)[1]
            try:
                dst = os.path.join(dst_dir, fname)
                n = 1
                while os.path.exists(dst):
                    base = os.path.splitext(fname)[0]
                    dst = os.path.join(dst_dir, f"{base}_{n}{ext}")
                    n += 1
                shutil.move(src, dst)
                self._undo_stack.append((src, dst))
                self._update_cat_count(cur_cat, -1, total_delta=0)
                self._update_cat_count(target_cat, +1, total_delta=0)
                moved += 1
            except Exception as e:
                QMessageBox.warning(self, "Move failed", f"{fname}: {e}")
                skipped += 1

        self._remove_paths_from_view(set(p for p in paths if p))
        self._update_undo_label()
        msg = f"Moved {moved} file{'s' if moved != 1 else ''} -> {target_cat}"
        if skipped:
            msg += f"  ({skipped} skipped)"
        self._status.setText(msg)
        if moved:
            self.main.append_log(f"[Reorganize] {moved} files -> {target_cat}")

    def _delete_items(self, paths: list):
        if not paths:
            return
        # Filter out missing files first.
        existing = [p for p in paths if p and os.path.isfile(p)]
        if not existing:
            return
        if len(existing) == 1:
            fname = os.path.basename(existing[0])
            r = QMessageBox.question(
                self, "Delete", f"Delete {fname}?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if r != QMessageBox.Yes:
                return
        else:
            r = QMessageBox.question(
                self, "Delete",
                f"Delete {len(existing)} files? This cannot be undone.",
                QMessageBox.Yes | QMessageBox.No,
            )
            if r != QMessageBox.Yes:
                return

        deleted = 0
        for p in existing:
            try:
                os.remove(p)
                cat = self._category_for(p)
                if cat:
                    self._update_cat_count(cat, -1)
                deleted += 1
            except Exception as e:
                QMessageBox.warning(self, "Delete failed", f"{os.path.basename(p)}: {e}")

        self._remove_paths_from_view(set(existing))
        self._status.setText(f"Deleted {deleted} file{'s' if deleted != 1 else ''}")

    def _remove_paths_from_view(self, paths: set):
        """Remove the given paths from the in-memory view and the grid."""
        if not paths:
            return
        self._all_files = [r for r in self._all_files if r[1] not in paths]
        self._visible = [r for r in self._visible if r[1] not in paths]
        # Drop cached thumbs for removed files.
        for p in list(paths):
            for key in [k for k in self._thumb_cache if k.startswith(f"{p}:")]:
                self._thumb_cache.pop(key, None)
            self._items.pop(p, None)
        # Clear & re-add (keeps ordering of the rest and applies filters).
        self._refresh_grid()
        self._update_stats()

    def _category_for(self, path: str) -> str:
        """Return the category folder a file currently lives under, or ''."""
        try:
            dest = s.resolve_dest_dir(s.load_settings())
            rel = os.path.relpath(os.path.dirname(path), dest)
            if rel.startswith("..") or os.path.isabs(rel):
                return ""
            top = rel.split(os.sep, 1)[0]
            if top in c.SPECIAL_FOLDERS or top.startswith("."):
                return ""
            return top
        except Exception:
            return ""

    # ------------------------ UNDO ------------------------

    def _open_destination(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        dest = s.resolve_dest_dir(s.load_settings())
        if not os.path.isdir(dest):
            QMessageBox.warning(self, "Missing folder", f"Destination does not exist:\n{dest}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(dest))

    def _update_undo_label(self):
        n = len(self._undo_stack)
        self._undo_btn.setText(f"Undo ({n})" if n else "Undo")
        self._undo_btn.setEnabled(n > 0)

    def _on_undo(self):
        if not self._undo_stack:
            return
        src, dst = self._undo_stack.pop()
        if not os.path.isfile(dst):
            QMessageBox.warning(
                self, "Cannot undo",
                f"Destination no longer exists:\n{dst}",
            )
            self._undo_btn.setEnabled(bool(self._undo_stack))
            return
        try:
            os.makedirs(os.path.dirname(src), exist_ok=True)
            shutil.move(dst, src)
            # Update counts: source category was decremented, dst incremented.
            # Reverting should restore both. Total stays the same.
            dst_cat = self._category_for(dst)
            src_cat = self._category_for(src)
            if dst_cat:
                self._update_cat_count(dst_cat, -1, total_delta=0)
            if src_cat:
                self._update_cat_count(src_cat, +1, total_delta=0)
            self._status.setText(f"Undone: {os.path.basename(dst)} -> {os.path.basename(src)}")
            self.main.append_log(f"[Reorganize] Undo: {os.path.basename(dst)} -> {src_cat or 'root'}")
        except Exception as e:
            QMessageBox.warning(self, "Undo failed", str(e))
        self._update_undo_label()
        # Reload from disk so the user sees the restored file in context.
        QTimer.singleShot(100, self._populate)

    # ------------------------ TOP-LEVEL ACTIONS ------------------------

    def _refresh(self):
        self._undo_stack.clear()
        self._update_undo_label()
        self._load_cats()

    def _on_rename(self):
        """Open rename dialog for the current category's files."""
        files = [r[1] for r in self._all_files]
        if not files:
            return
        from ..rename_dialog import RenameDialog
        strategy = self._rename_strat.currentData() or "sequential"
        dlg = RenameDialog(
            files,
            category=self._cur_cat,
            parent=self,
            default_strategy=strategy,
            max_tags=self._max_tags.value(),
        )
        if dlg.exec() == QDialog.Accepted:
            self._status.setText("Rename complete")
            self._refresh()
        else:
            self._status.setText("Rename cancelled")

    def _on_ai_rename(self):
        """Open the AI Rename dialog for the current selection/view."""
        paths = self._current_paths_for_rename()
        if not paths:
            QMessageBox.information(
                self, "No files",
                "Select files in the grid first, or pick a category "
                "in the sidebar to populate the view."
            )
            return
        from ..ai_rename_dialog import AIRenameDialog
        strategy = self._rename_strat.currentData() or "category_tags"
        # Default to heuristic — it's safe, instant, and the user
        # can switch to CLIP/Ollama from inside the dialog. This avoids
        # the OOM crash we saw when Auto tried Ollama on a large
        # selection before the user could intervene.
        dlg = AIRenameDialog(
            paths,
            category=self._cur_cat,
            parent=self,
            default_backend="heuristic",
            default_strategy=strategy,
            max_tags=self._max_tags.value(),
            preview_limit=min(50, len(paths)),
        )
        if dlg.exec() == QDialog.Accepted:
            self._status.setText("AI Rename complete")
            self._refresh()
        else:
            self._status.setText("AI Rename cancelled")

    def _on_ai_rename_category(self):
        """Run an AI-powered rename on every file in the chosen category.

        Pops up an AI Rename dialog seeded with the full category file
        list (regardless of the current filter/search), so the user can
        rename every wallpaper in the category in one shot.
        """
        # Build the file list from disk so filters/searches don't hide files.
        paths = self._category_files(self._cur_cat) if self._cur_cat else []
        if not paths:
            QMessageBox.information(
                self, "Empty category",
                "Pick a category in the sidebar first.\n"
                "Files inside it will be queued for AI rename."
            )
            return
        from ..ai_rename_dialog import AIRenameDialog
        # If the user has many files in the category, start the preview
        # cap low so the dialog opens instantly and the user can grow
        # the cap once they're sure the backend works.
        cap = 50 if len(paths) > 50 else len(paths)
        dlg = AIRenameDialog(
            paths,
            category=self._cur_cat,
            parent=self,
            default_backend="heuristic",
            default_strategy="category_tags",
            max_tags=self._max_tags.value(),
            preview_limit=cap,
        )
        if dlg.exec() == QDialog.Accepted:
            self._status.setText(
                f"AI Rename complete — {len(paths)} files in '{self._cur_cat}'"
            )
            self._refresh()
        else:
            self._status.setText("AI Rename cancelled")

    def _category_files(self, category: str) -> List[str]:
        """Return absolute paths of all media files in `category` on disk."""
        if not category:
            return []
        dest = s.resolve_dest_dir(s.load_settings())
        cat_dir = os.path.join(dest, category)
        if not os.path.isdir(cat_dir):
            return []
        exts = f.STATIC_EXTENSIONS | f.ANIMATED_EXTENSIONS
        out: List[str] = []
        for fn in sorted(os.listdir(cat_dir)):
            fp = os.path.join(cat_dir, fn)
            if not os.path.isfile(fp):
                continue
            if fn.startswith("."):
                continue
            if os.path.splitext(fn)[1].lower() not in exts:
                continue
            out.append(fp)
        return out

    def _on_rename_strat_changed(self):
        """Show strategy hint and toggle controls."""
        strat = self._rename_strat.currentData() or ""
        is_tag = strat in TAG_BASED_STRATEGIES
        self._cb_rename_prefix.setEnabled(strat == "category" or is_tag)
        if is_tag:
            self._rename_strategy_hint.setText(
                "Tag-based: tags are detected from the image using the "
                "heuristic CV pipeline (no AI). First-time use may take a few seconds."
            )
        elif strat == "category":
            self._rename_strategy_hint.setText(
                "Files are renamed to <Category>_NNN.jpg as they are moved."
            )
        elif strat == "sequential":
            self._rename_strategy_hint.setText(
                "Files are renamed to zero-padded sequential numbers in "
                "their destination folder."
            )
        else:
            self._rename_strategy_hint.setText("")
        self._refresh_rename_buttons()

    def _refresh_rename_buttons(self):
        """Enable rename buttons only when there are visible files."""
        has_files = bool(self._visible or self._all_files)
        self.btn_rename.setEnabled(has_files)
        self.btn_rename_only.setEnabled(has_files)
        self.btn_ai_rename.setEnabled(has_files)
        # The category-wide AI rename is enabled whenever a category is
        # selected — even with an empty visible selection, the user
        # may want to rename every file in the category.
        self.btn_ai_rename_cat.setEnabled(bool(self._cur_cat))

    def _on_rename_only(self):
        """Rename the currently visible files in place — no move.

        Uses the strategy / max-tags controls in the header. Tag-based
        strategies auto-detect tags via the heuristic CV pipeline.
        """
        paths = self._current_paths_for_rename()
        if not paths:
            return
        strategy = self._rename_strat.currentData() or "sequential"
        max_tags = self._max_tags.value()
        # Use the current category (if any) for tag boosting.
        category_hint = self._cur_cat or ""

        # Compute pairs in a background thread so tag analysis doesn't
        # freeze the UI on large selections.
        self._set_busy(True, "Detecting tags + building new names...")
        self._run_rename_job(
            paths=paths,
            target_dir=None,  # in-place
            strategy=strategy,
            category=category_hint,
            max_tags=max_tags,
            mode="rename_only",
        )

    def _current_paths_for_rename(self) -> List[str]:
        """Return the path list the user wants to operate on.

        Prefers the explicit selection in the grid; falls back to the
        currently filtered view; finally to the full unfiltered list.
        """
        sel = self._grid.selectedItems() if hasattr(self, "_grid") else []
        if sel:
            return [it.data(Qt.ItemDataRole.UserRole) for it in sel
                    if it.data(Qt.ItemDataRole.UserRole)]
        if self._visible:
            return [r[1] for r in self._visible]
        return [r[1] for r in self._all_files]

    def _run_rename_job(self, paths, target_dir, strategy, category, max_tags, mode):
        """Compute rename pairs in a worker and apply them.

        `mode` is 'rename_only' (in-place) or 'move' (move+rename).
        """
        if hasattr(self, "_rename_job") and self._rename_job and self._rename_job.isRunning():
            QMessageBox.information(self, "Busy", "A rename job is already running.")
            return

        self._rename_job = _RenameJob(
            paths=paths,
            target_dir=target_dir,
            strategy=strategy,
            category=category,
            max_tags=max_tags,
        )
        self._rename_job.progress.connect(self._on_rename_progress)
        self._rename_job.finished_ok.connect(self._on_rename_done)
        self._rename_job.failed.connect(self._on_rename_failed)
        self._rename_job.start()

    def _on_rename_progress(self, cur, total):
        self._status.setText(f"Renaming... {cur}/{total}")

    def _on_rename_done(self, result):
        renamed = result.get("renamed", 0)
        moved = result.get("moved", 0)
        errors = result.get("errors", 0)
        strategy = result.get("strategy", "")
        self._set_busy(False)
        target_cat = getattr(self, "_pending_move_target", None)
        src_cats = getattr(self, "_pending_move_src_cats", {}) or {}
        self._pending_move_target = None
        self._pending_move_src_cats = None

        # If this was a move job, update sidebar counts (one decrement per
        # source category, one increment for the target).
        if target_cat and moved:
            for sc in set(src_cats.values()):
                if sc:
                    self._update_cat_count(sc, -1, total_delta=0)
            self._update_cat_count(target_cat, +1, total_delta=0)

        # Format status message.
        if target_cat:
            msg = f"Moved {moved} files -> {target_cat}"
        else:
            msg = f"Renamed {renamed} files"
            if moved:
                msg += f"  ({moved} moved across folders)"
        if errors:
            msg += f"  |  {errors} errors"
        self._status.setText(msg)

        try:
            label = f"move -> {target_cat}" if target_cat else "rename-only"
            self.main.append_log(
                f"[Reorganize] {label} [{strategy}]: "
                f"renamed={renamed}, moved={moved}, errors={errors}"
            )
        except Exception:
            pass

        # Update undo stack with all the rename pairs (so user can reverse).
        if result.get("pairs"):
            # Skip pairs where src and dst are identical.
            real_pairs = [(o, n) for o, n in result["pairs"] if o != n]
            self._undo_stack.extend(real_pairs)
            self._update_undo_label()
            self._remove_paths_from_view({o for o, _ in real_pairs})
        else:
            QTimer.singleShot(100, self._populate)

    def _on_rename_failed(self, msg):
        QMessageBox.warning(self, "Rename failed", msg)
        self._status.setText("Rename failed")

    def _set_busy(self, busy: bool, message: str = ""):
        """Toggle the rename controls while a rename job is running."""
        self.cb_rename.setEnabled(not busy)
        self._rename_strat.setEnabled(not busy)
        self._max_tags.setEnabled(not busy)
        self.btn_rename_only.setEnabled(not busy and bool(self._all_files))
        self.btn_rename.setEnabled(not busy and bool(self._all_files))
        if busy and message:
            self._status.setText(message)
        elif not busy:
            self._status.setText("Ready")

    def _on_find_duplicates(self):
        """Switch to the Duplicates page to scan for duplicates."""
        self.main.switch_page(6)
        self._status.setText("Use 'Scan for Duplicates' to find duplicate files")

    # ------------------------ LIFECYCLE ------------------------

    def closeEvent(self, event):
        try:
            self._thumb_worker.stop()
            self._thumb_worker.quit()
            self._thumb_worker.wait(2000)
        except Exception:
            pass
        super().closeEvent(event)
