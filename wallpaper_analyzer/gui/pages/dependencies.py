"""Dependencies page: install optional packages."""
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QPlainTextEdit, QWidget, QMessageBox,
    QProgressBar,
)
from ..widgets import BTN_TEXT, BTN_KIND, BTN_DATA, setup_table_buttons, refresh_action_columns
from PySide6.QtCore import Qt
from ... import settings as s
from ..workers import DepWorker

HAS_CV2 = False
HAS_IMAGEHASH = False
HAS_SKLEARN = False
try:
    import cv2 as _
    HAS_CV2 = True
except Exception:
    pass
try:
    import imagehash as _
    HAS_IMAGEHASH = True
except Exception:
    pass
try:
    import sklearn as _
    HAS_SKLEARN = True
except Exception:
    pass


class DependenciesPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.worker = None
        self._build()

    def _build(self):
        l = QVBoxLayout(self)
        l.setContentsMargins(24, 24, 24, 24)

        tl = QLabel("Dependencies")
        tl.setObjectName("title")
        l.addWidget(tl)
        st = QLabel("Install or update optional packages for enhanced analysis.")
        st.setObjectName("subtitle")
        l.addWidget(st)

        tb = QHBoxLayout()
        self.btn_all = QPushButton("Install All")
        self.btn_all.setObjectName("primary")
        self.btn_all.clicked.connect(lambda: self._install_batch("all"))
        self.btn_missing = QPushButton("Install Missing")
        self.btn_missing.setObjectName("ghost")
        self.btn_missing.clicked.connect(lambda: self._install_batch("missing"))
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_install)
        tb.addWidget(self.btn_all)
        tb.addWidget(self.btn_missing)
        tb.addWidget(self.btn_stop)
        tb.addStretch()
        l.addLayout(tb)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Module", "Description", "Status", ""])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setMinimumSectionSize(100)
        self._table.setAlternatingRowColors(True)

        deps_data = [
            ("cv2", "OpenCV (edge detection, features, contours)"),
            ("imagehash", "Perceptual hashing (faster pHash, wavelet)"),
            ("sklearn", "KMeans palette extraction"),
        ]
        self._row_idx = {}
        self._table.setRowCount(len(deps_data))
        for i, (mod, desc) in enumerate(deps_data):
            self._table.setItem(i, 0, QTableWidgetItem(mod))
            self._table.setItem(i, 1, QTableWidgetItem(desc))
            av = {"cv2": HAS_CV2, "imagehash": HAS_IMAGEHASH, "sklearn": HAS_SKLEARN}.get(mod, False)
            self._table.setItem(i, 2, QTableWidgetItem("installed" if av else "missing"))
            act = QTableWidgetItem()
            act.setData(BTN_TEXT, "Reinstall" if av else "Install")
            act.setData(BTN_KIND, "action")
            act.setData(BTN_DATA, mod)
            self._table.setItem(i, 3, act)
            self._row_idx[mod] = i

        refresh_action_columns(self._table)
        l.addWidget(self._table, 1)

        self._slabel = QLabel("Ready")
        self._slabel.setObjectName("statSmall")
        l.addWidget(self._slabel)
        self._prog = QProgressBar()
        self._prog.setVisible(False)
        self._prog.setRange(0, 0)
        l.addWidget(self._prog)
        l.addWidget(QLabel("Install log:"))
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(500)
        self._log.setMinimumHeight(100)
        l.addWidget(self._log)

        def _dep_cb(r, c, mod):
            self._install_single(mod)

        setup_table_buttons(self._table, {"action": _dep_cb}, [3])

    def _set_busy(self, b):
        self.btn_all.setEnabled(not b)
        self.btn_missing.setEnabled(not b)
        self.btn_stop.setEnabled(b)
        self._prog.setVisible(b)
        for i in range(self._table.rowCount()):
            act = self._table.item(i, 3)
            if act:
                if b:
                    act.setFlags(act.flags() & ~Qt.ItemIsEnabled)
                else:
                    act.setFlags(act.flags() | Qt.ItemIsEnabled)
        self._table.viewport().update()

    def _install_single(self, mod):
        self._set_busy(True)
        self._log.clear()
        self._launch([mod])

    def _install_batch(self, which):
        avail_map = {"cv2": HAS_CV2, "imagehash": HAS_IMAGEHASH, "sklearn": HAS_SKLEARN}
        if which == "all":
            mods = list(avail_map.keys())
        else:
            mods = [m for m, av in avail_map.items() if not av]
        if not mods:
            QMessageBox.information(self, "Info", "All dependencies installed.")
            return
        self._set_busy(True)
        self._log.clear()
        self._launch(mods)

    def _launch(self, mods):
        self._slabel.setText(f"Installing {len(mods)} package(s)...")
        self.worker = DepWorker(mods)
        self.worker.log_line.connect(self._log.appendPlainText)
        self.worker.module_done.connect(self._on_mod_done)
        self.worker.all_done.connect(lambda: (self._set_busy(False), self._slabel.setText("Done.")))
        self.worker.start()

    def _stop_install(self):
        if self.worker:
            self.worker.cancel()
            self._slabel.setText("Stopping...")

    def _on_mod_done(self, mod, ok):
        idx = self._row_idx.get(mod)
        if idx is not None:
            self._table.item(idx, 2).setText("installed" if ok else "failed")
            act = self._table.item(idx, 3)
            if act:
                act.setData(BTN_TEXT, "Reinstall" if ok else "Retry")
                act.setData(BTN_KIND, "action")
