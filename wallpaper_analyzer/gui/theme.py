"""Red/Black/White theme for Wanalizer."""

QSS = """
QWidget {
    background-color: #000000;
    color: #ffffff;
    font-size: 10pt;
}
QMainWindow {
    background-color: #000000;
}
QStackedWidget {
    background-color: #000000;
}

/* Sidebar */
QFrame#sidebar {
    background: #050505;
    border-right: 1px solid #222222;
}
QPushButton#navBtn {
    background: transparent;
    color: #aaaaaa;
    border: none;
    border-radius: 6px;
    padding: 10px 16px;
    text-align: left;
    font-weight: 500;
    font-size: 10pt;
}
QPushButton#navBtn:hover {
    background: #1a1a1a;
    color: #ffffff;
}
QPushButton#navBtn:checked {
    background: #e01020;
    color: #ffffff;
    font-weight: 700;
}

/* Group boxes */
QGroupBox {
    background: #0a0a0a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    margin-top: 16px;
    padding: 16px 16px 16px 16px;
    font-weight: 600;
    color: #ffffff;
}
QGroupBox::title {
    color: #999999;
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 6px;
}

/* Buttons */
QPushButton {
    background: #1a1a1a;
    color: #ffffff;
    border: 1px solid #333333;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 500;
    min-height: 20px;
}
QPushButton:hover {
    background: #2a2a2a;
    border-color: #e01020;
}
QPushButton:pressed {
    background: #111111;
}
QPushButton:disabled {
    background: #0a0a0a;
    color: #555555;
    border-color: #1a1a1a;
}
QPushButton#primary {
    background: #e01020;
    color: #ffffff;
    border-color: #e01020;
    font-weight: 700;
}
QPushButton#primary:hover {
    background: #ff1a2e;
}
QPushButton#danger {
    background: #cc0000;
    color: #ffffff;
    border-color: #cc0000;
}
QPushButton#danger:hover {
    background: #ff0000;
}
QPushButton#ghost {
    background: transparent;
    border: 1px solid #333333;
    color: #999999;
}
QPushButton#ghost:hover {
    background: #1a1a1a;
    color: #ffffff;
}
QPushButton#success {
    background: #cc0000;
    color: #ffffff;
    border-color: #cc0000;
}

/* Inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #0a0a0a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 7px 10px;
    color: #ffffff;
    min-height: 20px;
    selection-background-color: #e01020;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #e01020;
}
QComboBox::drop-down {
    border: none;
    padding-right: 6px;
}
QComboBox QAbstractItemView {
    background: #0a0a0a;
    border: 1px solid #2a2a2a;
    selection-background-color: #e01020;
    color: #ffffff;
}

/* Checkboxes */
QCheckBox {
    spacing: 8px;
    color: #ffffff;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 2px solid #555555;
    background: #0a0a0a;
}
QCheckBox::indicator:checked {
    background: #e01020;
    border-color: #e01020;
}
QCheckBox::indicator:hover {
    border-color: #e01020;
}

/* Radio buttons */
QRadioButton {
    spacing: 6px;
    color: #ffffff;
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 10px;
    border: 2px solid #555555;
    background: #0a0a0a;
}
QRadioButton::indicator:checked {
    background: #e01020;
    border-color: #e01020;
}

/* Progress bar */
QProgressBar {
    background: #0a0a0a;
    border: 1px solid #1a1a1a;
    border-radius: 6px;
    text-align: center;
    height: 22px;
    font-size: 9pt;
    color: #aaaaaa;
}
QProgressBar::chunk {
    background: #e01020;
    border-radius: 5px;
}

/* Text areas */
QPlainTextEdit, QTextEdit {
    background: #050505;
    color: #ffffff;
    border: 1px solid #1a1a1a;
    border-radius: 6px;
    font-family: "JetBrains Mono", "Fira Code", monospace;
    font-size: 9pt;
    padding: 8px;
    selection-background-color: #e01020;
}
QPlainTextEdit:focus, QTextEdit:focus {
    border-color: #e01020;
}

/* Tables - FIXED: white text on dark bg, clear contrast */
QTableWidget {
    background: #050505;
    color: #ffffff;
    border: 1px solid #1a1a1a;
    gridline-color: #151515;
    outline: none;
    alternate-background-color: #0e0e0e;
}
QTableWidget::item {
    padding: 8px 12px;
    border-bottom: 1px solid #151515;
    color: #ffffff;
    background: #050505;
}
QTableWidget::item:alternate {
    background: #0e0e0e;
}
QTableWidget::item:selected {
    background: #e01020;
    color: #ffffff;
}
QTableWidget::item:hover {
    background: #1a1a1a;
}
QHeaderView {
    background: #0e0e0e;
}
QHeaderView::section {
    background: #0e0e0e;
    color: #999999;
    padding: 10px 12px;
    border: none;
    border-bottom: 2px solid #e01020;
    border-right: 1px solid #1a1a1a;
    font-weight: 700;
    font-size: 9pt;
    text-transform: uppercase;
}
QHeaderView::section:hover {
    background: #1a1a1a;
    color: #ffffff;
}

/* Table action buttons - clear, distinct, visible */
QTableWidget QPushButton {
    background: #1a1a1a;
    color: #ffffff;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 5px 10px;
    font-size: 9pt;
    font-weight: 600;
    min-width: 56px;
    min-height: 26px;
}
QTableWidget QPushButton:hover {
    background: #2a2a2a;
    border-color: #ffffff;
    color: #ffffff;
}
QTableWidget QPushButton:pressed {
    background: #000000;
    border-color: #e01020;
}
QTableWidget QPushButton:disabled {
    background: #0a0a0a;
    color: #444444;
    border-color: #1a1a1a;
}

/* Table: Edit button (neutral) */
QTableWidget QPushButton#tblEdit {
    background: #1a1a1a;
    color: #cccccc;
    border: 1px solid #555555;
}
QTableWidget QPushButton#tblEdit:hover {
    background: #2a2a2a;
    border-color: #999999;
    color: #ffffff;
}

/* Table: Generate button (red primary) */
QTableWidget QPushButton#tblGen {
    background: #e01020;
    color: #ffffff;
    border: 1px solid #e01020;
}
QTableWidget QPushButton#tblGen:hover {
    background: #ff1a2e;
    border-color: #ff1a2e;
}

/* Table: Delete button (danger) */
QTableWidget QPushButton#tblDel {
    background: transparent;
    color: #ff4444;
    border: 1px solid #991111;
}
QTableWidget QPushButton#tblDel:hover {
    background: #330000;
    border-color: #ff0000;
    color: #ff0000;
}

/* Table: Import button (ghost) */
QTableWidget QPushButton#tblImport {
    background: transparent;
    color: #888888;
    border: 1px solid #444444;
    min-width: 40px;
}
QTableWidget QPushButton#tblImport:hover {
    background: #1a1a1a;
    border-color: #888888;
    color: #ffffff;
}

/* Table: Remove button (danger small) */
QTableWidget QPushButton#tblRemove {
    background: transparent;
    color: #ff4444;
    border: 1px solid #991111;
    min-width: 32px;
    padding: 3px 6px;
}
QTableWidget QPushButton#tblRemove:hover {
    background: #330000;
    border-color: #ff0000;
    color: #ff0000;
}

/* Table: Activate/Pull/Install (compact primary) */
QTableWidget QPushButton#tblAction {
    background: #e01020;
    color: #ffffff;
    border: none;
}
QTableWidget QPushButton#tblAction:hover {
    background: #ff1a2e;
}
QTableWidget QPushButton#tblAction:disabled {
    background: #1a1a1a;
    color: #666666;
    border: 1px solid #333333;
}

/* Lists */
QListWidget {
    background: #050505;
    color: #ffffff;
    border: 1px solid #1a1a1a;
    border-radius: 6px;
    outline: none;
}
QListWidget::item {
    padding: 6px 10px;
    border-radius: 4px;
    color: #ffffff;
}
QListWidget::item:hover {
    background: #1a1a1a;
}
QListWidget::item:selected {
    background: #e01020;
    color: #ffffff;
}

/* Status bar */
QStatusBar {
    background: #050505;
    color: #888888;
    border-top: 1px solid #1a1a1a;
}

/* Splitter */
QSplitter::handle {
    background: #1a1a1a;
    width: 2px;
}

/* Scrollbars */
QScrollBar:vertical {
    background: #000000;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #333333;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: #e01020;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: #000000;
    height: 8px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #333333;
    border-radius: 4px;
    min-width: 24px;
}
QScrollBar::handle:horizontal:hover {
    background: #e01020;
}

/* Labels */
QLabel#title {
    font-size: 18pt;
    font-weight: 700;
    color: #ffffff;
    padding-bottom: 2px;
}
QLabel#subtitle {
    font-size: 10pt;
    color: #888888;
    padding-bottom: 8px;
}
QLabel#statBig {
    font-size: 24pt;
    font-weight: 700;
    color: #e01020;
}
QLabel#statSmall {
    font-size: 9pt;
    color: #888888;
}
QLabel#statusOk {
    color: #e01020;
    font-weight: 600;
}
QLabel#statusWarn {
    color: #cc6600;
    font-weight: 600;
}
QLabel#statusErr {
    color: #ff0000;
    font-weight: 600;
}

/* Menus */
QMenu {
    background: #0a0a0a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 4px;
    color: #ffffff;
}
QMenu::item {
    padding: 6px 28px 6px 14px;
    border-radius: 4px;
}
QMenu::item:hover {
    background: #e01020;
}
QMenu::separator {
    height: 1px;
    background: #1a1a1a;
    margin: 4px 8px;
}

/* Tooltips */
QToolTip {
    background: #1a1a1a;
    color: #ffffff;
    border: 1px solid #333333;
    padding: 6px 10px;
    border-radius: 6px;
}

/* Tabs */
QTabWidget::pane {
    border: 1px solid #2a2a2a;
    background: #0a0a0a;
    border-radius: 6px;
}
QTabBar::tab {
    background: #0a0a0a;
    color: #888888;
    border: 1px solid #1a1a1a;
    padding: 6px 16px;
    border-radius: 4px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #e01020;
    color: #ffffff;
    border-color: #e01020;
}
QTabBar::tab:hover:!selected {
    background: #1a1a1a;
    color: #ffffff;
}

/* Frames */
QFrame#card {
    background: #0a0a0a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 16px;
}

/* Slider */
QSlider::groove:horizontal {
    border: 1px solid #2a2a2a;
    height: 6px;
    background: #0a0a0a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #e01020;
    border: none;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #ff1a2e;
}

/* Scroll area */
QScrollArea {
    background: transparent;
    border: none;
}
"""


LIGHT_QSS = """
/* Light theme — placeholder. The dark theme is the only fully-styled
   variant; selecting "Light" in Settings falls back to the OS default
   palette so the UI is still usable without crashing on missing rules. */
QToolTip { color: #000; background: #ffffe0; border: 1px solid #888; }
"""


def apply_theme(app, mode: str = "dark"):
    from PySide6.QtGui import QPalette, QColor

    if mode == "light":
        app.setStyleSheet(LIGHT_QSS)
        app.setPalette(app.style().standardPalette())
        return

    app.setStyleSheet(QSS)

    p = QPalette()
    p.setColor(QPalette.Window, QColor("#000000"))
    p.setColor(QPalette.WindowText, QColor("#ffffff"))
    p.setColor(QPalette.Base, QColor("#050505"))
    p.setColor(QPalette.AlternateBase, QColor("#0e0e0e"))
    p.setColor(QPalette.Text, QColor("#ffffff"))
    p.setColor(QPalette.Button, QColor("#1a1a1a"))
    p.setColor(QPalette.ButtonText, QColor("#ffffff"))
    p.setColor(QPalette.BrightText, QColor("#ffffff"))
    p.setColor(QPalette.Highlight, QColor("#e01020"))
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.Link, QColor("#e01020"))
    p.setColor(QPalette.LinkVisited, QColor("#cc0000"))
    p.setColor(QPalette.ToolTipBase, QColor("#1a1a1a"))
    p.setColor(QPalette.ToolTipText, QColor("#ffffff"))
    app.setPalette(p)
