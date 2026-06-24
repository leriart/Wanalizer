"""Custom widgets: TableButtonDelegate for embedded table buttons.

Uses the proper QStyledItemDelegate.editorEvent() to handle clicks,
so cell selection, row navigation, and other table interactions work normally.
"""
from PySide6.QtCore import Qt, QRect, QRectF, QSize, QPoint, QEvent
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QMouseEvent
from PySide6.QtWidgets import QStyledItemDelegate, QStyle, QHeaderView

BTN_TEXT = Qt.ItemDataRole.UserRole + 100
BTN_KIND = Qt.ItemDataRole.UserRole + 101
BTN_DATA = Qt.ItemDataRole.UserRole + 102

KINDS = {
    "edit":   {"bg": QColor("#1a1a1a"), "fg": QColor("#cccccc"),
               "hover_bg": QColor("#2a2a2a"), "hover_fg": QColor("#ffffff"),
               "border": QColor("#555555"), "hover_border": QColor("#888888")},
    "gen":    {"bg": QColor("#e01020"), "fg": QColor("#ffffff"),
               "hover_bg": QColor("#ff1a2e"), "hover_fg": QColor("#ffffff"),
               "border": QColor("#e01020"), "hover_border": QColor("#ff1a2e")},
    "del":    {"bg": QColor(0, 0, 0, 0), "fg": QColor("#ff4444"),
               "hover_bg": QColor("#330000"), "hover_fg": QColor("#ffffff"),
               "border": QColor("#991111"), "hover_border": QColor("#ff0000")},
    "plus":   {"bg": QColor(0, 0, 0, 0), "fg": QColor("#888888"),
               "hover_bg": QColor("#1a1a1a"), "hover_fg": QColor("#ffffff"),
               "border": QColor("#444444"), "hover_border": QColor("#888888")},
    "remove": {"bg": QColor(0, 0, 0, 0), "fg": QColor("#ff4444"),
               "hover_bg": QColor("#330000"), "hover_fg": QColor("#ffffff"),
               "border": QColor("#991111"), "hover_border": QColor("#ff0000")},
    "action": {"bg": QColor("#e01020"), "fg": QColor("#ffffff"),
               "hover_bg": QColor("#ff1a2e"), "hover_fg": QColor("#ffffff"),
               "border": QColor("#e01020"), "hover_border": QColor("#ff1a2e")},
}


class TableButtonDelegate(QStyledItemDelegate):
    """Delegate that paints buttons in cells and handles clicks via editorEvent.

    Multi-button cells supported via pipe-separated BTN_TEXT and BTN_KIND.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hover = (-1, -1, -1)  # (row, col, sub_index)
        self._pressed = (-1, -1, -1)

    def set_hover(self, row, col, sub):
        new = (row, col, sub)
        if new != self._hover:
            old = self._hover
            self._hover = new
            p = self.parent()
            if p is not None:
                if old[0] >= 0:
                    p.viewport().update(self._cell_rect(p, old[0], old[1]))
                if row >= 0:
                    p.viewport().update(self._cell_rect(p, row, col))

    def _cell_rect(self, table, row, col):
        return table.visualRect(table.model().index(row, col))

    def _compute_rects(self, cell_rect, texts):
        m = 3
        spacing = 4
        n = len(texts)
        if n == 0 or cell_rect.width() <= 0:
            return []
        avail = cell_rect.width() - 2 * m
        spacing_total = spacing * (n - 1)
        btn_w = max(28, (avail - spacing_total) / n)
        rects = []
        x = cell_rect.left() + m
        for i in range(n):
            r = QRect(int(x), int(cell_rect.top() + 3),
                      int(btn_w), int(cell_rect.height() - 6))
            rects.append(r)
            x += btn_w + spacing
        return rects

    def _sub_at(self, cell_rect, texts, local_pos):
        """local_pos is in cell-relative coordinates."""
        # Compute rects in cell-relative coordinates
        m = 3
        spacing = 4
        n = len(texts)
        if n == 0 or cell_rect.width() <= 0:
            return -1
        avail = cell_rect.width() - 2 * m
        spacing_total = spacing * (n - 1)
        btn_w = max(28, (avail - spacing_total) / n)
        x = m
        for i in range(n):
            r = QRect(int(x), int(3), int(btn_w), int(cell_rect.height() - 6))
            if r.contains(local_pos):
                return i
            x += btn_w + spacing
        return -1

    def sizeHint(self, option, index):
        texts_raw = index.data(BTN_TEXT)
        if texts_raw is None:
            return super().sizeHint(option, index)
        texts = str(texts_raw).split("|")
        return QSize(min_button_column_width(len(texts)), option.rect.height())

    def paint(self, painter, option, index):
        texts_raw = index.data(BTN_TEXT)
        kinds_raw = index.data(BTN_KIND)
        if texts_raw is None or kinds_raw is None:
            return
        texts = str(texts_raw).split("|")
        kinds = str(kinds_raw).split("|")
        n = min(len(texts), len(kinds))
        if n == 0:
            return

        rects = self._compute_rects(option.rect, texts)
        is_disabled = not (option.state & QStyle.State_Enabled)

        row, col = index.row(), index.column()
        for i, (text, kind, r) in enumerate(zip(texts[:n], kinds[:n], rects[:n])):
            colors = KINDS.get(kind, KINDS["edit"])
            is_hovered = (row, col, i) == self._hover and not is_disabled
            is_pressed = (row, col, i) == self._pressed and not is_disabled

            bbg = colors["hover_bg"] if (is_hovered or is_pressed) else colors["bg"]
            bfg = colors["hover_fg"] if (is_hovered or is_pressed) else colors["fg"]
            bb = colors["hover_border"] if (is_hovered or is_pressed) else colors["border"]

            if is_disabled:
                bbg = QColor("#111111")
                bfg = QColor("#555555")
                bb = QColor("#1a1a1a")

            painter.save()
            painter.setRenderHint(QPainter.Antialiasing, True)
            if bbg.alpha() > 0:
                painter.setBrush(bbg)
                painter.setPen(QPen(bb, 1))
            else:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(bb, 1))
            painter.drawRoundedRect(QRectF(r), 4, 4)

            painter.setPen(bfg)
            f = QFont()
            f.setBold(True)
            f.setPixelSize(max(10, int(r.height() // 2.5)))
            painter.setFont(f)
            painter.drawText(QRectF(r), Qt.AlignmentFlag.AlignCenter, text)
            painter.restore()

    def editorEvent(self, event, model, option, index):
        """Handle mouse events on button cells. This is the proper Qt way."""
        if event is None or index is None:
            return False

        texts_raw = index.data(BTN_TEXT)
        kinds_raw = index.data(BTN_KIND)
        if texts_raw is None or kinds_raw is None:
            return False

        texts = str(texts_raw).split("|")
        kinds = str(kinds_raw).split("|")
        n = min(len(texts), len(kinds))
        if n == 0:
            return False

        cell_rect = self._cell_rect_from_option(option)
        if cell_rect is None:
            return False

        evt_type = event.type()
        if evt_type == QEvent.Type.MouseMove:
            local = event.pos() - cell_rect.topLeft()
            sub = self._sub_at(cell_rect, texts, local)
            if sub >= 0:
                self.set_hover(index.row(), index.column(), sub)
            else:
                self.set_hover(-1, -1, -1)
            return False  # Let table handle selection/movement
        elif evt_type == QEvent.Type.MouseButtonPress:
            if event.button() != Qt.MouseButton.LeftButton:
                return False
            local = event.pos() - cell_rect.topLeft()
            sub = self._sub_at(cell_rect, texts, local)
            if sub >= 0:
                self._pressed = (index.row(), index.column(), sub)
                return True
            return False
        elif evt_type == QEvent.Type.MouseButtonRelease:
            if self._pressed == (index.row(), index.column(), self._sub_at(
                    cell_rect, texts, event.pos() - cell_rect.topLeft())):
                if self._pressed[2] >= 0 and self._pressed[2] < len(kinds):
                    kind = kinds[self._pressed[2]]
                    data = index.data(BTN_DATA) or ""
                    p = self.parent()
                    if p is not None:
                        if hasattr(p, "button_callbacks") and kind in p.button_callbacks:
                            p.button_callbacks[kind](index.row(), index.column(), data)
                self._pressed = (-1, -1, -1)
                return True
            self._pressed = (-1, -1, -1)
            return False
        elif evt_type == QEvent.Type.Leave:
            self.set_hover(-1, -1, -1)
            return False
        return False

    def _cell_rect_from_option(self, option):
        if hasattr(option, "rect") and option.rect is not None:
            return option.rect
        return None


def min_button_column_width(n_buttons, btn_units=46, spacing=4, margin=3):
    """Minimum pixels needed for a column with n buttons."""
    return int(margin * 2 + n_buttons * btn_units + max(0, n_buttons - 1) * spacing)


def refresh_action_columns(table):
    """Resize only the action button columns to fit their content.
    Preserves user-adjusted widths for other columns.
    """
    header = table.horizontalHeader()
    for col in range(table.columnCount()):
        if header.sectionResizeMode(col) == QHeaderView.ResizeMode.ResizeToContents:
            sz = max(header.sectionSizeHint(col), header.minimumSectionSize())
            header.resizeSection(col, sz)


def setup_table_buttons(table, callbacks, button_cols):
    """Wire a QTableWidget to use TableButtonDelegate on button columns.

    Uses editorEvent for proper click handling that doesn't break other
    table interactions (selection, navigation, etc.).
    """
    delegate = TableButtonDelegate(table)
    for col in button_cols:
        table.setItemDelegateForColumn(col, delegate)
    # Store callbacks on the table for the delegate to access
    table.button_callbacks = callbacks
    return delegate
