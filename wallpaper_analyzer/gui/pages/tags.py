"""Tags page: manage the global tag registry."""
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView, QWidget,
    QMessageBox,
)
from ..widgets import BTN_TEXT, BTN_KIND, BTN_DATA, setup_table_buttons, refresh_action_columns
from ... import tags as t


class TagsPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self._build()
        self._refresh()

    def _build(self):
        l = QVBoxLayout(self)
        l.setContentsMargins(24, 24, 24, 24)
        tl = QLabel("Tags Registry")
        tl.setObjectName("title")
        l.addWidget(tl)
        st = QLabel("Manage global tags organized by group.")
        st.setObjectName("subtitle")
        l.addWidget(st)

        h = QHBoxLayout()
        self._cmb_group = QComboBox()
        self._cmb_group.currentTextChanged.connect(self._load_group)
        h.addWidget(QLabel("Group:"))
        h.addWidget(self._cmb_group, 1)
        h.addStretch()
        l.addLayout(h)

        ha = QHBoxLayout()
        self._tag_edit = QLineEdit()
        self._tag_edit.setPlaceholderText("New tag name...")
        self._btn_add = QPushButton("Add Tag")
        self._btn_add.setObjectName("primary")
        self._btn_add.clicked.connect(self._add_tag)
        self._btn_save = QPushButton("Save All Tags")
        self._btn_save.setObjectName("success")
        self._btn_save.clicked.connect(self._save_all)
        ha.addWidget(self._tag_edit, 1)
        ha.addWidget(self._btn_add)
        ha.addWidget(self._btn_save)
        l.addLayout(ha)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Tag", ""])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setMinimumSectionSize(40)
        self._table.setAlternatingRowColors(True)
        l.addWidget(self._table, 1)

        self._status = QLabel("")
        self._status.setObjectName("statSmall")
        l.addWidget(self._status)

        setup_table_buttons(self._table, {
            "remove": lambda r, c, d: self._rm_tag(d, self._table.item(r, 0).text()),
        }, [1])

    def _refresh(self):
        t.load_tags()
        groups = t.get_tag_groups()
        self._data = {"groups": groups}
        self._cmb_group.clear()
        for key, grp in sorted(groups.items()):
            self._cmb_group.addItem(f"{grp.get('label', key)} ({len(grp.get('tags', []))} tags)", key)
        if self._cmb_group.count() > 0:
            self._cmb_group.setCurrentIndex(0)
        self._status.setText(f"{len(t.get_all_tags())} total tags across {len(groups)} groups")

    def _load_group(self):
        key = self._cmb_group.currentData()
        if not key:
            return
        grp = self._data["groups"].get(key, {})
        tags_list = sorted(grp.get("tags", []))
        self._table.setRowCount(len(tags_list))
        for i, tag in enumerate(tags_list):
            self._table.setItem(i, 0, QTableWidgetItem(tag))
            act = QTableWidgetItem()
            act.setData(BTN_TEXT, "X")
            act.setData(BTN_KIND, "remove")
            act.setData(BTN_DATA, key)
            self._table.setItem(i, 1, act)
        refresh_action_columns(self._table)

    def _add_tag(self):
        key = self._cmb_group.currentData()
        tag = self._tag_edit.text().strip().lower()
        if not key or not tag:
            return
        grp = self._data["groups"].get(key, {})
        tags_list = list(grp.get("tags", []))
        if tag in tags_list:
            return
        tags_list.append(tag)
        grp["tags"] = sorted(set(tags_list))
        self._tag_edit.clear()
        self._load_group()
        self._status.setText(f"Added '{tag}'")

    def _rm_tag(self, group_key, tag):
        grp = self._data["groups"].get(group_key, {})
        tags_list = grp.get("tags", [])
        if tag in tags_list:
            tags_list.remove(tag)
            grp["tags"] = tags_list
        self._load_group()
        self._status.setText(f"Removed '{tag}'")

    def _save_all(self):
        t.save_tags(self._data)
        QMessageBox.information(self, "Saved", "Tags saved to tags.json")
