"""Categories page: manage category folders, tags, prompts."""
import os, shutil
from typing import Optional
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QHeaderView, QPlainTextEdit,
    QInputDialog, QMessageBox, QDialog, QFileDialog, QWidget, QLineEdit,
    QProgressBar, QFormLayout, QComboBox, QSpinBox, QCheckBox,
)
from PySide6.QtCore import QTimer
from ..workers import GenerateTagsWorker, RegenerateCategoryWorker
from ..widgets import BTN_TEXT, BTN_KIND, BTN_DATA, setup_table_buttons, min_button_column_width, refresh_action_columns
from ..category_config_dialog import CategoryConfigDialog, AIConfigDialog
from ... import settings as s
from ... import formats as f
from ... import categories as c


class CategoriesPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self._build()
        self._refresh()

    def _build(self):
        l = QVBoxLayout(self)
        l.setContentsMargins(24, 24, 24, 24)

        tl = QLabel("Categories")
        tl.setObjectName("title")
        l.addWidget(tl)
        st = QLabel("Each category = destination folder. Add sample images, then generate tags and style prompt via AI.")
        st.setObjectName("subtitle")
        l.addWidget(st)

        # Load settings once so we can default all spinners/dropdowns.
        cfg = s.load_settings()

        self._info = QLabel("")
        self._info.setObjectName("statSmall")
        l.addWidget(self._info)

        # AI selector row (mode + Vision model + Text model + samples)
        ai_row = QHBoxLayout()
        ai_row.addWidget(QLabel("AI source:"))
        self._ai_mode = QComboBox()
        self._ai_mode.addItem("Ollama (local)", "ollama")
        self._ai_mode.addItem("Low-Level CV (heuristic)", "lowlevel")
        self._ai_mode.currentIndexChanged.connect(self._on_ai_mode_changed)
        ai_row.addWidget(self._ai_mode, 1)

        self._ai_model_lbl = QLabel("Single model:")
        ai_row.addWidget(self._ai_model_lbl)
        self._ai_model = QComboBox()
        self._ai_model.setMinimumWidth(160)
        self._ai_model.setEditable(True)
        ai_row.addWidget(self._ai_model, 1)
        l.addLayout(ai_row)

        # Vision + Text model row (for the two-model regenerate pipeline)
        vt_row = QHBoxLayout()
        self._vt_lbl = QLabel("Vision model:")
        vt_row.addWidget(self._vt_lbl)
        self._vision_model = QComboBox()
        self._vision_model.setMinimumWidth(180)
        self._vision_model.setEditable(True)
        self._vision_model.setToolTip(
            "Vision-capable model used to extract structured info from each "
            "sample image (subject, style, tags, palette, mood)."
        )
        vt_row.addWidget(self._vision_model, 1)

        self._txt_lbl = QLabel("Text model:")
        vt_row.addWidget(self._txt_lbl)
        self._text_model = QComboBox()
        self._text_model.setMinimumWidth(160)
        self._text_model.setEditable(True)
        self._text_model.setToolTip(
            "Text-only model used to synthesise the 2-3 sentence unified "
            "aesthetic prompt from the per-image descriptions."
        )
        vt_row.addWidget(self._text_model, 1)

        vt_row.addWidget(QLabel("Samples:"))
        self._vt_samples = QSpinBox()
        self._vt_samples.setRange(1, 12)
        self._vt_samples.setValue(cfg.get("regen_samples", 3))
        self._vt_samples.setToolTip("How many images to sample per category")
        vt_row.addWidget(self._vt_samples)

        vt_row.addWidget(QLabel("Vision timeout:"))
        self._vt_vtimeout = QSpinBox()
        self._vt_vtimeout.setRange(30, 900)
        self._vt_vtimeout.setValue(cfg.get("regen_vision_timeout", 240))
        self._vt_vtimeout.setSuffix(" s")
        vt_row.addWidget(self._vt_vtimeout)

        vt_row.addWidget(QLabel("Text timeout:"))
        self._vt_ttimeout = QSpinBox()
        self._vt_ttimeout.setRange(15, 300)
        self._vt_ttimeout.setValue(cfg.get("regen_text_timeout", 120))
        self._vt_ttimeout.setSuffix(" s")
        vt_row.addWidget(self._vt_ttimeout)

        self._ai_status = QLabel("")
        self._ai_status.setObjectName("statSmall")
        vt_row.addWidget(self._ai_status)

        l.addLayout(vt_row)

        # Now that all three dropdowns exist, populate them.
        # Refresh button lives at the right of the vision+text row.
        self._btn_refresh_models = QPushButton("Refresh models")
        self._btn_refresh_models.setObjectName("ghost")
        self._btn_refresh_models.clicked.connect(self._populate_ollama_models)
        vt_row.addWidget(self._btn_refresh_models)

        self._populate_ollama_models()

        # Restore the legacy single-model selection from settings
        cfg = s.load_settings()
        current_model = cfg.get("ollama_model", "llava:7b")
        idx = self._ai_model.findText(current_model)
        if idx >= 0:
            self._ai_model.setCurrentIndex(idx)
        else:
            self._ai_model.setEditText(current_model)

        tb = QHBoxLayout()
        self.btn_add = QPushButton("New Category")
        self.btn_add.setObjectName("primary")
        self.btn_add.clicked.connect(self._add)
        self.btn_import = QPushButton("Import Images")
        self.btn_import.setObjectName("ghost")
        self.btn_import.clicked.connect(self._import_images)
        self.btn_gen = QPushButton("Generate (single AI)")
        self.btn_gen.setObjectName("ghost")
        self.btn_gen.setToolTip(
            "Use ONE Ollama model to both analyse images and write the prompt. "
            "Faster but tags tend to be generic."
        )
        self.btn_gen.clicked.connect(self._batch_gen)
        self.btn_regen = QPushButton("Regenerate (Vision + Text)")
        self.btn_regen.setObjectName("success")
        self.btn_regen.setToolTip(
            "Two-model pipeline: a vision LLM extracts structured tags + style "
            "from each sample, then a text LLM writes the unified prompt. "
            "Slower but produces much cleaner, more discriminating tags."
        )
        self.btn_regen.clicked.connect(self._toggle_regen)

        self._skip_videos = QCheckBox("Skip videos")
        self._skip_videos.setToolTip(
            "When checked, video files (mp4/mkv/webm/...) are excluded from "
            "the sample selection. Static images + GIFs only. Speeds up "
            "regeneration significantly on libraries heavy in videos."
        )
        self._skip_videos.setChecked(cfg.get("regen_skip_videos", False))
        self._skip_videos.stateChanged.connect(
            lambda _s: self._persist_regen_settings(skip_videos=self._skip_videos.isChecked())
        )
        self.btn_ref = QPushButton("Refresh")
        self.btn_ref.setObjectName("ghost")
        self.btn_ref.clicked.connect(self._refresh)
        self.btn_open_dest = QPushButton("Open destination")
        self.btn_open_dest.setObjectName("ghost")
        self.btn_open_dest.setToolTip("Open the destination folder in the system file manager.")
        self.btn_open_dest.clicked.connect(self._open_destination)
        tb.addWidget(self.btn_add)
        tb.addWidget(self.btn_import)
        tb.addWidget(self.btn_gen)
        tb.addWidget(self.btn_regen)
        tb.addWidget(self._skip_videos)
        tb.addStretch()
        tb.addWidget(self.btn_open_dest)
        tb.addWidget(self.btn_ref)
        l.addLayout(tb)

        # Secondary toolbar (config + AI helpers)
        tb2 = QHBoxLayout()
        self.btn_pattern = QPushButton("Build Patterns")
        self.btn_pattern.setObjectName("ghost")
        self.btn_pattern.setToolTip("Rebuild heuristic CV patterns from current sample images.")
        self.btn_pattern.clicked.connect(self._build_patterns)
        self.btn_config = QPushButton("Configure (Q&A)")
        self.btn_config.setObjectName("ghost")
        self.btn_config.setToolTip(
            "Open the questionnaire for the selected category. "
            "Sets aspect ratio / palette / style keywords that the "
            "classifier uses as a tie-breaker."
        )
        self.btn_config.clicked.connect(self._open_configure)
        self.btn_ai = QPushButton("AI Suggest")
        self.btn_ai.setObjectName("ghost")
        self.btn_ai.setToolTip(
            "Analyse sample images with CV + CLIP and propose an "
            "expected-content config for the selected category."
        )
        self.btn_ai.clicked.connect(self._open_ai_suggest)
        tb2.addWidget(self.btn_pattern)
        tb2.addWidget(self.btn_config)
        tb2.addWidget(self.btn_ai)
        tb2.addStretch()
        l.addLayout(tb2)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Category", "Images", "Tags / Prompt", "Actions", ""])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setMinimumSectionSize(min_button_column_width(4))
        self._table.setAlternatingRowColors(True)
        self._table.setColumnHidden(4, True)
        l.addWidget(self._table, 1)

        self._prog = QProgressBar()
        self._prog.setVisible(False)
        l.addWidget(self._prog)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(200)
        self._log.setFixedHeight(80)
        self._log.setVisible(False)
        l.addWidget(self._log)

        # Wire delegate for action buttons
        cbs = {
            "edit": lambda r, c, d: self._edit(d),
            "gen": lambda r, c, d: self._gen_single(d),
            "plus": lambda r, c, d: self._import_to(d),
            "del": lambda r, c, d: self._del(d),
        }
        setup_table_buttons(self._table, cbs, [3])

        self._on_ai_mode_changed()
        QTimer.singleShot(500, self._check_ai_status)

    def _populate_ollama_models(self):
        """Populate ALL three model dropdowns with locally installed models.

        Single-model dropdown (`_ai_model`) gets every model.
        Vision dropdown (`_vision_model`) only gets vision-capable models
        (those with `clip`/`mmproj` family or vision-suggesting name).
        Text dropdown (`_text_model`) gets every model (text models are
        a strict subset of installed models).

        We also annotate the vision list with a small marker so the user
        knows which one is for image analysis.
        """
        from ...ollama_client import OllamaClient
        cfg = s.load_settings()
        client = OllamaClient(
            base_url=cfg.get("ollama_url", "http://localhost:11434"),
            model="dummy",
            timeout=3,
        )

        def _extract_models(raw):
            out = []
            for m in raw:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("model") or ""
                    vision = bool(m.get("vision"))
                else:
                    name = str(m)
                    vision = client._supports_vision(name)
                if name:
                    out.append((name, vision))
            return out

        try:
            models = client.list_models()
            entries = _extract_models(models)
            if not entries:
                entries = [
                    ("llava:7b", True), ("llava:13b", True),
                    ("minicpm-v:8b", True), ("moondream:latest", True),
                    ("qwen2.5:3b", False), ("llama3.2:3b", False),
                ]
        except Exception as e:
            self._log.appendPlainText(f"[WARN] Could not list Ollama models: {e}")
            entries = [
                ("llava:7b", True), ("llava:13b", True),
                ("minicpm-v:8b", True), ("moondream:latest", True),
                ("qwen2.5:3b", False), ("llama3.2:3b", False),
            ]

        # 1) Single-model dropdown
        prev_ai = self._ai_model.currentText() if self._ai_model.count() > 0 else ""
        self._ai_model.clear()
        for name, _vision in entries:
            self._ai_model.addItem(name)
        if prev_ai:
            idx = self._ai_model.findText(prev_ai)
            if idx >= 0:
                self._ai_model.setCurrentIndex(idx)
            else:
                self._ai_model.setEditText(prev_ai)

        # 2) Vision dropdown (vision-only, annotated)
        prev_v = self._vision_model.currentText() if self._vision_model.count() > 0 else ""
        prev_v_clean = prev_v.split("  ")[0] if "  " in prev_v else prev_v
        self._vision_model.blockSignals(True)
        self._vision_model.clear()
        vision_entries = [(n, v) for n, v in entries if v]
        if not vision_entries:
            # Make sure the dropdown isn't empty even if Ollama reports no vision model
            vision_entries = [("llava-phi3:3.8b", True), ("llava:7b", True)]
        for name, _v in vision_entries:
            self._vision_model.addItem(f"{name}  (vision)")
        restored_v = False
        if prev_v_clean:
            for i in range(self._vision_model.count()):
                txt = self._vision_model.itemText(i)
                if txt.split("  ")[0] == prev_v_clean:
                    self._vision_model.setCurrentIndex(i)
                    restored_v = True
                    break
            if not restored_v:
                self._vision_model.setEditText(prev_v_clean)
                restored_v = bool(prev_v_clean.strip())
        self._vision_model.blockSignals(False)

        # 3) Text dropdown (all models)
        prev_t = self._text_model.currentText() if self._text_model.count() > 0 else ""
        prev_t_clean = prev_t.split("  ")[0] if "  " in prev_t else prev_t
        self._text_model.blockSignals(True)
        self._text_model.clear()
        for name, vision in entries:
            label = f"{name}  (vision)" if vision else name
            self._text_model.addItem(label)
        restored_t = False
        if prev_t_clean:
            for i in range(self._text_model.count()):
                txt = self._text_model.itemText(i)
                if txt.split("  ")[0] == prev_t_clean:
                    self._text_model.setCurrentIndex(i)
                    restored_t = True
                    break
            if not restored_t:
                self._text_model.setEditText(prev_t_clean)
                restored_t = bool(prev_t_clean.strip())
        self._text_model.blockSignals(False)

        # Apply first-time defaults ONLY if the user (or a previous session)
        # never picked anything. We track that with the restored_* flags above.
        cfg = s.load_settings()
        if not restored_v:
            v_pref = cfg.get("regen_vision_model", "llava-phi3:3.8b")
            # Try to find an exact match first, else fall back to first item
            for i in range(self._vision_model.count()):
                if self._vision_model.itemText(i).split("  ")[0] == v_pref:
                    self._vision_model.setCurrentIndex(i)
                    break
            else:
                self._vision_model.setCurrentIndex(0)
        if not restored_t:
            # Prefer a non-vision model as the default for synthesis.
            # qwen2.5:3b has been validated to produce good structured
            # prompts; 2B models like granite3.1-dense tend to truncate
            # or return prose-only outputs.
            v_pref = cfg.get("regen_text_model") or "qwen2.5:3b"
            text_only = [n for n, vis in entries if not vis]
            target = v_pref
            if target not in [n for n, _ in entries]:
                # Fallback: first non-vision model, else any model
                target = text_only[0] if text_only else (entries[0][0] if entries else "qwen2.5:3b")
            matched = False
            for i in range(self._text_model.count()):
                if self._text_model.itemText(i).split("  ")[0] == target:
                    self._text_model.setCurrentIndex(i)
                    matched = True
                    break
            if not matched:
                self._text_model.setEditText(target)

    def _on_ai_mode_changed(self):
        mode = self._ai_mode.currentData()
        if mode == "ollama":
            self._ai_model.setEnabled(True)
            self._ai_model_lbl.setText("Single model:")
        else:
            self._ai_model.setEnabled(False)
            self._ai_model_lbl.setText("Single model:")
        # Vision+Text row is always visible (it controls the regenerate button
        # which is its own pipeline, independent of the single-model row).

    def _check_ai_status(self):
        mode = self._ai_mode.currentData()
        if mode == "ollama":
            try:
                from ...ollama_client import OllamaClient
                cfg = s.load_settings()
                client = OllamaClient(
                    base_url=cfg.get("ollama_url", "http://localhost:11434"),
                    model=self._ai_model.currentText().strip(),
                    timeout=3,
                )
                if client.check():
                    self._ai_status.setText("Connected")
                    self._ai_status.setObjectName("statusOk")
                else:
                    self._ai_status.setText("Offline")
                    self._ai_status.setObjectName("statusErr")
            except Exception:
                self._ai_status.setText("Unavailable")
                self._ai_status.setObjectName("statusErr")
        else:
            self._ai_status.setText("Local analysis (no AI needed)")
            self._ai_status.setObjectName("statusOk")
        # re-apply stylesheet so objectName change takes effect
        self._ai_status.style().unpolish(self._ai_status)
        self._ai_status.style().polish(self._ai_status)

    def _persist_regen_settings(self, samples=None, skip_videos=None,
                              vision_model=None, text_model=None,
                              vision_timeout=None, text_timeout=None):
        """Persist the AI-regeneration UI state to disk so it survives restarts."""
        cfg = s.load_settings()
        if samples is not None:
            cfg["regen_samples"] = int(samples)
        if skip_videos is not None:
            cfg["regen_skip_videos"] = bool(skip_videos)
        if vision_model:
            cfg["regen_vision_model"] = vision_model
        if text_model:
            cfg["regen_text_model"] = text_model
        if vision_timeout is not None:
            cfg["regen_vision_timeout"] = int(vision_timeout)
        if text_timeout is not None:
            cfg["regen_text_timeout"] = int(text_timeout)
        s.save_settings(cfg)

    def _refresh(self):
        c.discover_categories()
        dest = s.resolve_dest_dir(s.load_settings())
        imgs_ext = f.STATIC_EXTENSIONS | f.ANIMATED_EXTENSIONS
        rows = []
        if os.path.isdir(dest):
            for e in sorted(os.listdir(dest)):
                p = os.path.join(dest, e)
                if not os.path.isdir(p) or e.startswith(".") or e in c.SPECIAL_FOLDERS:
                    continue
                hc = os.path.isfile(os.path.join(p, ".category.json"))
                cnt = sum(
                    1 for fn in os.listdir(p)
                    if os.path.isfile(os.path.join(p, fn))
                    and not fn.startswith(".")
                    and os.path.splitext(fn)[1].lower() in imgs_ext
                )
                rows.append((e, cnt, hc))
        conf = [(n, c2) for n, c2, h in rows if h]
        unconf = [(n, c2) for n, c2, h in rows if not h]
        self._table.setRowCount(len(rows))

        for i, (nm, cnt, hc) in enumerate(rows):
            cfg = c._read_category_config(nm) or {}
            exp = cfg.get("expected") or {}
            exp_source = exp.get("source", "")
            exp_marker = ""
            if exp_source and exp_source != "default":
                exp_marker = " \u2698"  # gear icon for "configured"
            self._table.setItem(i, 0, QTableWidgetItem(nm + (" *" if not hc else "") + exp_marker))
            self._table.setItem(i, 1, QTableWidgetItem(str(cnt)))
            tags = cfg.get("tags", [])
            prompt = cfg.get("prompt", "")
            desc_lines = []
            if tags:
                desc_lines.append(", ".join(tags[:6]))
            elif not hc:
                desc_lines.append("needs tags")
            if prompt:
                desc_lines.append(f"[prompt] {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
            if exp_source and exp_source != "default":
                exp_parts = []
                if exp.get("aspect_ratios"):
                    exp_parts.append("ar=" + "|".join(a[:3] for a in exp["aspect_ratios"]))
                if exp.get("color_palette"):
                    exp_parts.append("pal=" + "|".join(p[:3] for p in exp["color_palette"]))
                if exp.get("style_keywords"):
                    exp_parts.append("kw=" + ",".join(exp["style_keywords"][:4]))
                if exp_parts:
                    desc_lines.append("[" + exp_source + "] " + " ".join(exp_parts))
            desc = "\n".join(desc_lines) if desc_lines else ""
            self._table.setItem(i, 2, QTableWidgetItem(desc))
            self._table.setItem(i, 4, QTableWidgetItem(nm))

            act = QTableWidgetItem()
            act.setData(BTN_TEXT, "Edit|Gen|+|Del" if cnt > 0 else "Edit|+|Del")
            act.setData(BTN_KIND, "edit|gen|plus|del" if cnt > 0 else "edit|plus|del")
            act.setData(BTN_DATA, nm)
            self._table.setItem(i, 3, act)

        refresh_action_columns(self._table)
        self._info.setText(f"{len(conf)} configured, {len(unconf)} unconfigured categories")

    def _open_destination(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        dest = s.resolve_dest_dir(s.load_settings())
        os.makedirs(dest, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(dest))

    def _add(self):
        name, ok = QInputDialog.getText(self, "New Category", "Category name (folder name):")
        if not ok or not name.strip():
            return
        name = name.strip().replace(" ", "_")
        dest = s.resolve_dest_dir(s.load_settings())
        c.CATEGORIES_DIR = dest
        if c.create_category(name):
            c.discover_categories()
            self._refresh()
        else:
            QMessageBox.warning(self, "Exists", f"'{name}' already exists")

    def _selected_category(self) -> Optional[str]:
        """Return the name of the currently selected category, or None."""
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.information(
                self, "No selection",
                "Select a category in the table first."
            )
            return None
        idx = rows[0].row()
        item = self._table.item(idx, 4)  # hidden column with the name
        return item.text() if item else None

    def _open_configure(self):
        """Open the Q&A wizard for the selected category."""
        cat = self._selected_category()
        if not cat:
            return
        dlg = CategoryConfigDialog(cat, self)
        if dlg.exec():
            self._log.appendPlainText(f"[config] Updated expected config for '{cat}'")
            self._refresh()

    def _open_ai_suggest(self):
        """Open the AI-only suggest dialog for the selected category."""
        cat = self._selected_category()
        if not cat:
            return
        dlg = AIConfigDialog(cat, self)
        if dlg.exec():
            self._log.appendPlainText(f"[ai-config] Saved AI-suggested config for '{cat}'")
            self._refresh()

    def _import_images(self):
        src = QFileDialog.getExistingDirectory(self, "Choose folder with images")
        if not src:
            return
        imgs = [
            fn for fn in os.listdir(src)
            if os.path.isfile(os.path.join(src, fn))
            and not fn.startswith(".")
            and os.path.splitext(fn)[1].lower() in (f.STATIC_EXTENSIONS | f.ANIMATED_EXTENSIONS)
        ]
        if not imgs:
            QMessageBox.warning(self, "No Images", "No supported images found.")
            return
        dest = s.resolve_dest_dir(s.load_settings())
        c.CATEGORIES_DIR = dest
        c.discover_categories(dest)
        # Include all subfolders of dest (even those without .category.json)
        # so the user can import into freshly created but unconfigured folders.
        cats = list(c.CATEGORIES)
        if os.path.isdir(dest):
            for e in sorted(os.listdir(dest)):
                p = os.path.join(dest, e)
                if (os.path.isdir(p) and not e.startswith(".")
                        and e not in c.SPECIAL_FOLDERS and e not in cats):
                    cats.append(e)
        if not cats:
            QMessageBox.warning(self, "No Categories", "Create a category first.")
            return
        target, ok = QInputDialog.getItem(self, "Target Category", "Import into:", cats, 0, False)
        if not ok or not target:
            return
        self._import_to_cat(src, target)

    def _import_to(self, cat_name):
        src = QFileDialog.getExistingDirectory(self, f"Choose folder to import into '{cat_name}'")
        if not src:
            return
        self._import_to_cat(src, cat_name)

    def _import_to_cat(self, src, target):
        dest = s.resolve_dest_dir(s.load_settings())
        dst = os.path.join(dest, target)
        os.makedirs(dst, exist_ok=True)
        count = 0
        for fn in os.listdir(src):
            fp = os.path.join(src, fn)
            if not os.path.isfile(fp) or fn.startswith("."):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext not in (f.STATIC_EXTENSIONS | f.ANIMATED_EXTENSIONS):
                continue
            dp = os.path.join(dst, fn)
            if os.path.exists(dp):
                base, ex = os.path.splitext(fn)
                dp = os.path.join(dst, f"{base}_import{ex}")
            shutil.copy2(fp, dp)
            count += 1
        self._log.setVisible(True)
        self._log.appendPlainText(f"[OK] Imported {count} images into '{target}'")
        c.discover_categories()
        self._refresh()

    def _edit(self, name):
        cfg = c.get_category_config(name)
        dlg = CategoryFullDialog(self, name, cfg)
        if dlg.exec():
            nc = dlg.name_edit.text().strip()
            if not nc:
                return
            new_cfg = dict(cfg) if cfg else {}
            new_cfg["name"] = nc
            new_cfg["tags"] = dlg._tags
            new_cfg["prompt"] = dlg._prompt_edit.toPlainText().strip()
            dest = s.resolve_dest_dir(s.load_settings())
            c.CATEGORIES_DIR = dest
            c.write_category_config(name, new_cfg)
            if nc != name:
                nf = os.path.join(dest, nc)
                of = os.path.join(dest, name)
                if os.path.isdir(of) and not os.path.exists(nf):
                    os.rename(of, nf)
            c.discover_categories()
            self._refresh()

    def _gen_single(self, name):
        self._log.setVisible(True)
        self._prog.setVisible(True)
        self._log.clear()
        cfg = s.load_settings()
        mode = self._ai_mode.currentData() or "ollama"
        model = self._ai_model.currentText().strip() or "llava:7b"
        base_url = cfg.get("ollama_url", "http://localhost:11434")

        if hasattr(self, "_gen_worker") and self._gen_worker and self._gen_worker.isRunning():
            self._log.appendPlainText("[INFO] A generation is already running.")
            return

        self._gen_worker = GenerateTagsWorker(
            category_name=name,
            mode=mode,
            max_samples=5,
            model=model,
            base_url=base_url,
        )
        self._gen_worker.log.connect(self._log.appendPlainText)
        self._gen_worker.progress.connect(self._on_gen_progress)
        self._gen_worker.finished.connect(self._on_gen_finished)
        self._gen_worker.failed.connect(self._on_gen_failed)
        self._gen_worker.start()

    def _on_gen_progress(self, current, total, current_image):
        if current_image:
            self._prog.setMaximum(total)
            self._prog.setValue(current)
        self._log.appendPlainText(f"  [{current}/{total}] {current_image}")

    def _on_gen_finished(self, name, result):
        self._prog.setVisible(False)
        n_tags = len(result.get("tags", []))
        prompt_len = len(result.get("prompt", ""))
        samples = result.get("samples_used", 0)
        msg = f"[OK] {name}: {n_tags} tags"
        if prompt_len:
            msg += f", prompt {prompt_len} chars"
        msg += f" (from {samples} images)"
        self._log.appendPlainText(msg)
        c.discover_categories()
        self._refresh()
        if hasattr(self, "_gen_queue") and self._gen_queue:
            QTimer.singleShot(200, self._gen_next_in_queue)

    def _on_gen_failed(self, name, error):
        self._prog.setVisible(False)
        self._log.appendPlainText(f"[ERR] {name}: {error}")
        if not hasattr(self, "_gen_queue") or not self._gen_queue:
            QMessageBox.warning(self, "Generation Failed", f"Could not generate for '{name}':\n{error}")
        else:
            self._log.appendPlainText("  Continuing with next category...")
            QTimer.singleShot(200, self._gen_next_in_queue)

    # ------------------------------------------------------------------
    # Two-model pipeline (Vision LLM -> Text LLM)
    # ------------------------------------------------------------------

    def _vision_model_name(self) -> str:
        txt = self._vision_model.currentText().strip()
        # Drop the " (vision)" annotation suffix added in _populate_ollama_models
        if "  (" in txt:
            txt = txt.split("  (")[0]
        return txt

    def _text_model_name(self) -> str:
        txt = self._text_model.currentText().strip()
        if "  (" in txt:
            txt = txt.split("  (")[0]
        return txt

    def _regen_single(self, name):
        self._log.setVisible(True)
        self._prog.setVisible(True)
        self._prog.setMaximum(0)
        cfg = s.load_settings()
        base_url = cfg.get("ollama_url", "http://localhost:11434")
        vision_model = self._vision_model_name() or "llava-phi3:3.8b"
        text_model = self._text_model_name() or "qwen2.5:3b"
        samples = int(self._vt_samples.value())
        v_to = int(self._vt_vtimeout.value())
        t_to = int(self._vt_ttimeout.value())
        skip_videos = bool(self._skip_videos.isChecked())

        if hasattr(self, "_regen_worker") and self._regen_worker and self._regen_worker.isRunning():
            self._log.appendPlainText("[INFO] A regeneration is already running.")
            return

        self._persist_regen_settings(
            samples=samples,
            skip_videos=skip_videos,
            vision_model=vision_model,
            text_model=text_model,
            vision_timeout=v_to,
            text_timeout=t_to,
        )

        self._regen_worker = RegenerateCategoryWorker(
            category_name=name,
            vision_model=vision_model,
            text_model=text_model,
            base_url=base_url,
            max_samples=samples,
            vision_timeout=v_to,
            text_timeout=t_to,
            skip_videos=skip_videos,
        )
        self._regen_worker.log.connect(self._log.appendPlainText)
        self._regen_worker.progress.connect(self._on_gen_progress)
        self._regen_worker.finished.connect(self._on_regen_finished)
        self._regen_worker.failed.connect(self._on_regen_failed)
        self._regen_worker.start()
        # Update button to "Cancel" while running
        self.btn_regen.setText("Cancel regenerate")
        self.btn_regen.setObjectName("danger")

    def _toggle_regen(self):
        """If a regeneration is running, cancel it. Otherwise start a batch."""
        if hasattr(self, "_regen_worker") and self._regen_worker and self._regen_worker.isRunning():
            self._regen_worker.cancel()
            self._log.appendPlainText("[INFO] Cancellation requested - finishing current sample...")
            self.btn_regen.setEnabled(False)
            return
        self._batch_regen()

    def _reset_regen_button(self):
        self.btn_regen.setText("Regenerate (Vision + Text)")
        self.btn_regen.setObjectName("success")
        self.btn_regen.setEnabled(True)
        # Re-apply style so objectName change takes effect
        self.btn_regen.style().unpolish(self.btn_regen)
        self.btn_regen.style().polish(self.btn_regen)

    def _on_regen_finished(self, name, result):
        self._prog.setVisible(False)
        n_tags = len(result.get("tags", []))
        prompt_len = len(result.get("prompt", ""))
        samples = result.get("samples_used", 0)
        msg = f"[OK] {name}: {n_tags} tags"
        if prompt_len:
            msg += f", prompt {prompt_len} chars"
        msg += f" (from {samples} images, vision+text pipeline)"
        self._log.appendPlainText(msg)
        c.discover_categories()
        self._refresh()
        if hasattr(self, "_regen_queue") and self._regen_queue:
            QTimer.singleShot(200, self._regen_next_in_queue)
        else:
            self._reset_regen_button()

    def _on_regen_failed(self, name, error):
        self._prog.setVisible(False)
        self._log.appendPlainText(f"[ERR] {name}: {error}")
        if not hasattr(self, "_regen_queue") or not self._regen_queue:
            QMessageBox.warning(
                self, "Regenerate Failed",
                f"Could not regenerate '{name}':\n{error}",
            )
        else:
            self._log.appendPlainText("  Continuing with next category...")
            QTimer.singleShot(200, self._regen_next_in_queue)

    def _batch_regen(self):
        dest = s.resolve_dest_dir(s.load_settings())
        if not os.path.isdir(dest):
            return
        targets = []
        for e in sorted(os.listdir(dest)):
            p = os.path.join(dest, e)
            if not os.path.isdir(p) or e.startswith(".") or e in c.SPECIAL_FOLDERS:
                continue
            imgs = [
                fn for fn in os.listdir(p)
                if os.path.isfile(os.path.join(p, fn))
                and not fn.startswith(".")
                and os.path.splitext(fn)[1].lower() in (f.STATIC_EXTENSIONS | f.ANIMATED_EXTENSIONS)
            ]
            if imgs:
                targets.append(e)
        if not targets:
            QMessageBox.information(self, "Info", "No categories with images found.")
            return
        vm = self._vision_model_name()
        tm = self._text_model_name()
        n = self._vt_samples.value()
        if not QMessageBox.question(
            self, "Regenerate All (Vision + Text)",
            f"Regenerate tags + prompt for {len(targets)} category(ies) using:\n"
            f"  Vision model: {vm}\n"
            f"  Text model:   {tm}\n"
            f"  Samples per category: {n}\n\n"
            f"This uses TWO Ollama models per category and may take a while.\n"
            f"Existing tags + prompts will be overwritten.",
            QMessageBox.Yes | QMessageBox.No,
        ) == QMessageBox.Yes:
            return
        self._log.setVisible(True)
        self._prog.setVisible(False)
        self._log.clear()
        self._log.appendPlainText(
            f"Starting Vision+Text regeneration for {len(targets)} categories..."
        )
        self._regen_queue = list(targets)
        self._regen_next_in_queue()

    def _regen_next_in_queue(self):
        if not self._regen_queue:
            self._log.appendPlainText("[DONE] All categories regenerated.")
            c.discover_categories()
            self._refresh()
            self._reset_regen_button()
            return
        name = self._regen_queue.pop(0)
        self._log.appendPlainText(f"\n=== {name} ===")
        # Small delay between categories so the LLM server (and the
        # memory pressure on the host) has time to recover. Without
        # this, the 8th-10th categories in a batch tend to fail with
        # 'Ollama is not reachable' because the system is still busy
        # serving the previous one.
        QTimer.singleShot(5000, lambda: self._regen_single(name))

    def _batch_gen(self):
        dest = s.resolve_dest_dir(s.load_settings())
        if not os.path.isdir(dest):
            return
        targets = []
        for e in sorted(os.listdir(dest)):
            p = os.path.join(dest, e)
            if not os.path.isdir(p) or e.startswith(".") or e in c.SPECIAL_FOLDERS:
                continue
            imgs = [
                fn for fn in os.listdir(p)
                if os.path.isfile(os.path.join(p, fn))
                and not fn.startswith(".")
                and os.path.splitext(fn)[1].lower() in (f.STATIC_EXTENSIONS | f.ANIMATED_EXTENSIONS)
            ]
            if imgs:
                targets.append(e)
        if not targets:
            QMessageBox.information(self, "Info", "No categories with images found.")
            return
        if not QMessageBox.question(
            self, "Generate All",
            f"Generate tags and prompt for {len(targets)} category(ies) using "
            f"{self._ai_mode.currentText()}?\n\nThis may take several minutes.",
            QMessageBox.Yes | QMessageBox.No,
        ) == QMessageBox.Yes:
            return
        self._log.setVisible(True)
        self._prog.setVisible(False)
        self._log.clear()
        self._log.appendPlainText(f"Starting generation for {len(targets)} categories...")
        self._gen_queue = list(targets)
        self._gen_next_in_queue()

    def _gen_next_in_queue(self):
        if not self._gen_queue:
            self._log.appendPlainText("[DONE] All categories generated.")
            c.discover_categories()
            self._refresh()
            return
        name = self._gen_queue.pop(0)
        self._log.appendPlainText(f"\n--- {name} ---")
        self._gen_single(name)
        # _on_gen_finished triggers next via re-queue

    def _del(self, name):
        r = QMessageBox.question(
            self, "Delete", f"Delete category '{name}' and ALL files inside?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if r == QMessageBox.Yes:
            if c.delete_category(name):
                c.discover_categories()
                self._refresh()

    def _build_patterns(self):
        """Build CV heuristic patterns for all categories from sample images."""
        dest = s.resolve_dest_dir(s.load_settings())
        if not os.path.isdir(dest):
            QMessageBox.warning(self, "Error", "Destination folder not found.")
            return
        targets = []
        for e in sorted(os.listdir(dest)):
            p = os.path.join(dest, e)
            if not os.path.isdir(p) or e.startswith(".") or e in c.SPECIAL_FOLDERS:
                continue
            imgs = [
                fn for fn in os.listdir(p)
                if os.path.isfile(os.path.join(p, fn))
                and not fn.startswith(".")
                and os.path.splitext(fn)[1].lower() in (f.STATIC_EXTENSIONS | f.ANIMATED_EXTENSIONS)
            ]
            if imgs:
                targets.append(e)
        if not targets:
            QMessageBox.information(self, "Info", "No categories with images found.")
            return
        r = QMessageBox.question(
            self, "Build Patterns",
            f"Analyze {len(targets)} categories to build CV heuristic patterns?\n\n"
            "This scans up to 15 sample images per category and computes\n"
            "statistical fingerprints (color, texture, edges, composition).\n\n"
            "Patterns are saved to .category.json and will improve\n"
            "Low-Level CV classification accuracy.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if r != QMessageBox.Yes:
            return
        self._log.setVisible(True)
        self._prog.setVisible(True)
        self._log.clear()
        self._log.appendPlainText(f"Building patterns for {len(targets)} categories...")
        from ...lowlevel.category_profile import build_all_category_profiles
        c.CATEGORIES_DIR = dest
        c.discover_categories(dest)
        def progress(cur, total, fname):
            self._prog.setMaximum(total)
            self._prog.setValue(cur)
            self._log.appendPlainText(f"  [{cur}/{total}] {fname}")
        results = build_all_category_profiles(
            dest, max_samples=15, progress_callback=progress,
        )
        self._prog.setVisible(False)
        if results:
            self._log.appendPlainText(
                f"[OK] Built patterns for {len(results)} categories\n"
                f"  Pattern now used by Low-Level CV classification (+20% weight)"
            )
        else:
            self._log.appendPlainText("[WARN] Could not build any patterns")
        self._refresh()


class CategoryFullDialog(QDialog):
    def __init__(self, parent, name, cfg):
        super().__init__(parent)
        self.setWindowTitle(f"Edit Category: {name}")
        self.resize(520, 400)
        self._cfg = cfg
        self._tags = list(cfg.get("tags", []))

        l = QVBoxLayout(self)
        gf = QFormLayout()
        self.name_edit = QLineEdit(cfg.get("name", name))
        gf.addRow("Name:", self.name_edit)
        l.addLayout(gf)

        l.addWidget(QLabel("Tags (comma-separated):"))
        self._tags_edit = QPlainTextEdit()
        self._tags_edit.setPlainText(", ".join(self._tags))
        self._tags_edit.setFixedHeight(60)
        l.addWidget(self._tags_edit)

        l.addWidget(QLabel("Style Prompt (describes aesthetic for AI classification):"))
        self._prompt_edit = QPlainTextEdit()
        self._prompt_edit.setPlainText(cfg.get("prompt", ""))
        self._prompt_edit.setFixedHeight(80)
        self._prompt_edit.setPlaceholderText("e.g. Dark cyberpunk city with neon purple and teal lighting")
        l.addWidget(self._prompt_edit)

        l.addWidget(QLabel(
            "How to use: 1) Place sample images in the category folder  "
            '2) Click "Gen" in the category list to auto-generate tags + prompt via AI  '
            "3) Files matching these tags will be organized here",
            objectName="statSmall",
        ))
        l.addStretch()

        hb = QHBoxLayout()
        ok = QPushButton("Save")
        ok.setObjectName("primary")
        ok.clicked.connect(self._save)
        ca = QPushButton("Cancel")
        ca.setObjectName("ghost")
        ca.clicked.connect(self.reject)
        hb.addStretch()
        hb.addWidget(ca)
        hb.addWidget(ok)
        l.addLayout(hb)

    def _save(self):
        txt = self._tags_edit.toPlainText().strip()
        self._tags = [t.strip() for t in txt.split(",") if t.strip()]
        self.accept()
