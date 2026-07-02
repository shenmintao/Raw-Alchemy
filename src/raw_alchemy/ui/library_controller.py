import os

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QFileDialog, QListWidgetItem, QMessageBox, QTreeView
from qfluentwidgets import InfoBar

from raw_alchemy.i18n import tr
from raw_alchemy.workers.thumbnail_worker import ThumbnailWorker


MARK_PREFIX = "\u9983\u715d"


class LibraryControllerMixin:
    """Folder browser, gallery, navigation, and compare actions for MainWindow."""

    def browse_folder(self):
        start_dir = (
            self.last_folder_path
            if self.last_folder_path and os.path.exists(self.last_folder_path)
            else ""
        )
        folder = QFileDialog.getExistingDirectory(self, tr("select_folder"), start_dir)
        if folder:
            self._open_folder(folder)

    def _on_folder_tree_clicked(self, index):
        path = self.folder_model.filePath(index)
        if path and os.path.isdir(path):
            self._open_folder(path)

    def _open_folder(self, folder):
        self._persist_current_sidecar_now()
        self.current_folder = folder
        self.last_folder_path = folder
        self.gallery_items_by_path.clear()
        self.gallery_list.clear()
        self._load_sidecars_for_folder(folder)

        self._pending_scroll_path = folder
        idx = self.folder_model.index(folder)
        if idx.isValid():
            ancestors = []
            p = idx.parent()
            while p.isValid():
                ancestors.append(p)
                p = p.parent()
            if ancestors:
                for ancestor in reversed(ancestors):
                    self.folder_tree.expand(ancestor)
                QTimer.singleShot(100, lambda f=folder: self._try_finalize_tree_selection(f))
            else:
                self._pending_scroll_path = None
                self._expand_tree_to(folder)
        else:
            drive = os.path.splitdrive(folder)[0]
            if drive:
                root_path = drive + os.sep
                root_idx = self.folder_model.index(root_path)
                if root_idx.isValid():
                    self.folder_tree.expand(root_idx)

        self.start_thumbnail_scan(folder)

    def _try_finalize_tree_selection(self, folder):
        if self._pending_scroll_path == folder:
            self._pending_scroll_path = None
            self._expand_tree_to(folder)

    def _expand_tree_to(self, folder):
        idx = self.folder_model.index(folder)
        if not idx.isValid():
            return
        ancestors = []
        p = idx.parent()
        while p.isValid():
            ancestors.append(p)
            p = p.parent()
        for ancestor in reversed(ancestors):
            self.folder_tree.expand(ancestor)
        self.folder_tree.setCurrentIndex(idx)
        self._scroll_target_path = folder
        self._scroll_debounce_timer.start()

    def _do_final_scroll(self):
        if not self._scroll_target_path:
            return
        idx = self.folder_model.index(self._scroll_target_path)
        if idx.isValid():
            self.folder_tree.scrollTo(idx, QTreeView.ScrollHint.PositionAtCenter)
        self._scroll_target_path = None

    def start_thumbnail_scan(self, folder):
        # Never wait() on the UI thread: retire the old worker asynchronously
        # (stop flag + finished -> deleteLater) and start the new one at once.
        self._retire_thumbnail_worker()

        self.loading_label.setText(tr("loading_thumbnails"))
        self.loading_label.show()

        worker = ThumbnailWorker(folder)
        self.thumb_worker = worker
        worker.placeholders_ready.connect(self._on_thumbnail_placeholders)
        worker.thumbnail_ready.connect(self._on_thumbnail_image_ready)
        worker.progress_update.connect(self._on_thumbnail_progress_update)
        worker.finished_scanning.connect(self._on_thumbnail_scan_finished)
        worker.start(QThread.Priority.LowPriority)

    def _retire_thumbnail_worker(self):
        """Stop the current worker without blocking the UI thread."""
        worker = getattr(self, "thumb_worker", None)
        self.thumb_worker = None
        if worker is None:
            return
        worker.stop()
        if not worker.isRunning():
            worker.deleteLater()
            return
        # Keep a python reference until the thread finishes, otherwise the
        # QThread C++ object could be destroyed while still running.
        self._retired_thumb_workers.append(worker)
        worker.finished.connect(self._on_retired_thumb_worker_finished)
        if worker.isFinished():
            # finished may have been emitted before we connected
            self._reap_thumbnail_worker(worker)

    def _on_retired_thumb_worker_finished(self):
        worker = self.sender()
        if worker is not None:
            self._reap_thumbnail_worker(worker)

    def _reap_thumbnail_worker(self, worker):
        if worker in self._retired_thumb_workers:
            self._retired_thumb_workers.remove(worker)
            worker.deleteLater()

    # --- Worker signal handlers -------------------------------------------
    # A retired worker may still have queued emissions in flight; guard every
    # handler so stale signals from a previous folder never touch the gallery.

    def _on_thumbnail_placeholders(self, paths, image):
        if self.sender() is not self.thumb_worker:
            return
        self.add_gallery_items(paths, image)
        self._update_thumbnail_priority()

    def _on_thumbnail_image_ready(self, path, image):
        if self.sender() is not self.thumb_worker:
            return
        self.update_gallery_item(path, image)

    def _on_thumbnail_progress_update(self, current, total):
        if self.sender() is not self.thumb_worker:
            return
        self.on_thumbnail_progress(current, total)

    def _on_thumbnail_scan_finished(self):
        if self.sender() is not self.thumb_worker:
            return
        self.on_thumbnail_finished()

    # --- Viewport-priority scheduling --------------------------------------

    def _on_gallery_scrolled(self, _value=0):
        self._thumb_priority_timer.start()

    def _update_thumbnail_priority(self):
        worker = getattr(self, "thumb_worker", None)
        if worker is None:
            return
        worker.set_priority_paths(self._visible_gallery_paths())

    def _visible_gallery_paths(self):
        """Paths of gallery items in (or within one viewport height of) the
        visible area, in row order."""
        gallery = self.gallery_list
        viewport_rect = gallery.viewport().rect()
        if not viewport_rect.isValid():
            return []
        probe = viewport_rect.adjusted(
            0, -viewport_rect.height(), 0, viewport_rect.height()
        )
        paths = []
        seen_visible = False
        for row in range(gallery.count()):
            item = gallery.item(row)
            rect = gallery.visualItemRect(item)
            if rect.intersects(probe):
                paths.append(item.data(Qt.ItemDataRole.UserRole))
                seen_visible = True
            elif seen_visible and rect.top() > probe.bottom():
                break  # rows are laid out top-to-bottom; rest is below
        return paths

    def update_status(self, text):
        self.loading_label.setText(text)
        if text:
            self.loading_label.show()
        else:
            self.loading_label.hide()

    def on_thumbnail_progress(self, current, total):
        self.loading_label.setText(f"{tr('loading_thumbnails')}: {current}/{total}")

    def on_thumbnail_finished(self):
        self.loading_label.hide()
        self._preload_neighbors(0, count=2)

    def add_gallery_items(self, paths, image):
        """Add a batch of placeholder items sharing one QIcon (single relayout)."""
        icon = QIcon(QPixmap.fromImage(image))
        self.gallery_list.setUpdatesEnabled(False)
        try:
            for path in paths:
                name = os.path.basename(path)
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, path)
                item.setIcon(icon)
                is_marked = path in self.marked_files
                item.setText(f"{MARK_PREFIX} {name}" if is_marked else name)
                self.gallery_list.addItem(item)
                self.gallery_items_by_path[path] = item
        finally:
            self.gallery_list.setUpdatesEnabled(True)

    def add_gallery_item(self, path, image):
        name = os.path.basename(path)
        pixmap = QPixmap.fromImage(image)

        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setIcon(QIcon(pixmap))

        is_marked = path in self.marked_files
        item.setText(f"{MARK_PREFIX} {name}" if is_marked else name)

        self.gallery_list.addItem(item)
        self.gallery_items_by_path[path] = item

    def update_gallery_item(self, path, image):
        item = self.gallery_items_by_path.get(path)
        if item is None:
            return
        pixmap = QPixmap.fromImage(image)
        item.setIcon(QIcon(pixmap))
        rect = self.gallery_list.visualItemRect(item)
        self.gallery_list.viewport().update(rect)

    def on_gallery_item_clicked(self, item):
        if not item:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if path == self.current_raw_path:
            return

        if self.current_raw_path:
            self._persist_current_sidecar_now()

        self.current_raw_path = path
        self.update_window_title()
        self._load_sidecar_for_path(path)

        self.original.clear()
        self.current.clear()
        self.baseline.clear()

        if self.right_panel.auto_exp_radio.isChecked():
            self.right_panel.auto_ev_value = 0.0
            self.right_panel.exp_slider.blockSignals(True)
            self.right_panel.exp_slider.setValue(0)
            self.right_panel.exp_slider.update()
            self.right_panel.exp_slider.blockSignals(False)
            self.right_panel.exp_value_label.setText(f"{tr('exposure_ev')}: 0.0")

        if path in self.file_params_cache:
            self.right_panel.set_params(self.file_params_cache[path])
        else:
            current_sticky_params = self.right_panel.get_params()
            current_sticky_params["rotation"] = 0
            current_sticky_params["flip_horizontal"] = False
            current_sticky_params["flip_vertical"] = False
            current_sticky_params["crop"] = (0.0, 0.0, 1.0, 1.0)
            self.right_panel.set_params(current_sticky_params)

        self.update_mark_button_state()
        self.load_image(path)

        current_row = self.gallery_list.row(item)
        self._preload_neighbors(current_row)

        if path in self.file_baseline_params_cache:
            QTimer.singleShot(25, self.regenerate_baseline_for_current_image)

    def load_image(self, path):
        self.preview_lbl.setText(tr("loading"))
        self.viewport.clear_image()
        self.current_request_id = self.processor.load_image(path)

    def _preload_neighbors(self, current_index, count=2):
        self._preload_neighbors_args = (current_index, count, self.current_raw_path)
        self._preload_neighbors_timer.start()

    def _run_preload_neighbors(self):
        if not self._preload_neighbors_args:
            return
        current_index, count, expected_path = self._preload_neighbors_args
        self._preload_neighbors_args = None
        if expected_path != self.current_raw_path:
            return

        total = self.gallery_list.count()
        for offset in range(1, count + 1):
            for idx in [current_index + offset, current_index - offset]:
                if 0 <= idx < total:
                    path = self.gallery_list.item(idx).data(Qt.ItemDataRole.UserRole)
                    if not self.processor.cache_manager.get(path):
                        self.processor.preload_image(path)

    def prev_image(self):
        count = self.gallery_list.count()
        if count == 0:
            return
        row = self.gallery_list.currentRow()
        new_row = (row - 1) % count
        self.gallery_list.setCurrentRow(new_row)

    def next_image(self):
        count = self.gallery_list.count()
        if count == 0:
            return
        row = self.gallery_list.currentRow()
        new_row = (row + 1) % count
        self.gallery_list.setCurrentRow(new_row)

    def toggle_mark(self):
        if not self.current_raw_path:
            return
        if self.current_raw_path in self.marked_files:
            self.marked_files.remove(self.current_raw_path)
            InfoBar.info(tr("unmarked"), os.path.basename(self.current_raw_path), parent=self)
        else:
            self.marked_files.add(self.current_raw_path)
            InfoBar.success(tr("marked"), os.path.basename(self.current_raw_path), parent=self)
        self.update_mark_button_state()
        self.update_gallery_item_mark_indicator(self.current_raw_path)
        self._write_sidecar_for_path(self.current_raw_path)

    def update_mark_button_state(self):
        if not self.current_raw_path:
            self.btn_mark.setChecked(False)
            return
        self.btn_mark.blockSignals(True)
        self.btn_mark.setChecked(self.current_raw_path in self.marked_files)
        self.btn_mark.blockSignals(False)

    def update_gallery_item_mark_indicator(self, path):
        item = self.gallery_items_by_path.get(path)
        if item is None:
            return
        is_marked = path in self.marked_files
        name = os.path.basename(path)
        item.setText(f"{MARK_PREFIX} {name}" if is_marked else name)

    def delete_image(self):
        if not self.current_raw_path:
            return

        reply = QMessageBox.question(
            self,
            tr("delete_image"),
            tr("confirm_delete", filename=os.path.basename(self.current_raw_path)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                import send2trash

                send2trash.send2trash(os.path.normpath(self.current_raw_path))
                if self.current_raw_path in self.marked_files:
                    self.marked_files.remove(self.current_raw_path)
                if self.current_raw_path in self.file_params_cache:
                    del self.file_params_cache[self.current_raw_path]
                if self.current_raw_path in self.file_baseline_params_cache:
                    del self.file_baseline_params_cache[self.current_raw_path]

                current_row = self.gallery_list.currentRow()
                item = self.gallery_items_by_path.pop(self.current_raw_path, None)
                if item is not None:
                    row = self.gallery_list.row(item)
                    if row >= 0:
                        self.gallery_list.takeItem(row)

                if self.gallery_list.count() > 0:
                    if current_row >= self.gallery_list.count():
                        current_row = self.gallery_list.count() - 1
                    self.gallery_list.setCurrentRow(current_row)
                else:
                    self.current_raw_path = None
                    self.preview_lbl.setText(tr("no_image_selected"))
                    self.update_window_title()
                InfoBar.success(tr("delete_image"), tr("delete_image"), parent=self)
            except Exception:
                try:
                    os.remove(self.current_raw_path)
                    if self.current_raw_path in self.marked_files:
                        self.marked_files.remove(self.current_raw_path)
                    current_row = self.gallery_list.currentRow()
                    item = self.gallery_items_by_path.pop(self.current_raw_path, None)
                    if item is not None:
                        row = self.gallery_list.row(item)
                        if row >= 0:
                            self.gallery_list.takeItem(row)
                    if self.gallery_list.count() > 0:
                        if current_row >= self.gallery_list.count():
                            current_row = self.gallery_list.count() - 1
                        self.gallery_list.setCurrentRow(current_row)
                    else:
                        self.current_raw_path = None
                        self.preview_lbl.setText(tr("no_image_selected"))
                        self.update_window_title()
                    InfoBar.success(tr("delete_image"), tr("delete_image"), parent=self)
                except Exception as e2:
                    InfoBar.error(tr("delete_failed"), str(e2), parent=self)

    def show_original(self):
        img_to_show = self.baseline if self.baseline.full else self.original
        if not img_to_show.full:
            return
        if img_to_show.uint8_data is not None:
            self.viewport.set_image(img_to_show.uint8_data, source_size=img_to_show.source_size)
        InfoBar.info(tr("compare_showing_baseline"), "", parent=self)

    def show_processed(self):
        if not self.current.full:
            if self.original.uint8_data is not None:
                self.viewport.set_image(
                    self.original.uint8_data, source_size=self.original.source_size
                )
            return
        if self.current.uint8_data is not None:
            self.viewport.set_image(self.current.uint8_data, source_size=self.current.source_size)
