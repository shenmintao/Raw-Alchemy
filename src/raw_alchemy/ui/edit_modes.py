import numpy as np
from PySide6.QtGui import QImage, QPixmap
from loguru import logger

from raw_alchemy.i18n import tr


class EditModesMixin:
    """Crop/rotate and perspective mode orchestration for MainWindow."""

    def enter_crop_mode(self):
        if not self.current_raw_path:
            return

        if self.center_stack.currentWidget() == self.crop_viewer:
            self.crop_viewer._on_done()
            return

        logger.info("Entering Crop Mode...")
        self.update_timer.stop()

        params = self.file_params_cache.get(self.current_raw_path, {}).copy()
        base_params = params.copy()
        base_params["rotation"] = 0
        base_params["flip_horizontal"] = False
        base_params["flip_vertical"] = False
        base_params["crop"] = (0.0, 0.0, 1.0, 1.0)

        self.processor_connection_mode = "crop"
        self.crop_request_id = self.processor.update_preview(self.current_raw_path, base_params)
        self.update_status(tr("loading_crop_view"))

    def on_crop_ready(self, img_uint8, params_context):
        h, w, c = img_uint8.shape
        img = QImage(img_uint8.data, w, h, c * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)

        current_params = self.file_params_cache.get(self.current_raw_path, {})
        self.crop_viewer.set_image(pixmap, current_params)

        self.center_stack.setCurrentWidget(self.crop_viewer)
        self.loading_label.hide()

    def on_crop_applied(self, rotation, flip_h, flip_v, crop_rect):
        self.right_panel.update_crop_params(rotation, flip_h, flip_v, crop_rect)
        self.exit_crop_mode()

    def exit_crop_mode(self):
        logger.info("Exiting crop mode...")

        try:
            if self.crop_viewer.display_pixmap:
                full_pix = self.crop_viewer.display_pixmap
                w = full_pix.width()
                h = full_pix.height()
                cx, cy, cw, ch = self.crop_viewer.crop_rect
                px = int(cx * w)
                py = int(cy * h)
                pw = int(cw * w)
                ph = int(ch * h)
                if pw > 0 and ph > 0:
                    temp_preview = full_pix.copy(px, py, pw, ph)
                    qimg = temp_preview.toImage().convertToFormat(QImage.Format.Format_RGB888)
                    ptr = qimg.bits()
                    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
                        qimg.height(), qimg.width(), 3
                    ).copy()
                    self.viewport.set_image(arr)
        except Exception as e:
            logger.warning(f"Failed to generate temp preview: {e}")

        self.processor_connection_mode = "normal"
        self.center_stack.setCurrentWidget(self.page_preview)
        self.update_timer.stop()

        if self.current_raw_path:
            self.file_params_cache[self.current_raw_path] = self.right_panel.get_params()
            self._schedule_current_sidecar_write()
            self.trigger_processing()

    def enter_perspective_mode(self):
        if not self.current_raw_path:
            return

        if self.center_stack.currentWidget() == self.perspective_viewer:
            self.perspective_viewer._on_done()
            return

        logger.info("Entering Perspective Mode...")
        self.update_timer.stop()

        params = self.file_params_cache.get(self.current_raw_path, {}).copy()
        base_params = params.copy()
        base_params["rotation"] = 0
        base_params["flip_horizontal"] = False
        base_params["flip_vertical"] = False
        base_params["perspective_corners"] = None
        base_params["crop"] = (0.0, 0.0, 1.0, 1.0)

        self.processor_connection_mode = "perspective"
        self.perspective_request_id = self.processor.update_preview(
            self.current_raw_path, base_params
        )
        self.update_status(tr("loading_perspective_view"))

    def on_perspective_ready(self, img_uint8, params_context):
        h, w, c = img_uint8.shape
        img = QImage(img_uint8.data, w, h, c * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)

        current_params = self.file_params_cache.get(self.current_raw_path, {})
        self.perspective_viewer.set_image(pixmap, current_params)

        self.center_stack.setCurrentWidget(self.perspective_viewer)
        self.loading_label.hide()

    def on_perspective_applied(self, corners):
        if self.current_raw_path:
            current_params = self.file_params_cache.get(self.current_raw_path, {})
            current_params["perspective_corners"] = corners
            self.file_params_cache[self.current_raw_path] = current_params
            self.right_panel.set_params(current_params)

        self.exit_perspective_mode()

    def exit_perspective_mode(self):
        logger.info("Exiting perspective mode...")

        self.processor_connection_mode = "normal"
        self.center_stack.setCurrentWidget(self.page_preview)
        self.update_timer.stop()

        if self.current_raw_path:
            self.file_params_cache[self.current_raw_path] = self.right_panel.get_params()
            self._schedule_current_sidecar_write()
            self.trigger_processing()
