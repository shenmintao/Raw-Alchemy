import os

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QFileDialog
from qfluentwidgets import (
    BodyLabel,
    CheckBox,
    ComboBox,
    InfoBar,
    MessageBoxBase,
    SubtitleLabel,
)

from raw_alchemy import orchestrator
from raw_alchemy.exporting import resolve_export_exposure
from raw_alchemy.i18n import tr


class BatchExportDialog(MessageBoxBase):
    """Dialog for batch export settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.titleLabel = SubtitleLabel(tr("batch_export_settings"), self)

        self.formatLabel = BodyLabel(tr("select_format"), self)
        self.formatComboBox = ComboBox(self)
        self.formatComboBox.addItems(["JPEG", "HEIF", "HDR HEIF", "TIFF", "DNG"])
        self.formatComboBox.setCurrentIndex(0)

        self.disableLutCheckBox = CheckBox(tr("disable_lut_globally"), self)
        self.disableLutCheckBox.setChecked(False)

        self.viewLayout.addWidget(self.titleLabel)
        self.viewLayout.addWidget(self.formatLabel)
        self.viewLayout.addWidget(self.formatComboBox)
        self.viewLayout.addWidget(self.disableLutCheckBox)

        self.widget.setMinimumWidth(350)

    def get_settings(self):
        return {
            "format": self.formatComboBox.currentText(),
            "ignore_lut": self.disableLutCheckBox.isChecked(),
        }


class ExportControllerMixin:
    """Single-image and batch export orchestration for MainWindow."""

    def export_current(self):
        if not self.current_raw_path:
            return
        base = os.path.splitext(os.path.basename(self.current_raw_path))[0]
        start_dir = self.last_export_path if self.last_export_path else (
            self.last_folder_path if self.last_folder_path else ""
        )
        default_path = os.path.join(start_dir, base) if start_dir else base

        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("export_image"),
            default_path,
            "JPEG (*.jpg);;HEIF (*.heif);;HDR HEIF (*.hdr.heif);;TIFF (*.tif);;DNG (*.dng)",
        )

        if path:
            self.last_export_path = os.path.dirname(path)
            self.saving_infobar = InfoBar.info(
                tr("saving"), tr("saving_image"), duration=-1, parent=self
            )
            self.btn_export_curr.setEnabled(False)

            cached_data = self.processor.get_cached_for_export()
            if cached_data is not None:
                self.run_export(
                    self.current_raw_path,
                    path,
                    is_single_export=True,
                    cached_img=cached_data["corrected"],
                    cached_exif_data=cached_data.get("exif_data"),
                    cached_exif_metadata=cached_data.get("exif_metadata"),
                )
            else:
                self.run_export(self.current_raw_path, path, is_single_export=True)

    def export_all(self):
        if not self.marked_files:
            InfoBar.warning(tr("no_files_marked"), tr("please_mark_files"), parent=self)
            return

        dialog = BatchExportDialog(self)
        if dialog.exec():
            settings = dialog.get_settings()
            format_str = settings["format"]
            self.batch_export_ignore_lut = settings["ignore_lut"]
        else:
            return

        start_dir = self.last_export_path if self.last_export_path else (
            self.last_folder_path if self.last_folder_path else ""
        )
        folder = QFileDialog.getExistingDirectory(self, tr("select_export_folder"), start_dir)

        if folder:
            self.last_export_path = folder
            self.batch_export_list = list(self.marked_files)
            self.batch_export_folder = folder
            fmt_map = {
                "JPEG": "jpg",
                "HEIF": "heif",
                "HDR HEIF": "hdr.heif",
                "TIFF": "tif",
                "DNG": "dng",
            }
            self.batch_export_ext = fmt_map.get(format_str, "jpg")

            self.batch_saving_infobar = InfoBar.info(
                tr("saving"), tr("batch_exporting"), duration=-1, parent=self
            )
            self.export_progress.setRange(0, len(self.batch_export_list))
            self.export_progress.setValue(0)
            self.export_progress.show()
            self.btn_export_all.setEnabled(False)
            self.batch_export_idx = 0
            self.batch_export_next()

    def batch_export_next(self):
        if self.batch_export_idx >= len(self.batch_export_list):
            if hasattr(self, "batch_saving_infobar") and self.batch_saving_infobar:
                try:
                    self.batch_saving_infobar.close()
                except RuntimeError:
                    pass
                self.batch_saving_infobar = None
            InfoBar.success(tr("batch_export"), tr("all_exported"), parent=self)
            self.export_progress.hide()
            self.btn_export_all.setEnabled(True)
            return

        self.export_progress.setValue(self.batch_export_idx)
        input_path = self.batch_export_list[self.batch_export_idx]
        filename = os.path.basename(input_path)
        output_path = os.path.join(
            self.batch_export_folder,
            os.path.splitext(filename)[0] + "." + self.batch_export_ext,
        )

        params = None
        if input_path == self.current_raw_path:
            params = self.right_panel.get_params().copy()
        else:
            params = self.file_params_cache.get(input_path)
            if params is None:
                params = self._load_sidecar_for_path(input_path)
            if params:
                params = params.copy()

        if params and hasattr(self, "batch_export_ignore_lut") and self.batch_export_ignore_lut:
            params["lut_path"] = None

        self.batch_export_idx += 1
        self.run_export(input_path, output_path, params=params, callback=self.batch_export_next)

    def run_export(
        self,
        input_path,
        output_path,
        params=None,
        callback=None,
        is_single_export=False,
        cached_img=None,
        cached_exif_data=None,
        cached_exif_metadata=None,
    ):
        p = params if params else self.right_panel.get_params()

        hdr_output = output_path.lower().endswith(".hdr.heif")
        ext = os.path.splitext(output_path)[1].lower().replace(".", "")
        if ext not in ["jpg", "heif", "tif", "tiff", "dng"]:
            ext = "jpg"

        _fast_path_exposure, _full_path_exposure = resolve_export_exposure(
            p,
            getattr(self.processor, "last_applied_ev", p.get("exposure", 0.0)),
        )
        _perspective_corners = p.get("perspective_corners")

        def on_finish(success, msg):
            if is_single_export:
                if hasattr(self, "saving_infobar") and self.saving_infobar:
                    try:
                        self.saving_infobar.close()
                    except RuntimeError:
                        pass
                    self.saving_infobar = None
                self.btn_export_curr.setEnabled(True)

            if success:
                if not callback:
                    InfoBar.success(
                        tr("export_success"),
                        tr("saved_to", path=os.path.basename(output_path)),
                        parent=self,
                    )
                if callback:
                    callback()
            else:
                InfoBar.error(tr("export_failed"), msg, parent=self)

        if cached_img is not None:
            export_payload = {
                "cached_img": cached_img,
                "output_path": output_path,
                "exif_data": cached_exif_data,
                "exif_metadata": cached_exif_metadata,
                "log_space": p.get("log_space"),
                "lut_path": p.get("lut_path"),
                "exposure": _fast_path_exposure,
                "metering_mode": p.get("metering_mode", "matrix"),
                "wb_temp": p.get("wb_temp", 0.0),
                "wb_tint": p.get("wb_tint", 0.0),
                "saturation": p.get("saturation", 1.0),
                "contrast": p.get("contrast", 1.0),
                "highlight": p.get("highlight", 0.0),
                "shadow": p.get("shadow", 0.0),
                "rotation": p.get("rotation", 0),
                "flip_horizontal": p.get("flip_horizontal", False),
                "flip_vertical": p.get("flip_vertical", False),
                "perspective_corners": _perspective_corners,
                "crop": p.get("crop", (0.0, 0.0, 1.0, 1.0)),
                "sharpen_strength": p.get("sharpen_strength", 0.0),
                "hdr_output": hdr_output,
            }

            previous_handler = getattr(self, "_processor_export_finished_handler", None)
            if previous_handler is not None:
                try:
                    self.processor.export_finished.disconnect(previous_handler)
                except (TypeError, RuntimeError):
                    pass
            self._processor_export_finished_handler = on_finish
            self.processor.export_finished.connect(on_finish)
            self.processor.export_from_cache(input_path, export_payload)
            return

        class ExportThread(QThread):
            finished_sig = Signal(bool, str)

            def run(self):
                try:
                    orchestrator.process_path(
                        input_path=input_path,
                        output_path=output_path,
                        log_space=p.get("log_space"),
                        lut_path=p.get("lut_path"),
                        exposure=_full_path_exposure,
                        lens_correct=p.get("lens_correct", True),
                        custom_db_path=p.get("custom_db_path"),
                        metering_mode=p.get("metering_mode", "matrix"),
                        jobs=1,
                        logger_func=lambda msg: None,
                        output_format="hdr-heif" if hdr_output else ext,
                        wb_temp=p.get("wb_temp", 0.0),
                        wb_tint=p.get("wb_tint", 0.0),
                        saturation=p.get("saturation", 1.0),
                        contrast=p.get("contrast", 1.0),
                        highlight=p.get("highlight", 0.0),
                        shadow=p.get("shadow", 0.0),
                        rotation=p.get("rotation", 0),
                        flip_horizontal=p.get("flip_horizontal", False),
                        flip_vertical=p.get("flip_vertical", False),
                        perspective_corners=_perspective_corners,
                        crop=p.get("crop", (0.0, 0.0, 1.0, 1.0)),
                        denoise_enabled=p.get("denoise_enabled", False),
                        sharpen_strength=p.get("sharpen_strength", 0.0),
                    )
                    self.finished_sig.emit(True, "")
                except Exception as e:
                    self.finished_sig.emit(False, str(e))

        self.export_thread = ExportThread()
        self.export_thread.finished_sig.connect(on_finish)
        self.export_thread.start()
