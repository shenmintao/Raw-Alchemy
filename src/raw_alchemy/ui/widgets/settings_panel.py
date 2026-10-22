from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QMessageBox, QFileDialog
from PySide6.QtCore import Signal, QThread, QObject

from qfluentwidgets import (
    SubtitleLabel, StrongBodyLabel, BodyLabel, CaptionLabel,
    SimpleCardWidget, ScrollArea, ComboBox, PrimaryPushButton,
    PushButton, ProgressBar, InfoBar, InfoBarPosition, SwitchButton, SpinBox
)

from raw_alchemy import config, i18n
from raw_alchemy.i18n import tr


class CudaDownloadWorker(QObject):
    """Download or import a CUDA runtime away from the UI thread."""
    progress = Signal(object, object)  # bytes copied, total; packages may exceed 2 GiB
    finished = Signal(bool)  # success

    def __init__(self, archive_path=None):
        super().__init__()
        self._cancel_requested = False
        self.archive_path = archive_path
        self.error = ''

    def cancel(self):
        self._cancel_requested = True
    
    def run(self):
        try:
            from raw_alchemy.onnx import gpu_runtime
            install = gpu_runtime.import_cuda_runtime if self.archive_path else gpu_runtime.download_cuda_runtime
            args = (self.archive_path,) if self.archive_path else ()
            success = install(
                *args,
                progress_callback=lambda d, t: self.progress.emit(d, t),
                cancel_callback=lambda: self._cancel_requested,
            )
            self.finished.emit(success)
        except Exception as e:
            from loguru import logger
            self.error = str(e)
            logger.error(f"CUDA runtime installation failed: {e}")
            self.finished.emit(False)


class SettingsPanel(QWidget):
    """Settings panel widget"""
    write_sidecar_changed = Signal(bool)
    cache_limit_changed = Signal(int)
    thumb_cache_changed = Signal(bool)
    thumb_cache_clear_requested = Signal()
    native_base_changed = Signal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("settingsWidget")
        self.setStyleSheet("#settingsWidget { background-color: transparent; }")
        
        self._download_thread = None
        self._download_worker = None
        self._importing_cuda = False
        
        self._init_ui()
        self._update_cuda_status()
    
    def _init_ui(self):
        """Initialize UI"""
        # Create scroll area
        settings_scroll = ScrollArea(self)
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")
        
        # Content widget
        settings_content = QWidget()
        settings_content.setObjectName("settingsContent")
        settings_content.setStyleSheet("#settingsContent { background-color: transparent; }")
        settings_layout = QVBoxLayout(settings_content)
        settings_layout.setContentsMargins(40, 40, 40, 40)
        settings_layout.setSpacing(20)
        
        # Title
        title = SubtitleLabel(tr('settings'))
        settings_layout.addWidget(title)
        
        # Language Settings Card
        lang_card = SimpleCardWidget()
        lang_layout = QVBoxLayout(lang_card)
        lang_layout.setSpacing(10)
        lang_title = StrongBodyLabel(tr('language'))
        self.lang_combo = ComboBox()
        self.lang_combo.addItems([tr('english'), tr('chinese')])
        current_lang = i18n.get_current_language()
        self.lang_combo.setCurrentIndex(1 if current_lang == 'zh' else 0)
        self.lang_combo.currentIndexChanged.connect(self.on_language_changed)
        lang_layout.addWidget(lang_title)
        lang_layout.addWidget(self.lang_combo)
        settings_layout.addWidget(lang_card)

        # Editing Settings Card
        editing_card = SimpleCardWidget()
        editing_layout = QVBoxLayout(editing_card)
        editing_layout.setSpacing(10)
        editing_title = StrongBodyLabel(tr('editing'))
        self.write_sidecar_switch = SwitchButton(text=tr('write_sidecar'))
        self.write_sidecar_switch.setChecked(True)
        self.write_sidecar_switch.checkedChanged.connect(self.write_sidecar_changed.emit)
        editing_layout.addWidget(editing_title)
        editing_layout.addWidget(self.write_sidecar_switch)
        settings_layout.addWidget(editing_card)

        # Performance / memory Settings Card (T7.6)
        perf_card = SimpleCardWidget()
        perf_layout = QVBoxLayout(perf_card)
        perf_layout.setSpacing(10)
        perf_title = StrongBodyLabel(tr('performance'))
        perf_layout.addWidget(perf_title)

        cache_row = QHBoxLayout()
        cache_row.addWidget(BodyLabel(tr('cache_limit')))
        self.cache_limit_spin = SpinBox()
        self.cache_limit_spin.setRange(1024, 65536)
        self.cache_limit_spin.setSingleStep(1024)
        self.cache_limit_spin.setSuffix(" MB")
        self.cache_limit_spin.setValue(int(config.CACHE_LIMIT_MB))
        self.cache_limit_spin.valueChanged.connect(self.cache_limit_changed.emit)
        cache_row.addWidget(self.cache_limit_spin)
        cache_row.addStretch()
        perf_layout.addLayout(cache_row)

        cache_hint = CaptionLabel(tr('cache_limit_hint'))
        cache_hint.setStyleSheet("color: gray;")
        perf_layout.addWidget(cache_hint)

        self.native_base_switch = SwitchButton(text=tr('native_base_enable'))
        self.native_base_switch.setChecked(True)
        self.native_base_switch.checkedChanged.connect(self.native_base_changed.emit)
        perf_layout.addWidget(self.native_base_switch)
        native_hint = CaptionLabel(tr('native_base_hint'))
        native_hint.setStyleSheet("color: gray;")
        native_hint.setWordWrap(True)
        perf_layout.addWidget(native_hint)
        settings_layout.addWidget(perf_card)

        # Thumbnail cache Settings Card (T7.8)
        thumb_card = SimpleCardWidget()
        thumb_layout = QVBoxLayout(thumb_card)
        thumb_layout.setSpacing(10)
        thumb_title = StrongBodyLabel(tr('thumb_cache'))
        thumb_layout.addWidget(thumb_title)

        self.thumb_cache_switch = SwitchButton(text=tr('thumb_cache_enable'))
        self.thumb_cache_switch.setChecked(True)
        self.thumb_cache_switch.checkedChanged.connect(self.thumb_cache_changed.emit)
        thumb_layout.addWidget(self.thumb_cache_switch)

        thumb_btn_row = QHBoxLayout()
        self.clear_thumb_cache_btn = PushButton(tr('clear_thumb_cache'))
        self.clear_thumb_cache_btn.clicked.connect(self.thumb_cache_clear_requested.emit)
        thumb_btn_row.addWidget(self.clear_thumb_cache_btn)
        thumb_btn_row.addStretch()
        thumb_layout.addLayout(thumb_btn_row)

        thumb_hint = CaptionLabel(tr('thumb_cache_hint'))
        thumb_hint.setStyleSheet("color: gray;")
        thumb_layout.addWidget(thumb_hint)
        settings_layout.addWidget(thumb_card)


        # GPU Acceleration Settings Card
        gpu_card = SimpleCardWidget()
        gpu_layout = QVBoxLayout(gpu_card)
        gpu_layout.setSpacing(10)
        
        gpu_title = StrongBodyLabel(tr('gpu_acceleration'))
        gpu_layout.addWidget(gpu_title)
        
        # GPU detection info
        self.gpu_info_label = BodyLabel("")
        self.gpu_info_label.setWordWrap(True)
        gpu_layout.addWidget(self.gpu_info_label)

        self.backend_status_label = BodyLabel("")
        self.backend_status_label.setWordWrap(True)
        gpu_layout.addWidget(self.backend_status_label)
        self.backend_hint_label = CaptionLabel("")
        self.backend_hint_label.setWordWrap(True)
        gpu_layout.addWidget(self.backend_hint_label)
        
        # CUDA status
        cuda_status_row = QHBoxLayout()
        cuda_status_row.addWidget(BodyLabel(tr('cuda_status') + ":"))
        self.cuda_status_label = BodyLabel("")
        cuda_status_row.addWidget(self.cuda_status_label)
        cuda_status_row.addStretch()
        gpu_layout.addLayout(cuda_status_row)
        
        # Download progress bar (hidden by default)
        self.cuda_progress = ProgressBar()
        self.cuda_progress.setVisible(False)
        gpu_layout.addWidget(self.cuda_progress)
        
        # Download size hint
        self.download_hint = CaptionLabel(tr('cuda_download_size'))
        self.download_hint.setStyleSheet("color: gray;")
        self.download_hint.setWordWrap(True)
        gpu_layout.addWidget(self.download_hint)

        self.cuda_details_label = CaptionLabel('')
        self.cuda_details_label.setWordWrap(True)
        gpu_layout.addWidget(self.cuda_details_label)
        
        # Download/Remove button
        btn_row = QHBoxLayout()
        self.cuda_download_btn = PrimaryPushButton(tr('download_cuda'))
        self.cuda_download_btn.clicked.connect(self.on_cuda_download_clicked)
        btn_row.addWidget(self.cuda_download_btn)

        self.cuda_import_btn = PushButton(tr('import_cuda'))
        self.cuda_import_btn.clicked.connect(self.on_cuda_import_clicked)
        btn_row.addWidget(self.cuda_import_btn)

        self.cuda_detect_btn = PushButton(tr('detect_cuda'))
        self.cuda_detect_btn.clicked.connect(self._update_cuda_status)
        btn_row.addWidget(self.cuda_detect_btn)
        
        self.cuda_remove_btn = PushButton(tr('remove_cuda'))
        self.cuda_remove_btn.clicked.connect(self.on_cuda_remove_clicked)
        btn_row.addWidget(self.cuda_remove_btn)
        btn_row.addStretch()
        gpu_layout.addLayout(btn_row)
        
        settings_layout.addWidget(gpu_card)
        
        settings_layout.addStretch()
        
        # Set scroll area widget
        settings_scroll.setWidget(settings_content)
        
        # Set main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(settings_scroll)

    def set_write_sidecar_enabled(self, enabled: bool):
        self.write_sidecar_switch.blockSignals(True)
        self.write_sidecar_switch.setChecked(bool(enabled))
        self.write_sidecar_switch.blockSignals(False)

    def set_cache_limit_mb(self, limit_mb: int):
        self.cache_limit_spin.blockSignals(True)
        self.cache_limit_spin.setValue(int(limit_mb))
        self.cache_limit_spin.blockSignals(False)

    def set_thumb_cache_enabled(self, enabled: bool):
        self.thumb_cache_switch.blockSignals(True)
        self.thumb_cache_switch.setChecked(bool(enabled))
        self.thumb_cache_switch.blockSignals(False)

    def set_native_base_enabled(self, enabled: bool):
        self.native_base_switch.blockSignals(True)
        self.native_base_switch.setChecked(bool(enabled))
        self.native_base_switch.blockSignals(False)
    
    def _update_backend_status(self):
        """Show provider preference without initializing a model or claiming placement."""
        try:
            from raw_alchemy.onnx.denoiser import _get_providers

            provider = _get_providers()[0]
            backend = {
                'DmlExecutionProvider': 'DirectML',
                'ROCMExecutionProvider': 'ROCm',
            }.get(provider, provider.removesuffix('ExecutionProvider'))
            if provider in (
                'CUDAExecutionProvider', 'DmlExecutionProvider',
                'ROCMExecutionProvider', 'MIGraphXExecutionProvider',
            ):
                backend = tr('inference_backend_gpu', backend=backend)
            self.backend_status_label.setText(tr('inference_backend', backend=backend))
            hint = 'inference_backend_cpu_hint' if provider == 'CPUExecutionProvider' else 'inference_backend_gpu_hint'
            self.backend_hint_label.setText(tr(hint))
        except Exception as exc:
            from loguru import logger
            logger.warning(f"Could not detect inference backend: {exc}")
            self.backend_status_label.setText(tr('inference_backend_unknown'))
            self.backend_hint_label.setText('')

    def _update_cuda_status(self):
        """Update CUDA status display based on current state."""
        self._update_backend_status()
        try:
            from raw_alchemy.onnx import gpu_runtime
            
            # Inspect installed libraries before offering any installation.
            status = gpu_runtime.detect_cuda_runtime()
            gpu_info = gpu_runtime.detect_gpu_vendor()
            provider_missing = gpu_runtime.cuda_provider_bundled() is False
            can_install = gpu_info['cuda_compatible'] and not provider_missing
            if gpu_info['cuda_compatible']:
                self.gpu_info_label.setText(f"✅ {tr('nvidia_gpu')}: {gpu_info['name']}")
            elif gpu_info['vendor'] == 'amd':
                self.gpu_info_label.setText(f"⚠️ {tr('amd_gpu')}")
            elif gpu_info['vendor'] == 'intel':
                self.gpu_info_label.setText(f"⚠️ {tr('intel_gpu')}")
            else:
                self.gpu_info_label.setText(f"❓ {tr('no_gpu_detected')}")

            self.cuda_download_btn.setEnabled(can_install)
            self.cuda_import_btn.setEnabled(can_install)
            self.cuda_download_btn.setVisible(not status['installed'] and can_install)
            self.cuda_import_btn.setVisible(not status['installed'] and can_install)
            # Removal only affects files managed by this application.
            self.cuda_remove_btn.setVisible(status['local_present'])
            self.download_hint.setVisible(not status['installed'] and can_install)
            if provider_missing:
                # Installed CUDA libraries do not change this build's backend.
                self.cuda_status_label.setText(tr('cuda_not_required'))
                self.cuda_status_label.setStyleSheet("color: gray;")
                self.cuda_details_label.setText(tr('cuda_provider_missing'))
            elif status['installed']:
                key = 'cuda_system_installed' if status['source'] == 'system' else 'cuda_installed'
                self.cuda_status_label.setText(f"✅ {tr(key)}")
                self.cuda_status_label.setStyleSheet("color: green;")
                self.cuda_details_label.setText('\n'.join(status['library_dirs']))
            else:
                key = 'cuda_incomplete' if status['detected'] else 'cuda_not_installed'
                self.cuda_status_label.setText(f"❌ {tr(key)}")
                self.cuda_status_label.setStyleSheet("color: orange;")
                details = tr('cuda_install_hint')
                if status['detected']:
                    details += '\n' + tr('cuda_missing_libraries', libraries=', '.join(status['missing_libraries']))
                self.cuda_details_label.setText(details)
                
        except Exception as e:
            self.gpu_info_label.setText(f"Error: {e}")
            self.cuda_download_btn.setEnabled(False)
            self.cuda_import_btn.setEnabled(False)
    
    def on_language_changed(self, index):
        """Handle language change"""
        lang_code = 'en' if index == 0 else 'zh'
        current = i18n.get_current_language()
        if lang_code == current:
            return
        i18n.set_language(lang_code)
        QMessageBox.information(self, tr('restart_required'), tr('restart_message'))
    
    def on_cuda_download_clicked(self):
        """Start CUDA download."""
        if self._download_thread is not None:
            return
        from raw_alchemy.onnx import gpu_runtime
        if gpu_runtime.is_cuda_runtime_installed():
            self._update_cuda_status()
            return
        if not self._can_install_cuda():
            return
        self._start_cuda_install()

    def _can_install_cuda(self):
        try:
            from raw_alchemy.onnx import gpu_runtime
            gpu_info = gpu_runtime.detect_gpu_vendor()
            if not gpu_info['cuda_compatible']:
                InfoBar.warning(
                    title=tr('cuda_not_supported'),
                    content=tr('no_gpu_detected') if gpu_info['vendor'] == 'unknown' else tr(f"{gpu_info['vendor']}_gpu"),
                    parent=self,
                    position=InfoBarPosition.TOP
                )
                return False
            if gpu_runtime.cuda_provider_bundled() is False:
                InfoBar.warning(
                    title=tr('cuda_not_supported'),
                    content=tr('cuda_provider_missing'),
                    parent=self,
                    position=InfoBarPosition.TOP
                )
                return False
            return True
        except Exception as exc:
            self.cuda_details_label.setText(str(exc))
            return False

    def on_cuda_import_clicked(self):
        if self._download_thread is not None:
            return
        from raw_alchemy.onnx import gpu_runtime
        if gpu_runtime.is_cuda_runtime_installed():
            self._update_cuda_status()
            return
        if not self._can_install_cuda():
            return
        archive_path, _ = QFileDialog.getOpenFileName(
            self, tr('import_cuda'), '', tr('cuda_archive_filter'),
        )
        if archive_path:
            self._start_cuda_install(archive_path)

    def _start_cuda_install(self, archive_path=None):
        self._importing_cuda = archive_path is not None
        self.cuda_download_btn.setEnabled(False)
        self.cuda_import_btn.setEnabled(False)
        self.cuda_detect_btn.setEnabled(False)
        self.cuda_remove_btn.setEnabled(False)
        if self._importing_cuda:
            self.cuda_import_btn.setText(tr('cuda_importing'))
        else:
            self.cuda_download_btn.setText(tr('cuda_downloading'))
        self.cuda_progress.setVisible(True)
        self.cuda_progress.setValue(0)
        
        # Start download in background thread
        self._download_thread = QThread()
        self._download_worker = CudaDownloadWorker(archive_path)
        self._download_worker.moveToThread(self._download_thread)
        
        self._download_thread.started.connect(self._download_worker.run)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_thread.finished.connect(self._download_worker.deleteLater)
        
        self._download_thread.start()
    
    def _on_download_progress(self, downloaded: int, total: int):
        """Update download progress."""
        if total > 0:
            percent = min(100, int(downloaded * 100 / total))
            self.cuda_progress.setValue(percent)
    
    def _on_download_finished(self, success: bool):
        """Handle download or local import completion."""
        self.cuda_progress.setVisible(False)
        self.cuda_download_btn.setText(tr('download_cuda'))
        self.cuda_import_btn.setText(tr('import_cuda'))
        self.cuda_detect_btn.setEnabled(True)
        self.cuda_remove_btn.setEnabled(True)
        error = self._download_worker.error if self._download_worker else ''
        
        if self._download_thread:
            self._download_thread.quit()
            self._download_thread.wait()
            self._download_thread.deleteLater()
            self._download_thread = None
            self._download_worker = None
        
        if success:
            InfoBar.success(
                title=tr('cuda_import_success' if self._importing_cuda else 'cuda_download_success'),
                content=tr('cuda_restart_hint'),
                parent=self,
                position=InfoBarPosition.TOP
            )
        else:
            InfoBar.error(
                title=tr('cuda_import_failed' if self._importing_cuda else 'cuda_download_failed'),
                content=error,
                parent=self,
                position=InfoBarPosition.TOP
            )
        
        self._update_cuda_status()

    def shutdown_worker(self):
        """Cancel and join an active runtime installation before window teardown."""
        worker = self._download_worker
        thread = self._download_thread
        if worker is not None:
            worker.cancel()
            try:
                worker.progress.disconnect(self._on_download_progress)
                worker.finished.disconnect(self._on_download_finished)
            except (TypeError, RuntimeError):
                pass
        if thread is not None:
            thread.quit()
            if thread.isRunning():
                thread.wait()
            thread.deleteLater()
        self._download_thread = None
        self._download_worker = None
    
    def on_cuda_remove_clicked(self):
        """Remove installed CUDA runtime."""
        try:
            from raw_alchemy.onnx import gpu_runtime
            if not gpu_runtime.remove_cuda_runtime():
                raise OSError(tr('cuda_remove_failed'))
            InfoBar.success(
                title=tr('cuda_removed'),
                content="",
                parent=self,
                position=InfoBarPosition.TOP
            )
        except Exception as e:
            InfoBar.error(
                title=tr('error'),
                content=str(e),
                parent=self,
                position=InfoBarPosition.TOP
            )
        
        self._update_cuda_status()

