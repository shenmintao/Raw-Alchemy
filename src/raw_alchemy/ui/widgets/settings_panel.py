from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QMessageBox
from PySide6.QtCore import Signal, QThread, QObject

from qfluentwidgets import (
    SubtitleLabel, StrongBodyLabel, BodyLabel, CaptionLabel,
    SimpleCardWidget, ScrollArea, ComboBox, PrimaryPushButton,
    PushButton, ProgressBar, InfoBar, InfoBarPosition, SwitchButton, SpinBox
)

from raw_alchemy import config, i18n
from raw_alchemy.i18n import tr


class CudaDownloadWorker(QObject):
    """Worker thread for downloading CUDA runtime."""
    progress = Signal(int, int)  # downloaded, total
    finished = Signal(bool)  # success
    
    def run(self):
        try:
            from raw_alchemy.onnx import gpu_runtime
            success = gpu_runtime.download_cuda_runtime(
                progress_callback=lambda d, t: self.progress.emit(d, t)
            )
            self.finished.emit(success)
        except Exception as e:
            from loguru import logger
            logger.error(f"CUDA download failed: {e}")
            self.finished.emit(False)


class SettingsPanel(QWidget):
    """Settings panel widget"""
    write_sidecar_changed = Signal(bool)
    cache_limit_changed = Signal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("settingsWidget")
        self.setStyleSheet("#settingsWidget { background-color: transparent; }")
        
        self._download_thread = None
        self._download_worker = None
        
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
        settings_layout.addWidget(perf_card)


        # GPU Acceleration Settings Card
        gpu_card = SimpleCardWidget()
        gpu_layout = QVBoxLayout(gpu_card)
        gpu_layout.setSpacing(10)
        
        gpu_title = StrongBodyLabel(tr('gpu_acceleration'))
        gpu_layout.addWidget(gpu_title)
        
        # GPU detection info
        self.gpu_info_label = BodyLabel("")
        gpu_layout.addWidget(self.gpu_info_label)
        
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
        gpu_layout.addWidget(self.download_hint)
        
        # Download/Remove button
        btn_row = QHBoxLayout()
        self.cuda_download_btn = PrimaryPushButton(tr('download_cuda'))
        self.cuda_download_btn.clicked.connect(self.on_cuda_download_clicked)
        btn_row.addWidget(self.cuda_download_btn)
        
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
    
    def _update_cuda_status(self):
        """Update CUDA status display based on current state."""
        try:
            from raw_alchemy.onnx import gpu_runtime
            
            # Detect GPU
            gpu_info = gpu_runtime.detect_gpu_vendor()
            if gpu_info['cuda_compatible']:
                self.gpu_info_label.setText(f"✅ {tr('nvidia_gpu')}: {gpu_info['name']}")
                self.cuda_download_btn.setEnabled(True)
            elif gpu_info['vendor'] == 'amd':
                self.gpu_info_label.setText(f"⚠️ {tr('amd_gpu')}")
                self.cuda_download_btn.setEnabled(False)
            elif gpu_info['vendor'] == 'intel':
                self.gpu_info_label.setText(f"⚠️ {tr('intel_gpu')}")
                self.cuda_download_btn.setEnabled(False)
            else:
                self.gpu_info_label.setText(f"❓ {tr('no_gpu_detected')}")
                self.cuda_download_btn.setEnabled(False)
            
            # Check CUDA installation
            if gpu_runtime.is_cuda_runtime_installed():
                version = gpu_runtime.get_installed_cuda_version() or ""
                self.cuda_status_label.setText(f"✅ {tr('cuda_installed')} (v{version})")
                self.cuda_status_label.setStyleSheet("color: green;")
                self.cuda_download_btn.setVisible(False)
                self.cuda_remove_btn.setVisible(True)
                self.download_hint.setVisible(False)
            else:
                self.cuda_status_label.setText(f"❌ {tr('cuda_not_installed')}")
                self.cuda_status_label.setStyleSheet("color: orange;")
                self.cuda_download_btn.setVisible(True)
                self.cuda_remove_btn.setVisible(False)
                self.download_hint.setVisible(True)
                
        except Exception as e:
            self.gpu_info_label.setText(f"Error: {e}")
            self.cuda_download_btn.setEnabled(False)
    
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
        try:
            from raw_alchemy.onnx import gpu_runtime
            gpu_info = gpu_runtime.detect_gpu_vendor()
            if not gpu_info['cuda_compatible']:
                InfoBar.warning(
                    title=tr('cuda_not_supported'),
                    content=tr('amd_gpu') if gpu_info['vendor'] == 'amd' else tr('intel_gpu'),
                    parent=self,
                    position=InfoBarPosition.TOP
                )
                return
        except Exception:
            pass
        
        # Disable button and show progress
        self.cuda_download_btn.setEnabled(False)
        self.cuda_download_btn.setText(tr('cuda_downloading'))
        self.cuda_progress.setVisible(True)
        self.cuda_progress.setValue(0)
        
        # Start download in background thread
        self._download_thread = QThread()
        self._download_worker = CudaDownloadWorker()
        self._download_worker.moveToThread(self._download_thread)
        
        self._download_thread.started.connect(self._download_worker.run)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.finished.connect(self._on_download_finished)
        
        self._download_thread.start()
    
    def _on_download_progress(self, downloaded: int, total: int):
        """Update download progress."""
        if total > 0:
            percent = int(downloaded * 100 / total)
            self.cuda_progress.setValue(percent)
    
    def _on_download_finished(self, success: bool):
        """Handle download completion."""
        self.cuda_progress.setVisible(False)
        self.cuda_download_btn.setText(tr('download_cuda'))
        
        if self._download_thread:
            self._download_thread.quit()
            self._download_thread.wait()
            self._download_thread = None
            self._download_worker = None
        
        if success:
            InfoBar.success(
                title=tr('cuda_download_success'),
                content="",
                parent=self,
                position=InfoBarPosition.TOP
            )
        else:
            InfoBar.error(
                title=tr('cuda_download_failed'),
                content="",
                parent=self,
                position=InfoBarPosition.TOP
            )
        
        self._update_cuda_status()
    
    def on_cuda_remove_clicked(self):
        """Remove installed CUDA runtime."""
        try:
            from raw_alchemy.onnx import gpu_runtime
            gpu_runtime.remove_cuda_runtime()
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

