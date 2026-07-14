import os
import shutil
import time
from PySide6.QtWidgets import QWidget, QVBoxLayout, QFileDialog
from PySide6.QtCore import Signal

from qfluentwidgets import (
    SubtitleLabel, StrongBodyLabel, BodyLabel,
    SimpleCardWidget, ScrollArea, InfoBar, PushButton
)

from raw_alchemy import utils
from raw_alchemy.i18n import tr
from raw_alchemy.workers.version_worker import VersionCheckWorker


class AboutPanel(QWidget):
    """About panel widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("aboutWidget")
        self.setStyleSheet("#aboutWidget { background-color: transparent; }")
        
        self.version_worker = None
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI"""
        # Create scroll area
        about_scroll = ScrollArea(self)
        about_scroll.setWidgetResizable(True)
        about_scroll.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")
        
        # Content widget
        about_content = QWidget()
        about_content.setObjectName("aboutContent")
        about_content.setStyleSheet("#aboutContent { background-color: transparent; }")
        about_layout = QVBoxLayout(about_content)
        about_layout.setContentsMargins(40, 40, 40, 40)
        about_layout.setSpacing(20)
        
        # Get version and license info
        version = '0.0.0'
        license_info = 'AGPL-V3'
        try:
            from raw_alchemy import __version__
            version = __version__
        except (ImportError, AttributeError):
            pass  # 使用默认版本号
        
        # Try to get license info from utils
        try:
            _, license_info = utils.get_version_info()
        except (AttributeError, Exception):
            pass  # 使用默认许可证信息
        
        self.current_version = version
        
        # Title
        title = SubtitleLabel(tr('about_title'))
        about_layout.addWidget(title)
        
        # Version Card with Check Update Button
        version_card = SimpleCardWidget()
        version_layout = QVBoxLayout(version_card)
        version_layout.setSpacing(10)
        version_title = StrongBodyLabel(tr('about_version'))
        version_text = BodyLabel(version)
        
        # Check Update Button
        self.check_update_btn = PushButton(tr('check_update'))
        self.check_update_btn.clicked.connect(self.check_for_updates)
        
        # Export Logs Button
        self.export_logs_btn = PushButton(tr('export_logs'))
        self.export_logs_btn.clicked.connect(self.export_logs)
        
        version_layout.addWidget(version_title)
        version_layout.addWidget(version_text)
        version_layout.addWidget(self.check_update_btn)
        version_layout.addWidget(self.export_logs_btn)
        about_layout.addWidget(version_card)
        
        # License Card
        license_card = SimpleCardWidget()
        license_layout = QVBoxLayout(license_card)
        license_layout.setSpacing(10)
        license_title = StrongBodyLabel(tr('about_license'))
        license_text = BodyLabel(license_info)
        license_layout.addWidget(license_title)
        license_layout.addWidget(license_text)
        about_layout.addWidget(license_card)
        
        # Features Card
        features_card = SimpleCardWidget()
        features_layout = QVBoxLayout(features_card)
        features_layout.setSpacing(10)
        features_title = StrongBodyLabel(tr('about_features'))
        features_text = BodyLabel(tr('about_features_list'))
        features_text.setWordWrap(True)
        features_layout.addWidget(features_title)
        features_layout.addWidget(features_text)
        about_layout.addWidget(features_card)
        
        # GitHub Card
        github_card = SimpleCardWidget()
        github_layout = QVBoxLayout(github_card)
        github_layout.setSpacing(10)
        github_title = StrongBodyLabel(tr('about_github'))
        github_link = BodyLabel('<a href="https://github.com/shenmintao/Raw-alchemy">https://github.com/shenmintao/Raw-alchemy</a>')
        github_link.setOpenExternalLinks(True)
        github_layout.addWidget(github_title)
        github_layout.addWidget(github_link)
        about_layout.addWidget(github_card)
        
        # Contact Card
        contact_card = SimpleCardWidget()
        contact_layout = QVBoxLayout(contact_card)
        contact_layout.setSpacing(10)
        contact_title = StrongBodyLabel(tr('about_contact'))
        contact_text = BodyLabel(tr('about_contact_info'))
        contact_text.setWordWrap(True)
        contact_text.setOpenExternalLinks(True)
        contact_layout.addWidget(contact_title)
        contact_layout.addWidget(contact_text)
        about_layout.addWidget(contact_card)
        
        # Support Card (Optional - 如果你需要赞助支持)
        support_card = SimpleCardWidget()
        support_layout = QVBoxLayout(support_card)
        support_layout.setSpacing(10)
        support_title = StrongBodyLabel(tr('about_support'))
        support_text = BodyLabel(tr('about_support_info'))
        support_text.setWordWrap(True)
        support_text.setOpenExternalLinks(True)
        support_layout.addWidget(support_title)
        support_layout.addWidget(support_text)
        about_layout.addWidget(support_card)
        
        about_layout.addStretch()
        
        # Set scroll area widget
        about_scroll.setWidget(about_content)
        
        # Set main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(about_scroll)
    
    def check_for_updates(self):
        """Check for updates"""
        self.check_update_btn.setEnabled(False)
        self.check_update_btn.setText(tr('checking_update'))
        
        self.version_worker = VersionCheckWorker(self.current_version)
        self.version_worker.version_checked.connect(self.on_version_checked)
        self.version_worker.start()
    
    def on_version_checked(self, has_update, latest_version, download_url):
        """Handle version check result"""
        self.check_update_btn.setEnabled(True)
        self.check_update_btn.setText(tr('check_update'))
        
        # If latest_version is empty, it means the check failed
        if not latest_version:
            InfoBar.error(
                tr('update_check_failed'),
                tr('update_check_error', error=tr('network_error')),
                parent=self
            )
            return
        
        if has_update:
            # New version available
            InfoBar.success(
                tr('update_available'),
                tr('update_available_message', version=latest_version, url=download_url),
                parent=self
            )
        else:
            # Already up to date
            InfoBar.success(
                tr('no_update'),
                tr('no_update_message', version=latest_version),
                parent=self
            )

    def shutdown_worker(self):
        """Join an in-flight update check so its QThread is not destroyed live."""
        worker = self.version_worker
        if worker is None:
            return
        try:
            worker.version_checked.disconnect(self.on_version_checked)
        except (TypeError, RuntimeError):
            pass
        if worker.isRunning():
            worker.wait()
        worker.deleteLater()
        self.version_worker = None
    
    def export_logs(self):
        """Export logs to file"""
        from raw_alchemy.logger import get_log_file_path
        log_file = get_log_file_path()
        if not os.path.exists(log_file):
            InfoBar.warning(tr('no_logs_found'), tr('no_logs_found'), parent=self)
            return
        
        default_name = f"raw_alchemy_logs_{time.strftime('%Y%m%d_%H%M%S')}.log"
        save_path, _ = QFileDialog.getSaveFileName(
            self, 
            tr('export_logs'), 
            default_name, 
            "Log Files (*.log);;All Files (*)"
        )
        
        if save_path:
            try:
                shutil.copy2(log_file, save_path)
                InfoBar.success(tr('export_logs_success'), tr('logs_saved_to', path=save_path), parent=self)
            except Exception as e:
                InfoBar.error(tr('export_logs_failed'), str(e), parent=self)

