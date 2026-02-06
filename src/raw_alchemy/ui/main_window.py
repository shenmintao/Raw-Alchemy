import sys
import os
import shutil
import time
from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QFileDialog, QListWidget, QListWidgetItem, QFrame,
    QSplitter, QSizePolicy, QGraphicsDropShadowEffect, QGridLayout,
    QInputDialog, QMessageBox
)
from PySide6.QtCore import Qt, QSize, QThread, Signal, QObject, QTimer, QEvent, QRect
from PySide6.QtGui import QIcon, QPixmap, QImage, QPainter, QColor, QResizeEvent

from qfluentwidgets import (
    FluentWindow, SubtitleLabel, PrimaryPushButton, 
    CaptionLabel, StrongBodyLabel, BodyLabel, 
    SimpleCardWidget, ScrollArea, InfoBar, Theme, setTheme, 
    FluentIcon as FIF, ProgressRing, ComboBox,
    ToolButton, PushButton
)

from raw_alchemy import config, utils, orchestrator, lensfun_wrapper, i18n
from raw_alchemy.i18n import tr
from loguru import logger

# New structure imports
from raw_alchemy.pipeline.state import ImageState
from raw_alchemy.pipeline.processor import ImageProcessor
from raw_alchemy.workers.thumbnail_worker import ThumbnailWorker
from raw_alchemy.workers.version_worker import VersionCheckWorker
from raw_alchemy.ui.widgets.histogram import HistogramWidget
from raw_alchemy.ui.widgets.waveform import WaveformWidget
from raw_alchemy.ui.widgets.gallery_item import GalleryItem
from raw_alchemy.ui.widgets.inspector_panel import InspectorPanel
from raw_alchemy.ui.widgets.title_bar import CenteredFluentTitleBar
from raw_alchemy.ui.widgets.crop_rotate_viewer import CropRotateViewer
from raw_alchemy.ui.widgets.perspective_viewer import PerspectiveViewer
from raw_alchemy.ui.widgets.help_panel import HelpPanel
from raw_alchemy.ui.widgets.about_panel import AboutPanel
from raw_alchemy.ui.widgets.settings_panel import SettingsPanel
from PySide6.QtWidgets import QStackedWidget

class MainWindow(FluentWindow):
    def __init__(self):
        # Initialize image states BEFORE super().__init__() to avoid resizeEvent issues
        # FluentWindow.__init__ may trigger resizeEvent during initialization
        self.original = ImageState()  # RAW decoded
        self.current = ImageState()   # Processed with current params
        self.baseline = ImageState()  # Saved baseline (optional)
        
        super().__init__()
        # 替换为“真正居中”的标题栏，避免 resize 拖动期间跳左
        self.setTitleBar(CenteredFluentTitleBar(self))

        self.base_title = "Raw Alchemy Studio"
        self.setWindowTitle(self.base_title)
        self.setWindowIcon(QIcon(self._get_icon_path()))
        self.resize(1900, 1200)
        
        # State
        self.current_folder = None
        self.current_raw_path = None
        self.marked_files = set()
        self.file_params_cache = {}  # path -> params dict
        self.file_baseline_params_cache = {}  # path -> baseline params dict
        
        # Last used paths for file dialogs
        self.last_folder_path = None  # Last opened gallery folder
        self.last_lut_folder_path = None  # Last LUT folder
        self.last_lensfun_db_path = None  # Last Lensfun DB path
        self.last_export_path = None  # Last export folder
        
        # Request tracking
        self.current_request_id = 0
        
        # 预加载lensfun数据库（在后台线程中）
        self._preload_lensfun_database()
        
        self.create_ui()
        self.create_settings_interface()
        self.create_help_interface()
        self.create_about_interface()

        # Workers
        self.thumb_worker = None
        self.processor = ImageProcessor()
        self.processor.result_ready.connect(self.on_process_result)
        self.processor.load_complete.connect(self.on_load_complete)
        self.processor.error_occurred.connect(self.on_error)
        self.processor.denoise_started.connect(self.on_denoise_started)
        self.processor.denoise_progress.connect(self.on_denoise_progress)
        self.processor.denoise_finished.connect(self.on_denoise_finished)
        
        # Denoise progress dialog
        self.denoise_progress_dialog = None
        
        # Baseline processor
        self.baseline_processor = ImageProcessor()
        self.baseline_processor.result_ready.connect(self.on_baseline_result)
        
        # Processing Debounce
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(25) # 100ms debounce
        self.update_timer.timeout.connect(self.trigger_processing)
        
        # Load saved settings
        self.load_settings()
        
        # Restore UI state from saved settings
        self.restore_ui()
        
        self.processor_connection_mode = 'normal' # 'normal', 'crop', or 'perspective'

    def systemTitleBarRect(self, size: QSize):
        """macOS: 预留系统红黄绿按钮区域（避免被自绘标题栏覆盖）"""
        if sys.platform != "darwin":
            return super().systemTitleBarRect(size)

        y = 0 if self.isFullScreen() else 9
        return QRect(0, y, 100, size.height())

    def update_window_title(self):
        """更新窗口标题以显示当前文件名"""
        if self.current_raw_path:
            filename = os.path.basename(self.current_raw_path)
            self.setWindowTitle(f"{self.base_title} - {filename}")
        else:
            self.setWindowTitle(self.base_title)

    def _get_icon_path(self):
        """Get the path to the application icon (supports PyInstaller and Nuitka)."""
        if getattr(sys, 'frozen', False):
            if hasattr(sys, '_MEIPASS'):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(sys.executable)
        else:
            base_path = os.path.abspath(".")
        
        if sys.platform == 'win32':
            icon_path = os.path.join(base_path, "icon.ico")
            if not os.path.exists(icon_path):
                icon_path = os.path.join(base_path, "icon.png")
        else:
            icon_path = os.path.join(base_path, "icon.png")
            if not os.path.exists(icon_path):
                icon_path = os.path.join(base_path, "icon.ico")
            
        return icon_path

    def _preload_lensfun_database(self):
        """在后台线程中预加载lensfun数据库，避免阻塞GUI启动"""
        def preload():
            try:
                lensfun_wrapper._get_or_create_database(custom_db_path=None)
            except Exception as e:
                logger.error(f"  ⚠️ [Lensfun] Failed to preload database: {e}")
        
        import threading
        preload_thread = threading.Thread(target=preload, daemon=True)
        preload_thread.start()
    
    def load_settings(self):
        """Load saved application settings"""
        settings = i18n.load_app_settings()
        
        if 'window_geometry' in settings:
            geom = settings['window_geometry']
            if all(k in geom for k in ['x', 'y', 'width', 'height']):
                self.setGeometry(geom['x'], geom['y'], geom['width'], geom['height'])
        
        if settings.get('window_maximized', False):
            self.showMaximized()
        
        if 'last_folder_path' in settings:
            self.last_folder_path = settings['last_folder_path']
        
        if 'last_lut_folder_path' in settings:
            self.last_lut_folder_path = settings['last_lut_folder_path']
        
        if 'last_lensfun_db_path' in settings:
            self.last_lensfun_db_path = settings['last_lensfun_db_path']
        
        if 'last_export_path' in settings:
            self.last_export_path = settings['last_export_path']
    
    def restore_ui(self):
        """Restore UI state from saved settings"""
        if self.last_lut_folder_path and os.path.exists(self.last_lut_folder_path):
            self.right_panel.lut_folder = self.last_lut_folder_path
            self.right_panel.refresh_lut_list()
    
    def save_settings(self):
        """Save current application settings"""
        is_maximized = self.isMaximized()
        
        if is_maximized:
            geom = self.normalGeometry()
        else:
            geom = self.geometry()
        
        settings = {
            'window_geometry': {
                'x': geom.x(),
                'y': geom.y(),
                'width': geom.width(),
                'height': geom.height()
            },
            'window_maximized': is_maximized,
            'last_folder_path': self.current_folder,
            'last_lut_folder_path': self.last_lut_folder_path,
            'last_lensfun_db_path': self.last_lensfun_db_path,
            'last_export_path': self.last_export_path
        }
        i18n.save_app_settings(settings)

    def create_ui(self):
        # Central Layout
        self.main_widget = QWidget()
        self.main_widget.setObjectName("mainWidget")
        self.h_layout = QHBoxLayout(self.main_widget)
        self.h_layout.setContentsMargins(0, 0, 0, 0)
        self.h_layout.setSpacing(0)
        
        try:
            if hasattr(self, 'navigationInterface') and hasattr(self.navigationInterface, 'panel'):
                panel = self.navigationInterface.panel
                if hasattr(panel, 'returnButton'):
                    panel.returnButton.hide()
                if hasattr(panel, 'topLayout') and panel.topLayout is not None:
                    panel.topLayout.insertSpacing(0, 43)
        except Exception:
            pass
        
        # 1. Left Panel (Gallery)
        self.left_panel = QWidget()
        self.left_panel.setFixedWidth(400)
        self.left_panel.setStyleSheet("background-color: transparent;")
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(5, 10, 5, 10)
        
        self.gallery_list = QListWidget()
        self.gallery_list.setIconSize(QSize(130, 100))
        self.gallery_list.setGridSize(QSize(160, 140))
        self.gallery_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.gallery_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.gallery_list.setSpacing(10)
        self.gallery_list.setDragEnabled(False)
        self.gallery_list.setAcceptDrops(False)
        self.gallery_list.setDropIndicatorShown(False)
        self.gallery_list.setDragDropMode(QListWidget.DragDropMode.NoDragDrop)
        self.gallery_list.setDefaultDropAction(Qt.DropAction.IgnoreAction)

        self.gallery_list.itemClicked.connect(self.on_gallery_item_clicked)
        self.gallery_list.currentItemChanged.connect(lambda current, prev: self.on_gallery_item_clicked(current))
        self.gallery_list.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                border: none;
                outline: none;
            }
            QListWidget::item {
                color: white;
                border-radius: 8px;
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: rgba(255, 255, 255, 0.1);
                color: white;
            }
            QListWidget::item:hover {
                background-color: rgba(255, 255, 255, 0.05);
            }
        """)
        
        self.open_btn = PrimaryPushButton(FIF.FOLDER, tr('open_folder'))
        self.open_btn.clicked.connect(self.browse_folder)
        
        self.loading_label = CaptionLabel("")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.loading_label.hide()
        
        self.left_layout.addWidget(SubtitleLabel(tr('library')))
        self.left_layout.addWidget(self.gallery_list)
        self.left_layout.addWidget(self.loading_label)
        self.left_layout.addWidget(self.open_btn)
        
        # 2. Center Panel (Preview Stack)
        self.center_panel = QWidget()
        self.center_layout = QVBoxLayout(self.center_panel)
        self.center_layout.setContentsMargins(0, 0, 0, 0)
        
        self.center_stack = QStackedWidget()
        
        # --- Page 1: Standard Preview ---
        self.page_preview = QWidget()
        self.preview_layout = QVBoxLayout(self.page_preview)
        self.preview_layout.setContentsMargins(10, 10, 10, 10)
        
        self.preview_lbl = QLabel(tr('no_image_selected'))
        self.preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_lbl.setStyleSheet("background-color: #202020; border-radius: 8px; color: white;")
        self.preview_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_lbl.mousePressEvent = self.show_original
        self.preview_lbl.mouseReleaseEvent = self.show_processed
        
        self.toolbar = QFrame()
        self.toolbar.setFixedHeight(60)
        self.toolbar_layout = QHBoxLayout(self.toolbar)
        
        self.btn_prev = ToolButton(FIF.LEFT_ARROW)
        self.btn_next = ToolButton(FIF.RIGHT_ARROW)
        self.btn_mark = ToolButton(FIF.TAG)
        self.btn_mark.setCheckable(True)
        self.btn_delete = ToolButton(FIF.DELETE)
        self.btn_compare = PushButton(tr('hold_to_compare'))
        self.btn_compare.setToolTip(tr('hold_to_compare'))
        
        self.btn_export_curr = PushButton(tr('export_current'))
        self.btn_export_all = PrimaryPushButton(tr('export_all_marked'))
        
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_mark.clicked.connect(self.toggle_mark)
        self.btn_delete.clicked.connect(self.delete_image)
        self.btn_export_curr.clicked.connect(self.export_current)
        self.btn_export_all.clicked.connect(self.export_all)
        
        self.export_progress = ProgressRing()
        self.export_progress.setFixedSize(40, 40)
        self.export_progress.setTextVisible(True)
        self.export_progress.hide()

        self.btn_compare.pressed.connect(lambda: self.show_original(None))
        self.btn_compare.released.connect(lambda: self.show_processed(None))
        
        self.toolbar_layout.addWidget(self.btn_prev)
        self.toolbar_layout.addWidget(self.btn_next)
        self.toolbar_layout.addStretch()
        self.toolbar_layout.addWidget(self.btn_mark)
        self.toolbar_layout.addWidget(self.btn_delete)
        self.toolbar_layout.addStretch()
        self.toolbar_layout.addWidget(self.btn_compare)
        self.toolbar_layout.addWidget(self.export_progress)
        self.toolbar_layout.addWidget(self.btn_export_curr)
        self.toolbar_layout.addWidget(self.btn_export_all)
        
        self.preview_layout.addWidget(self.preview_lbl)
        self.preview_layout.addWidget(self.toolbar)
        
        # --- Page 2: Crop Viewer ---
        self.crop_viewer = CropRotateViewer()
        self.crop_viewer.applied.connect(self.on_crop_applied)
        self.crop_viewer.cancelled.connect(self.exit_crop_mode)
        
        # --- Page 3: Perspective Viewer ---
        self.perspective_viewer = PerspectiveViewer()
        self.perspective_viewer.applied.connect(self.on_perspective_applied)
        self.perspective_viewer.cancelled.connect(self.exit_perspective_mode)
        
        self.center_stack.addWidget(self.page_preview)
        self.center_stack.addWidget(self.crop_viewer)
        self.center_stack.addWidget(self.perspective_viewer)
        
        self.center_layout.addWidget(self.center_stack)
        
        # 3. Right Panel (Inspector)
        self.right_panel = InspectorPanel()
        self.right_panel.setFixedWidth(360) 
        self.right_panel.param_changed.connect(self.on_param_changed)
        self.right_panel.enter_crop_mode.connect(self.enter_crop_mode)
        self.right_panel.enter_perspective_mode.connect(self.enter_perspective_mode)
        self.right_panel.save_baseline_btn.clicked.connect(self.save_baseline_image)

        self.h_layout.addWidget(self.left_panel)
        self.h_layout.addWidget(self.center_panel, 1)
        self.h_layout.addWidget(self.right_panel)
        
        self.addSubInterface(self.main_widget, FIF.PHOTO, tr('editor'))
        setTheme(Theme.DARK)
        QApplication.instance().installEventFilter(self)

    def create_about_interface(self):
        self.about_widget = AboutPanel()
        self.about_widget.setObjectName("aboutInterface")
        self.addSubInterface(self.about_widget, FIF.INFO, tr('about'))

    def create_help_interface(self):
        self.help_widget = HelpPanel()
        self.help_widget.setObjectName("helpInterface")
        self.addSubInterface(self.help_widget, FIF.HELP, tr('help'))

    def create_settings_interface(self):
        self.settings_widget = SettingsPanel()
        self.settings_widget.setObjectName("settingsInterface")
        self.addSubInterface(self.settings_widget, FIF.SETTING, tr('settings'))

    def eventFilter(self, obj, event):
        if isinstance(obj, QWidget) and obj.window() == self:
            if event.type() == QEvent.Type.KeyPress:
                key = event.key()
                if key == Qt.Key.Key_Left:
                    self.prev_image()
                    return True
                elif key == Qt.Key.Key_Right:
                    self.next_image()
                    return True
                elif key == Qt.Key.Key_Space:
                    if not event.isAutoRepeat():
                        self.show_original(None)
                    return True
                elif key == Qt.Key.Key_Delete:
                    self.delete_image()
                    return True
                elif key == Qt.Key.Key_T:
                    self.toggle_mark()
                    return True
            elif event.type() == QEvent.Type.KeyRelease:
                if event.key() == Qt.Key.Key_Space:
                    if not event.isAutoRepeat():
                        self.show_processed(None)
                    return True
        return super().eventFilter(obj, event)

    def browse_folder(self):
        start_dir = self.last_folder_path if self.last_folder_path and os.path.exists(self.last_folder_path) else ""
        folder = QFileDialog.getExistingDirectory(self, tr('select_folder'), start_dir)
        if folder:
            self.current_folder = folder
            self.last_folder_path = folder
            self.gallery_list.clear()
            self.start_thumbnail_scan(folder)

    def start_thumbnail_scan(self, folder):
        if self.thumb_worker:
            self.thumb_worker.stop()
            self.thumb_worker.wait()
        
        self.loading_label.setText(tr('loading_thumbnails'))
        self.loading_label.show()
        
        self.thumb_worker = ThumbnailWorker(folder)
        self.thumb_worker.thumbnail_ready.connect(self.add_gallery_item)
        self.thumb_worker.progress_update.connect(self.on_thumbnail_progress)
        self.thumb_worker.finished_scanning.connect(self.on_thumbnail_finished)
        self.thumb_worker.start()
    
    def update_status(self, text):
        self.loading_label.setText(text)
        if text:
            self.loading_label.show()
        else:
            self.loading_label.hide()
            
    # --- Crop Mode Handling ---
    
    def enter_crop_mode(self):
        """Switch to Crop/Rotate view"""
        if not self.current_raw_path: return
        
        # [Fix Issue 2] If already in crop mode, this button acts as "Done"
        if self.center_stack.currentWidget() == self.crop_viewer:
            self.crop_viewer._on_done()
            return
            
        logger.info("Entering Crop Mode...")
         
        # 1. Disable standard processing updates temporarily
        self.update_timer.stop()
        
        # 2. Get current params
        params = self.file_params_cache.get(self.current_raw_path, {}).copy()
        
        # 3. We need the "Base" image (Linear + Lens Corrected ONLY).
        # The crop viewer needs to see the UN-ROTATED, UN-CROPPED image to allow user to edit it.
        # However, for performance, we should ask the processor for the cached linear/corrected data.
        # But Processor communicates via signals.
        # Strategy: Send a special request or leverage the fact that we process in layers?
        # Actually, we can just grab the current 'linear' or 'corrected' from processor if we had access, 
        # but thread safety.
        # `self.original` in MainWindow is RAW decoded? No, `ImageState` just holds generic objects.
        # `self.original` is populated in `on_load_complete`? No, it's not currently used much.
        
        # Let's request a special "preview_for_crop" from processor?
        # Or just use the original raw loading logic?
        # 
        # Hack/Fast path: Access processor.cached_corrected directly? (Not thread safe technically but Python GIL helps)
        # Proper way: request_id.
        
        # Let's implement a "request_base_image" signal/slot.
        # For now, let's just trigger a process with "Geometry=Defaults" and "Color=Defaults"?
        # But that applies display transform etc.
        # The CropViewer expects a PIXMAP.
        
        # Let's add a sync method or callback to get the current base image from processor.
        # Or... since we are in Python, let's cheat slightly and read the cache if the worker is idle?
        # No, let's do it properly:
        # We emit a signal to processor "get_crop_source(path)". Processor replies with signals.
        
        # Simpler: The `CropRotateViewer` takes the current display image? No, that's already cropped.
        # 
        # NEW PLAN: 
        # When entering crop mode, we take the *current* params, 
        # set rotation=0, flip=False, crop=None,
        # request a preview update with a special flag `_for_crop=True`?
        # The processor generates the image (which will hit the Corrected Cache) and emits result.
        # We intercept this result.
        
        # To distinguish this result from normal preview: use `request_id`.
        # Track ID for crop mode
        # self.crop_request_id = self.processor.current_request_id + 100000 
        # Actually we can't force ID.
        # We can just track the ID returned by update_preview? 
        # But update_preview returns nothing, just queues.
        
        # Let's assume we request an update with specific params. Parameters are the key.
        base_params = params.copy()
        base_params['rotation'] = 0
        base_params['flip_horizontal'] = False
        base_params['flip_vertical'] = False
        base_params['crop'] = (0.0, 0.0, 1.0, 1.0)
        # Keep Lens Correction!
        # Disable others for speed? Or user wants to see exposure while cropping?
        # Usually we want to see exposure. So keep exposure/color.
        # Just reset Geometry.
        
        self.processor_connection_mode = 'crop' # State flag
        self.crop_request_id = self.processor.update_preview(self.current_raw_path, base_params)
        
        self.update_status(tr('loading_crop_view'))
        
    def on_crop_ready(self, img_uint8, params_context):
        """Called when processor finishes the image for Crop Viewer"""
        # Convert to QPixmap
        h, w, c = img_uint8.shape
        img = QImage(img_uint8.data, w, h, c * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        
        # Original Params (to retrieve current rotation/crop state)
        current_params = self.file_params_cache.get(self.current_raw_path, {})
        
        # Initialize Viewer
        self.crop_viewer.set_image(pixmap, current_params)
        
        # Switch View
        self.center_stack.setCurrentWidget(self.crop_viewer)
        self.loading_label.hide()

    def on_crop_applied(self, rotation, flip_h, flip_v, crop_rect):
        """User clicked Done"""
        # Update Params
        # We update the Inspector Panel, which updates its internal model and UI.
        # Accessing private method or expose public setter? 
        # `update_crop_params` was added to InspectorPanel.
        self.right_panel.update_crop_params(rotation, flip_h, flip_v, crop_rect)
        
        self.exit_crop_mode()
        
    def exit_crop_mode(self):
        """Return to normal view"""
        logger.info("Exiting crop mode...")
        
        # [Fix Issue 1] Generate temporary preview to prevent "jump" back to old state
        # effectively masking the lag while the processor works in background
        try:
            if self.crop_viewer.display_pixmap:
                # 1. Get the full rotated image
                full_pix = self.crop_viewer.display_pixmap
                w = full_pix.width()
                h = full_pix.height()
                
                # 2. Calculate crop rect in pixels
                cx, cy, cw, ch = self.crop_viewer.crop_rect
                px = int(cx * w)
                py = int(cy * h)
                pw = int(cw * w)
                ph = int(ch * h)
                
                # 3. Sanity check
                if pw > 0 and ph > 0:
                    # 4. Crop
                    temp_preview = full_pix.copy(px, py, pw, ph)
                    
                    # 5. Scale to preview label size for display
                    # Use SmoothTransformation and KeepAspectRatio to match normal behavior
                    # This prevents the window from resizing if the crop is larger than the viewport
                    scaled_preview = temp_preview.scaled(
                        self.preview_lbl.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    
                    self.preview_lbl.setPixmap(scaled_preview)
                    # self.preview_lbl.setText("") # Clear text if any
        except Exception as e:
            logger.warning(f"Failed to generate temp preview: {e}")

        self.processor_connection_mode = 'normal'
        self.center_stack.setCurrentWidget(self.page_preview)
        
        # Re-enable standard processing updates
        self.update_timer.stop() # Stop any pending
        
        # Trigger re-process with current params from the panel
        # This ensuring we use the latest state (including crop if applied, or old if cancelled)
        if self.current_raw_path:
            # Update cache immediately with new crop params
            self.file_params_cache[self.current_raw_path] = self.right_panel.get_params()
            # Trigger processing (handles request_id increment and viewport optimization)
            self.trigger_processing()
        
    # --- Perspective Mode Handling ---
    
    def enter_perspective_mode(self):
        """Switch to Perspective Correction view"""
        if not self.current_raw_path: return
        
        # If already in perspective mode, this button acts as "Done"
        if self.center_stack.currentWidget() == self.perspective_viewer:
            self.perspective_viewer._on_done()
            return
            
        logger.info("Entering Perspective Mode...")
         
        # 1. Disable standard processing updates temporarily
        self.update_timer.stop()
        
        # 2. Get current params
        params = self.file_params_cache.get(self.current_raw_path, {}).copy()
        
        # 3. Request base image (without geometry/perspective/crop applied)
        base_params = params.copy()
        base_params['rotation'] = 0
        base_params['flip_horizontal'] = False
        base_params['flip_vertical'] = False
        base_params['perspective_corners'] = None  # Reset perspective
        base_params['crop'] = (0.0, 0.0, 1.0, 1.0)
        
        self.processor_connection_mode = 'perspective'
        self.perspective_request_id = self.processor.update_preview(self.current_raw_path, base_params)
        
        self.update_status(tr('loading_perspective_view'))
        
    def on_perspective_ready(self, img_uint8, params_context):
        """Called when processor finishes the image for Perspective Viewer"""
        # Convert to QPixmap
        h, w, c = img_uint8.shape
        from PySide6.QtGui import QImage, QPixmap
        img = QImage(img_uint8.data, w, h, c * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        
        # Original Params (to retrieve current perspective state)
        current_params = self.file_params_cache.get(self.current_raw_path, {})
        
        # Initialize Viewer
        self.perspective_viewer.set_image(pixmap, current_params)
        
        # Switch View
        self.center_stack.setCurrentWidget(self.perspective_viewer)
        self.loading_label.hide()

    def on_perspective_applied(self, corners):
        """User clicked Done in perspective viewer"""
        # Update Params in cache
        if self.current_raw_path:
            current_params = self.file_params_cache.get(self.current_raw_path, {})
            current_params['perspective_corners'] = corners
            self.file_params_cache[self.current_raw_path] = current_params
            # Also update InspectorPanel internal state
            self.right_panel.set_params(current_params)
        
        self.exit_perspective_mode()
        
    def exit_perspective_mode(self):
        """Return to normal view from perspective mode"""
        logger.info("Exiting perspective mode...")

        self.processor_connection_mode = 'normal'
        self.center_stack.setCurrentWidget(self.page_preview)
        
        # Re-enable standard processing updates
        self.update_timer.stop()
        
        # Trigger re-process with current params
        if self.current_raw_path:
            self.file_params_cache[self.current_raw_path] = self.right_panel.get_params()
            self.trigger_processing()
        
    # Modify on_process_result to handle specific crop flow?
    # We need to intercept the result.
    
    # ... In next edit, I will patch `on_process_result` ...
    def on_thumbnail_progress(self, current, total):
        self.loading_label.setText(f"{tr('loading_thumbnails')}: {current}/{total}")
    
    def on_thumbnail_finished(self):
        self.loading_label.hide()

    def add_gallery_item(self, path, image):
        name = os.path.basename(path)
        pixmap = QPixmap.fromImage(image)
        
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setIcon(QIcon(pixmap))
        
        is_marked = path in self.marked_files
        item.setText(f"🟢 {name}" if is_marked else name)
        
        self.gallery_list.addItem(item)

    def on_gallery_item_clicked(self, item):
        if not item: return
        path = item.data(Qt.ItemDataRole.UserRole)
        if path == self.current_raw_path: return
        
        if self.current_raw_path:
            self.file_params_cache[self.current_raw_path] = self.right_panel.get_params()
        
        self.current_raw_path = path
        self.update_window_title()
        
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
            # [Fix Issue 11] New image: Reset geometry (crop/rotate) but keep other sticky settings
            # We get the current params from the panel (which still holds the previous image's state or defaults)
            current_sticky_params = self.right_panel.get_params()
            
            # Reset geometry defaults
            current_sticky_params['rotation'] = 0
            current_sticky_params['flip_horizontal'] = False
            current_sticky_params['flip_vertical'] = False
            current_sticky_params['crop'] = (0.0, 0.0, 1.0, 1.0)
            
            # Apply back to panel so the UI reflects the reset and next get_params() is correct
            self.right_panel.set_params(current_sticky_params)
        
        self.update_mark_button_state()
        self.load_image(path)
        
        if path in self.file_baseline_params_cache:
            QTimer.singleShot(25, self.regenerate_baseline_for_current_image)

    def load_image(self, path):
        self.preview_lbl.setText(tr('loading'))
        self.current_request_id = self.processor.load_image(path)
        
    def on_param_changed(self, params):
        self.update_timer.start()
    
    def trigger_processing(self):
        if not self.current_raw_path: return
        params = self.right_panel.get_params()
        # Pass viewport size for view-based LUT optimization
        size = self.preview_lbl.size()
        params['viewport_size'] = (size.width(), size.height())
        params['viewport_size'] = (size.width(), size.height())
        self.current_request_id = self.processor.update_preview(self.current_raw_path, params)
    
    def save_baseline_image(self):
        if not self.current_raw_path: return
        current_params = self.right_panel.get_params()
        self.file_baseline_params_cache[self.current_raw_path] = current_params.copy()
        
        if self.processor.cached_linear is not None:
            self.baseline_processor.cached_linear = self.processor.cached_linear
            self.baseline_processor.cached_corrected = self.processor.cached_corrected
            self.baseline_processor.cached_lens_key = self.processor.cached_lens_key
            self.baseline_processor.exif_data = self.processor.exif_data
            self.baseline_processor.current_path = self.current_raw_path
            self.baseline_processor.update_preview(self.current_raw_path, current_params)
        
        InfoBar.success(tr('baseline_saved'), tr('baseline_saved_message'), parent=self)
    
        
    def on_baseline_result(self, img_uint8, img_float, image_path, request_id):
        if image_path != self.current_raw_path: return
        h, w, c = img_uint8.shape
        qimg = QImage(img_uint8.data, w, h, c * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        self.baseline.update_full(pixmap, img_float)
    
    def regenerate_baseline_for_current_image(self):
        if not self.current_raw_path or self.current_raw_path not in self.file_baseline_params_cache: return
        if self.processor.cached_linear is None: return
        
        baseline_params = self.file_baseline_params_cache[self.current_raw_path]
        self.baseline_processor.cached_linear = self.processor.cached_linear
        self.baseline_processor.cached_corrected = self.processor.cached_corrected
        self.baseline_processor.cached_lens_key = self.processor.cached_lens_key
        self.baseline_processor.exif_data = self.processor.exif_data
        self.baseline_processor.current_path = self.current_raw_path
        self.baseline_processor.update_preview(self.current_raw_path, baseline_params)

    def on_process_result(self, img_uint8, img_float, image_path, request_id, applied_ev):
        """Handle processing result"""
        if image_path != self.current_raw_path:
            return

        # Check connection mode
        if hasattr(self, 'processor_connection_mode') and self.processor_connection_mode == 'crop':
            # Route to Crop Viewer
            if request_id == self.crop_request_id:
                self.on_crop_ready(img_uint8, None)
            return
        
        if hasattr(self, 'processor_connection_mode') and self.processor_connection_mode == 'perspective':
            # Route to Perspective Viewer
            if request_id == self.perspective_request_id:
                self.on_perspective_ready(img_uint8, None)
            return

        h, w, c = img_uint8.shape
        img = QImage(img_uint8.data, w, h, c * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        
        # Update Thumbnail
        for i in range(self.gallery_list.count()):
            item = self.gallery_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == image_path:
                # Align with ThumbnailWorker: scale to height 300, preserve aspect ratio, no padding.
                # This prevents the thumbnail from shrinking/changing layout when updated.
                scaled = pixmap.scaledToHeight(300, Qt.TransformationMode.SmoothTransformation)
                item.setIcon(QIcon(scaled))
                
                # Force repaint of the item area to prevent ghosting artifacts
                # when aspect ratio changes (e.g. rotation) on a transparent background
                rect = self.gallery_list.visualItemRect(item)
                self.gallery_list.viewport().update(rect)
                break
        
        if request_id != self.current_request_id or image_path != self.current_raw_path: return
        
        self.current.update_full(pixmap, img_float.copy())
        # Update params cache to current state
        if self.current_raw_path:
             self.file_params_cache[self.current_raw_path] = self.right_panel.get_params()
        
        if self.original.full is None:
            self.original.update_full(pixmap.copy(), img_float.copy())
        
        if self.right_panel.auto_exp_radio.isChecked():
            self.right_panel.auto_ev_value = applied_ev
            try:
                self.right_panel.exp_slider.valueChanged.disconnect(self.right_panel.exp_slider_callback)
            except (TypeError, RuntimeError):
                pass  # 信号未连接或已断开
            self.right_panel.exp_slider.setValue(int(applied_ev * 10))
            self.right_panel.exp_slider.update()
            self.right_panel.exp_value_label.setText(f"{tr('exposure_ev')}: {applied_ev:+.1f}")
            self.right_panel.exp_slider.valueChanged.connect(self.right_panel.exp_slider_callback)
        
        display_pixmap = self.current.get_display(self.preview_lbl.size())
        if display_pixmap:
            self.preview_lbl.setPixmap(display_pixmap)
            self.preview_lbl.update()
        
        self.right_panel.hist_widget.update_data(img_float)
        self.right_panel.waveform_widget.update_data(img_float)

    def on_load_complete(self, image_path, request_id):
        if request_id != self.current_request_id or image_path != self.current_raw_path: return
        self.preview_lbl.setText(tr('processing'))
        if self.update_timer.isActive(): self.update_timer.stop()
        if image_path == self.current_raw_path:
            QTimer.singleShot(0, lambda: self._trigger_processing_for_path(image_path))
    
    def _trigger_processing_for_path(self, path):
        if path != self.current_raw_path: return
        params = self.right_panel.get_params()
        # Pass viewport size for view-based LUT optimization
        size = self.preview_lbl.size()
        params['viewport_size'] = (size.width(), size.height())
        params['viewport_size'] = (size.width(), size.height())
        self.current_request_id = self.processor.update_preview(path, params)

    def on_error(self, msg):
        self.preview_lbl.setText(f"{tr('error')}: {msg}")
        InfoBar.error(tr('error'), msg, parent=self)

    # --- Toolbar Actions ---
    def prev_image(self):
        count = self.gallery_list.count()
        if count == 0: return
        row = self.gallery_list.currentRow()
        new_row = (row - 1) % count
        self.gallery_list.setCurrentRow(new_row)

    def next_image(self):
        count = self.gallery_list.count()
        if count == 0: return
        row = self.gallery_list.currentRow()
        new_row = (row + 1) % count
        self.gallery_list.setCurrentRow(new_row)

    def toggle_mark(self):
        if not self.current_raw_path: return
        if self.current_raw_path in self.marked_files:
            self.marked_files.remove(self.current_raw_path)
            InfoBar.info(tr('unmarked'), os.path.basename(self.current_raw_path), parent=self)
        else:
            self.marked_files.add(self.current_raw_path)
            InfoBar.success(tr('marked'), os.path.basename(self.current_raw_path), parent=self)
        self.update_mark_button_state()
        self.update_gallery_item_mark_indicator(self.current_raw_path)
    
    def update_mark_button_state(self):
        if not self.current_raw_path:
            self.btn_mark.setChecked(False)
            return
        self.btn_mark.blockSignals(True)
        self.btn_mark.setChecked(self.current_raw_path in self.marked_files)
        self.btn_mark.blockSignals(False)

    def update_gallery_item_mark_indicator(self, path):
        for i in range(self.gallery_list.count()):
            item = self.gallery_list.item(i)
            item_path = item.data(Qt.ItemDataRole.UserRole)
            if item_path == path:
                is_marked = path in self.marked_files
                name = os.path.basename(path)
                item.setText(f"🟢 {name}" if is_marked else name)
                break

    def delete_image(self):
        if not self.current_raw_path: return
        
        reply = QMessageBox.question(self, tr('delete_image'), tr('confirm_delete', filename=os.path.basename(self.current_raw_path)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                import send2trash
                send2trash.send2trash(os.path.normpath(self.current_raw_path))
                if self.current_raw_path in self.marked_files: self.marked_files.remove(self.current_raw_path)
                if self.current_raw_path in self.file_params_cache: del self.file_params_cache[self.current_raw_path]
                if self.current_raw_path in self.file_baseline_params_cache: del self.file_baseline_params_cache[self.current_raw_path]
                
                
                current_row = self.gallery_list.currentRow()
                for i in range(self.gallery_list.count()):
                    item = self.gallery_list.item(i)
                    if item.data(Qt.ItemDataRole.UserRole) == self.current_raw_path:
                        self.gallery_list.takeItem(i)
                        break
                
                if self.gallery_list.count() > 0:
                    if current_row >= self.gallery_list.count(): current_row = self.gallery_list.count() - 1
                    self.gallery_list.setCurrentRow(current_row)
                else:
                    self.current_raw_path = None
                    self.preview_lbl.setText(tr('no_image_selected'))
                    self.update_window_title()
                InfoBar.success(tr('delete_image'), tr('delete_image'), parent=self)
            except Exception as e:
                try:
                    # Fallback to os.remove if send2trash fails
                    os.remove(self.current_raw_path)
                    if self.current_raw_path in self.marked_files: self.marked_files.remove(self.current_raw_path)
                    # ... same cleanup logic ...
                    current_row = self.gallery_list.currentRow()
                    for i in range(self.gallery_list.count()):
                        item = self.gallery_list.item(i)
                        if item.data(Qt.ItemDataRole.UserRole) == self.current_raw_path:
                            self.gallery_list.takeItem(i)
                            break
                    if self.gallery_list.count() > 0:
                        if current_row >= self.gallery_list.count(): current_row = self.gallery_list.count() - 1
                        self.gallery_list.setCurrentRow(current_row)
                    else:
                        self.current_raw_path = None
                        self.preview_lbl.setText(tr('no_image_selected'))
                        self.update_window_title()
                    InfoBar.success(tr('delete_image'), tr('delete_image'), parent=self)
                except Exception as e2:
                    InfoBar.error(tr('delete_failed'), str(e2), parent=self)

    def show_original(self, event):
        img_to_show = self.baseline if self.baseline.full else self.original
        if not img_to_show.full: return
        display_pixmap = img_to_show.get_display(self.preview_lbl.size())
        if display_pixmap: self.preview_lbl.setPixmap(display_pixmap)
        InfoBar.info(tr('compare_showing_baseline'), "", parent=self)

    def show_processed(self, event):
        if not self.current.full:
            if self.original.full:
                display_pixmap = self.original.get_display(self.preview_lbl.size())
                if display_pixmap: self.preview_lbl.setPixmap(display_pixmap)
            return
        display_pixmap = self.current.get_display(self.preview_lbl.size())
        if display_pixmap: self.preview_lbl.setPixmap(display_pixmap)

    def export_current(self):
        if not self.current_raw_path: return
        base = os.path.splitext(os.path.basename(self.current_raw_path))[0]
        start_dir = self.last_export_path if self.last_export_path else (self.last_folder_path if self.last_folder_path else "")
        default_path = os.path.join(start_dir, base) if start_dir else base
        
        path, _ = QFileDialog.getSaveFileName(self, tr('export_image'), default_path, "JPEG (*.jpg);;HEIF (*.heif);;TIFF (*.tif);;DNG (*.dng)")
        
        if path:
            self.last_export_path = os.path.dirname(path)
            self.saving_infobar = InfoBar.info(tr('saving'), tr('saving_image'), duration=-1, parent=self)
            self.btn_export_curr.setEnabled(False)
            self.run_export(self.current_raw_path, path, is_single_export=True)

    def export_all(self):
        if not self.marked_files:
            InfoBar.warning(tr('no_files_marked'), tr('please_mark_files'), parent=self)
            return
        
        formats = ["JPEG", "HEIF", "TIFF", "DNG"]
        format_str, ok = QInputDialog.getItem(self, tr('select_export_format'), "Format:", formats, 0, False)
        if not ok: return
        
        start_dir = self.last_export_path if self.last_export_path else (self.last_folder_path if self.last_folder_path else "")
        folder = QFileDialog.getExistingDirectory(self, tr('select_export_folder'), start_dir)
        
        if folder:
            self.last_export_path = folder
            self.batch_export_list = list(self.marked_files)
            self.batch_export_folder = folder
            fmt_map = {"JPEG": "jpg", "HEIF": "heif", "TIFF": "tif", "DNG": "dng"}
            self.batch_export_ext = fmt_map.get(format_str, "jpg")
            
            self.batch_saving_infobar = InfoBar.info(tr('saving'), tr('batch_exporting'), duration=-1, parent=self)
            self.export_progress.setRange(0, len(self.batch_export_list))
            self.export_progress.setValue(0)
            self.export_progress.show()
            self.btn_export_all.setEnabled(False)
            self.batch_export_idx = 0
            self.batch_export_next()

    def batch_export_next(self):
        if self.batch_export_idx >= len(self.batch_export_list):
            if hasattr(self, 'batch_saving_infobar') and self.batch_saving_infobar:
                try:
                    self.batch_saving_infobar.close()
                except RuntimeError:
                    pass
                self.batch_saving_infobar = None
            InfoBar.success(tr('batch_export'), tr('all_exported'), parent=self)
            self.export_progress.hide()
            self.btn_export_all.setEnabled(True)
            return
        
        self.export_progress.setValue(self.batch_export_idx)
        input_path = self.batch_export_list[self.batch_export_idx]
        filename = os.path.basename(input_path)
        output_path = os.path.join(self.batch_export_folder, os.path.splitext(filename)[0] + "." + self.batch_export_ext)
        
        params = None
        if input_path == self.current_raw_path:
            params = self.right_panel.get_params()
        else:
            params = self.file_params_cache.get(input_path)
            if not params:
                pass
        
        self.batch_export_idx += 1
        self.run_export(input_path, output_path, params=params, callback=self.batch_export_next)

    def run_export(self, input_path, output_path, params=None, callback=None, is_single_export=False):
        p = params if params else self.right_panel.get_params()
        
        ext = os.path.splitext(output_path)[1].lower().replace('.', '')
        if ext not in ['jpg', 'heif', 'tif', 'tiff', 'dng']: ext = 'jpg'
        
        class ExportThread(QThread):
            finished_sig = Signal(bool, str)
            def run(self):
                try:
                    orchestrator.process_path(
                        input_path=input_path,
                        output_path=output_path,
                        log_space=p.get('log_space'),
                        lut_path=p.get('lut_path'),
                        exposure=p.get('exposure', 0.0),
                        lens_correct=p.get('lens_correct', True),
                        custom_db_path=p.get('custom_db_path'),
                        metering_mode=p.get('metering_mode', 'matrix'),
                        jobs=1,
                        logger_func=lambda msg: None,
                        output_format=ext,
                        wb_temp=p.get('wb_temp', 0.0),
                        wb_tint=p.get('wb_tint', 0.0),
                        saturation=p.get('saturation', 1.0),
                        contrast=p.get('contrast', 1.0),
                        highlight=p.get('highlight', 0.0),
                        shadow=p.get('shadow', 0.0),
                        # Geometry
                        rotation=p.get('rotation', 0),
                        flip_horizontal=p.get('flip_horizontal', False),
                        flip_vertical=p.get('flip_vertical', False),
                        crop=p.get('crop', (0.0, 0.0, 1.0, 1.0)),
                        # Denoising
                        denoise_strength=p.get('denoise_strength', 0.0),
                        # Sharpening
                        sharpen_strength=p.get('sharpen_strength', 0.0),
                    )
                    self.finished_sig.emit(True, "")
                except Exception as e:
                    self.finished_sig.emit(False, str(e))
        
        self.export_thread = ExportThread() # User self to keep reference
        
        def on_finish(success, msg):
            if is_single_export:
                if hasattr(self, 'saving_infobar') and self.saving_infobar:
                    try:
                        self.saving_infobar.close()
                    except RuntimeError:
                        pass
                    self.saving_infobar = None
                self.btn_export_curr.setEnabled(True)
            
            if success:
                if not callback: InfoBar.success(tr('export_success'), tr('saved_to', path=os.path.basename(output_path)), parent=self)
                if callback: callback()
            else:
                InfoBar.error(tr('export_failed'), msg, parent=self)
        
        self.export_thread.finished_sig.connect(on_finish)
        self.export_thread.start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.original.display = None
        self.current.display = None
        self.baseline.display = None
        if not hasattr(self, 'resize_timer'):
            self.resize_timer = QTimer()
            self.resize_timer.setSingleShot(True)
            self.resize_timer.setInterval(25)
            self.resize_timer.timeout.connect(self._on_resize_complete)
        self.resize_timer.start()

    def _on_resize_complete(self):
        if self.current.full:
            display_pixmap = self.current.get_display(self.preview_lbl.size())
            if display_pixmap: self.preview_lbl.setPixmap(display_pixmap)

    def on_denoise_started(self):
        """Called when denoising starts - show progress dialog"""
        from qfluentwidgets import StateToolTip
        
        if self.denoise_progress_dialog is None:
            self.denoise_progress_dialog = StateToolTip(
                tr('denoising'),
                tr('denoise_progress').format(current=0, total=0),
                self
            )
            self.denoise_progress_dialog.move(
                self.width() - self.denoise_progress_dialog.width() - 20,
                self.height() - self.denoise_progress_dialog.height() - 20
            )
            self.denoise_progress_dialog.show()
    
    def on_denoise_progress(self, current: int, total: int):
        """Called to update denoising progress"""
        if self.denoise_progress_dialog:
            self.denoise_progress_dialog.setContent(
                tr('denoise_progress').format(current=current, total=total)
            )
    
    def on_denoise_finished(self):
        """Called when denoising finishes - hide progress dialog"""
        if self.denoise_progress_dialog:
            self.denoise_progress_dialog.setContent(tr('done'))
            self.denoise_progress_dialog.setState(True)
            self.denoise_progress_dialog = None

    def closeEvent(self, event):
        self.save_settings()
        if self.thumb_worker and self.thumb_worker.isRunning():
            self.thumb_worker.stop()
            self.thumb_worker.quit()
            self.thumb_worker.wait()
        if self.processor.isRunning():
            self.processor.quit()
            self.processor.wait()
        super().closeEvent(event)
