import sys
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QListWidget, QFrame, QSplitter, QSizePolicy, QTreeView,
)
from PySide6.QtWidgets import QFileSystemModel as _QFileSystemModel
from PySide6.QtCore import QDir
from PySide6.QtCore import Qt, QSize, QTimer, QEvent, QRect
from PySide6.QtGui import QIcon, QPixmap, QImage

from qfluentwidgets import (
    FluentWindow, SubtitleLabel, PrimaryPushButton,
    CaptionLabel, InfoBar, Theme, setTheme,
    FluentIcon as FIF, ProgressRing,
    ToolButton, PushButton,
)

from raw_alchemy import config, lensfun_wrapper, i18n, sidecar
from raw_alchemy.i18n import tr
from raw_alchemy.thumb_cache import ThumbnailCache
from loguru import logger

# New structure imports
from raw_alchemy.ui.image_state import ImageState
from raw_alchemy.workers.image_processor import ImageProcessor
from raw_alchemy.ui.widgets.inspector_panel import InspectorPanel
from raw_alchemy.ui.widgets.title_bar import CenteredFluentTitleBar
from raw_alchemy.ui.widgets.crop_rotate_viewer import CropRotateViewer
from raw_alchemy.ui.widgets.perspective_viewer import PerspectiveViewer
from raw_alchemy.ui.widgets.help_panel import HelpPanel
from raw_alchemy.ui.widgets.about_panel import AboutPanel
from raw_alchemy.ui.viewport_gl import ImageViewportGL
from raw_alchemy.ui.widgets.settings_panel import SettingsPanel
from raw_alchemy.ui.edit_modes import EditModesMixin
from raw_alchemy.ui.export_controller import ExportControllerMixin
from raw_alchemy.ui.library_controller import LibraryControllerMixin
from PySide6.QtWidgets import QStackedWidget


class MainWindow(ExportControllerMixin, EditModesMixin, LibraryControllerMixin, FluentWindow):
    def __init__(self):
        # Initialize image states BEFORE super().__init__() to avoid resizeEvent issues
        # FluentWindow.__init__ may trigger resizeEvent during initialization
        self.original = ImageState()  # RAW decoded
        self.current = ImageState()   # Processed with current params
        self.baseline = ImageState()  # Saved baseline (optional)
        
        super().__init__()
        # Use a centered custom title bar to avoid left jumps while resizing.
        self.setTitleBar(CenteredFluentTitleBar(self))

        self.base_title = "Raw Alchemy Studio"
        self.setWindowTitle(self.base_title)
        self.setWindowIcon(QIcon(self._get_icon_path()))
        self.resize(1900, 1200)
        
        # State
        self.current_folder = None
        self.current_raw_path = None
        self.marked_files = set()
        self.gallery_items_by_path = {}
        self.file_params_cache = {}  # path -> params dict
        self.file_baseline_params_cache = {}  # path -> baseline params dict
        self.write_sidecar_enabled = True

        # Persistent thumbnail cache (T7.8); shared across folder switches
        # and toggled from Settings.
        self.thumb_cache = ThumbnailCache()
        self.thumb_cache_enabled = True

        # Last used paths for file dialogs
        self.last_folder_path = None  # Last opened gallery folder
        self.last_lut_folder_path = None  # Last LUT folder
        self.last_lensfun_db_path = None  # Last Lensfun DB path
        self.last_export_path = None  # Last export folder
        
        # Request tracking
        self.current_request_id = 0
        self._suppress_gallery_change = False
        
        # Preload lensfun database in the background.
        self._preload_lensfun_database()
        
        self.create_ui()
        self.create_settings_interface()
        self.create_help_interface()
        self.create_about_interface()

        # Workers
        self.thumb_worker = None
        self._retired_thumb_workers = []

        # Debounced viewport-priority updates for the thumbnail worker
        self._thumb_priority_timer = QTimer()
        self._thumb_priority_timer.setSingleShot(True)
        self._thumb_priority_timer.setInterval(100)
        self._thumb_priority_timer.timeout.connect(self._update_thumbnail_priority)
        self.gallery_list.verticalScrollBar().valueChanged.connect(
            self._on_gallery_scrolled
        )
        self.processor = ImageProcessor()
        self.processor.result_ready.connect(self.on_process_result)
        self.processor.load_complete.connect(self.on_load_complete)
        self.processor.error_occurred.connect(self.on_error)
        self.processor.denoise_started.connect(self.on_denoise_started)
        self.processor.denoise_progress.connect(self.on_denoise_progress)
        self.processor.denoise_finished.connect(self.on_denoise_finished)
        self.processor.preview_source_changed.connect(self.right_panel.update_scope_source)
        
        # Denoise progress dialog
        self.denoise_progress_dialog = None
        
        # Baseline processor
        self.baseline_processor = ImageProcessor()
        self.baseline_processor.result_ready.connect(self.on_baseline_result)
        
        # Processing Debounce
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(150) # 150ms debounce
        self.update_timer.timeout.connect(self.trigger_processing)

        self._preload_neighbors_args = None
        self._preload_neighbors_timer = QTimer()
        self._preload_neighbors_timer.setSingleShot(True)
        self._preload_neighbors_timer.setInterval(350)
        self._preload_neighbors_timer.timeout.connect(self._run_preload_neighbors)

        self._zoom_update_timer = QTimer()
        self._zoom_update_timer.setSingleShot(True)
        self._zoom_update_timer.setInterval(220)
        self._zoom_update_timer.timeout.connect(self.trigger_processing)

        self._sidecar_write_timer = QTimer()
        self._sidecar_write_timer.setSingleShot(True)
        self._sidecar_write_timer.setInterval(2000)
        self._sidecar_write_timer.timeout.connect(self._persist_current_sidecar_now)
        
        # Load saved settings
        self.load_settings()
        
        # Restore UI state from saved settings
        self.restore_ui()
        
        self.processor_connection_mode = 'normal' # 'normal', 'crop', or 'perspective'

    def systemTitleBarRect(self, size: QSize):
        """macOS: reserve the system traffic-light button area."""
        if sys.platform != "darwin":
            return super().systemTitleBarRect(size)

        y = 0 if self.isFullScreen() else 9
        return QRect(0, y, 100, size.height())

    def update_window_title(self):
        """Update the window title with the current filename."""
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
        """Preload the lensfun database without blocking GUI startup."""
        def preload():
            try:
                lensfun_wrapper._get_or_create_database(custom_db_path=None)
            except Exception as e:
                logger.error(f"[Lensfun] Failed to preload database: {e}")
        
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

        self.write_sidecar_enabled = settings.get('write_sidecar', True)
        if hasattr(self, 'settings_widget'):
            self.settings_widget.set_write_sidecar_enabled(self.write_sidecar_enabled)

        try:
            self.cache_limit_mb = int(settings.get('cache_limit_mb', config.CACHE_LIMIT_MB))
        except (TypeError, ValueError):
            self.cache_limit_mb = int(config.CACHE_LIMIT_MB)
        self._apply_cache_limit(self.cache_limit_mb)
        if hasattr(self, 'settings_widget'):
            self.settings_widget.set_cache_limit_mb(self.cache_limit_mb)

        self.thumb_cache_enabled = bool(settings.get('thumb_cache', True))
        self.thumb_cache.set_enabled(self.thumb_cache_enabled)
        if hasattr(self, 'settings_widget'):
            self.settings_widget.set_thumb_cache_enabled(self.thumb_cache_enabled)

    def restore_ui(self):
        """Restore UI state from saved settings"""
        if self.last_lut_folder_path and os.path.exists(self.last_lut_folder_path):
            self.right_panel.lut_folder = self.last_lut_folder_path
            self.right_panel.refresh_lut_list()

        # Folder restore is deferred to showEvent to ensure tree viewport is laid out
        self._need_restore_folder = True

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, '_need_restore_folder', False):
            self._need_restore_folder = False
            if self.last_folder_path and os.path.exists(self.last_folder_path):
                QTimer.singleShot(300, lambda: self._open_folder(self.last_folder_path))

    def _on_directory_loaded(self, loaded_path):
        """When a directory is loaded, check if we need to keep expanding towards the target."""
        if not self._pending_scroll_path:
            # New directories loading may shift layout; debounce a final scroll.
            if self._scroll_target_path:
                self._scroll_debounce_timer.start()
            return
        norm_loaded = os.path.normpath(loaded_path).lower()
        norm_target = os.path.normpath(self._pending_scroll_path).lower()

        if norm_loaded == norm_target:
            # Target directory loaded; select and scroll.
            target = self._pending_scroll_path
            self._pending_scroll_path = None
            self._expand_tree_to(target)
        else:
            # Check if loaded_path is an ancestor of target
            # Use rstrip to handle drive roots like "C:\" correctly
            norm_loaded_prefix = norm_loaded.rstrip(os.sep) + os.sep
            if norm_target.startswith(norm_loaded_prefix):
                # An ancestor was loaded; expand the next level toward target.
                remaining = os.path.normpath(self._pending_scroll_path)[len(norm_loaded_prefix):]
                next_name = remaining.split(os.sep)[0]
                next_path = os.path.join(loaded_path, next_name)
                next_idx = self.folder_model.index(next_path)
                if next_idx.isValid():
                    self.folder_tree.expand(next_idx)
    
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
            'last_export_path': self.last_export_path,
            'write_sidecar': self.write_sidecar_enabled,
            'cache_limit_mb': int(getattr(self, 'cache_limit_mb', config.CACHE_LIMIT_MB)),
            'thumb_cache': bool(getattr(self, 'thumb_cache_enabled', True)),
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
        
        # 1. Left Panel (Folder Tree + Gallery): Bridge-style browser
        self.left_panel = QWidget()
        self.left_panel.setFixedWidth(400)
        self.left_panel.setStyleSheet("background-color: transparent;")
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(5, 10, 5, 10)

        # --- Folder Tree ---
        self.folder_model = _QFileSystemModel()
        self.folder_model.setFilter(
            QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot | QDir.Filter.AllDirs | QDir.Filter.Drives
        )
        self.folder_model.setRootPath("")
        self._pending_scroll_path = None
        self._scroll_target_path = None
        self._scroll_debounce_timer = QTimer()
        self._scroll_debounce_timer.setSingleShot(True)
        self._scroll_debounce_timer.setInterval(500)
        self._scroll_debounce_timer.timeout.connect(self._do_final_scroll)
        self.folder_model.directoryLoaded.connect(self._on_directory_loaded)

        self.folder_tree = QTreeView()
        self.folder_tree.setModel(self.folder_model)
        self.folder_tree.setRootIndex(self.folder_model.index(""))
        # Only show the Name column
        for col in range(1, self.folder_model.columnCount()):
            self.folder_tree.hideColumn(col)
        self.folder_tree.setHeaderHidden(True)
        self.folder_tree.setAnimated(True)
        self.folder_tree.setIndentation(16)
        self.folder_tree.clicked.connect(self._on_folder_tree_clicked)
        self.folder_tree.setStyleSheet("""
            QTreeView {
                background-color: transparent;
                border: none;
                outline: none;
                color: white;
            }
            QTreeView::item {
                padding: 4px 2px;
            }
            QTreeView::item:selected {
                background-color: rgba(255, 255, 255, 0.1);
            }
            QTreeView::item:hover {
                background-color: rgba(255, 255, 255, 0.05);
            }
            QTreeView::branch {
                background-color: transparent;
            }
        """)

        # --- Gallery Thumbnail Grid ---
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
        self.gallery_list.currentItemChanged.connect(self._on_gallery_current_changed)
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

        # --- Splitter: Folder Tree (top) | Gallery (bottom) ---
        self.left_splitter = QSplitter(Qt.Orientation.Vertical)
        self.left_splitter.setStyleSheet("QSplitter::handle { background-color: rgba(255,255,255,0.08); height: 3px; }")

        # Top: folder tree
        tree_container = QWidget()
        tree_layout = QVBoxLayout(tree_container)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        tree_layout.setSpacing(4)
        tree_layout.addWidget(SubtitleLabel(tr('library')))
        tree_layout.addWidget(self.folder_tree)
        tree_layout.addWidget(self.open_btn)

        # Bottom: gallery thumbnails
        gallery_container = QWidget()
        gallery_layout = QVBoxLayout(gallery_container)
        gallery_layout.setContentsMargins(0, 0, 0, 0)
        gallery_layout.setSpacing(4)
        gallery_layout.addWidget(self.gallery_list)
        gallery_layout.addWidget(self.loading_label)

        self.left_splitter.addWidget(tree_container)
        self.left_splitter.addWidget(gallery_container)
        self.left_splitter.setSizes([250, 400])

        self.left_layout.addWidget(self.left_splitter)
        
        # 2. Center Panel (Preview Stack)
        self.center_panel = QWidget()
        self.center_layout = QVBoxLayout(self.center_panel)
        self.center_layout.setContentsMargins(0, 0, 0, 0)
        
        self.center_stack = QStackedWidget()
        
        # --- Page 1: Standard Preview ---
        self.page_preview = QWidget()
        self.preview_layout = QVBoxLayout(self.page_preview)
        self.preview_layout.setContentsMargins(10, 10, 10, 10)
        
        # OpenGL Viewport for GPU-accelerated image display
        self.viewport = ImageViewportGL()
        self.viewport.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.viewport.compare_pressed.connect(self.show_original)
        self.viewport.compare_released.connect(self.show_processed)
        self.viewport.viewport_resized.connect(self._on_viewport_resized)
        self.viewport.zoom_changed.connect(self._on_viewport_zoom_changed)
        self.viewport.roi_update_needed.connect(self._on_viewport_roi_update_needed)

        # Overlay label for status text (on top of viewport)
        self.preview_lbl = QLabel(tr('no_image_selected'))
        self.preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_lbl.setStyleSheet("background-color: transparent; color: white; font-size: 16px;")
        self.preview_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.preview_lbl.setParent(self.viewport)
        self.preview_lbl.raise_()
        
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

        self.btn_compare.pressed.connect(self.show_original)
        self.btn_compare.released.connect(self.show_processed)
        
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
        
        self.preview_layout.addWidget(self.viewport)
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
        self.settings_widget.write_sidecar_changed.connect(self.on_write_sidecar_changed)
        self.settings_widget.cache_limit_changed.connect(self.on_cache_limit_changed)
        self.settings_widget.thumb_cache_changed.connect(self.on_thumb_cache_changed)
        self.settings_widget.thumb_cache_clear_requested.connect(
            self.on_thumb_cache_clear_requested
        )
        self.addSubInterface(self.settings_widget, FIF.SETTING, tr('settings'))

    def on_cache_limit_changed(self, limit_mb):
        self.cache_limit_mb = int(limit_mb)
        self._apply_cache_limit(self.cache_limit_mb)

    def on_thumb_cache_changed(self, enabled):
        """Toggle the persistent thumbnail cache (T7.8). Off = legacy
        behavior: every folder visit re-extracts thumbnails."""
        self.thumb_cache_enabled = bool(enabled)
        self.thumb_cache.set_enabled(self.thumb_cache_enabled)

    def on_thumb_cache_clear_requested(self):
        removed = self.thumb_cache.clear()
        InfoBar.success(tr('thumb_cache_cleared'), str(removed), parent=self)

    def _apply_cache_limit(self, limit_mb):
        """Apply the image-cache memory cap to all processors (T7.6)."""
        for processor in (self.processor, self.baseline_processor):
            manager = getattr(processor, 'cache_manager', None)
            if manager is not None:
                manager.set_limit_mb(limit_mb)

    def on_write_sidecar_changed(self, enabled):
        self.write_sidecar_enabled = bool(enabled)
        if self.write_sidecar_enabled:
            if self.current_folder:
                self._load_sidecars_for_folder(self.current_folder)
            self._persist_current_sidecar_now()

    def _load_sidecars_for_folder(self, folder):
        if not self.write_sidecar_enabled:
            return
        for path, data in sidecar.load_folder_sidecars(folder).items():
            if data.params and path not in self.file_params_cache:
                self.file_params_cache[path] = data.params.copy()
            if data.marked:
                self.marked_files.add(path)

    def _load_sidecar_for_path(self, path):
        if not self.write_sidecar_enabled:
            return None
        data = sidecar.read_sidecar(path)
        if data is None:
            return None
        if data.params and path not in self.file_params_cache:
            self.file_params_cache[path] = data.params.copy()
        if data.marked:
            self.marked_files.add(path)
        return data.params.copy()

    def _cache_current_params(self):
        if not self.current_raw_path or not hasattr(self, 'right_panel'):
            return None
        params = self.right_panel.get_params().copy()
        self.file_params_cache[self.current_raw_path] = params
        return params

    def _write_sidecar_for_path(self, path, params=None):
        if not self.write_sidecar_enabled or not path:
            return
        try:
            if params is None:
                if path == self.current_raw_path:
                    params = self._cache_current_params()
                else:
                    params = self.file_params_cache.get(path, {})
            sidecar.write_sidecar(path, params or {}, path in self.marked_files)
        except Exception as e:
            logger.warning(f"Failed to write sidecar for {path}: {e}")

    def _persist_current_sidecar_now(self):
        if not self.current_raw_path:
            return
        params = self._cache_current_params()
        self._write_sidecar_for_path(self.current_raw_path, params)

    def _schedule_current_sidecar_write(self):
        if self.write_sidecar_enabled and self.current_raw_path:
            self._sidecar_write_timer.start()

    def changeEvent(self, event):
        """Track window minimize/restore to suppress spurious gallery signals."""
        if event.type() == QEvent.Type.WindowStateChange:
            old_minimized = bool(event.oldState() & Qt.WindowState.WindowMinimized)
            new_minimized = bool(self.windowState() & Qt.WindowState.WindowMinimized)
            if old_minimized and not new_minimized:
                # Restoring from minimize; suppress currentItemChanged briefly.
                self._suppress_gallery_change = True
                QTimer.singleShot(500, self._unsuppress_gallery_change)
        super().changeEvent(event)

    def _unsuppress_gallery_change(self):
        self._suppress_gallery_change = False

    def _on_gallery_current_changed(self, current, prev):
        """Wrapper that suppresses gallery changes during minimize/restore."""
        if getattr(self, '_suppress_gallery_change', False):
            return
        self.on_gallery_item_clicked(current)

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
                        self.show_original()
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
                        self.show_processed()
                    return True
        return super().eventFilter(obj, event)

    def on_param_changed(self, params):
        if self.current_raw_path:
            self.file_params_cache[self.current_raw_path] = params.copy()
            self._schedule_current_sidecar_write()
        self.update_timer.start()

    def _on_viewport_zoom_changed(self, zoom):
        if not self.current_raw_path:
            return
        if getattr(self, 'processor_connection_mode', 'normal') != 'normal':
            return
        self._zoom_update_timer.start()

    def _on_viewport_resized(self, w, h):
        self._sync_preview_label_geometry(w, h)
        if not self.current_raw_path:
            return
        if getattr(self, 'processor_connection_mode', 'normal') != 'normal':
            return
        self._zoom_update_timer.start()

    def _on_viewport_roi_update_needed(self):
        """Pan left the rendered ROI coverage (T7.5): re-render, debounced."""
        if not self.current_raw_path:
            return
        if getattr(self, 'processor_connection_mode', 'normal') != 'normal':
            return
        self._zoom_update_timer.start()

    def _add_preview_output_params(self, params, include_visible_rect=True):
        params = params.copy()
        size = self.viewport.size()
        params['viewport_size'] = (size.width(), size.height())
        params['preview_zoom'] = self.viewport.preview_zoom()
        try:
            params['device_pixel_ratio'] = float(self.viewport.devicePixelRatioF())
        except Exception:
            params['device_pixel_ratio'] = 1.0
        if include_visible_rect:
            # Lets the worker do zoom>fit ROI rendering (T7.5). Excluded for
            # baseline renders, which must always be full-frame.
            visible_rect = self.viewport.visible_source_rect()
            if visible_rect:
                params['preview_visible_rect'] = visible_rect
        return params

    def _get_preview_params(self):
        return self._add_preview_output_params(self.right_panel.get_params())
    
    def trigger_processing(self):
        if not self.current_raw_path: return
        params = self._get_preview_params()
        self.current_request_id = self.processor.update_preview(self.current_raw_path, params)
    
    def save_baseline_image(self):
        if not self.current_raw_path: return
        current_params = self.right_panel.get_params()
        self.file_baseline_params_cache[self.current_raw_path] = current_params.copy()

        if self.processor.cpu_linear is not None:
            # Share CPU cache so baseline processor can load from it
            self.baseline_processor.cache_manager = self.processor.cache_manager
            self.baseline_processor.update_preview(
                self.current_raw_path,
                self._add_preview_output_params(current_params, include_visible_rect=False),
            )

        InfoBar.success(tr('baseline_saved'), tr('baseline_saved_message'), parent=self)
    
        
    def on_baseline_result(self, img_uint8, image_path, request_id, applied_ev, source_size=None):
        if image_path != self.current_raw_path: return
        h, w, c = img_uint8.shape
        qimg = QImage(img_uint8.data, w, h, c * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())
        self.baseline.update_full(pixmap, None, img_uint8.copy(), source_size=source_size)
    
    def regenerate_baseline_for_current_image(self):
        if not self.current_raw_path or self.current_raw_path not in self.file_baseline_params_cache: return
        if not self.processor.cpu_linear is not None: return

        baseline_params = self.file_baseline_params_cache[self.current_raw_path]
        self.baseline_processor.cache_manager = self.processor.cache_manager
        self.baseline_processor.update_preview(
            self.current_raw_path,
            self._add_preview_output_params(baseline_params, include_visible_rect=False),
        )

    def on_process_result(self, img_uint8, image_path, request_id, applied_ev, source_size=None):
        """Handle processing result"""
        # Close denoise/processing progress dialog
        if self.denoise_progress_dialog:
            self.denoise_progress_dialog.setContent(tr('done'))
            self.denoise_progress_dialog.setState(True)
            self.denoise_progress_dialog = None

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

        # zoom>fit ROI results (T7.5) carry the ROI rect on source_size and
        # contain only the visible region: they update the viewport's ROI
        # overlay but must not feed full-frame consumers (gallery thumbnail,
        # compare/baseline states).
        roi_rect = getattr(source_size, 'roi_rect', None)

        h, w, c = img_uint8.shape
        img = QImage(img_uint8.data, w, h, c * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)

        if roi_rect is None:
            # Update thumbnail without scanning the full gallery.
            item = self.gallery_items_by_path.get(image_path)
            if item is not None:
                scaled = pixmap.scaledToHeight(300, Qt.TransformationMode.FastTransformation)
                item.setIcon(QIcon(scaled))
                rect = self.gallery_list.visualItemRect(item)
                self.gallery_list.viewport().update(rect)

        if request_id != self.current_request_id or image_path != self.current_raw_path: return

        if roi_rect is None:
            self.current.update_full(pixmap, None, img_uint8.copy(), source_size=source_size)
            if self.original.full is None:
                self.original.update_full(pixmap.copy(), None, img_uint8.copy(), source_size=source_size)
        if self.current_raw_path:
             self.file_params_cache[self.current_raw_path] = self.right_panel.get_params()

        if self.right_panel.auto_exp_radio.isChecked():
            self.right_panel.auto_ev_value = applied_ev
            try:
                self.right_panel.exp_slider.valueChanged.disconnect(self.right_panel.exp_slider_callback)
            except (TypeError, RuntimeError):
                pass
            self.right_panel.exp_slider.setValue(int(applied_ev * 10))
            self.right_panel.exp_slider.update()
            self.right_panel.exp_value_label.setText(f"{tr('exposure_ev')}: {applied_ev:+.1f}")
            self.right_panel.exp_slider.valueChanged.connect(self.right_panel.exp_slider_callback)

        # Display via OpenGL viewport first.
        if roi_rect is not None:
            self.viewport.set_roi_image(img_uint8, source_size, roi_rect)
        else:
            self.viewport.set_image(img_uint8, source_size=source_size)
        self.preview_lbl.setText("")

        # Defer visible scope update to the next event loop iteration.
        img_for_hist = img_uint8
        QTimer.singleShot(0, lambda: self._update_histogram_async(img_for_hist))

    def _update_histogram_async(self, img_uint8):
        """Deferred scope update; runs after viewport display."""
        self.right_panel.update_scope_data(img_uint8)

    def on_load_complete(self, image_path, request_id):
        if request_id != self.current_request_id or image_path != self.current_raw_path: return
        self.preview_lbl.setText(tr('processing'))
        if self.update_timer.isActive(): self.update_timer.stop()
        if image_path == self.current_raw_path:
            QTimer.singleShot(0, lambda: self._trigger_processing_for_path(image_path))
    
    def _trigger_processing_for_path(self, path):
        if path != self.current_raw_path: return
        params = self._get_preview_params()
        self.current_request_id = self.processor.update_preview(path, params)

    def on_error(self, msg):
        self.preview_lbl.setText(f"{tr('error')}: {msg}")
        InfoBar.error(tr('error'), msg, parent=self)

    def _sync_preview_label_geometry(self, w=None, h=None):
        if hasattr(self, 'preview_lbl') and hasattr(self, 'viewport'):
            self.preview_lbl.setGeometry(0, 0, self.viewport.width(), self.viewport.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_preview_label_geometry()

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
        """Called when denoising finishes - keep dialog showing pipeline progress"""
        if self.denoise_progress_dialog:
            self.denoise_progress_dialog.setTitle(tr('processing'))
            self.denoise_progress_dialog.setContent(tr('denoise_done_processing'))

    def closeEvent(self, event):
        if hasattr(self, '_sidecar_write_timer'):
            self._sidecar_write_timer.stop()
        self._persist_current_sidecar_now()
        self.save_settings()
        if hasattr(self, '_preload_neighbors_timer'):
            self._preload_neighbors_timer.stop()
        if hasattr(self, '_zoom_update_timer'):
            self._zoom_update_timer.stop()
        if hasattr(self, '_thumb_priority_timer'):
            self._thumb_priority_timer.stop()
        thumb_workers = list(getattr(self, '_retired_thumb_workers', []))
        if self.thumb_worker:
            thumb_workers.append(self.thumb_worker)
        for worker in thumb_workers:
            worker.stop()
        for worker in thumb_workers:
            if worker.isRunning():
                worker.quit()
                # stop() is checked before each decode, so this is bounded
                worker.wait(2000)
        if hasattr(self, 'right_panel'):
            self.right_panel.shutdown_scope_workers()
        self.processor.stop_and_cleanup()
        if hasattr(self, 'baseline_processor'):
            self.baseline_processor.stop_and_cleanup()
        super().closeEvent(event)
