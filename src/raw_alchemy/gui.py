import sys
import os
import time
import threading
import multiprocessing
import numpy as np
import rawpy
import colour
import gc
from typing import Optional, Dict

from PyQt6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QFileDialog, QListWidget, QListWidgetItem, QFrame,
    QSplitter, QSizePolicy, QGraphicsDropShadowEffect, QGridLayout,
    QInputDialog
)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QObject, QTimer, QEvent
from PyQt6.QtGui import QIcon, QPixmap, QImage, QPainter, QColor, QResizeEvent, QTransform

from qfluentwidgets import (
    FluentWindow, SubtitleLabel, PrimaryPushButton, PushButton,
    ComboBox, Slider, CaptionLabel, SwitchButton, StrongBodyLabel,
    BodyLabel, LineEdit, ToolButton, FluentIcon as FIF,
    CardWidget, SimpleCardWidget, ScrollArea, IndeterminateProgressRing,
    InfoBar, InfoBarPosition, Theme, setTheme, CheckBox, ProgressRing
)

from raw_alchemy import config, utils, orchestrator, metering, lensfun_wrapper, i18n
from raw_alchemy.i18n import tr
from raw_alchemy.orchestrator import SUPPORTED_RAW_EXTENSIONS

# ==============================================================================
#                               Worker Threads
# ==============================================================================

class ThumbnailWorker(QThread):
    """Scan folder and generate thumbnails"""
    thumbnail_ready = pyqtSignal(str, QImage)
    finished_scanning = pyqtSignal()

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.stopped = False

    def run(self):
        if not os.path.exists(self.folder_path):
            return

        files = sorted([f for f in os.listdir(self.folder_path) 
                        if os.path.splitext(f)[1].lower() in SUPPORTED_RAW_EXTENSIONS])
        
        for f in files:
            if self.stopped: break
            full_path = os.path.join(self.folder_path, f)
            try:
                # Attempt to use rawpy to extract thumbnail
                with rawpy.imread(full_path) as raw:
                    try:
                        thumb = raw.extract_thumb()
                    except rawpy.LibRawNoThumbnailError:
                        thumb = None
                    
                    if thumb:
                        if thumb.format == rawpy.ThumbFormat.JPEG:
                            image = QImage()
                            image.loadFromData(thumb.data)
                        elif thumb.format == rawpy.ThumbFormat.BITMAP:
                             # Handle bitmap if necessary, or skip
                             continue
                    else:
                        # Fallback or skip
                        continue

                    if not image.isNull():
                        # Handle rotation based on flip value
                        orientation = raw.sizes.flip
                        if orientation == 3:
                            image = image.transformed(QTransform().rotate(180))
                        elif orientation == 5:
                            image = image.transformed(QTransform().rotate(-90))
                        elif orientation == 6:
                            image = image.transformed(QTransform().rotate(90))

                        # Scale down
                        scaled = image.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                        self.thumbnail_ready.emit(full_path, scaled)
            except Exception as e:
                print(f"Error generating thumb for {f}: {e}")
                continue
        
        self.finished_scanning.emit()

    def stop(self):
        self.stopped = True

class ImageProcessor(QThread):
    """
    Handles the heavy lifting of the image pipeline.
    Modes:
    1. 'load': Load RAW -> Demosaic -> Lens Correction (Cache this stage)
    2. 'process': Cached Linear -> Exp/WB/Effects -> Log/LUT -> Display
    """
    result_ready = pyqtSignal(np.ndarray, np.ndarray, str) # display_image, histogram_data, image_path
    original_ready = pyqtSignal(np.ndarray, str) # original_image, image_path
    load_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.mode = 'idle'
        self.running_mode = 'idle'
        self.params = {}
        self.params_dirty = False
        self.raw_path = None
        self.processing_path = None  # Track which image is being processed
        
        # Caches
        self.cached_linear = None # After demosaic
        self.cached_corrected = None # After lens correction
        self.cached_lens_key = None # (enable, db_path)
        self.exif_data = None

    def load_image(self, path):
        self.raw_path = path
        self.processing_path = path  # Mark which image we're processing
        self.mode = 'load'
        if not self.isRunning():
            self.start()

    def update_preview(self, params):
        self.params = params
        self.params_dirty = True
        self.mode = 'process'
        if not self.isRunning():
            self.params_dirty = False
            self.start()

    def run(self):
        self.running_mode = self.mode
        try:
            if self.mode == 'load':
                self._do_load()
            elif self.mode == 'process':
                self._do_process()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))

    def _do_load(self):
        if not self.raw_path: return
        
        # Store the path we're processing
        current_path = self.raw_path
        
        with rawpy.imread(self.raw_path) as raw:
            self.exif_data = utils.extract_lens_exif(raw)
            # Use half_size for faster preview
            raw_post = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=True,
                output_bps=16,
                output_color=rawpy.ColorSpace.ProPhoto,
                bright=1.0,
                highlight_mode=2,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AAHD,
                half_size=True
            )
            img = raw_post.astype(np.float32) / 65535.0
            
            # Downscale if still too big for 1080p preview (optional optimization)
            h, w = img.shape[:2]
            max_dim = 2048
            if max(h, w) > max_dim:
                from scipy.ndimage import zoom
                scale = max_dim / max(h, w)
                img = zoom(img, (scale, scale, 1), order=1)
                
            self.cached_linear = img
            self.cached_corrected = None
            self.cached_lens_key = None
            
            # Generate Original Preview with Auto Exposure, Lens Correction, and Camera Boost
            # This represents what the camera would show with auto settings
            orig_preview = img.copy()
            
            # 1. Apply Lens Correction (if EXIF data available)
            if self.exif_data:
                orig_preview = utils.apply_lens_correction(
                    orig_preview,
                    self.exif_data,
                    custom_db_path=None
                )
            
            # 2. Apply Auto Exposure (Matrix metering by default)
            source_cs = colour.RGB_COLOURSPACES['ProPhoto RGB']
            class DummyLogger:
                def info(self, *args, **kwargs):
                    pass
            metering.apply_auto_exposure(orig_preview, source_cs, 'matrix', logger=DummyLogger())
            
            # 3. Apply Camera Boost (Saturation and Contrast)
            utils.apply_saturation_and_contrast(orig_preview, saturation=1.25, contrast=1.1)
            
            # 4. Convert to sRGB for display
            utils.bt709_to_srgb_inplace(orig_preview)
            orig_preview = np.clip(orig_preview, 0, 1)
            orig_uint8 = (orig_preview * 255).astype(np.uint8)
            self.original_ready.emit(orig_uint8, current_path)
            
            self.load_finished.emit()

    def _do_process(self):
        # Fallback: If we are in process mode but haven't loaded data yet (race condition fix)
        if self.cached_linear is None:
            if self.raw_path:
                self._do_load()
            
            if self.cached_linear is None:
                return
        
        # Make a local copy of params to avoid race conditions
        params = self.params.copy()
        
        # 1. Lens Correction Check
        current_lens_key = (params.get('lens_correct'), params.get('custom_db_path'))
        
        img = None
        if current_lens_key != self.cached_lens_key or self.cached_corrected is None:
            # Re-run lens correction
            temp = self.cached_linear.copy()
            if params.get('lens_correct') and self.exif_data:
                temp = utils.apply_lens_correction(
                    temp,
                    self.exif_data,
                    custom_db_path=params.get('custom_db_path')
                )
            self.cached_corrected = temp
            self.cached_lens_key = current_lens_key
        
        # Start from corrected base
        img = self.cached_corrected.copy()
        
        # 2. Exposure
        if params.get('exposure_mode') == 'Manual':
            gain = 2.0 ** params.get('exposure', 0.0)
            utils.apply_gain_inplace(img, gain)
        else:
             # Auto exposure
             source_cs = colour.RGB_COLOURSPACES['ProPhoto RGB']
             mode = params.get('metering_mode', 'matrix')
             # Create a dummy logger object with info method
             class DummyLogger:
                 def info(self, *args, **kwargs):
                     pass
             metering.apply_auto_exposure(img, source_cs, mode, logger=DummyLogger())

        # 3. White Balance
        temp_val = params.get('wb_temp', 0.0)
        tint = params.get('wb_tint', 0.0)
        utils.apply_white_balance(img, temp_val, tint)

        # 4. Highlight / Shadow
        hl = params.get('highlight', 0.0)
        sh = params.get('shadow', 0.0)
        utils.apply_highlight_shadow(img, hl, sh)

        # 5. Saturation / Contrast
        sat = params.get('saturation', 1.0)
        con = params.get('contrast', 1.0)
        utils.apply_saturation_and_contrast(img, saturation=sat, contrast=con)

        # 6. Log Transform
        log_space = params.get('log_space')
        if log_space and log_space != 'None':
             log_color_space = config.LOG_TO_WORKING_SPACE.get(log_space)
             log_curve = config.LOG_ENCODING_MAP.get(log_space, log_space)
             
             if log_color_space:
                 M = colour.matrix_RGB_to_RGB(
                     colour.RGB_COLOURSPACES['ProPhoto RGB'],
                     colour.RGB_COLOURSPACES[log_color_space]
                 )
                 if not img.flags['C_CONTIGUOUS']: img = np.ascontiguousarray(img)
                 utils.apply_matrix_inplace(img, M)
                 np.maximum(img, 1e-6, out=img)
                 img = colour.cctf_encoding(img, function=log_curve)
        
        # 7. LUT
        lut_path = params.get('lut_path')
        if lut_path and os.path.exists(lut_path):
            try:
                lut = colour.read_LUT(lut_path)
                if isinstance(lut, colour.LUT3D):
                    if not img.flags['C_CONTIGUOUS']: img = np.ascontiguousarray(img)
                    if img.dtype != np.float32: img = img.astype(np.float32)
                    if lut.table.dtype != np.float32: lut.table = lut.table.astype(np.float32)
                    utils.apply_lut_inplace(img, lut.table, lut.domain[0], lut.domain[1])
                else:
                    img = lut.apply(img)
            except:
                pass

        # 8. Display Transform (to sRGB)
        # We need sRGB for display. If we are in Log space, we just display as is (Log looks flat)
        # If user selected "None" log space, we are in ProPhoto Linear. We need to convert to sRGB for display.
        # However, usually Log/LUT pipeline outputs display referred or Log.
        # For this preview:
        # If Log is applied -> Display as is (or if LUT applied, it handles it)
        # If Log is None -> Convert ProPhoto Linear -> sRGB
        
        if not log_space or log_space == 'None':
             utils.bt709_to_srgb_inplace(img)
        
        img = np.clip(img, 0, 1)
        
        # Store float version for histogram before converting to uint8
        img_float = img.copy()
        
        # Convert to uint8 for QImage
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Emit with the path being processed
        self.result_ready.emit(img_uint8, img_float, self.processing_path)


# ==============================================================================
#                               UI Components
# ==============================================================================

class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(150)
        self.hist_data = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def update_data(self, img_array):
        # 使用 numba 加速的直方图计算
        if img_array is None:
            return
        
        try:
             # img_array is float 0-1
             # 使用 utils 中的快速直方图计算函数
             self.hist_data = utils.compute_histogram_fast(img_array, bins=100, sample_rate=4)
             self.repaint()  # 使用 repaint() 强制立即重绘，而不是 update()
        except Exception as e:
             print(f"Histogram update error: {e}")
             import traceback
             traceback.print_exc()

    def paintEvent(self, event):
        if not self.hist_data:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # 检查是否有有效数据
        try:
            max_val = max([np.max(hist) for hist in self.hist_data]) if self.hist_data else 1
            if max_val == 0 or max_val < 1e-10:
                max_val = 1
        except Exception as e:
            print(f"Error computing max_val: {e}")
            return
        
        colors = [QColor(255, 50, 50, 180), QColor(50, 255, 50, 180), QColor(50, 50, 255, 180)]
        
        for i, hist in enumerate(self.hist_data):
            if len(hist) == 0:
                continue
                
            path_color = colors[i]
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(path_color)
            
            bin_w = w / len(hist)
            
            # Draw bars (or polygon)
            from PyQt6.QtGui import QPolygonF
            from PyQt6.QtCore import QPointF
            
            points = [QPointF(0, h)]
            for j, val in enumerate(hist):
                x = j * bin_w
                y = h - (float(val) / max_val * h)
                points.append(QPointF(x, y))
            points.append(QPointF(w, h))
            
            painter.drawPolygon(QPolygonF(points))


class GalleryItem(QWidget):
    """Custom widget for gallery item (Image + Text)"""
    def __init__(self, path, pixmap, parent=None):
        super().__init__(parent)
        self.path = path
        self.base_name = os.path.basename(path)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.img_label = QLabel()
        self.img_label.setPixmap(pixmap)
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setFixedSize(140, 100)
        self.img_label.setScaledContents(True)
        
        # Text label with green dot indicator
        self.text_label = CaptionLabel(self.base_name)
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(self.img_label)
        layout.addWidget(self.text_label)
    
    def set_marked(self, marked):
        """Show or hide the green dot indicator in the filename"""
        if marked:
            self.text_label.setText(f"🟢 {self.base_name}")
        else:
            self.text_label.setText(self.base_name)

class InspectorPanel(ScrollArea):
    """Right side control panel"""
    param_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = QWidget()
        self.view.setObjectName("view")
        self.setWidget(self.view)
        self.setWidgetResizable(True)
        self.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")
        self.view.setStyleSheet("#view { background-color: transparent; }")
        
        self.v_layout = QVBoxLayout(self.view)
        self.v_layout.setSpacing(20)
        self.v_layout.setContentsMargins(20, 20, 20, 20)
        
        # --- Histogram ---
        self.hist_widget = HistogramWidget()
        self.add_section(tr('histogram'), self.hist_widget)

        # --- Exposure ---
        self.exp_card = SimpleCardWidget()
        exp_layout = QVBoxLayout(self.exp_card)
        
        self.auto_exp_radio = SwitchButton(text=tr('auto_exposure'))
        self.auto_exp_radio.setChecked(True)  # Default to Auto Exposure
        self.auto_exp_radio.checkedChanged.connect(self._on_exposure_mode_changed)
        
        self.metering_lbl = BodyLabel(tr('metering_mode'))
        self.metering_combo = ComboBox()
        # Store metering mode mapping: display text -> internal key
        self.metering_mode_map = {
            tr('matrix'): 'matrix',
            tr('average'): 'average',
            tr('center_weighted'): 'center-weighted',
            tr('highlight_safe'): 'highlight-safe',
            tr('hybrid'): 'hybrid'
        }
        # Reverse mapping: internal key -> display text
        self.metering_mode_reverse_map = {v: k for k, v in self.metering_mode_map.items()}
        
        self.metering_combo.addItems([tr('matrix'), tr('average'), tr('center_weighted'), tr('highlight_safe'), tr('hybrid')])
        self.metering_combo.setCurrentText(tr('matrix'))
        self.metering_combo.currentTextChanged.connect(self._on_param_change)
        
        self.exp_slider = Slider(Qt.Orientation.Horizontal)
        self.exp_slider.setRange(-50, 50) # -5.0 to 5.0
        self.exp_slider.setValue(0)
        
        # Add exposure value label
        self.exp_value_label = BodyLabel(tr('exposure_ev') + ": 0.0")
        
        def update_exp_label(val):
            real_val = val / 10.0
            self.exp_value_label.setText(f"{tr('exposure_ev')}: {real_val:+.1f}")
            self._on_param_change()
        
        self.exp_slider.valueChanged.connect(update_exp_label)
        
        exp_layout.addWidget(self.auto_exp_radio)
        exp_layout.addWidget(self.metering_lbl)
        exp_layout.addWidget(self.metering_combo)
        exp_layout.addWidget(self.exp_value_label)
        exp_layout.addWidget(self.exp_slider)
        
        self._update_exposure_ui_state()
        
        self.add_section(tr('exposure'), self.exp_card)
        
        # --- Color / Log ---
        self.color_card = SimpleCardWidget()
        color_layout = QVBoxLayout(self.color_card)
        
        # Log Space
        color_layout.addWidget(BodyLabel(tr('log_space')))
        self.log_combo = ComboBox()
        log_items = [tr('none')] + list(config.LOG_TO_WORKING_SPACE.keys())
        self.log_combo.addItems(log_items)
        self.log_combo.setCurrentText('None')
        self.log_combo.currentTextChanged.connect(self._on_param_change)
        color_layout.addWidget(self.log_combo)
        
        # LUT
        color_layout.addWidget(BodyLabel(tr('lut')))
        lut_layout = QHBoxLayout()
        self.lut_combo = ComboBox()
        self.lut_combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.lut_combo.addItem(tr('none'))
        self.lut_combo.currentTextChanged.connect(self._on_param_change)
        
        self.lut_btn = ToolButton(FIF.FOLDER)
        self.lut_btn.clicked.connect(self._browse_lut_folder)
        
        lut_layout.addWidget(self.lut_combo, 1)
        lut_layout.addWidget(self.lut_btn)
        color_layout.addLayout(lut_layout)
        
        self.add_section(tr('color_management'), self.color_card)
        
        # --- Lens Correction ---
        self.lens_card = SimpleCardWidget()
        lens_layout = QVBoxLayout(self.lens_card)
        
        self.lens_correct_switch = SwitchButton(text=tr('enable_lens_correction'))
        self.lens_correct_switch.setChecked(True)  # Default enabled
        self.lens_correct_switch.checkedChanged.connect(self._on_param_change)
        lens_layout.addWidget(self.lens_correct_switch)
        
        # Custom Lensfun DB
        lens_layout.addWidget(BodyLabel(tr('custom_lensfun_db')))
        db_layout = QHBoxLayout()
        self.db_path_edit = LineEdit()
        self.db_path_edit.setPlaceholderText(tr('optional_db_path'))
        self.db_path_edit.setReadOnly(True)
        self.db_path_edit.textChanged.connect(self._on_param_change)
        
        self.db_browse_btn = ToolButton(FIF.FOLDER)
        self.db_browse_btn.clicked.connect(self._browse_lensfun_db)
        
        self.db_clear_btn = ToolButton(FIF.CLOSE)
        self.db_clear_btn.clicked.connect(self._clear_lensfun_db)
        
        db_layout.addWidget(self.db_path_edit, 1)
        db_layout.addWidget(self.db_browse_btn)
        db_layout.addWidget(self.db_clear_btn)
        lens_layout.addLayout(db_layout)
        
        self.add_section(tr('lens_correction'), self.lens_card)
        
        # --- Adjustments ---
        self.adj_card = SimpleCardWidget()
        adj_layout = QVBoxLayout(self.adj_card)
        
        self.sliders = {}
        self.slider_labels = {}  # 存储标签引用以便更新
        
        def add_slider(key, name, min_v, max_v, default_v, scale=1.0):
            layout = QVBoxLayout()
            lbl = BodyLabel(f"{name}: {default_v}")
            slider = Slider(Qt.Orientation.Horizontal)
            # slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            slider.setRange(int(min_v*scale), int(max_v*scale))
            slider.setValue(int(default_v*scale))
            
            def update_lbl(val):
                real_val = val / scale
                lbl.setText(f"{name}: {real_val:.2f}")
                self._on_param_change()
                
            slider.valueChanged.connect(update_lbl)
            
            layout.addWidget(lbl)
            layout.addWidget(slider)
            adj_layout.addLayout(layout)
            self.sliders[key] = (slider, scale, default_v, name)  # 添加 name 到元组
            self.slider_labels[key] = lbl  # 存储标签引用

        add_slider('wb_temp', tr('temp'), -100, 100, 0, 1)
        add_slider('wb_tint', tr('tint'), -100, 100, 0, 1)
        add_slider('saturation', tr('saturation'), 0, 3, 1.25, 100)
        add_slider('contrast', tr('contrast'), 0, 3, 1.1, 100)
        add_slider('highlight', tr('highlights'), -100, 100, 0, 1)
        add_slider('shadow', tr('shadows'), -100, 100, 0, 1)
        
        self.reset_btn = PushButton(tr('reset_all'))
        self.reset_btn.clicked.connect(self.reset_adjustments)
        adj_layout.addWidget(self.reset_btn)
        
        self.add_section(tr('adjustments'), self.adj_card)
        
        # Filler
        self.v_layout.addStretch()
        
        self.lut_folder = None

    def set_params(self, params):
        """Update UI controls from params dict"""
        if not params: return
        
        self.blockSignals(True) # Pause signals to avoid triggering processing loops
        
        # Exposure
        if 'exposure_mode' in params:
            self.auto_exp_radio.setChecked(params['exposure_mode'] == 'Auto')
        
        if 'metering_mode' in params:
            # Use reverse map to convert internal key to display text
            display_text = self.metering_mode_reverse_map.get(params['metering_mode'], tr('matrix'))
            self.metering_combo.setCurrentText(display_text)

        self._update_exposure_ui_state()

        if 'exposure' in params:
            exp_val = params['exposure']
            self.exp_slider.setValue(int(exp_val * 10))
            # Update the exposure value label
            self.exp_value_label.setText(f"{tr('exposure_ev')}: {exp_val:+.1f}")
            
        # Color
        if 'log_space' in params:
            self.log_combo.setCurrentText(params['log_space'])
        
        # LUT (Path reconstruction logic needed if we only store path)
        # Assuming lut_path is full path
        if 'lut_path' in params and params['lut_path']:
            lut_name = os.path.basename(params['lut_path'])
            idx = self.lut_combo.findText(lut_name)
            if idx >= 0:
                self.lut_combo.setCurrentIndex(idx)
            else:
                 # Maybe LUT folder changed? For now set to None or handle gracefully
                 pass
        else:
            self.lut_combo.setCurrentIndex(0)
        
        # Lens Correction
        if 'lens_correct' in params:
            self.lens_correct_switch.setChecked(params['lens_correct'])
        
        if 'custom_db_path' in params and params['custom_db_path']:
            self.db_path_edit.setText(params['custom_db_path'])
        else:
            self.db_path_edit.clear()
            
        # Sliders
        for key, (slider, scale, _, name) in self.sliders.items():
            if key in params:
                slider.setValue(int(params[key] * scale))
                # 更新标签文本
                if key in self.slider_labels:
                    real_val = params[key]
                    self.slider_labels[key].setText(f"{name}: {real_val:.2f}")
                
        self.blockSignals(False)
        # Emit one signal to update view if needed?
        # Usually calling code handles the logic update, here we just update UI.

    def add_section(self, title, widget):
        self.v_layout.addWidget(StrongBodyLabel(title))
        self.v_layout.addWidget(widget)

    def _browse_lut_folder(self):
        folder = QFileDialog.getExistingDirectory(self, tr('select_lut_folder'))
        if folder:
            self.lut_folder = folder
            self.refresh_lut_list()

    def refresh_lut_list(self):
        if not self.lut_folder: return
        self.lut_combo.clear()
        self.lut_combo.addItem(tr('none'))
        files = sorted([f for f in os.listdir(self.lut_folder) if f.lower().endswith('.cube')])
        self.lut_combo.addItems(files)
    
    def _browse_lensfun_db(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr('select_lensfun_db'),
            "",
            "XML Files (*.xml);;All Files (*)"
        )
        if file_path:
            self.db_path_edit.setText(file_path)
            # 重新加载lensfun数据库
            try:
                lensfun_wrapper.reload_lensfun_database(custom_db_path=file_path, logger=print)
                InfoBar.success(tr('db_loaded'), tr('using_custom_db', name=os.path.basename(file_path)), parent=self)
            except Exception as e:
                InfoBar.error(tr('db_load_failed'), tr('failed_to_load_db', error=str(e)), parent=self)
                self.db_path_edit.clear()
    
    def _clear_lensfun_db(self):
        self.db_path_edit.clear()
        # 重新加载默认lensfun数据库
        try:
            lensfun_wrapper.reload_lensfun_database(custom_db_path=None, logger=print)
            InfoBar.info(tr('db_cleared'), tr('using_default_db'), parent=self)
        except Exception as e:
            InfoBar.warning(tr('db_cleared'), f"Warning: {str(e)}", parent=self)

    def _update_exposure_ui_state(self):
        is_auto = self.auto_exp_radio.isChecked()
        self.metering_combo.setEnabled(is_auto)
        self.metering_lbl.setEnabled(is_auto)
        self.exp_slider.setEnabled(not is_auto)

    def _on_exposure_mode_changed(self):
        self._update_exposure_ui_state()
        self._on_param_change()

    def _on_param_change(self):
        self.param_changed.emit(self.get_params())

    def reset_adjustments(self):
        for key, (slider, scale, default, name) in self.sliders.items():
            slider.setValue(int(default * scale))
        self._on_param_change()

    def reset_params(self):
        self.auto_exp_radio.setChecked(True)  # Default to Auto Exposure
        self.metering_combo.setCurrentText('Matrix')
        self._update_exposure_ui_state()
        self.exp_slider.setValue(0)
        self.exp_value_label.setText(tr('exposure_ev') + ": 0.0")
        self.log_combo.setCurrentText('None')
        self.lut_combo.setCurrentIndex(0)
        self.lens_correct_switch.setChecked(True)  # Default enabled
        self.db_path_edit.clear()
        
        # 重置所有滑块,不阻塞信号以确保UI正确更新
        for key, (slider, scale, default, name) in self.sliders.items():
            slider.setValue(int(default * scale))
            # 标签会通过valueChanged信号自动更新,无需手动设置
        
        self._on_param_change()

    def get_params(self):
        # Get internal metering mode key from display text
        metering_display_text = self.metering_combo.currentText()
        metering_internal_key = self.metering_mode_map.get(metering_display_text, 'matrix')
        
        p = {
            'exposure_mode': 'Auto' if self.auto_exp_radio.isChecked() else 'Manual',
            'metering_mode': metering_internal_key,
            'exposure': self.exp_slider.value() / 10.0, # Scale factor for EV
            'log_space': self.log_combo.currentText(),
            'lut_path': os.path.join(self.lut_folder, self.lut_combo.currentText()) if self.lut_folder and self.lut_combo.currentIndex() > 0 else None,
            'lens_correct': self.lens_correct_switch.isChecked(),
            'custom_db_path': self.db_path_edit.text() if self.db_path_edit.text() else None
        }
        
        for key, (slider, scale, _, _) in self.sliders.items():
            p[key] = slider.value() / scale
            
        return p


class MainWindow(FluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Raw Alchemy")
        self.setWindowIcon(QIcon(self._get_icon_path()))
        self.resize(1900, 1200)
        
        # State
        self.current_folder = None
        self.current_raw_path = None
        self.marked_files = set()
        self.file_params_cache = {} # path -> params dict
        self.thumbnail_cache = {} # path -> original QPixmap (without green dot)
        
        # 预加载lensfun数据库（在后台线程中）
        self._preload_lensfun_database()
        
        self.create_ui()
        self.create_settings_interface()
        
        # Workers
        self.thumb_worker = None
        self.processor = ImageProcessor()
        self.processor.result_ready.connect(self.on_process_result)
        self.processor.original_ready.connect(self.on_original_ready)
        self.processor.finished.connect(self.on_processor_finished)
        self.processor.error_occurred.connect(self.on_error)
        
        # Processing Debounce
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(100) # 100ms debounce
        self.update_timer.timeout.connect(self.trigger_processing)
    
    def _get_icon_path(self):
        """Get the path to the application icon."""
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        
        icon_path = os.path.join(base_path, "icon.png")
        if not os.path.exists(icon_path):
            # Fallback for development environment if running from src
            icon_path = os.path.join(base_path, "icon.ico")
            
        return icon_path

    def _preload_lensfun_database(self):
        """在后台线程中预加载lensfun数据库，避免阻塞GUI启动"""
        def preload():
            try:
                # 预加载默认数据库
                lensfun_wrapper._get_or_create_database(custom_db_path=None, logger=lambda msg: None)
            except Exception as e:
                print(f"  ⚠️ [Lensfun] Failed to preload database: {e}")
        
        # 在后台线程中执行，不阻塞GUI
        import threading
        preload_thread = threading.Thread(target=preload, daemon=True)
        preload_thread.start()

    def create_ui(self):
        # Central Layout
        self.main_widget = QWidget()
        self.main_widget.setObjectName("mainWidget")
        self.h_layout = QHBoxLayout(self.main_widget)
        self.h_layout.setContentsMargins(0, 0, 0, 0)
        self.h_layout.setSpacing(0)
        
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
        
        self.left_layout.addWidget(SubtitleLabel(tr('library')))
        self.left_layout.addWidget(self.gallery_list)
        self.left_layout.addWidget(self.open_btn)
        
        # 2. Center Panel (Preview)
        self.center_panel = QWidget()
        self.center_layout = QVBoxLayout(self.center_panel)
        self.center_layout.setContentsMargins(10, 10, 10, 10)
        
        # Preview Area
        self.preview_lbl = QLabel(tr('no_image_selected'))
        self.preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_lbl.setStyleSheet("background-color: #202020; border-radius: 8px; color: white;")
        self.preview_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Handle compare (Mouse Press/Release)
        self.preview_lbl.mousePressEvent = self.show_original
        self.preview_lbl.mouseReleaseEvent = self.show_processed
        
        # Toolbar
        self.toolbar = QFrame()
        self.toolbar.setFixedHeight(60)
        self.toolbar_layout = QHBoxLayout(self.toolbar)
        
        self.btn_prev = ToolButton(FIF.LEFT_ARROW)
        self.btn_next = ToolButton(FIF.RIGHT_ARROW)
        self.btn_mark = ToolButton(FIF.TAG)
        self.btn_mark.setCheckable(True)  # Make it a toggle button
        self.btn_delete = ToolButton(FIF.DELETE)
        self.btn_compare = PushButton(tr('hold_to_compare')) # Visual cue
        self.btn_compare.setToolTip(tr('hold_to_compare'))
        
        self.btn_export_curr = PushButton(tr('export_current'))
        self.btn_export_all = PrimaryPushButton(tr('export_all_marked'))
        
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_mark.clicked.connect(self.toggle_mark)
        self.btn_delete.clicked.connect(self.delete_image)
        self.btn_export_curr.clicked.connect(self.export_current)
        self.btn_export_all.clicked.connect(self.export_all)
        
        # Progress Ring for Batch Export
        self.export_progress = ProgressRing()
        self.export_progress.setFixedSize(40, 40)
        self.export_progress.setTextVisible(True)
        self.export_progress.hide()

        # Make compare button toggle between original and processed
        self.btn_compare.pressed.connect(lambda: self.show_original(None))
        self.btn_compare.released.connect(lambda: self.show_processed(None))
        
        self.toolbar_layout.addWidget(self.btn_prev)
        self.toolbar_layout.addWidget(self.btn_next)
        self.toolbar_layout.addStretch()
        self.toolbar_layout.addWidget(self.btn_mark)
        self.toolbar_layout.addWidget(self.btn_delete)
        self.toolbar_layout.addStretch()
        self.toolbar_layout.addWidget(self.btn_compare)
        self.toolbar_layout.addWidget(self.export_progress) # Add progress ring
        self.toolbar_layout.addWidget(self.btn_export_curr)
        self.toolbar_layout.addWidget(self.btn_export_all)
        
        self.center_layout.addWidget(self.preview_lbl)
        self.center_layout.addWidget(self.toolbar)

        # 3. Right Panel (Inspector)
        self.right_panel = InspectorPanel()
        self.right_panel.setFixedWidth(400)
        self.right_panel.param_changed.connect(self.on_param_changed)

        # Assemble
        self.h_layout.addWidget(self.left_panel)
        self.h_layout.addWidget(self.center_panel, 1) # Expand
        self.h_layout.addWidget(self.right_panel)
        
        self.addSubInterface(self.main_widget, FIF.PHOTO, tr('editor'))
        
        # Apply Dark Theme
        setTheme(Theme.DARK)

        # Install event filter to capture keys globally
        QApplication.instance().installEventFilter(self)
    
    def create_settings_interface(self):
        """Create settings interface with language selection"""
        self.settings_widget = QWidget()
        self.settings_widget.setObjectName("settingsWidget")
        settings_layout = QVBoxLayout(self.settings_widget)
        settings_layout.setContentsMargins(40, 40, 40, 40)
        settings_layout.setSpacing(20)
        
        # Title
        title = SubtitleLabel(tr('settings'))
        settings_layout.addWidget(title)
        
        # Language Card
        lang_card = SimpleCardWidget()
        lang_layout = QVBoxLayout(lang_card)
        lang_layout.setSpacing(10)
        
        lang_label = StrongBodyLabel(tr('language'))
        lang_layout.addWidget(lang_label)
        
        # Language ComboBox
        self.lang_combo = ComboBox()
        self.lang_combo.addItems([tr('english'), tr('chinese')])
        
        # Set current language
        current_lang = i18n.get_current_language()
        if current_lang == 'zh':
            self.lang_combo.setCurrentIndex(1)
        else:
            self.lang_combo.setCurrentIndex(0)
        
        self.lang_combo.currentIndexChanged.connect(self.on_language_changed)
        lang_layout.addWidget(self.lang_combo)
        
        settings_layout.addWidget(lang_card)
        settings_layout.addStretch()
        
        # Add settings interface to navigation
        self.addSubInterface(self.settings_widget, FIF.SETTING, tr('settings'))
    
    def on_language_changed(self, index):
        """Handle language change"""
        # Map index to language code
        lang_code = 'en' if index == 0 else 'zh'
        
        # Set language
        i18n.set_language(lang_code)
        
        # Show restart message
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            tr('restart_required'),
            tr('restart_message')
        )

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
            elif event.type() == QEvent.Type.KeyRelease:
                if event.key() == Qt.Key.Key_Space:
                    if not event.isAutoRepeat():
                        self.show_processed(None)
                    return True
        
        return super().eventFilter(obj, event)

    # --- Actions ---

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, tr('select_folder'))
        if folder:
            self.current_folder = folder
            self.gallery_list.clear()
            self.start_thumbnail_scan(folder)

    def start_thumbnail_scan(self, folder):
        if self.thumb_worker:
            self.thumb_worker.stop()
            self.thumb_worker.wait()
        
        self.thumb_worker = ThumbnailWorker(folder)
        self.thumb_worker.thumbnail_ready.connect(self.add_gallery_item)
        self.thumb_worker.start()

    def add_gallery_item(self, path, image):
        name = os.path.basename(path)
        
        # Convert QImage to QPixmap in main thread
        pixmap = QPixmap.fromImage(image)
        
        # Store original pixmap in cache
        self.thumbnail_cache[path] = pixmap

        # Custom Item with icon and text
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setIcon(QIcon(pixmap))
        
        # Set the text with or without green dot based on marked status
        is_marked = path in self.marked_files
        if is_marked:
            item.setText(f"🟢 {name}")
        else:
            item.setText(name)
        
        self.gallery_list.addItem(item)

    def on_gallery_item_clicked(self, item):
        if not item: return
        path = item.data(Qt.ItemDataRole.UserRole)
        if path == self.current_raw_path: return

        # 1. Save current params before switching
        if self.current_raw_path:
            self.file_params_cache[self.current_raw_path] = self.right_panel.get_params()

        # 2. Switch path
        self.current_raw_path = path
        
        # 3. Clear old pixmaps to prevent showing wrong image
        if hasattr(self, 'original_pixmap_scaled'):
            self.original_pixmap_scaled = None
        if hasattr(self, 'last_processed_pixmap'):
            self.last_processed_pixmap = None
        
        # 4. Restore params or Reset
        if path in self.file_params_cache:
            self.right_panel.set_params(self.file_params_cache[path])
        else:
            # Keep previous params (inherit from previous image)
            # We do NOT reset params here, so the new image inherits current UI settings
            # Just trigger a param change to ensure pipeline picks it up for new image
            # This ensures ALL settings (Exposure, WB, LUT, etc.) are carried over
            self.right_panel._on_param_change()
        
        # 5. Update mark button state
        self.update_mark_button_state()
        
        # 6. Load Image
        self.load_image(path)

    def load_image(self, path):
        self.preview_lbl.setText(tr('loading'))
        self.processor.load_image(path)
        
    def on_processor_finished(self):
        # Determine if we need to restart processing based on pending tasks
        
        # Case 1: We wanted to load, but just finished processing (or something else)
        if self.processor.mode == 'load' and self.processor.running_mode != 'load':
            self.processor.start()
            return

        # Case 2: We finished loading, now we need to trigger initial processing
        if self.processor.mode == 'load':
            # Finished loading, start processing
            QTimer.singleShot(0, self.trigger_processing)
            return
            
        # Case 3: We are in process mode
        if self.processor.mode == 'process':
            # If we just finished load, or params are dirty, restart
            if self.processor.running_mode != 'process' or self.processor.params_dirty:
                self.processor.params_dirty = False
                self.processor.start()
                return

    def on_param_changed(self, params):
        # Debounce
        self.update_timer.start()
        
    def trigger_processing(self):
        if not self.current_raw_path: return
        params = self.right_panel.get_params()
        self.processor.update_preview(params)

    def on_process_result(self, img_uint8, img_float, image_path):
        # Verify this result is for the currently selected image
        if image_path != self.current_raw_path:
            # Ignore results from previous image selections
            return
            
        h, w, c = img_uint8.shape
        # Create QImage from numpy array, ensure data is persistent during copy
        # Using bytes copy to be safe against garbage collection of array view
        # We store the bytes object in self to guarantee it lives until QPixmap conversion is done
        self._current_img_data = img_uint8.data.tobytes()
        qimg = QImage(self._current_img_data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale to fit label
        scaled = pixmap.scaled(self.preview_lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.preview_lbl.setPixmap(scaled)
        self.preview_lbl.repaint() # Force update
        
        # Store for compare
        self.last_processed_pixmap = scaled
        
        # Update Thumbnail in Gallery
        # Find the item corresponding to image_path
        current_item = self.gallery_list.currentItem()
        if current_item and current_item.data(Qt.ItemDataRole.UserRole) == image_path:
            thumb_pixmap = pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            current_item.setIcon(QIcon(thumb_pixmap))
        else:
            # Fallback: search for item
            for i in range(self.gallery_list.count()):
                item = self.gallery_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == image_path:
                    thumb_pixmap = pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    item.setIcon(QIcon(thumb_pixmap))
                    break

        # Update Histogram
        self.right_panel.hist_widget.update_data(img_float)

    def on_original_ready(self, img_uint8, image_path):
        # Verify this result is for the currently selected image
        if image_path != self.current_raw_path:
            # Ignore results from previous image selections
            return
            
        h, w, c = img_uint8.shape
        self._current_orig_data = img_uint8.data.tobytes()
        qimg = QImage(self._current_orig_data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.original_pixmap_raw = pixmap
        
        # Show immediate preview and store scaled version for compare
        scaled = pixmap.scaled(self.preview_lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.original_pixmap_scaled = scaled  # Store for compare functionality
        self.preview_lbl.setPixmap(scaled)
        self.preview_lbl.repaint()

    def on_error(self, msg):
        self.preview_lbl.setText(f"{tr('error')}: {msg}")
        InfoBar.error(tr('error'), msg, parent=self)


    # --- Toolbar Actions ---
    
    def prev_image(self):
        count = self.gallery_list.count()
        if count == 0: return
        
        row = self.gallery_list.currentRow()
        # Loop to the end if at beginning
        new_row = (row - 1) % count
        self.gallery_list.setCurrentRow(new_row)

    def next_image(self):
        count = self.gallery_list.count()
        if count == 0: return

        row = self.gallery_list.currentRow()
        # Loop to start if at end
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
        
        # Update button state and gallery item indicator
        self.update_mark_button_state()
        self.update_gallery_item_mark_indicator(self.current_raw_path)
    
    def update_mark_button_state(self):
        """Update the mark button's checked state based on whether current image is marked"""
        if not self.current_raw_path:
            self.btn_mark.setChecked(False)
            return
        
        # Block signals to prevent triggering toggle_mark when programmatically setting state
        self.btn_mark.blockSignals(True)
        self.btn_mark.setChecked(self.current_raw_path in self.marked_files)
        self.btn_mark.blockSignals(False)

    def update_gallery_item_mark_indicator(self, path):
        """Update the green dot indicator on a gallery item"""
        for i in range(self.gallery_list.count()):
            item = self.gallery_list.item(i)
            item_path = item.data(Qt.ItemDataRole.UserRole)
            if item_path == path:
                # Update the item's text to show/hide the green dot
                is_marked = path in self.marked_files
                name = os.path.basename(path)
                
                if is_marked:
                    item.setText(f"🟢 {name}")
                else:
                    item.setText(name)
                break

    def delete_image(self):
        if not self.current_raw_path: return
        
        from PyQt6.QtWidgets import QMessageBox
        
        # Ask for confirmation
        reply = QMessageBox.question(
            self,
            tr('delete_image'),
            tr('confirm_delete', filename=os.path.basename(self.current_raw_path)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Move to recycle bin using send2trash
                import send2trash
                # Normalize path to fix mixed slashes issue on Windows
                normalized_path = os.path.normpath(self.current_raw_path)
                send2trash.send2trash(normalized_path)
                
                # Remove from marked files if it was marked
                if self.current_raw_path in self.marked_files:
                    self.marked_files.remove(self.current_raw_path)
                
                # Remove from params cache
                if self.current_raw_path in self.file_params_cache:
                    del self.file_params_cache[self.current_raw_path]
                
                # Find and remove from gallery
                current_row = self.gallery_list.currentRow()
                for i in range(self.gallery_list.count()):
                    item = self.gallery_list.item(i)
                    if item.data(Qt.ItemDataRole.UserRole) == self.current_raw_path:
                        self.gallery_list.takeItem(i)
                        break
                
                # Move to next image or previous if at end
                if self.gallery_list.count() > 0:
                    if current_row >= self.gallery_list.count():
                        current_row = self.gallery_list.count() - 1
                    self.gallery_list.setCurrentRow(current_row)
                else:
                    # No more images
                    self.current_raw_path = None
                    self.preview_lbl.setText(tr('no_image_selected'))
                
                InfoBar.success(tr('delete_image'), tr('delete_image'), parent=self)
                
            except ImportError:
                # Fallback if send2trash is not installed
                InfoBar.error(tr('error'), tr('send2trash_error'), parent=self)
            except Exception as e:
                InfoBar.error(tr('delete_failed'), str(e), parent=self)

    def show_original(self, event):
        """Show original image when mouse is pressed or button is held"""
        if hasattr(self, 'original_pixmap_scaled') and self.original_pixmap_scaled:
            self.preview_lbl.setPixmap(self.original_pixmap_scaled)
            InfoBar.info(tr('hold_to_compare'), tr('compare_showing_original'), duration=1000, parent=self)
        else:
            # Debug: Image not ready yet
            if self.current_raw_path:
                InfoBar.warning(tr('hold_to_compare'), tr('compare_loading'), duration=1500, parent=self)

    def show_processed(self, event):
        """Show processed image when mouse is released or button is released"""
        if hasattr(self, 'last_processed_pixmap') and self.last_processed_pixmap:
            self.preview_lbl.setPixmap(self.last_processed_pixmap)
        elif hasattr(self, 'original_pixmap_scaled') and self.original_pixmap_scaled:
            # Fallback to original if processed not ready yet
            self.preview_lbl.setPixmap(self.original_pixmap_scaled)

    def export_current(self):
        if not self.current_raw_path: return
        
        # Save Dialog with HEIF support
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr('export_image'),
            os.path.basename(self.current_raw_path),
            "JPEG (*.jpg);;HEIF (*.heif);;TIFF (*.tif)"
        )
        if path:
            self.run_export(
                input_path=self.current_raw_path,
                output_path=path
            )

    def export_all(self):
        if not self.marked_files:
             InfoBar.warning(tr('no_files_marked'), tr('please_mark_files'), parent=self)
             return
        
        # Ask for format
        formats = ["JPEG", "HEIF", "TIFF"]
        format_str, ok = QInputDialog.getItem(self, tr('select_export_format'), "Format:", formats, 0, False)
        
        if not ok:
            return
             
        folder = QFileDialog.getExistingDirectory(self, tr('select_export_folder'))
        if folder:
             # Batch export marked files
             self.batch_export_list = list(self.marked_files)
             self.batch_export_folder = folder
             
             # Map format string to extension
             fmt_map = {"JPEG": "jpg", "HEIF": "heif", "TIFF": "tif"}
             self.batch_export_ext = fmt_map.get(format_str, "jpg")
             
             # Initialize Progress UI
             self.export_progress.setRange(0, len(self.batch_export_list))
             self.export_progress.setValue(0)
             self.export_progress.show()
             self.btn_export_all.setEnabled(False) # Disable button during export
             
             self.batch_export_idx = 0
             self.batch_export_next()

    def batch_export_next(self):
         if self.batch_export_idx >= len(self.batch_export_list):
             InfoBar.success(tr('batch_export'), tr('all_exported'), parent=self)
             self.export_progress.hide()
             self.btn_export_all.setEnabled(True)
             return
         
         # Update Progress
         self.export_progress.setValue(self.batch_export_idx)

         input_path = self.batch_export_list[self.batch_export_idx]
         filename = os.path.basename(input_path)
         
         # Use selected extension
         output_path = os.path.join(self.batch_export_folder, os.path.splitext(filename)[0] + "." + self.batch_export_ext)
         
         # Determine params for this file
         if input_path == self.current_raw_path:
             # Use current UI params
             params = self.right_panel.get_params()
         else:
             # Use cached params
             params = self.file_params_cache.get(input_path)
             if not params:
                 # Fallback to current if not found (should not happen for marked files)
                 params = self.right_panel.get_params()

         self.batch_export_idx += 1
         
         # Trigger single export but chain the next one
         # We'll use a modified run_export that accepts a callback
         self.run_export(input_path, output_path, params=params, callback=self.batch_export_next)


    def run_export(self, input_path, output_path, params=None, callback=None):
        # Gather params
        p = params if params else self.right_panel.get_params()
        
        # Determine format from extension
        ext = os.path.splitext(output_path)[1].lower().replace('.', '')
        if ext not in ['jpg', 'heif', 'tif', 'tiff']: ext = 'jpg'
        
        # Create a thread to run orchestrator (it blocks otherwise)
        # Using a simple QThread wrapper
        
        class ExportThread(QThread):
            finished_sig = pyqtSignal(bool, str)
            
            def run(self):
                try:
                    orchestrator.process_path(
                        input_path=input_path,
                        output_path=output_path,
                        log_space=p['log_space'],
                        lut_path=p['lut_path'],
                        exposure=p['exposure'] if p['exposure_mode'] == 'Manual' else None,
                        lens_correct=p['lens_correct'],
                        custom_db_path=p['custom_db_path'],
                        metering_mode=p.get('metering_mode', 'matrix'),
                        jobs=1,
                        logger_func=lambda msg: None, # Mute log for now
                        output_format=ext,
                        wb_temp=p['wb_temp'],
                        wb_tint=p['wb_tint'],
                        saturation=p['saturation'],
                        contrast=p['contrast'],
                        highlight=p['highlight'],
                        shadow=p['shadow']
                    )
                    self.finished_sig.emit(True, "")
                except Exception as e:
                    self.finished_sig.emit(False, str(e))
        
        self.export_thread = ExportThread()
        
        def on_finish(success, msg):
            if success:
                if not callback:
                    InfoBar.success(tr('export_success'), tr('saved_to', path=os.path.basename(output_path)), parent=self)
                if callback: callback()
            else:
                InfoBar.error(tr('export_failed'), msg, parent=self)
        
        self.export_thread.finished_sig.connect(on_finish)
        self.export_thread.start()


def launch_gui():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    launch_gui()
