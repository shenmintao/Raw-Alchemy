import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy, QFileDialog
)
from PySide6.QtCore import Qt, Signal, QTimer, QEvent
from qfluentwidgets import (
    ScrollArea, SimpleCardWidget, StrongBodyLabel, BodyLabel, 
    SwitchButton, ComboBox, Slider, LineEdit, ToolButton, 
    PushButton, InfoBar, FluentIcon as FIF
)


class NoWheelSlider(Slider):
    """Slider that ignores mouse wheel events to prevent accidental changes"""
    def wheelEvent(self, event):
        event.ignore()  # Let the parent ScrollArea handle the wheel event
from raw_alchemy import config, lensfun_wrapper
from raw_alchemy.i18n import tr
from raw_alchemy.ui.widgets.histogram import HistogramWidget
from raw_alchemy.ui.widgets.waveform import WaveformWidget

class InspectorPanel(ScrollArea):
    """Right side control panel"""
    param_changed = Signal(dict)
    enter_crop_mode = Signal()
    enter_perspective_mode = Signal()
    
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
        
        # 保存的基准参数
        self.saved_baseline_params = None
        
        # 保存各模式的EV值
        self.manual_ev_value = 0.0  # 手动模式的EV
        self.auto_ev_value = 0.0    # 自动模式计算的EV（只读）
        
        # --- Histogram / Waveform with Switch ---
        self.hist_widget = HistogramWidget()
        self.waveform_widget = WaveformWidget()
        self.waveform_widget.hide()  # Initially hidden
        
        # Create a container for the display mode switch
        display_mode_card = SimpleCardWidget()
        display_mode_layout = QVBoxLayout(display_mode_card)
        display_mode_layout.setSpacing(5)
        
        # Switch button for histogram/waveform
        self.display_mode_switch = SwitchButton()
        self.display_mode_switch.setChecked(False)  # False = Histogram, True = Waveform
        self.display_mode_switch.checkedChanged.connect(self._on_display_mode_changed)
        self._update_display_mode_switch_text()
        
        display_mode_layout.addWidget(self.display_mode_switch)
        self.scope_source_label = BodyLabel("Scopes: full")
        display_mode_layout.addWidget(self.scope_source_label)
        
        self.add_section(tr('histogram_waveform'), display_mode_card)
        self.v_layout.addWidget(self.hist_widget)
        self.v_layout.addWidget(self.waveform_widget)

        # --- Geometry ---
        self.geo_card = SimpleCardWidget()
        geo_layout = QVBoxLayout(self.geo_card)
        
        # Crop & Rotate Mode Button
        self.btn_crop_mode = PushButton(tr('enter_crop_rotate'))
        self.btn_crop_mode.setIcon(FIF.CUT)
        self.btn_crop_mode.clicked.connect(self._on_enter_crop_mode)
        
        # Perspective Correction Button
        self.btn_perspective_mode = PushButton(tr('perspective_correction'))
        self.btn_perspective_mode.setIcon(FIF.ZOOM)  # Using ZOOM as proxy for perspective
        self.btn_perspective_mode.clicked.connect(self._on_enter_perspective_mode)
        
        geo_layout.addWidget(self.btn_crop_mode)
        geo_layout.addWidget(self.btn_perspective_mode)
        
        self.add_section(tr('geometry'), self.geo_card)
        
        # Initialize internal rotation state
        self.base_rotation = 0 # 0, 90, 180, 270
        self.fine_rotation = 0 # -45 to 45

        # --- Exposure ---
        self.exp_card = SimpleCardWidget()
        exp_layout = QVBoxLayout(self.exp_card)
        
        self.auto_exp_radio = SwitchButton()
        self.auto_exp_radio.setChecked(True)  # Default to Auto Exposure
        self.auto_exp_radio.checkedChanged.connect(self._on_exposure_mode_changed)
        self._update_exposure_switch_text()
        
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
        
        self.exp_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.exp_slider.setRange(-100, 100) # -10.0 to 10.0
        self.exp_slider.setValue(0)
        self.exp_slider.update()
        
        # Add exposure value label
        self.exp_value_label = BodyLabel(tr('exposure_ev') + ": 0.0")
        
        def update_exp_label(val):
            """Update label and trigger debounced parameter change"""
            real_val = val / 10.0
            self.exp_value_label.setText(f"{tr('exposure_ev')}: {real_val:+.1f}")
            # Trigger parameter change - will be debounced by 100ms timer in on_param_changed
            self._on_param_change()
        
        # 保存回调函数引用，以便后续临时断开连接
        self.exp_slider_callback = update_exp_label
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
        # Store log space mapping: display text -> internal key
        self.log_space_map = {tr('none'): 'None'}
        # Map actual log space names to themselves
        for log_name in config.LOG_TO_WORKING_SPACE.keys():
            self.log_space_map[log_name] = log_name
        # Reverse mapping: internal key -> display text
        self.log_space_reverse_map = {v: k for k, v in self.log_space_map.items()}
        
        log_items = [tr('none')] + list(config.LOG_TO_WORKING_SPACE.keys())
        self.log_combo.addItems(log_items)
        self.log_combo.setCurrentText(tr('none'))
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
        self.slider_revert_btns = {}  # 存储撤回按钮引用
        
        def add_slider(key, name, min_v, max_v, default_v, scale=1.0):
            layout = QVBoxLayout()
            
            # 标签和撤回按钮布局
            header_layout = QHBoxLayout()
            lbl = BodyLabel(f"{name}: {default_v}")
            
            # 撤回按钮
            revert_btn = ToolButton(FIF.HISTORY)
            revert_btn.setFixedSize(24, 24)
            revert_btn.setToolTip(tr('revert_to_baseline_or_default'))
            revert_btn.clicked.connect(lambda checked, k=key: self._revert_slider(k))
            
            header_layout.addWidget(lbl)
            header_layout.addStretch()
            header_layout.addWidget(revert_btn)
            
            slider = NoWheelSlider(Qt.Orientation.Horizontal)
            # slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            slider.setRange(int(min_v*scale), int(max_v*scale))
            slider.setValue(int(default_v*scale))
            
            def update_lbl(val):
                """Update label and trigger debounced parameter change"""
                real_val = val / scale
                lbl.setText(f"{name}: {real_val:.2f}")
                # Trigger parameter change - will be debounced by 100ms timer in on_param_changed
                self._on_param_change()
            
            slider.valueChanged.connect(update_lbl)
                
            layout.addLayout(header_layout)
            layout.addWidget(slider)
            adj_layout.addLayout(layout)
            self.sliders[key] = (slider, scale, default_v, name)  # 添加 name 到元组
            self.slider_labels[key] = lbl  # 存储标签引用
            self.slider_revert_btns[key] = revert_btn  # 存储撤回按钮引用

        add_slider('wb_temp', tr('temp'), -100, 100, 0, 1)
        add_slider('wb_tint', tr('tint'), -100, 100, 0, 1)
        add_slider('saturation', tr('saturation'), 0, 3, 1.25, 100)
        add_slider('contrast', tr('contrast'), 0, 3, 1.1, 100)
        add_slider('highlight', tr('highlights'), -100, 100, 0, 1)
        add_slider('shadow', tr('shadows'), -100, 100, 0, 1)
        
        # 按钮布局：保存参数，重置到基准，重置所有
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(10)
        
        self.save_baseline_btn = PushButton(tr('save_baseline'))
        self.save_baseline_btn.clicked.connect(self.save_baseline_params)
        btn_layout.addWidget(self.save_baseline_btn)
        
        row2_layout = QHBoxLayout()
        self.reset_baseline_btn = PushButton(tr('reset_baseline'))
        self.reset_baseline_btn.clicked.connect(self.reset_to_baseline)
        self.reset_baseline_btn.setEnabled(False)  # Disabled until baseline is saved
        
        self.reset_defaults_btn = PushButton(tr('reset_defaults'))
        self.reset_defaults_btn.clicked.connect(self.reset_to_defaults)
        
        row2_layout.addWidget(self.reset_baseline_btn)
        row2_layout.addWidget(self.reset_defaults_btn)
        
        btn_layout.addLayout(row2_layout)
        adj_layout.addLayout(btn_layout)
        
        self.add_section(tr('adjustments'), self.adj_card)
        
        # --- AI Denoise (disabled — new RAW-to-RAW model in progress) ---
        self.denoise_card = SimpleCardWidget()
        denoise_layout = QVBoxLayout(self.denoise_card)

        self.denoise_switch = SwitchButton()
        self.denoise_switch.setChecked(False)
        self.denoise_switch.setEnabled(self._denoise_available())
        self.denoise_switch.checkedChanged.connect(self._on_denoise_toggled)
        self._update_denoise_switch_text()
        denoise_layout.addWidget(self.denoise_switch)

        self.denoise_info_label = BodyLabel(tr('denoise_info'))
        self.denoise_info_label.setWordWrap(True)
        self.denoise_info_label.setStyleSheet("color: gray; font-size: 11px;")
        denoise_layout.addWidget(self.denoise_info_label)

        # Sharpen strength slider (Richardson-Lucy deconvolution)
        sharpen_header_layout = QHBoxLayout()
        self.sharpen_label = BodyLabel(f"{tr('sharpen_strength')}: 0.00")

        self.sharpen_revert_btn = ToolButton(FIF.HISTORY)
        self.sharpen_revert_btn.setFixedSize(24, 24)
        self.sharpen_revert_btn.setToolTip(tr('revert_to_baseline_or_default'))
        self.sharpen_revert_btn.clicked.connect(self._revert_sharpen)

        sharpen_header_layout.addWidget(self.sharpen_label)
        sharpen_header_layout.addStretch()
        sharpen_header_layout.addWidget(self.sharpen_revert_btn)

        self.sharpen_slider = NoWheelSlider(Qt.Orientation.Horizontal)
        self.sharpen_slider.setRange(0, 100)
        self.sharpen_slider.setValue(0)

        def update_sharpen_label(val):
            real_val = val / 100.0
            self.sharpen_label.setText(f"{tr('sharpen_strength')}: {real_val:.2f}")
            self._on_param_change()

        self.sharpen_slider.valueChanged.connect(update_sharpen_label)

        denoise_layout.addLayout(sharpen_header_layout)
        denoise_layout.addWidget(self.sharpen_slider)

        self.add_section(tr('ai_denoise'), self.denoise_card)
        
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
            self.exp_slider.update()
            # Update the exposure value label
            self.exp_value_label.setText(f"{tr('exposure_ev')}: {exp_val:+.1f}")
            
        # Color
        if 'log_space' in params:
            # Use reverse map to convert internal key to display text
            display_text = self.log_space_reverse_map.get(params['log_space'], tr('none'))
            self.log_combo.setCurrentText(display_text)
        
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
            
        if 'custom_db_path' in params:
            self.db_path_edit.setText(params['custom_db_path'])
        else:
            self.db_path_edit.clear()
            
        # Geometry
        if 'rotation' in params:
            # We need to decompose total rotation back into base (90 steps) + fine
            # This is tricky because 95 could be 90 + 5 or 0 + 95 (out of slider range)
            # Let's simple logic: find nearest 90 degree step
            rot = params['rotation']
            base = round(rot / 90.0) * 90
            fine = rot - base
            
            self.base_rotation = int(base % 360)
            self.fine_rotation = int(fine)
            
            # Update UI
            # self.rot_slider.blockSignals(True)
            # self.rot_slider.setValue(self.fine_rotation)
            # self.rot_slider.blockSignals(False)
            # self.rot_value_label.setText(f"{self.fine_rotation}°")
        
        if 'flip_horizontal' in params:
             self.flip_h = params['flip_horizontal'] # Save internal state
            
        if 'flip_vertical' in params:
             self.flip_v = params['flip_vertical'] # Save internal state
             
        if 'crop' in params:
             self.crop_rect = params.get('crop', (0.0, 0.0, 1.0, 1.0))
             
        if 'perspective_corners' in params:
             self.perspective_corners = params.get('perspective_corners')
            
        # Sliders
        for key, (slider, scale, _, name) in self.sliders.items():
            if key in params:
                slider.setValue(int(params[key] * scale))
                # 更新标签文本
                if key in self.slider_labels:
                    real_val = params[key]
                    self.slider_labels[key].setText(f"{name}: {real_val:.2f}")
        
        # Denoise
        if 'denoise_enabled' in params:
            self.denoise_switch.setChecked(params['denoise_enabled'])
            self._update_denoise_switch_text()
        
        # Sharpen
        if 'sharpen_strength' in params:
            sharpen_val = params['sharpen_strength']
            self.sharpen_slider.setValue(int(sharpen_val * 100))
            self.sharpen_label.setText(f"{tr('sharpen_strength')}: {sharpen_val:.2f}")
                
        self.blockSignals(False)

    def add_section(self, title, widget):
        self.v_layout.addWidget(StrongBodyLabel(title))
        self.v_layout.addWidget(widget)

    def _browse_lut_folder(self):
        # Get the main window to access last_lut_folder_path
        main_window = self.window()
        
        # Use last LUT folder path, or fall back to last gallery folder, or home
        start_dir = ""
        if hasattr(main_window, 'last_lut_folder_path') and main_window.last_lut_folder_path and os.path.exists(main_window.last_lut_folder_path):
            start_dir = main_window.last_lut_folder_path
        elif hasattr(main_window, 'last_folder_path') and main_window.last_folder_path and os.path.exists(main_window.last_folder_path):
            start_dir = main_window.last_folder_path
        
        folder = QFileDialog.getExistingDirectory(self, tr('select_lut_folder'), start_dir)
        if folder:
            self.lut_folder = folder
            # Remember this path in main window
            if hasattr(main_window, 'last_lut_folder_path'):
                main_window.last_lut_folder_path = folder
            self.refresh_lut_list()

    def refresh_lut_list(self):
        if not self.lut_folder: return
        self.lut_combo.clear()
        self.lut_combo.addItem(tr('none'))
        files = sorted([f for f in os.listdir(self.lut_folder) if f.lower().endswith('.cube')])
        self.lut_combo.addItems(files)
    
    def _browse_lensfun_db(self):
        # Get the main window to access last_lensfun_db_path
        main_window = self.window()
        
        # Use last Lensfun DB path's directory, or fall back to last gallery folder, or home
        start_dir = ""
        if hasattr(main_window, 'last_lensfun_db_path') and main_window.last_lensfun_db_path and os.path.exists(main_window.last_lensfun_db_path):
            start_dir = os.path.dirname(main_window.last_lensfun_db_path)
        elif hasattr(main_window, 'last_folder_path') and main_window.last_folder_path and os.path.exists(main_window.last_folder_path):
            start_dir = main_window.last_folder_path
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr('select_lensfun_db'),
            start_dir,
            "XML Files (*.xml);;All Files (*)"
        )
        if file_path:
            self.db_path_edit.setText(file_path)
            # Remember this path in main window
            if hasattr(main_window, 'last_lensfun_db_path'):
                main_window.last_lensfun_db_path = file_path
            # 重新加载lensfun数据库
            try:
                lensfun_wrapper.reload_lensfun_database(custom_db_path=file_path)
                InfoBar.success(tr('db_loaded'), tr('using_custom_db', name=os.path.basename(file_path)), parent=self)
            except Exception as e:
                InfoBar.error(tr('db_load_failed'), tr('failed_to_load_db', error=str(e)), parent=self)
                self.db_path_edit.clear()
    
    def _clear_lensfun_db(self):
        self.db_path_edit.clear()
        # 重新加载默认lensfun数据库
        try:
            lensfun_wrapper.reload_lensfun_database(custom_db_path=None)
            InfoBar.info(tr('db_cleared'), tr('using_default_db'), parent=self)
        except Exception as e:
            InfoBar.warning(tr('db_cleared'), f"Warning: {str(e)}", parent=self)

    def _update_display_mode_switch_text(self):
        """Update the display mode switch button text based on its state"""
        if self.display_mode_switch.isChecked():
            self.display_mode_switch.setText(tr('waveform'))
        else:
            self.display_mode_switch.setText(tr('histogram'))
    
    def _on_display_mode_changed(self):
        """Handle display mode switch between histogram and waveform"""
        is_waveform = self.display_mode_switch.isChecked()
        
        if is_waveform:
            # Switch to waveform
            self.hist_widget.hide()
            self.waveform_widget.show()
        else:
            # Switch to histogram
            self.waveform_widget.hide()
            self.hist_widget.show()
        
        self._update_display_mode_switch_text()
        main_window = self.window()
        current = getattr(main_window, 'current', None)
        if current is not None and current.uint8_data is not None:
            self.update_scope_data(current.uint8_data)

    def update_scope_data(self, img_uint8):
        """Update only the visible scope."""
        if self.display_mode_switch.isChecked():
            self.waveform_widget.update_data(img_uint8)
        else:
            self.hist_widget.update_data(img_uint8)

    def update_scope_source(self, source_name):
        source_name = "proxy" if source_name == "proxy" else "full"
        self.scope_source_label.setText(f"Scopes: {source_name}")

    def shutdown_scope_workers(self):
        self.hist_widget.shutdown_worker()
        self.waveform_widget.shutdown_worker()

    def _update_exposure_switch_text(self):
        """Update the switch button text based on its state"""
        if self.auto_exp_radio.isChecked():
            self.auto_exp_radio.setText(tr('auto_exposure'))
        else:
            self.auto_exp_radio.setText(tr('manual_exposure'))
    
    def _update_exposure_ui_state(self):
        is_auto = self.auto_exp_radio.isChecked()
        self.metering_combo.setEnabled(is_auto)
        self.metering_lbl.setEnabled(is_auto)
        self.exp_slider.setEnabled(not is_auto)
        self._update_exposure_switch_text()

    def _on_exposure_mode_changed(self):
        is_auto = self.auto_exp_radio.isChecked()
        
        # Update UI state FIRST (enable/disable controls)
        self._update_exposure_ui_state()
        
        if is_auto:
            # Switching to auto mode: save current manual value
            self.manual_ev_value = self.exp_slider.value() / 10.0
            # Display auto mode EV
            self.exp_slider.setValue(int(self.auto_ev_value * 10))
            self.exp_slider.update()  # Force immediate visual refresh
            # Update label manually
            self.exp_value_label.setText(f"{tr('exposure_ev')}: {self.auto_ev_value:+.1f}")
        else:
            # Switching to manual mode: save current auto value, restore manual value
            self.auto_ev_value = self.exp_slider.value() / 10.0
            self.exp_slider.setValue(int(self.manual_ev_value * 10))
            self.exp_slider.update()  # Force immediate visual refresh
            # Update label manually
            self.exp_value_label.setText(f"{tr('exposure_ev')}: {self.manual_ev_value:+.1f}")
        
        self._on_param_change()

    def _on_enter_crop_mode(self):
        self.enter_crop_mode.emit()
    
    def _on_enter_perspective_mode(self):
        self.enter_perspective_mode.emit()

    def update_crop_params(self, rotation, flip_h, flip_v, crop_rect):
        params = self.get_params()
        params['rotation'] = rotation
        params['flip_horizontal'] = flip_h
        params['flip_vertical'] = flip_v
        params['crop'] = crop_rect
        
        # Update internal state so get_params returns correct values next time
        self.set_params(params) 
        
        from loguru import logger
        logger.info(f"[Inspector] update_crop_params called with rotation={rotation}, flip_h={flip_h}, flip_v={flip_v}")
        logger.info(f"[Inspector] New params rotation check: {self.get_params().get('rotation')}")
        
        self._on_param_change()
    
    # Removed old rotation handlers
    # ...

    def _on_rot_slider_changed(self, value):
        self.fine_rotation = value
        self.rot_value_label.setText(f"{value}°")
        self._on_param_change() # This needs debounce for slider!
        # Slider valueChanged connects to this, but we might want debouncing for heavy rotate ops
        # InspectorPanel uses _on_param_change which emits param_changed
        # MainWindow connects param_changed to update_timer (debounce 100ms)
        # So it should be fine.

    def _on_param_change(self):
        self.param_changed.emit(self.get_params())
    
    def _revert_slider(self, key):
        """撤回单个滑条到基准值（如果有）或默认值"""
        if key not in self.sliders:
            return
        
        slider, scale, default_v, name = self.sliders[key]
        
        # 检查基准参数中是否有该值
        if self.saved_baseline_params and key in self.saved_baseline_params:
            target_value = self.saved_baseline_params[key]
        else:
            target_value = default_v
        
        # 设置滑条值
        slider.setValue(int(target_value * scale))
        
        # 更新标签
        if key in self.slider_labels:
            self.slider_labels[key].setText(f"{name}: {target_value:.2f}")
        
        # 触发参数变更
        self._on_param_change()
        
    def get_params(self):
        """Get parameters from UI to send to processor"""
        is_auto = self.auto_exp_radio.isChecked()
        
        params = {
            'exposure_mode': 'Auto' if is_auto else 'Manual',
            'metering_mode': self.metering_mode_map.get(self.metering_combo.currentText(), 'matrix'),
            # Always pass the slider value as 'exposure' - in auto mode this is visualized but respected if passed?
            # Actually, in auto mode, the processor calculates gain. The slider shows it.
            # But if we send 'exposure', it might override?
            # Looking at ImageProcessor:
            # if params.get('exposure_mode') == 'Manual': use 'exposure'
            # else: use auto exposure
            'exposure': self.exp_slider.value() / 10.0,
            
            'log_space': self.log_space_map.get(self.log_combo.currentText(), 'None'),
            'lut_path': os.path.join(self.lut_folder, self.lut_combo.currentText()) if self.lut_folder and self.lut_combo.currentText() != tr('none') else None,
            
            'lens_correct': self.lens_correct_switch.isChecked(),
            'custom_db_path': self.db_path_edit.text() if self.db_path_edit.text() else None,
            
            # Geometry
            # Combine base and fine rotation
            'rotation': (self.base_rotation + self.fine_rotation) % 360, 
            
            'flip_horizontal': getattr(self, 'flip_h', False),
            'flip_vertical': getattr(self, 'flip_v', False),
            'crop': getattr(self, 'crop_rect', (0.0, 0.0, 1.0, 1.0)),
            'perspective_corners': getattr(self, 'perspective_corners', None)
        }
        
        # Add sliders
        for key, (slider, scale, _, _) in self.sliders.items():
            params[key] = slider.value() / scale
        
        # Add denoise toggle
        params['denoise_enabled'] = self.denoise_switch.isChecked()
        
        # Add sharpen strength
        params['sharpen_strength'] = self.sharpen_slider.value() / 100.0
            
        return params

    def save_baseline_params(self):
        """保存当前参数作为基准点"""
        self.saved_baseline_params = self.get_params().copy()
        self.reset_baseline_btn.setEnabled(True)
        InfoBar.success(tr('baseline_saved'), tr('baseline_saved_message'), parent=self)

    def reset_to_baseline(self):
        """重置到保存的基准点"""
        if self.saved_baseline_params:
            self.set_params(self.saved_baseline_params)
            self._on_param_change()
            # InfoBar.success(tr('reset_to_default'), tr('compare_showing_baseline'), parent=self)

    def reset_to_defaults(self):
        self.auto_exp_radio.setChecked(True)  # Default to Auto Exposure
        self.metering_combo.setCurrentText(tr('matrix'))
        self._update_exposure_ui_state()
        self.exp_slider.setValue(0)
        self.exp_slider.update()
        self.exp_value_label.setText(tr('exposure_ev') + ": 0.0")
        self.log_combo.setCurrentText(tr('none'))
        self.lut_combo.setCurrentIndex(0)
        self.lens_correct_switch.setChecked(True)  # Default enabled
        self.db_path_edit.clear()
        
        # Geometry defaults
        self.base_rotation = 0
        self.fine_rotation = 0
        self.flip_h = False
        self.flip_v = False
        self.crop_rect = (0.0, 0.0, 1.0, 1.0)
        self.perspective_corners = None
        
        # Reset sliders
        for key, (slider, scale, default, name) in self.sliders.items():
            slider.setValue(int(default * scale))
            # 更新标签
            if key in self.slider_labels:
                self.slider_labels[key].setText(f"{name}: {default:.2f}")
        
        # Note: Denoise is NOT reset here (similar to lens correction)
        # It's a heavy operation and should be preserved across resets
        
        # Clear saved baseline
        self.saved_baseline_params = None
        self.reset_baseline_btn.setEnabled(False)
        
        # Trigger parameter change
        self._on_param_change()
        
        InfoBar.success(tr('reset_to_default'), tr('reset_to_default_message'), parent=self)

    @staticmethod
    def _denoise_available():
        try:
            from raw_alchemy.onnx.denoiser import is_available

            return is_available()
        except Exception:
            return False

    def _revert_denoise(self):
        """Revert denoise toggle to baseline or default"""
        if self.saved_baseline_params and 'denoise_enabled' in self.saved_baseline_params:
            self.denoise_switch.setChecked(self.saved_baseline_params['denoise_enabled'])
        else:
            self.denoise_switch.setChecked(False)
        self._update_denoise_switch_text()
        self._on_param_change()

    def _on_denoise_toggled(self):
        """Handle denoise switch toggle"""
        self._update_denoise_switch_text()
        self._on_param_change()

    def _update_denoise_switch_text(self):
        """Update the denoise switch button text based on its state"""
        if self.denoise_switch.isChecked():
            self.denoise_switch.setText(tr('denoise_on'))
        else:
            self.denoise_switch.setText(tr('denoise_off'))

    def _revert_sharpen(self):
        """Revert sharpen slider to baseline or default"""
        if self.saved_baseline_params and 'sharpen_strength' in self.saved_baseline_params:
            # Revert to baseline
            val = self.saved_baseline_params['sharpen_strength']
            self.sharpen_slider.setValue(int(val * 100))
            self.sharpen_label.setText(f"{tr('sharpen_strength')}: {val:.2f}")
        else:
            # Revert to default (0)
            self.sharpen_slider.setValue(0)
            self.sharpen_label.setText(f"{tr('sharpen_strength')}: 0.00")
        self._on_param_change()
