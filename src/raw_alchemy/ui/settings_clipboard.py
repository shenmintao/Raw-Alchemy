"""Copy and paste develop settings between images.

Settings are copied in groups. Geometry is left out by default because a crop
or perspective fit belongs to one frame. A paste goes to the current image or
to every marked image, and the most recent paste can be undone from its
notification (the app has no general undo history).
"""
import copy
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from qfluentwidgets import (
    Action,
    CheckBox,
    FluentIcon as FIF,
    InfoBar,
    InfoBarIcon,
    InfoBarPosition,
    MessageBox,
    MessageBoxBase,
    PushButton,
    RoundMenu,
    SplitToolButton,
    SubtitleLabel,
    ToolButton,
)

from raw_alchemy.i18n import tr


# (group, parameter keys, copied by default)
SETTINGS_GROUPS = (
    ("exposure", ("exposure_mode", "metering_mode", "exposure"), True),
    ("white_balance", ("wb_temp", "wb_tint"), True),
    ("tone_color", ("saturation", "contrast", "highlight", "shadow"), True),
    ("log_lut", ("log_space", "lut_path"), True),
    ("lens", ("lens_correct", "custom_db_path"), True),
    ("denoise", ("denoise_enabled", "denoise_strength"), True),
    ("sharpen", ("sharpen_strength",), True),
    ("geometry",
     ("rotation", "flip_horizontal", "flip_vertical", "crop", "perspective_corners"),
     False),
)


def default_group_selection():
    return {group for group, _keys, default_on in SETTINGS_GROUPS if default_on}


def pick_settings(params, groups):
    """The parameters of ``params`` that belong to the selected groups."""
    return {
        key: copy.deepcopy(params[key])
        for group, keys, _default_on in SETTINGS_GROUPS if group in groups
        for key in keys if key in params
    }


def merge_settings(base, copied):
    merged = dict(base)
    merged.update(copy.deepcopy(copied))
    return merged


class CopySettingsDialog(MessageBoxBase):
    """Choose which setting groups to copy."""

    def __init__(self, selected, parent=None):
        super().__init__(parent)
        self.viewLayout.addWidget(SubtitleLabel(tr('copy_settings_title'), self))
        self.checks = {}
        for group, _keys, _default_on in SETTINGS_GROUPS:
            box = CheckBox(tr(f'settings_group_{group}'), self)
            box.setChecked(group in selected)
            self.viewLayout.addWidget(box)
            self.checks[group] = box
        self.yesButton.setText(tr('copy_settings'))
        self.cancelButton.setText(tr('cancel'))
        self.widget.setMinimumWidth(340)

    def selected_groups(self):
        return {group for group, box in self.checks.items() if box.isChecked()}


class SettingsClipboardMixin:
    """Copy / paste of settings for MainWindow."""

    def _init_settings_clipboard(self):
        self._settings_clipboard = None  # {"params", "groups", "source"}
        self._copy_groups = default_group_selection()
        self._paste_undo = None  # path -> params before the last paste (None = had none)
        self._settings_shortcuts = []
        for sequence, slot in (
            ("Ctrl+Shift+C", self.copy_settings),
            ("Ctrl+Shift+V", self.paste_settings_to_current),
        ):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(slot)
            self._settings_shortcuts.append(shortcut)

    def _create_settings_clipboard_buttons(self):
        self.btn_copy_settings = ToolButton(FIF.COPY)
        self.btn_copy_settings.setToolTip(f"{tr('copy_settings')} (Ctrl+Shift+C)")
        self.btn_copy_settings.clicked.connect(self.copy_settings)

        self.btn_paste_settings = SplitToolButton(FIF.PASTE)
        self.btn_paste_settings.setToolTip(f"{tr('paste_settings')} (Ctrl+Shift+V)")
        self.btn_paste_settings.clicked.connect(self.paste_settings_to_current)
        menu = RoundMenu(parent=self)
        menu.addAction(Action(FIF.PASTE, tr('paste_settings'),
                              triggered=self.paste_settings_to_current))
        menu.addAction(Action(FIF.TAG, tr('paste_settings_to_marked'),
                              triggered=self.paste_settings_to_marked))
        self.btn_paste_settings.setFlyout(menu)
        self.btn_paste_settings.setEnabled(False)  # nothing copied yet
        return [self.btn_copy_settings, self.btn_paste_settings]

    def _settings_editable(self):
        return bool(
            self.current_raw_path
            and self.main_widget.isVisible()
            and getattr(self, 'processor_connection_mode', 'normal') == 'normal'
            and self.center_stack.currentWidget() is self.page_preview
        )

    def copy_settings(self):
        if not self._settings_editable():
            return
        dialog = CopySettingsDialog(self._copy_groups, self)
        if not dialog.exec():
            return
        groups = dialog.selected_groups()
        if not groups:
            return
        self._copy_groups = groups
        source = os.path.basename(self.current_raw_path)
        self._settings_clipboard = {
            "params": pick_settings(self.right_panel.get_params(), groups),
            "groups": groups,
            "source": source,
        }
        self.btn_paste_settings.setEnabled(True)
        InfoBar.success(
            tr('settings_copied'),
            tr('settings_copied_detail', count=len(groups), filename=source),
            parent=self,
        )

    def _apply_params_to_current(self, params):
        self.right_panel.set_params(params)
        # set_params blocks the panel's signals; run the normal edit path so
        # the cache, sidecar and preview all follow.
        self.on_param_changed(self.right_panel.get_params())

    def paste_settings_to_current(self):
        clipboard = self._settings_clipboard
        if clipboard is None:
            InfoBar.info(tr('paste_settings'), tr('no_settings_copied'), parent=self)
            return
        if not self._settings_editable():
            return
        before = self.right_panel.get_params()
        self._apply_params_to_current(merge_settings(before, clipboard["params"]))
        self._paste_undo = {self.current_raw_path: before}
        self._notify_paste(1)

    def paste_settings_to_marked(self):
        clipboard = self._settings_clipboard
        if clipboard is None:
            InfoBar.info(tr('paste_settings'), tr('no_settings_copied'), parent=self)
            return
        if not self._settings_editable():
            return
        targets = sorted(p for p in self.marked_files if os.path.exists(p))
        if not targets:
            InfoBar.warning(tr('no_files_marked'), tr('please_mark_files'), parent=self)
            return
        box = MessageBox(
            tr('paste_settings_to_marked'),
            tr('paste_to_marked_confirm',
               groups=len(clipboard["groups"]), count=len(targets)),
            self,
        )
        if not box.exec():
            return

        undo = {}
        for path in targets:
            if path == self.current_raw_path:
                before = self.right_panel.get_params()
                self._apply_params_to_current(merge_settings(before, clipboard["params"]))
            else:
                before = copy.deepcopy(self.file_params_cache.get(path))
                base = before if before is not None else self.right_panel.default_params()
                merged = merge_settings(base, clipboard["params"])
                self.file_params_cache[path] = merged
                self._write_sidecar_for_path(path, merged)
            undo[path] = before
        self._paste_undo = undo
        self._notify_paste(len(targets))

    def undo_last_paste(self):
        undo, self._paste_undo = self._paste_undo, None
        if not undo:
            return
        for path, before in undo.items():
            if path == self.current_raw_path:
                self._apply_params_to_current(
                    before if before is not None else self.right_panel.default_params()
                )
            elif before is None:
                self.file_params_cache.pop(path, None)
                self._write_sidecar_for_path(path, {})
            else:
                self.file_params_cache[path] = before
                self._write_sidecar_for_path(path, before)
        InfoBar.info(tr('paste_undone'), '', parent=self)

    def _notify_paste(self, count):
        bar = InfoBar(
            InfoBarIcon.SUCCESS,
            tr('settings_pasted'),
            tr('settings_pasted_detail', count=count),
            orient=Qt.Orientation.Horizontal,
            isClosable=True,
            duration=8000,
            position=InfoBarPosition.TOP_RIGHT,
            parent=self,
        )
        undo_button = PushButton(tr('undo'))
        undo_button.clicked.connect(self.undo_last_paste)
        undo_button.clicked.connect(bar.close)
        bar.addWidget(undo_button)
        bar.show()
