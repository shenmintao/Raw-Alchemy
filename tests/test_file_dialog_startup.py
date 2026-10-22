"""Exercise the GUI entry point with real Qt dialogs in a fresh process."""
import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize('system', ['Windows', 'Linux'])
def test_startup_uses_qt_file_dialogs_on_windows_only(tmp_path, system):
    script = textwrap.dedent('''
        import sys
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtWidgets import QApplication, QFileDialog, QWidget
        from raw_alchemy import main as startup

        system = sys.argv[1]
        startup.platform.system = lambda: system
        expected = system == 'Windows'
        flag = Qt.ApplicationAttribute.AA_DontUseNativeDialogs
        assert not QApplication.testAttribute(flag)

        class ProbeWindow(QWidget):
            def show(self):
                app = QApplication.instance()
                assert app.testAttribute(flag) == expected
                if expected:
                    cancelled = []

                    def cancel_dialog():
                        dialog = app.activeModalWidget()
                        if isinstance(dialog, QFileDialog):
                            cancelled.append(True)
                            dialog.reject()

                    # These are the same static APIs used by the LUT folder
                    # picker and log export; neither may enter a native dialog.
                    QTimer.singleShot(0, cancel_dialog)
                    assert QFileDialog.getExistingDirectory(self, 'LUT folder') == ''
                    QTimer.singleShot(0, cancel_dialog)
                    assert QFileDialog.getSaveFileName(self, 'Export logs')[0] == ''
                    assert len(cancelled) == 2
                print('dialog-probe-ok', flush=True)
                QTimer.singleShot(0, app.quit)

        startup.MainWindow = ProbeWindow
        sys.argv = [sys.argv[0]]
        startup.main()
    ''')
    env = {
        **os.environ,
        'PYTHONIOENCODING': 'utf-8',
        'QT_QPA_PLATFORM': 'offscreen',
        'RAW_ALCHEMY_LOG_DIR': str(tmp_path / 'logs'),
        'RAW_ALCHEMY_CONFIG_DIR': str(tmp_path / 'config'),
    }
    result = subprocess.run(
        [sys.executable, '-c', script, system],
        env=env, capture_output=True, text=True, encoding='utf-8', timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'dialog-probe-ok' in result.stdout
