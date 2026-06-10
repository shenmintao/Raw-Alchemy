from PySide6.QtWidgets import QApplication, QMessageBox


def show_crash_dialog(log_path: str, error_type: str, error_msg: str):
    app = QApplication.instance()
    if not app:
        return

    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setWindowTitle("Fatal Error")
    msg.setText(f"Application encountered an unhandled exception.\nType: {error_type}")
    msg.setInformativeText(f"Check log for details: {log_path}")
    msg.setDetailedText(error_msg)
    msg.exec()
