import sys
import os
from PySide6.QtWidgets import QApplication

from raw_alchemy import i18n
from raw_alchemy.ui.main_window import MainWindow

def main():
    # --- Taichi is initialized on import of math_ops ---
    # No separate cache directory needed (Taichi manages its own cache)

    # 解决部分Windows环境下缩放问题
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    app_font = app.font()
    if app_font.pointSize() <= 0:
        app_font.setPointSize(9)
        app.setFont(app_font)

    # Load i18n
    i18n.init_i18n()

    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
