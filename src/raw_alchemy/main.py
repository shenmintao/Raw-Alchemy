import sys
from multiprocessing import freeze_support

# Frozen worker processes must exit into multiprocessing before importing Qt.
if __name__ == "__main__":
    freeze_support()

import os
import platform
from PySide6.QtWidgets import QApplication

from raw_alchemy import i18n
from raw_alchemy.ui.main_window import MainWindow


def _configure_opengl():
    # macOS defaults to an OpenGL 2.1 legacy context which cannot compile
    # #version 330 core shaders. Request Core Profile before QApplication so
    # all surfaces inherit it. Not needed on Windows/Linux where Qt negotiates
    # a compatible context automatically.
    if platform.system() == 'Darwin':
        from PySide6.QtGui import QSurfaceFormat
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        QSurfaceFormat.setDefaultFormat(fmt)


def main():
    from multiprocessing import freeze_support
    freeze_support()
    # --- Taichi is initialized on import of math_ops ---
    # No separate cache directory needed (Taichi manages its own cache)

    # Must be called before QApplication to take effect on all surfaces.
    _configure_opengl()

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
