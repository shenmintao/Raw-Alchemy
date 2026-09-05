"""Manual macOS OpenGL smoke test. Run from an active desktop, not offscreen.

    python tools/check_macos_opengl.py

Uses a synthetic frame, opens a small window briefly, and checks the actual
framebuffer; no RAW files or existing application settings are touched.
"""
import sys

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from raw_alchemy.ui.viewport_gl import ImageViewportGL


def main():
    if sys.platform != "darwin":
        raise SystemExit("This smoke test requires macOS and an active desktop.")
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication([])
    widget = ImageViewportGL()
    widget.setWindowTitle("Raw Alchemy — OpenGL smoke test")
    widget.resize(400, 300)
    frame = np.full((300, 400, 3), (220, 40, 80), dtype=np.uint8)
    widget.set_image(frame)
    widget.show()
    result = 1

    def verify():
        nonlocal result
        try:
            assert widget.isValid(), "OpenGL widget has no valid context"
            context_format = widget.context().format()
            assert context_format.profile() == QSurfaceFormat.OpenGLContextProfile.CoreProfile
            assert (context_format.majorVersion(), context_format.minorVersion()) >= (3, 3)
            assert widget._shader_program.isLinked(), widget._shader_program.log()
            image = widget.grabFramebuffer()
            assert not image.isNull(), "Empty framebuffer"
            rgb = image.pixelColor(image.width() // 2, image.height() // 2).getRgb()[:3]
            assert max(abs(int(a) - int(b)) for a, b in zip(rgb, (220, 40, 80))) <= 3, rgb
            print(f"PASS: OpenGL {context_format.majorVersion()}.{context_format.minorVersion()} "
                  f"Core Profile, shaders linked, rendered center pixel={rgb}", flush=True)
            result = 0
        except Exception as exc:
            print(f"FAIL: {exc}", file=sys.stderr, flush=True)
        finally:
            widget.close()
            app.quit()

    QTimer.singleShot(1500, verify)
    QTimer.singleShot(15000, app.quit)
    app.exec()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
