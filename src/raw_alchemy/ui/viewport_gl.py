"""
OpenGL Viewport Widget for GPU-accelerated image display.

Uses PBO (Pixel Buffer Object) for efficient GPU→texture upload:
  Taichi GPU buffer → to_numpy() → PBO (async) → OpenGL texture → screen

Data flow:
  GpuImage (ti.ndarray, GPU) → numpy (CPU) → PBO → texture → fullscreen quad
  Only viewport-sized data crosses GPU→CPU (e.g. 1920x1080 = ~6MB, <1ms)
"""
import numpy as np
from typing import Optional
from loguru import logger

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt, Signal, QSize, QPointF, QTimer
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLShaderProgram,
    QOpenGLShader,
    QOpenGLTexture,
)
from OpenGL import GL


# Vertex shader: fullscreen quad
_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform vec2 u_offset;   // pan offset in NDC
uniform vec2 u_scale;    // (scale_x, scale_y) for aspect-correct zoom

void main() {
    vec2 pos = aPos * u_scale + u_offset;
    gl_Position = vec4(pos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
"""

# Fragment shader: texture display
_FRAGMENT_SHADER = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D u_texture;

void main() {
    FragColor = texture(u_texture, TexCoord);
}
"""


class ImageViewportGL(QOpenGLWidget):
    """
    OpenGL-based image viewport with zoom/pan.

    Accepts uint8 RGB numpy arrays and displays them via OpenGL texture.
    Uses PBO for async pixel upload.

    Controls:
      - Left drag: pan (when zoomed in)
      - Middle drag: pan
      - Mouse wheel: zoom
      - Double click: toggle fit / 100%
      - Right press/release: compare (show original / show processed)
    """
    # Signal emitted when viewport size changes (for processor to know target size)
    viewport_resized = Signal(int, int)  # width, height
    # Signals for A/B comparison (right-click hold)
    compare_pressed = Signal()
    compare_released = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)

        # Image state
        self._img_width = 0
        self._img_height = 0
        self._has_image = False

        # Zoom / Pan
        self._zoom = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._dragging = False
        self._last_mouse_pos = QPointF()

        # OpenGL objects (created in initializeGL)
        self._texture_id = 0
        self._pbo_id = 0
        self._vao = 0
        self._vbo = 0
        self._shader_program = None
        self._initialized = False

        # Pending image data to upload
        self._pending_data: Optional[np.ndarray] = None
        # Last uploaded image (kept for GL context recovery after minimize/restore)
        self._last_image: Optional[np.ndarray] = None

    def initializeGL(self):
        """Set up OpenGL state, shaders, buffers."""
        GL.glClearColor(0.12, 0.12, 0.12, 1.0)
        GL.glDisable(GL.GL_DEPTH_TEST)

        # --- Compile shaders ---
        self._shader_program = QOpenGLShaderProgram(self)
        self._shader_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, _VERTEX_SHADER)
        self._shader_program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, _FRAGMENT_SHADER)
        self._shader_program.link()

        # --- Fullscreen quad geometry ---
        # Two triangles covering [-1, 1] NDC with UV coords
        vertices = np.array([
            # position    texcoord
            -1.0, -1.0,   0.0, 1.0,   # bottom-left  (UV flipped Y)
             1.0, -1.0,   1.0, 1.0,   # bottom-right
            -1.0,  1.0,   0.0, 0.0,   # top-left
             1.0,  1.0,   1.0, 0.0,   # top-right
        ], dtype=np.float32)

        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)

        # Position attribute (location=0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, GL.ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(0)
        # TexCoord attribute (location=1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, GL.ctypes.c_void_p(8))
        GL.glEnableVertexAttribArray(1)

        GL.glBindVertexArray(0)

        # --- Texture ---
        self._texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        # --- PBO for async pixel upload ---
        self._pbo_id = GL.glGenBuffers(1)

        self._initialized = True
        logger.debug("[ViewportGL] OpenGL initialized.")

        # Recover texture after GL context recreation (e.g. minimize/restore)
        if self._last_image is not None:
            self._pending_data = self._last_image

    def resizeGL(self, w, h):
        GL.glViewport(0, 0, w, h)
        self.viewport_resized.emit(w, h)

    def paintGL(self):
        """Render the current texture."""
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        if not self._has_image:
            return

        # Upload pending data via PBO
        if self._pending_data is not None:
            self._upload_via_pbo(self._pending_data)
            self._pending_data = None

        # Draw textured quad
        self._shader_program.bind()

        # Compute aspect-correct zoom
        vp_w, vp_h = self.width(), self.height()
        if vp_w > 0 and vp_h > 0 and self._img_width > 0 and self._img_height > 0:
            img_aspect = self._img_width / self._img_height
            vp_aspect = vp_w / vp_h

            if img_aspect > vp_aspect:
                scale_x = self._zoom
                scale_y = self._zoom * vp_aspect / img_aspect
            else:
                scale_x = self._zoom * img_aspect / vp_aspect
                scale_y = self._zoom
        else:
            scale_x = self._zoom
            scale_y = self._zoom

        # Set uniforms via raw GL calls (PySide6 setUniformValue doesn't support str+float)
        scale_loc = self._shader_program.uniformLocation("u_scale")
        offset_loc = self._shader_program.uniformLocation("u_offset")
        tex_loc = self._shader_program.uniformLocation("u_texture")

        GL.glUniform2f(scale_loc, scale_x, scale_y)
        GL.glUniform2f(offset_loc, self._offset_x, self._offset_y)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture_id)
        GL.glUniform1i(tex_loc, 0)

        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)

        self._shader_program.release()

    def _upload_via_pbo(self, data: np.ndarray):
        """Upload pixel data to texture via PBO (async-friendly)."""
        h, w = data.shape[:2]
        data_size = data.nbytes

        # RGB rows may not be 4-byte aligned; tell OpenGL to expect tight packing
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)

        # Resize texture if image size changed
        if w != self._img_width or h != self._img_height:
            self._img_width = w
            self._img_height = h

            GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture_id)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8,
                            w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        # Upload via PBO
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self._pbo_id)
        GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, data_size, None, GL.GL_STREAM_DRAW)

        # Map PBO, copy data
        ptr = GL.glMapBuffer(GL.GL_PIXEL_UNPACK_BUFFER, GL.GL_WRITE_ONLY)
        if ptr:
            import ctypes
            ctypes.memmove(ptr, data.ctypes.data, data_size)
            GL.glUnmapBuffer(GL.GL_PIXEL_UNPACK_BUFFER)
        else:
            logger.warning(f"[ViewportGL] glMapBuffer failed for {w}x{h} image, skipping frame")
            GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
            return

        # PBO → texture (GPU-side transfer)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture_id)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
                           GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)  # None = read from PBO
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

    def set_image(self, img_uint8: np.ndarray):
        """
        Update the displayed image.
        img_uint8: HxWx3 uint8 numpy array (RGB).
        This schedules a PBO upload on the next paintGL.
        """
        if img_uint8 is None or img_uint8.size == 0:
            return

        if not img_uint8.flags['C_CONTIGUOUS']:
            img_uint8 = np.ascontiguousarray(img_uint8)

        self._pending_data = img_uint8
        self._last_image = img_uint8
        self._has_image = True
        self.update()  # Schedule repaint

    def set_image_float(self, img_float: np.ndarray):
        """
        Convenience: accept float32 [0,1] image, convert to uint8.
        For better performance, prefer doing this conversion on GPU
        and calling set_image() with uint8 data directly.
        """
        if img_float is None or img_float.size == 0:
            return
        img_uint8 = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
        self.set_image(img_uint8)

    def clear_image(self):
        """Clear the displayed image and reset zoom."""
        self._has_image = False
        self._pending_data = None
        self._last_image = None
        self._zoom = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self.update()

    # --- Zoom / Pan ---

    def fit_to_view(self):
        """Reset zoom to fit image in viewport."""
        self._zoom = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self.update()

    def zoom_to_100(self):
        """Zoom to 100% (1 image pixel = 1 screen pixel)."""
        if self._img_width > 0 and self._img_height > 0:
            vp_w, vp_h = self.width(), self.height()
            self._zoom = max(self._img_width / vp_w, self._img_height / vp_h)
            self._offset_x = 0.0
            self._offset_y = 0.0
            self.update()

    def _clamp_offset(self):
        if self._img_width <= 0 or self._img_height <= 0:
            return
        vp_w, vp_h = self.width(), self.height()
        if vp_w <= 0 or vp_h <= 0:
            return
        img_aspect = self._img_width / self._img_height
        vp_aspect = vp_w / vp_h

        if img_aspect > vp_aspect:
            sx, sy = self._zoom, self._zoom * vp_aspect / img_aspect
        else:
            sx, sy = self._zoom * img_aspect / vp_aspect, self._zoom

        if sx <= 1.0:
            self._offset_x = 0.0
        else:
            max_off = sx - 1.0
            self._offset_x = max(-max_off, min(self._offset_x, max_off))

        if sy <= 1.0:
            self._offset_y = 0.0
        else:
            max_off = sy - 1.0
            self._offset_y = max(-max_off, min(self._offset_y, max_off))

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else (1 / 1.1)
        old_zoom = self._zoom
        self._zoom = max(0.1, min(self._zoom * factor, 50.0))
        real_factor = self._zoom / old_zoom

        mx = (event.position().x() / self.width()) * 2.0 - 1.0
        my = -((event.position().y() / self.height()) * 2.0 - 1.0)

        self._offset_x = mx - (mx - self._offset_x) * real_factor
        self._offset_y = my - (my - self._offset_y) * real_factor
        self._clamp_offset()
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            self.compare_pressed.emit()
        elif event.button() == Qt.MouseButton.MiddleButton or event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_mouse_pos = event.position()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            self.compare_released.emit()
        self._dragging = False

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            pos = event.position()
            dx = (pos.x() - self._last_mouse_pos.x()) / self.width() * 2.0
            dy = -(pos.y() - self._last_mouse_pos.y()) / self.height() * 2.0
            self._offset_x += dx
            self._offset_y += dy
            self._last_mouse_pos = pos
            self._clamp_offset()
            self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Double click to toggle fit/100%."""
        if abs(self._zoom - 1.0) < 0.01:
            self.zoom_to_100()
        else:
            self.fit_to_view()
