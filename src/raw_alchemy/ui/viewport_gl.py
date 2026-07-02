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
    zoom_changed = Signal(float)
    # Signals for A/B comparison (right-click hold)
    compare_pressed = Signal()
    compare_released = Signal()
    # Emitted while panning when the visible region leaves the rendered ROI
    # texture's coverage (T7.5) — the owner should request a re-render.
    roi_update_needed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)

        # Image state
        self._img_width = 0
        self._img_height = 0
        self._source_width = 0
        self._source_height = 0
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

        # ===== zoom>fit ROI overlay (T7.5) =====
        # A second texture holding just the visible region (plus margins) at
        # native resolution, drawn on top of the full-frame texture (which
        # acts as the backdrop so pans past the margin never flash black).
        self._roi_texture_id = 0
        self._roi_img_width = 0
        self._roi_img_height = 0
        self._roi_pending_data: Optional[np.ndarray] = None
        self._roi_last_image: Optional[np.ndarray] = None
        # ROI rect (x, y, w, h) in full-frame pixel coordinates.
        self._roi_rect_px: Optional[tuple] = None
        self._has_roi = False

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

        # --- Textures (base full-frame + ROI overlay) ---
        self._texture_id = self._create_texture()
        self._roi_texture_id = self._create_texture()
        # Fresh GL context: force (re)allocation on the next upload.
        self._img_width = self._img_height = 0
        self._roi_img_width = self._roi_img_height = 0

        # --- PBO for async pixel upload ---
        self._pbo_id = GL.glGenBuffers(1)

        self._initialized = True
        logger.debug("[ViewportGL] OpenGL initialized.")

        # Recover textures after GL context recreation (e.g. minimize/restore)
        if self._last_image is not None:
            self._pending_data = self._last_image
        if self._roi_last_image is not None:
            self._roi_pending_data = self._roi_last_image

    @staticmethod
    def _create_texture():
        texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return texture_id

    def resizeGL(self, w, h):
        GL.glViewport(0, 0, w, h)
        self._clamp_offset()
        self.viewport_resized.emit(w, h)

    def _display_dimensions(self):
        """Return the stable image dimensions used for viewport geometry."""
        if self._source_width > 0 and self._source_height > 0:
            return self._source_width, self._source_height
        if self._pending_data is not None:
            h, w = self._pending_data.shape[:2]
            return w, h
        return self._img_width, self._img_height

    def _display_scale(self, zoom=None):
        vp_w, vp_h = self.width(), self.height()
        display_w, display_h = self._display_dimensions()
        if vp_w <= 0 or vp_h <= 0 or display_w <= 0 or display_h <= 0:
            z = self._zoom if zoom is None else zoom
            return z, z

        z = self._zoom if zoom is None else zoom
        img_aspect = display_w / display_h
        vp_aspect = vp_w / vp_h

        if img_aspect > vp_aspect:
            return z, z * vp_aspect / img_aspect
        return z * img_aspect / vp_aspect, z

    def _roi_rect_norm(self) -> Optional[tuple]:
        """ROI rect normalized to the displayed frame, or None."""
        if self._roi_rect_px is None:
            return None
        display_w, display_h = self._display_dimensions()
        if display_w <= 0 or display_h <= 0:
            return None
        x, y, w, h = self._roi_rect_px
        return (x / display_w, y / display_h, (x + w) / display_w, (y + h) / display_h)

    def paintGL(self):
        """Render the base texture, then the ROI overlay on top (T7.5)."""
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        if not self._has_image and not self._has_roi:
            return

        # Upload pending data via PBO
        if self._pending_data is not None:
            self._img_width, self._img_height = self._upload_via_pbo(
                self._pending_data, self._texture_id,
                self._img_width, self._img_height,
            )
            self._pending_data = None
        if self._roi_pending_data is not None:
            self._roi_img_width, self._roi_img_height = self._upload_via_pbo(
                self._roi_pending_data, self._roi_texture_id,
                self._roi_img_width, self._roi_img_height,
            )
            self._roi_pending_data = None

        self._shader_program.bind()

        # Compute aspect-correct zoom from the full source dimensions, not the
        # current preview texture. Preview/detail texture swaps must not move
        # the image while the user is zooming.
        scale_x, scale_y = self._display_scale()

        # Full-frame backdrop quad.
        if self._has_image:
            self._draw_quad(
                self._texture_id,
                scale_x, scale_y,
                self._offset_x, self._offset_y,
            )

        # ROI overlay quad: the same unit quad, positioned over the ROI's
        # sub-rectangle of the image via per-draw offset/scale uniforms.
        roi_norm = self._roi_rect_norm() if self._has_roi else None
        if roi_norm is not None:
            x0, y0, x1, y1 = roi_norm
            roi_scale_x = (x1 - x0) * scale_x
            roi_scale_y = (y1 - y0) * scale_y
            roi_offset_x = (x0 + x1 - 1.0) * scale_x + self._offset_x
            roi_offset_y = (1.0 - y0 - y1) * scale_y + self._offset_y
            self._draw_quad(
                self._roi_texture_id,
                roi_scale_x, roi_scale_y,
                roi_offset_x, roi_offset_y,
            )

        self._shader_program.release()

    def _draw_quad(self, texture_id, scale_x, scale_y, offset_x, offset_y):
        """Draw the unit quad with the given texture and NDC placement."""
        # Set uniforms via raw GL calls (PySide6 setUniformValue doesn't support str+float)
        scale_loc = self._shader_program.uniformLocation("u_scale")
        offset_loc = self._shader_program.uniformLocation("u_offset")
        tex_loc = self._shader_program.uniformLocation("u_texture")

        GL.glUniform2f(scale_loc, scale_x, scale_y)
        GL.glUniform2f(offset_loc, offset_x, offset_y)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glUniform1i(tex_loc, 0)

        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)

    def _upload_via_pbo(self, data: np.ndarray, texture_id, tex_width, tex_height):
        """Upload pixel data to a texture via PBO (async-friendly).

        Returns the texture's (width, height) after the upload so the caller
        can track per-texture allocation state.
        """
        h, w = data.shape[:2]
        data_size = data.nbytes

        # RGB rows may not be 4-byte aligned; tell OpenGL to expect tight packing
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)

        # Resize texture if image size changed
        if w != tex_width or h != tex_height:
            tex_width = w
            tex_height = h

            GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
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
            return tex_width, tex_height

        # PBO → texture (GPU-side transfer)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
                           GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)  # None = read from PBO
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)
        return tex_width, tex_height

    def set_image(self, img_uint8: np.ndarray, source_size=None):
        """
        Update the displayed image.
        img_uint8: HxWx3 uint8 numpy array (RGB).
        This schedules a PBO upload on the next paintGL.

        A full-frame image supersedes any ROI overlay (T7.5): the overlay is
        dropped so a stale detail crop is never drawn over fresh content.
        """
        if img_uint8 is None or img_uint8.size == 0:
            return

        if not img_uint8.flags['C_CONTIGUOUS']:
            img_uint8 = np.ascontiguousarray(img_uint8)

        self._pending_data = img_uint8
        self._last_image = img_uint8
        if source_size:
            self._source_width = max(1, int(source_size[0]))
            self._source_height = max(1, int(source_size[1]))
        else:
            self._source_width = img_uint8.shape[1]
            self._source_height = img_uint8.shape[0]
        self._has_image = True
        self._clear_roi_state()
        self.update()  # Schedule repaint

    def set_roi_image(self, img_uint8: np.ndarray, source_size, roi_rect):
        """Update the ROI overlay texture (T7.5).

        img_uint8: the rendered ROI (HxWx3 uint8 RGB).
        source_size: (w, h) of the *full* frame the ROI belongs to — keeps
            the viewport geometry (fit/zoom/pan) identical to a full render.
        roi_rect: (x, y, w, h) ROI placement in full-frame pixels.

        The existing full-frame texture is kept underneath as the backdrop,
        so panning past the rendered margin degrades to the backdrop instead
        of flashing black.
        """
        if img_uint8 is None or img_uint8.size == 0 or not roi_rect:
            return
        if not img_uint8.flags['C_CONTIGUOUS']:
            img_uint8 = np.ascontiguousarray(img_uint8)

        if source_size:
            self._source_width = max(1, int(source_size[0]))
            self._source_height = max(1, int(source_size[1]))
        self._roi_pending_data = img_uint8
        self._roi_last_image = img_uint8
        self._roi_rect_px = tuple(int(v) for v in roi_rect)
        self._has_roi = True
        self.update()

    def _clear_roi_state(self):
        self._has_roi = False
        self._roi_pending_data = None
        self._roi_last_image = None
        self._roi_rect_px = None

    def has_roi_image(self) -> bool:
        return self._has_roi

    def visible_source_rect(self) -> Optional[tuple]:
        """Visible region of the displayed frame, normalized to [0, 1].

        Returns (x0, y0, x1, y1) with y pointing down (image convention),
        clamped to the frame; None when nothing is displayed yet. This is
        what the worker inverts into the zoom>fit source ROI (T7.5).
        """
        if not self._has_image and not self._has_roi:
            return None
        scale_x, scale_y = self._display_scale()
        if scale_x <= 0.0 or scale_y <= 0.0:
            return None

        x0 = ((-1.0 - self._offset_x) / scale_x + 1.0) / 2.0
        x1 = ((1.0 - self._offset_x) / scale_x + 1.0) / 2.0
        y0 = (1.0 - (1.0 - self._offset_y) / scale_y) / 2.0
        y1 = (1.0 - (-1.0 - self._offset_y) / scale_y) / 2.0

        x0, x1 = max(0.0, min(x0, 1.0)), max(0.0, min(x1, 1.0))
        y0, y1 = max(0.0, min(y0, 1.0)), max(0.0, min(y1, 1.0))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def _visible_within_roi(self) -> bool:
        """True when the ROI texture still covers the visible region."""
        if not self._has_roi:
            return True
        roi_norm = self._roi_rect_norm()
        if roi_norm is None:
            return True
        visible = self.visible_source_rect()
        if visible is None:
            return True
        tol = 1e-3
        return (
            visible[0] >= roi_norm[0] - tol
            and visible[1] >= roi_norm[1] - tol
            and visible[2] <= roi_norm[2] + tol
            and visible[3] <= roi_norm[3] + tol
        )

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
        self._clear_roi_state()
        self._source_width = 0
        self._source_height = 0
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
        self.zoom_changed.emit(self._zoom)

    def zoom_to_100(self):
        """Zoom to 100% (1 image pixel = 1 screen pixel)."""
        source_w = self._source_width if self._source_width > 0 else self._img_width
        source_h = self._source_height if self._source_height > 0 else self._img_height
        if source_w > 0 and source_h > 0:
            vp_w, vp_h = self.width(), self.height()
            if vp_w <= 0 or vp_h <= 0:
                return
            self._zoom = max(source_w / vp_w, source_h / vp_h)
            self._offset_x = 0.0
            self._offset_y = 0.0
            self.update()
            self.zoom_changed.emit(self._zoom)

    def preview_zoom(self):
        return self._zoom

    def display_is_full_resolution(self):
        if not self._has_image:
            return False
        if self._pending_data is not None:
            img_h, img_w = self._pending_data.shape[:2]
        else:
            img_w, img_h = self._img_width, self._img_height
        if img_w <= 0 or img_h <= 0:
            return False
        source_w = self._source_width if self._source_width > 0 else self._img_width
        source_h = self._source_height if self._source_height > 0 else self._img_height
        return img_w >= source_w and img_h >= source_h

    def _clamp_offset(self):
        scale_x, scale_y = self._display_scale()
        if scale_x <= 0.0 or scale_y <= 0.0:
            return

        if scale_x <= 1.0:
            self._offset_x = 0.0
        else:
            max_off = scale_x - 1.0
            self._offset_x = max(-max_off, min(self._offset_x, max_off))

        if scale_y <= 1.0:
            self._offset_y = 0.0
        else:
            max_off = scale_y - 1.0
            self._offset_y = max(-max_off, min(self._offset_y, max_off))

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else (1 / 1.1)
        old_zoom = self._zoom
        old_scale_x, old_scale_y = self._display_scale(old_zoom)
        new_zoom = max(0.1, min(self._zoom * factor, 50.0))
        new_scale_x, new_scale_y = self._display_scale(new_zoom)

        mx = (event.position().x() / self.width()) * 2.0 - 1.0
        my = -((event.position().y() / self.height()) * 2.0 - 1.0)

        if old_scale_x > 0.0 and old_scale_y > 0.0:
            local_x = (mx - self._offset_x) / old_scale_x
            local_y = (my - self._offset_y) / old_scale_y
            self._offset_x = mx - local_x * new_scale_x
            self._offset_y = my - local_y * new_scale_y

        self._zoom = new_zoom
        self._clamp_offset()
        self.update()
        self.zoom_changed.emit(self._zoom)

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
            # Pans inside the ROI margin cost nothing (quads just move);
            # once the visible region leaves the rendered coverage, ask the
            # owner for an incremental ROI re-render (T7.5).
            if self._has_roi and not self._visible_within_roi():
                self.roi_update_needed.emit()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Double click to toggle fit/100%."""
        if abs(self._zoom - 1.0) < 0.01:
            self.zoom_to_100()
        else:
            self.fit_to_view()
