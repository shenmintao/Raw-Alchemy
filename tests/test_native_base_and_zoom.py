"""Native-resolution base map gate and viewport zoom/filter helpers."""

import pytest


def _force(**extra):
    params = {"_force_full_preview": True, "viewport_size": (2560, 1440)}
    params.update(extra)
    return params


@pytest.fixture
def plenty_of_ram(monkeypatch):
    from raw_alchemy.workers import image_processor

    class VM:
        available = 64 * 2**30

    monkeypatch.setattr(image_processor.psutil, "virtual_memory", lambda: VM)


def test_native_base_keeps_even_100mp_frames_when_the_gpu_can(plenty_of_ram):
    from raw_alchemy.workers.image_processor import ImageProcessor

    gfx100 = (11648, 8736)
    size = ImageProcessor._make_preview_target_size(
        *gfx100, _force(native_base=True, max_texture_size=16384, free_vram_mb=12000)
    )
    assert size == gfx100


def test_native_base_falls_back_to_the_quality_base(plenty_of_ram):
    from raw_alchemy.workers.image_processor import NATIVE_PREVIEW_MAX_SIDE, ImageProcessor

    frame = (7952, 5304)
    fits = dict(native_base=True, max_texture_size=16384, free_vram_mb=8000)
    assert ImageProcessor._make_preview_target_size(*frame, _force(**fits)) == frame

    for reason in (
        dict(fits, native_base=False),                 # user switched it off
        dict(fits, max_texture_size=0),                # GL not initialised yet
        dict(fits, max_texture_size=4096),             # texture limit too small
        dict(fits, free_vram_mb=100),                  # too little free VRAM
    ):
        tw, th = ImageProcessor._make_preview_target_size(*frame, _force(**reason))
        assert tw == NATIVE_PREVIEW_MAX_SIDE, reason

    # Unknown VRAM (driver without a memory query) does not block it.
    unknown = dict(fits, free_vram_mb=None)
    assert ImageProcessor._make_preview_target_size(*frame, _force(**unknown)) == frame


def test_native_base_respects_free_ram(monkeypatch):
    from raw_alchemy.workers import image_processor
    from raw_alchemy.workers.image_processor import NATIVE_PREVIEW_MAX_SIDE, ImageProcessor

    class VM:
        available = 512 * 2**20

    monkeypatch.setattr(image_processor.psutil, "virtual_memory", lambda: VM)
    tw, _th = ImageProcessor._make_preview_target_size(
        7952, 5304, _force(native_base=True, max_texture_size=16384)
    )
    assert tw == NATIVE_PREVIEW_MAX_SIDE


def test_hundred_percent_zoom_counts_device_pixels():
    from raw_alchemy.ui.viewport_gl import hundred_percent_zoom

    # 7952 px wide image in a 1000-logical-px viewport.
    assert hundred_percent_zoom(7952, 5304, 1000, 800, 1.0) == pytest.approx(7.952)
    # At 150% Windows scaling the viewport has 1500 device px: 1:1 needs less zoom.
    assert hundred_percent_zoom(7952, 5304, 1000, 800, 1.5) == pytest.approx(7952 / 1500)


def test_magnification_switches_to_nearest_for_pixel_peeping():
    from OpenGL import GL

    from raw_alchemy.ui.viewport_gl import mag_filter_for

    assert mag_filter_for(0.3) == GL.GL_LINEAR
    assert mag_filter_for(1.0) == GL.GL_LINEAR
    assert mag_filter_for(1.99) == GL.GL_LINEAR
    assert mag_filter_for(2.0) == GL.GL_NEAREST
    assert mag_filter_for(8.0) == GL.GL_NEAREST
