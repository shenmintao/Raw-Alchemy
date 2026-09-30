"""Slider renders complete while newer values coalesce on the GUI lane."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from raw_alchemy.pipeline.request import ProcessRequest
from raw_alchemy.workers.image_processor import ImageProcessor


def test_slider_changes_wait_for_current_result_without_superseding_it():
    from raw_alchemy.ui.main_window import MainWindow, _as_hashable

    busy = [True]
    params = {"exposure": 1.0}
    timer = SimpleNamespace(start=Mock(), setInterval=Mock())
    submitted = []
    harness = SimpleNamespace(current_raw_path="same.raw", current_request_id=10,
        _last_param_submit_key=_as_hashable({"exposure": 0.0}), update_timer=timer,
        right_panel=SimpleNamespace(get_params=lambda: dict(params)),
        processor=SimpleNamespace(has_interactive_request=lambda: busy[0]))
    harness._interactive_params = lambda: (dict(params), 0.0)

    def submit():
        submitted.append(dict(params))
        harness._last_param_submit_key = _as_hashable(params)
        harness.current_request_id += 1

    harness.trigger_processing = submit
    MainWindow._trigger_param_processing(harness)
    params["exposure"] = 2.0
    MainWindow._trigger_param_processing(harness)
    assert submitted == [] and harness.current_request_id == 10
    timer.start.assert_called_with(16)
    busy[0] = False
    MainWindow._trigger_param_processing(harness)
    assert submitted == [{"exposure": 2.0}] and harness.current_request_id == 11
    MainWindow._trigger_param_processing(harness)
    assert len(submitted) == 1


def test_preloads_and_idle_quality_refinement_remain_preemptible():
    worker = ImageProcessor()
    for params in ({"_preload": True}, {"_force_full_preview": True}):
        worker._active_request = ProcessRequest("same.raw", params, 1)
        assert not worker.has_interactive_request()
    worker._active_request = ProcessRequest("same.raw", {"exposure": 1.0}, 1)
    assert worker.has_interactive_request()
    worker._active_request = None
    worker.pending_request = ProcessRequest("same.raw", {"exposure": 2.0}, 2)
    assert worker.has_interactive_request()


def _denoise_harness(params, cached):
    from raw_alchemy.ui.main_window import MainWindow

    submitted = []
    harness = SimpleNamespace(
        current_raw_path="same.raw", current_request_id=0,
        _last_param_submit_key=None, _param_interaction_active=True,
        _denoise_ui_key=None, _denoise_settle_deadline=0.0,
        update_timer=SimpleNamespace(start=Mock(), stop=Mock(), setInterval=Mock()),
        right_panel=SimpleNamespace(get_params=lambda: dict(params)),
        processor=SimpleNamespace(has_interactive_request=lambda: False,
            preview_denoise_strength=lambda _: cached[0]),
        file_params_cache={}, _schedule_current_sidecar_write=lambda: None,
        _add_preview_output_params=lambda p: p.copy(),
    )

    def submit(path, actual):
        submitted.append(actual.copy())
        return len(submitted)

    harness.processor.update_preview = submit
    for name in ("_note_denoise_change", "_denoise_settle_remaining", "_interactive_params",
                 "_trigger_param_processing", "trigger_processing"):
        setattr(harness, name, getattr(MainWindow, name).__get__(harness))
    return harness, submitted


@pytest.mark.parametrize("initial", [None, 0.25])
def test_denoise_waits_for_release_and_quiet_input_but_colour_keeps_up(monkeypatch, initial):
    from raw_alchemy.ui import main_window

    now = [1.0]
    monkeypatch.setattr(main_window.time, "monotonic", lambda: now[0])
    params = dict(exposure=1.0, denoise_enabled=True, denoise_strength=0.3)
    harness, submitted = _denoise_harness(params, [initial])
    for value in (0.3, 0.35, 0.4):
        params.update(denoise_strength=value, exposure=params["exposure"] + 0.1)
        harness._note_denoise_change(params)
        harness._trigger_param_processing()
        now[0] += 0.05
    assert len(submitted) == 3
    assert all(p["denoise_enabled"] == (initial is not None) for p in submitted)
    assert all(p["denoise_strength"] == (initial or 0.25) for p in submitted)
    assert submitted[-1]["exposure"] == params["exposure"]
    # A pause while holding the mouse must not launch a full-frame inference.
    now[0] += 10
    harness._trigger_param_processing()
    assert len(submitted) == 3
    main_window.MainWindow._on_param_interaction_finished(harness, params)
    now[0] += 0.19
    harness._trigger_param_processing()
    assert len(submitted) == 3
    now[0] += 0.011
    harness._trigger_param_processing()
    assert submitted[-1] == dict(params, _denoise_target=0.4, _denoise_deferred=False)
    assert len(submitted) == 4
    harness._trigger_param_processing()
    assert len(submitted) == 4


def test_denoise_keyboard_updates_coalesce_and_disable_is_immediate(monkeypatch):
    from raw_alchemy.ui import main_window

    now = [1.0]
    monkeypatch.setattr(main_window.time, "monotonic", lambda: now[0])
    params = dict(exposure=1.0, denoise_enabled=True, denoise_strength=0.3)
    harness, submitted = _denoise_harness(params, [None])
    harness._param_interaction_active = False
    harness._note_denoise_change(params)
    harness._trigger_param_processing()
    now[0] += 0.15
    params["denoise_strength"] = 0.4
    harness._note_denoise_change(params)
    harness._trigger_param_processing()
    now[0] += 0.15
    harness._trigger_param_processing()
    assert len(submitted) == 1 and not submitted[0]["denoise_enabled"]
    now[0] += 0.051
    harness._trigger_param_processing()
    assert len(submitted) == 2
    assert submitted[-1] == dict(params, _denoise_target=0.4, _denoise_deferred=False)
    params["denoise_enabled"] = False
    harness._note_denoise_change(params)
    harness._trigger_param_processing()
    assert not submitted[-1]["denoise_enabled"]
    assert harness._denoise_settle_deadline == 0.0


def test_denoise_snapshot_never_reuses_another_image():
    worker = ImageProcessor()
    worker.current_path = "same.raw"
    worker.cached_denoise_full = np.zeros((2, 2, 3), np.float32)
    worker.last_denoise_key = ("same.raw", "denoise", 0.25)
    assert worker.preview_denoise_strength("same.raw") == 0.25
    assert worker.preview_denoise_strength("other.raw") is None
