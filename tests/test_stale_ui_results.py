"""A late queued result must not mutate the UI for a newer request."""

from types import SimpleNamespace

import pytest


class _Progress:
    def __init__(self):
        self.calls = []

    def setContent(self, content):
        self.calls.append(("content", content))

    def setState(self, state):
        self.calls.append(("state", state))


@pytest.mark.parametrize("mode", [None, "crop", "perspective"])
@pytest.mark.parametrize("wrong_path", [False, True])
def test_stale_result_rejected_before_progress_or_image_access(mode, wrong_path):
    from raw_alchemy.ui.main_window import MainWindow

    progress = _Progress()
    harness = SimpleNamespace(
        current_raw_path="current.raw", current_request_id=10,
        crop_request_id=20, perspective_request_id=30,
        processor_connection_mode=mode, denoise_progress_dialog=progress,
    )
    expected = {None: 10, "crop": 20, "perspective": 30}[mode]
    # Deliberately no ndarray: a rejected result must not even inspect it.
    MainWindow.on_process_result(
        harness, None, "old.raw" if wrong_path else "current.raw",
        expected if wrong_path else expected - 1, 0.0,
    )
    assert harness.denoise_progress_dialog is progress
    assert progress.calls == []


@pytest.mark.parametrize("mode", ["crop", "perspective"])
def test_edit_mode_accepts_its_own_request_id(mode):
    from raw_alchemy.ui.main_window import MainWindow

    progress = _Progress()
    routed = []
    harness = SimpleNamespace(
        current_raw_path="current.raw", current_request_id=10,
        crop_request_id=20, perspective_request_id=30,
        processor_connection_mode=mode, denoise_progress_dialog=progress,
        on_crop_ready=lambda *args: routed.append(("crop", args)),
        on_perspective_ready=lambda *args: routed.append(("perspective", args)),
    )
    image = object()
    MainWindow.on_process_result(
        harness, image, "current.raw", 20 if mode == "crop" else 30, 0.0,
    )
    assert harness.denoise_progress_dialog is None
    assert progress.calls[-1] == ("state", True)
    assert routed == [(mode, (image, None))]
