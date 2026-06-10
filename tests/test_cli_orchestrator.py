import shutil
from pathlib import Path

from click.testing import CliRunner


def _scratch_dir(name):
    root = Path.cwd() / ".test-output" / "cli-orchestrator-tests" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def test_cli_single_file_forwards_entrypoint_options(monkeypatch):
    from raw_alchemy import cli

    scratch = _scratch_dir("single")
    input_path = scratch / "image.dng"
    output_path = scratch / "out.tif"
    input_path.write_bytes(b"raw")
    captured = {}

    def fake_process_path(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli.orchestrator, "process_path", fake_process_path)

    result = CliRunner().invoke(
        cli.main,
        [
            str(input_path),
            str(output_path),
            "--log-space",
            "F-Log",
            "--exposure",
            "0.5",
            "--metering",
            "matrix",
            "--jobs",
            "2",
            "--format",
            "hdr-heif",
        ],
    )

    assert result.exit_code == 0
    assert captured["input_path"] == str(input_path)
    assert captured["output_path"] == str(output_path)
    assert captured["log_space"] == "F-Log"
    assert captured["exposure"] == 0.5
    assert captured["lens_correct"] is True
    assert captured["metering_mode"] == "matrix"
    assert captured["jobs"] == 2
    assert captured["output_format"] == "hdr-heif"


def test_cli_safe_echo_falls_back_for_legacy_console(monkeypatch):
    from raw_alchemy import cli

    calls = []

    class FakeStream:
        encoding = "gbk"

    def fake_echo(message):
        calls.append(message)
        if len(calls) == 1:
            raise UnicodeEncodeError("gbk", "⚙", 0, 1, "illegal multibyte sequence")

    monkeypatch.setattr(cli.click, "echo", fake_echo)
    monkeypatch.setattr(cli.click, "get_text_stream", lambda _name: FakeStream())

    cli._safe_echo("⚙️ Processing single file...")

    assert calls == [
        "⚙️ Processing single file...",
        "?? Processing single file...",
    ]


class _FakeFuture:
    def result(self):
        return None


class _FakeExecutor:
    def __init__(self):
        self.max_workers = None
        self.submitted = []

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False

    def submit(self, fn, *args, **kwargs):
        future = _FakeFuture()
        self.submitted.append((fn, args, kwargs, future))
        return future


class _FakeQueueLogger:
    def __init__(self):
        self.messages = []

    def put(self, message):
        self.messages.append(message)


def test_orchestrator_batch_submits_supported_raw_files(monkeypatch):
    from raw_alchemy import core, orchestrator

    scratch = _scratch_dir("batch")
    input_dir = scratch / "input"
    output_dir = scratch / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "a.dng").write_bytes(b"raw")
    (input_dir / "b.NEF").write_bytes(b"raw")
    (input_dir / "ignore.jpg").write_bytes(b"jpg")

    fake_executor = _FakeExecutor()
    logger = _FakeQueueLogger()

    def fake_executor_factory(max_workers):
        fake_executor.max_workers = max_workers
        return fake_executor

    monkeypatch.setattr(orchestrator, "SUPPORTED_RAW_EXTENSIONS", (".dng", ".nef"))
    monkeypatch.setattr(
        orchestrator.concurrent.futures,
        "ProcessPoolExecutor",
        fake_executor_factory,
    )
    monkeypatch.setattr(
        orchestrator.concurrent.futures,
        "as_completed",
        lambda futures: list(futures),
    )

    orchestrator.process_path(
        input_path=str(input_dir),
        output_path=str(output_dir),
        log_space="F-Log",
        lut_path=None,
        exposure=None,
        lens_correct=False,
        custom_db_path=None,
        metering_mode="hybrid",
        jobs=3,
        logger_func=logger,
        output_format="jpg",
        wb_temp=100.0,
        wb_tint=-10.0,
        saturation=0.95,
        contrast=1.05,
        highlight=-0.2,
        shadow=0.3,
        rotation=90,
        flip_horizontal=True,
        flip_vertical=False,
        perspective_corners=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        crop=(0.1, 0.2, 0.8, 0.9),
        denoise_enabled=True,
        sharpen_strength=0.4,
    )

    assert fake_executor.max_workers == 3
    assert len(fake_executor.submitted) == 2
    assert logger.messages.count({"status": "done"}) == 2
    assert {"total_files": 2} in logger.messages

    submitted = {
        kwargs["raw_path"]: kwargs
        for fn, _args, kwargs, _future in fake_executor.submitted
        if fn is core.process_image
    }
    assert set(submitted) == {
        str(input_dir / "a.dng"),
        str(input_dir / "b.NEF"),
    }

    first = submitted[str(input_dir / "a.dng")]
    assert first["output_path"] == str(output_dir / "a.jpg")
    assert first["log_space"] == "F-Log"
    assert first["exposure"] is None
    assert first["lens_correct"] is False
    assert first["metering_mode"] == "hybrid"
    assert first["hdr_output"] is False
    assert first["wb_temp"] == 100.0
    assert first["wb_tint"] == -10.0
    assert first["saturation"] == 0.95
    assert first["contrast"] == 1.05
    assert first["highlight"] == -0.2
    assert first["shadow"] == 0.3
    assert first["rotation"] == 90
    assert first["flip_horizontal"] is True
    assert first["flip_vertical"] is False
    assert first["perspective_corners"] == (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    )
    assert first["crop"] == (0.1, 0.2, 0.8, 0.9)
    assert first["denoise_enabled"] is True
    assert first["sharpen_strength"] == 0.4
