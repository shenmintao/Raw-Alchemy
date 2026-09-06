"""Regression coverage for lossless, identity-checked preview/export artifacts."""
from pathlib import Path

import numpy as np
import pytest

from raw_alchemy.pipeline import denoise_disk_cache as disk
from raw_alchemy.pipeline import source_artifacts as artifacts
from raw_alchemy.pipeline import stage_identity as identity
from raw_alchemy.pipeline.executor import PipelineAborted


@pytest.fixture
def cache_source(tmp_path, monkeypatch):
    monkeypatch.setenv("RAWALCHEMY_DENOISE_CACHE_DIR", str(tmp_path / "cache"))
    raw = tmp_path / "image.raw"
    raw.write_bytes(b"raw-image-a")
    return str(raw)


def test_float32_cache_is_bit_exact(cache_source):
    image = np.random.default_rng(18).uniform(-0.2, 8, (9, 13, 3)).astype(np.float32)
    assert not np.array_equal(image, image.astype(np.float16).astype(np.float32))
    disk.save(cache_source, "stage-a", image)
    restored = disk.load(cache_source, "stage-a")
    assert restored.dtype == np.float32
    np.testing.assert_array_equal(restored, image)
    assert disk.load(cache_source, "stage-b") is None


def test_source_replacement_invalidates_cache(cache_source):
    disk.save(cache_source, "stage", np.ones((2, 3, 3), np.float32))
    Path(cache_source).write_bytes(b"raw-image-b")
    assert disk.load(cache_source, "stage") is None


@pytest.mark.parametrize("payload", [b"RADC1\n", b"RADC2\n\x02\x00\x00\x00\x03\x00\x00\x00truncated"])
def test_legacy_or_corrupt_cache_is_never_promoted(cache_source, payload):
    target = disk._cache_dir() / f"{disk._key(cache_source, 'stage')}.radc"
    target.write_bytes(payload)
    assert disk.load(cache_source, "stage") is None
    assert not target.exists()


def test_canonical_artifact_skips_decode_and_inference_on_export(cache_source, monkeypatch):
    monkeypatch.setattr(artifacts, "denoise_tag", lambda strength, *, decode_variant: f"{strength}:{decode_variant}")
    image = np.ones((3, 5, 3), np.float32)
    calls = []

    def denoise(src, *, strength, progress_callback=None):
        calls.append(strength)
        if progress_callback is not None:
            progress_callback(1, 1)
        return src + np.float32(0.1234567)

    preview = artifacts.resolve_denoised_source(cache_source, 0.5, source=image, denoise=denoise)

    def no_decode(raw_path):
        pytest.fail("cache hit should skip RAW decode")

    exported = artifacts.resolve_denoised_source(cache_source, 0.5, decode=no_decode, denoise=denoise)
    np.testing.assert_array_equal(exported, preview)
    assert calls == [0.5]
    artifacts.resolve_denoised_source(cache_source, 0.5, source=image, denoise=denoise, decode_variant=identity.DECODE_PRELOAD)
    assert calls == [0.5, 0.5]


def test_cancellation_after_last_tile_never_saves(cache_source, monkeypatch):
    monkeypatch.setattr(artifacts, "denoise_tag", lambda *a, **kw: "stage")
    cancelled = [False]

    def denoise(src, **kwargs):
        cancelled[0] = True
        return src

    with pytest.raises(PipelineAborted):
        artifacts.resolve_denoised_source(cache_source, 1.0, source=np.ones((2, 3, 3), np.float32), denoise=denoise, should_abort=lambda: cancelled[0])
    assert disk.load(cache_source, "stage") is None


@pytest.mark.parametrize("changed", ["source", "model"])
def test_identity_change_during_inference_never_publishes(cache_source, monkeypatch, changed):
    policy = ["model-a"]
    monkeypatch.setattr(artifacts, "denoise_tag", lambda *a, **kw: policy[0])

    def denoise(src, **kwargs):
        if changed == "source":
            Path(cache_source).write_bytes(b"replacement")
        else:
            policy[0] = "model-b"
        return src

    with pytest.raises(PipelineAborted):
        artifacts.resolve_denoised_source(cache_source, 1.0, source=np.ones((2, 3, 3), np.float32), denoise=denoise)
    assert not list(disk._cache_dir().glob("*.radc"))


def test_real_pipeline_identity_is_available_and_parameter_sensitive(tmp_path, monkeypatch):
    # Exercise actual source dependency names; a typo silently disables reuse.
    from raw_alchemy.onnx import denoiser
    model = tmp_path / "weights.onnx"
    model.write_bytes(b"model-a")
    monkeypatch.setattr(denoiser, "_find_model", lambda name: str(model))
    tag = identity.denoise_tag(0.5)
    assert tag is not None
    assert tag != identity.denoise_tag(0.6)
    assert tag != identity.denoise_tag(0.5, decode_variant=identity.DECODE_PRELOAD)
    monkeypatch.setenv("RAWALCHEMY_COREML_DENOISE", "cpu")
    assert tag != identity.denoise_tag(0.5)
    monkeypatch.delenv("RAWALCHEMY_COREML_DENOISE")
    model.write_bytes(b"model-b")
    assert tag != identity.denoise_tag(0.5)