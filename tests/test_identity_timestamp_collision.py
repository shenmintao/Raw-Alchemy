"""Same-size replacements must invalidate identity even with colliding stat times."""
import os

import pytest

from raw_alchemy.pipeline import stage_identity as identity


@pytest.mark.parametrize("identify", [identity.file_digest, identity.source_identity])
def test_rewrite_with_identical_metadata_changes_content_identity(tmp_path, monkeypatch, identify):
    raw = tmp_path / "same-size.raw"
    raw.write_bytes(b"old source")
    old_stat = raw.stat()
    real_stat = os.stat

    def colliding_stat(path, *args, **kwargs):
        if os.fspath(path) == str(raw):
            return old_stat
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(identity.os, "stat", colliding_stat)
    before = identify(raw)
    raw.write_bytes(b"new source")
    assert identify(raw) != before