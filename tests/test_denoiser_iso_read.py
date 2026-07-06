"""_read_iso must survive pyexiv2 hard failures via the TIFF fallback walker.

pyexiv2 raises RuntimeError ("Directory Sony2 ... considered invalid") on
Sony-converted DNGs instead of degrading gracefully. The old code swallowed
that and returned the 100.0 default — feeding log10(100) into the v14 FiLM
ISO condition and wrecking the denoise output on high-ISO night shots.
The fallback parses the standard EXIF ISO tag (0x8827) straight out of the
TIFF structure, which is intact in those files.
"""
import struct

import pytest

from raw_alchemy.onnx import denoiser as D


def _build_tiff_with_iso(iso: int, via_exif_ifd: bool = True, bo: str = "<") -> bytes:
    """Assemble a minimal valid TIFF: IFD0 (+optional ExifIFD) carrying ISO."""

    def u16(v):
        return struct.pack(bo + "H", v)

    def u32(v):
        return struct.pack(bo + "I", v)

    header = (b"II" if bo == "<" else b"MM") + u16(42) + u32(8)
    if via_exif_ifd:
        # IFD0 @8: one entry -> ExifIFD pointer @26; ExifIFD @26: ISO entry.
        ifd0 = u16(1) + u16(0x8769) + u16(4) + u32(1) + u32(26) + u32(0)
        exif_ifd = u16(1) + u16(0x8827) + u16(3) + u32(1) + u16(iso) + u16(0) + u32(0)
        return header + ifd0 + exif_ifd
    ifd0 = u16(1) + u16(0x8827) + u16(3) + u32(1) + u16(iso) + u16(0) + u32(0)
    return header + ifd0


@pytest.mark.parametrize("bo", ["<", ">"], ids=["little-endian", "big-endian"])
def test_fallback_reads_iso_from_exif_ifd(tmp_path, bo):
    p = tmp_path / "synthetic.dng"
    p.write_bytes(_build_tiff_with_iso(12800, via_exif_ifd=True, bo=bo))
    assert D._read_iso_tiff_fallback(str(p)) == 12800.0


def test_fallback_reads_iso_from_ifd0(tmp_path):
    p = tmp_path / "synthetic.dng"
    p.write_bytes(_build_tiff_with_iso(8000, via_exif_ifd=False))
    assert D._read_iso_tiff_fallback(str(p)) == 8000.0


def test_fallback_is_safe_on_non_tiff_and_garbage(tmp_path):
    raf = tmp_path / "fake.raf"
    raf.write_bytes(b"FUJIFILMCCD-RAW 0201FF129502")
    assert D._read_iso_tiff_fallback(str(raf)) == 0.0

    truncated = tmp_path / "trunc.dng"
    truncated.write_bytes(b"II\x2a\x00")
    assert D._read_iso_tiff_fallback(str(truncated)) == 0.0

    # Self-referencing IFD chain must not loop forever.
    looped = tmp_path / "loop.dng"
    ifd = struct.pack("<H", 0) + struct.pack("<I", 8)  # 0 entries, next=self
    looped.write_bytes(b"II" + struct.pack("<H", 42) + struct.pack("<I", 8) + ifd)
    assert D._read_iso_tiff_fallback(str(looped)) == 0.0

    assert D._read_iso_tiff_fallback("/no/such/file.dng") == 0.0


def test_read_iso_uses_fallback_when_pyexiv2_raises(tmp_path, monkeypatch):
    """End-to-end: pyexiv2 blowing up must not degrade ISO to the 100 default."""
    p = tmp_path / "sony_converted.dng"
    p.write_bytes(_build_tiff_with_iso(12800))

    import pyexiv2

    def _boom(*a, **k):
        raise RuntimeError("Directory Sony2 with 25665 entries considered invalid; not read.")

    monkeypatch.setattr(pyexiv2, "Image", _boom)
    assert D._read_iso(str(p)) == 12800.0


def test_read_iso_defaults_to_100_when_everything_fails(tmp_path, monkeypatch):
    p = tmp_path / "opaque.raw"
    p.write_bytes(b"not a tiff at all")

    import pyexiv2

    def _boom(*a, **k):
        raise RuntimeError("unreadable")

    monkeypatch.setattr(pyexiv2, "Image", _boom)
    assert D._read_iso(str(p)) == 100.0
