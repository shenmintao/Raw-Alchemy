# HDR Gain Map Research

`UPGRADE_PLAN.md` asks for two HDR outputs:

- PQ HEIF: 10-bit BT.2020/PQ HEIF.
- Gain-map JPEG: ISO 21496-1 compatible JPEG with SDR fallback.

Raw Alchemy currently implements the first path only. The `hdr-heif` exporter
adds a `pq_out` pipeline operation and writes a 10-bit HEIF file tagged with
BT.2020 primaries and ST 2084/PQ transfer metadata.

## Gain-Map JPEG Requirements

Ultra HDR / gain-map JPEG is not equivalent to writing a brightened JPEG. A
compatible file needs at least:

- An SDR base JPEG image.
- A gain-map secondary image, normally stored as an MPF auxiliary JPEG image.
- XMP metadata describing gain-map math, min/max boost, gamma, offsets, and
HDR capacity.
- ISO 21496-1 metadata compatibility for readers that expect the standardized
gain-map form.

The Android Ultra HDR documentation describes the format as a backwards
compatible JPEG container: SDR decoders show the base image, HDR-aware decoders
combine the base and gain map. Google `libultrahdr` is the reference codec for
that format.

## Local Dependency Check

Checked in the current Windows development environment:

- `pillow-heif 1.3.0`: available and used for HEIF/PQ output.
- `Pillow 12.2.0`: available for ordinary JPEG writing.
- `pyexiv2 2.15.5`: available for EXIF/XMP metadata writes.
- `ultrahdr`: not installed.

`pillow-heif` does not provide a gain-map JPEG encoder. Pillow can write JPEG
pixels but does not assemble the MPF auxiliary image and gain-map metadata
needed for Ultra HDR / ISO 21496-1.

## Decision For `studio-v0.6.0-pre2`

Do not ship a custom partial encoder. A non-conformant file would be worse than
no feature because Windows Photos, Chrome, Android, and editing tools may treat
it inconsistently.

For this prerelease:

- Keep `hdr-heif` as the only HDR output mode.
- Keep ordinary `jpg` output SDR-only.
- Document gain-map JPEG as not implemented.

## Proposed Implementation Path

1. Add an optional `libultrahdr`-backed encoder when a maintained Python wheel or
   bundled CLI path is available for Windows/macOS/Linux.
2. Feed it an SDR rendition and a linear/PQ HDR rendition from the same op list.
3. Preserve regular JPEG as SDR fallback and add a new explicit export format,
   for example `hdr-jpg`, instead of changing existing `jpg` semantics.
4. Add fixture-level validation by decoding metadata and smoke-test display in
   Chrome and Windows Photos.

References:

- Android Ultra HDR image format: https://developer.android.com/media/platform/hdr-image-format
- Google libultrahdr: https://github.com/google/libultrahdr

