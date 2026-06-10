# AI Culling Research

This spike evaluates local image-culling assistance for Raw Alchemy. It does
not add product code. The goal is to identify a low-risk path for ranking
sharpness, closed eyes, and duplicate groups using deterministic signals first,
then optional ONNX models.

## Summary

Recommended first implementation:

1. Decode or reuse the existing 2-4MP proxy preview.
2. Compute deterministic focus and duplicate signals immediately:
   Laplacian variance, perceptual hash, color histogram, exposure clipping.
3. Add optional face/eye analysis only when a local landmark model is bundled.
4. Add embedding-based grouping later, after model-size and license review.

This keeps culling fast without adding network calls or a heavyweight model to
the first release.

## Local Measurements

Measured on this workspace using Python 3.12, OpenCV 4.13.0, and synthetic RGB
arrays. Timings are wall-clock medians after warmup.

| Signal | Input | Median | P95 | Notes |
| --- | ---: | ---: | ---: | --- |
| Laplacian focus score | 24MP resized to 1.7MP | 28.34 ms | 31.28 ms | Good first-pass blur score. |
| Laplacian focus score | 3MP | 28.68 ms | 34.28 ms | Proxy-sized cost is stable. |
| dHash duplicate key | 24MP to 9x8 | 46.26 ms | 49.19 ms | Resize dominates; batch with proxy to reduce cost. |
| 16x16x16 RGB histogram | 3MP | 2.04 ms | 2.63 ms | Cheap similarity and exposure color signal. |
| OpenCV Haar face + eye scan | 3MP | 208.49 ms | 215.55 ms | Too slow/noisy as the primary closed-eye path. |

The deterministic pass is viable for interactive background culling. Haar face
and eye detection should be treated as a fallback or removed once a landmark
model is selected.

## Candidate Signals

### Sharpness

Use Laplacian variance on the proxy luminance image. Normalize within a burst or
folder because absolute values vary by scene, ISO noise, resize scale, and lens.

Suggested stored fields:

```json
{
  "focus_score": 183.2,
  "focus_percentile_in_group": 0.83,
  "likely_blurry": false
}
```

OpenCV's Laplacian operator is a standard edge/second-derivative primitive:
https://docs.opencv.org/4.x/d5/db5/tutorial_laplace_operator.html

### Duplicate Groups

Start with a two-stage grouping pass:

1. dHash or pHash bucket for near-identical frames.
2. Histogram cosine distance inside each bucket to split false matches.

Later, replace or augment this with image embeddings. DINOv2 is a strong
candidate for visual-only embeddings because it is designed to produce general
visual features without task-specific fine-tuning:
https://arxiv.org/abs/2304.07193

CLIP/OpenCLIP-style embeddings are useful when text tags or semantic search are
planned, but they are heavier and less directly aligned with duplicate grouping:
https://arxiv.org/abs/2103.00020

### Closed Eyes

Do not ship Haar cascades as the main closed-eye detector. The local benchmark
shows about 200 ms on a 3MP proxy, and Haar eye hits are brittle for glasses,
profile faces, and low light.

Preferred route:

1. Use a face landmark model with eye landmarks or eye-blink blendshapes.
2. Compute eye-aspect-ratio or use the model's blink coefficients.
3. Score closed-eye risk only for frames with confident face detections.

MediaPipe Face Landmarker is a reference candidate because it exposes face
landmarks and optional blendshape outputs:
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

Before bundling it, verify model license, Windows packaging size, and whether it
can run through the existing ONNX Runtime dependency. If not, keep the first
release to focus/duplicate grouping and leave closed-eye detection behind a
disabled feature flag.

### Aesthetic Ranking

Defer aesthetic scoring. CLIP aesthetic predictors can work for rough ranking,
but they introduce subjective taste, model licensing, and calibration problems.
Raw Alchemy should first expose objective assistive signals and let users make
the final pick.

## Proposed Architecture

Add a future `culling/` package with:

```text
culling/
  analyzer.py          deterministic signals and orchestration
  duplicate.py         hashes, histogram distance, burst grouping
  focus.py             focus metrics
  face.py              optional landmark/blink adapter
  models.py            model availability and ONNX Runtime sessions
```

Processing should run on existing decoded proxies, not full RAW data. Results
belong in sidecar metadata so the gallery can restore them without recomputing:

```json
{
  "version": 1,
  "params": {},
  "marked": false,
  "culling": {
    "focus_score": 183.2,
    "duplicate_group": "20260610-00042",
    "duplicate_rank": 1,
    "closed_eye_risk": null
  }
}
```

## UI Draft

Gallery:

- Add small non-blocking badges: sharp, soft, duplicate, eye warning.
- Sort/filter by culling score, but never auto-delete or auto-hide files.
- Show duplicate groups as collapsible stacks with the highest-ranked frame
  first.

Inspector:

- Add a read-only culling section with focus percentile, duplicate group size,
  and closed-eye confidence when available.
- Add "mark best in group" and "mark all except best" commands only after the
  user expands a duplicate group.

Background work:

- Trigger culling analysis after proxy generation.
- Use a cancellable queue with latest-folder priority.
- Cache results in sidecars and invalidate only when the source file timestamp
  changes or the analyzer version changes.

## Recommendation

Implement culling in two phases:

1. Deterministic MVP: focus score, dHash/pHash duplicate groups, histogram
   similarity, sidecar persistence, gallery badges.
2. Optional model pass: face landmark closed-eye score and DINOv2/CLIP
   embeddings if model size, license, and ONNX Runtime performance are
   acceptable.

This delivers useful culling assistance with the dependencies already present
and avoids blocking the app on heavyweight inference.
