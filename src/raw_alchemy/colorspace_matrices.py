"""
Camera→working RGB matrix derivation (no rawpy.postprocess needed).

Port of darktable's ``dt_colorspaces_conversion_matrices_rgb()`` from
``src/common/colorspaces.c`` (which itself is a port of dcraw's
``cam_xyz_coeff()``).

The derived matrix is numerically equivalent to the old
``postprocess(raw) -> postprocess(ProPhoto) -> lstsq`` approach (within
~0.5% per cell, sub-pixel on the resulting image), but ~1000× faster: ~1 ms
vs ~500-3000 ms per RAW.

Usage:
    from raw_alchemy.colorspace_matrices import cam_to_prophoto_matrix

    with rawpy.imread(path) as raw:
        xyz_to_cam = np.array(raw.rgb_xyz_matrix, dtype=np.float64)  # (4, 3)

    cam_to_pro = cam_to_prophoto_matrix(xyz_to_cam)  # (3, 3)
"""

from __future__ import annotations

import numpy as np


# sRGB(D65) → XYZ(D65) — exactly the matrix darktable uses (constants from
# colorspaces.c so we match its numerical behaviour bit-for-bit).
_RGB_TO_XYZ_D65 = np.array([
    [0.412453, 0.357580, 0.180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227],
], dtype=np.float64)

# ProPhoto-RGB(D50) → XYZ(D50). Bruce Lindbloom canonical constants.
_PRO_TO_XYZ_D50 = np.array([
    [0.7976749, 0.1351917, 0.0313534],
    [0.2880402, 0.7118741, 0.0000857],
    [0.0000000, 0.0000000, 0.8252100],
], dtype=np.float64)

# Bradford chromatic adaptation D65 → D50.
_D65_TO_D50_BRADFORD = np.array([
    [ 1.0478112,  0.0228866, -0.0501270],
    [ 0.0295424,  0.9904844, -0.0170491],
    [-0.0092345,  0.0150436,  0.7521316],
], dtype=np.float64)

# Pre-computed ProPhoto(D65) → XYZ(D65). rawpy's rgb_xyz_matrix is XYZ_D65 →
# cam, so the working-space input also needs to live in D65.
_D50_TO_D65 = np.linalg.inv(_D65_TO_D50_BRADFORD)
PRO_TO_XYZ_D65 = _D50_TO_D65 @ _PRO_TO_XYZ_D50
SRGB_TO_XYZ_D65 = _RGB_TO_XYZ_D65


def _pseudoinverse(M: np.ndarray) -> np.ndarray:
    """Port of darktable's ``dt_colorspaces_pseudoinverse()``.

    Takes a (size, 3) matrix and returns its (size, 3) pseudoinverse.
    Functionally equivalent to ``np.linalg.pinv(M).T`` but uses the exact
    Gauss-Jordan procedure from colorspaces.c:2277 so we match darktable
    bit-for-bit.
    """
    size = M.shape[0]
    work = np.zeros((3, 6))
    for i in range(3):
        work[i, i + 3] = 1.0
        for j in range(3):
            for k in range(size):
                work[i, j] += M[k, i] * M[k, j]
    for i in range(3):
        num = work[i, i]
        work[i] /= num
        for k in range(3):
            if k == i:
                continue
            num = work[k, i]
            work[k] -= work[i] * num
    out = np.zeros((size, 3))
    for i in range(size):
        for j in range(3):
            for k in range(3):
                out[i, j] += work[j, k + 3] * M[i, k]
    return out


def cam_to_working_matrix(
    xyz_to_cam_4x3: np.ndarray,
    working_rgb_to_xyz_d65: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the CAM→working RGB matrix the dcraw / darktable way.

    Args:
        xyz_to_cam_4x3: rawpy's ``rgb_xyz_matrix`` (shape (4, 3) float). For
            3-color sensors the 4th row is all-zero.
        working_rgb_to_xyz_d65: (3, 3) matrix mapping the working colour
            space (in D65) to XYZ_D65. Use :data:`PRO_TO_XYZ_D65` for
            ProPhoto (after D50→D65 adaptation) or :data:`SRGB_TO_XYZ_D65`
            for sRGB.

    Returns:
        Tuple ``(cam_to_rgb, daylight_mul)`` where ``cam_to_rgb`` is a
        (3, 3) float64 matrix and ``daylight_mul`` is a length-3 vector of
        daylight white-balance multipliers (the inverse of each row sum of
        the un-normalised RGB→CAM matrix). For 4-color sensors the G and
        G2 channels are folded together.
    """
    xyz_to_cam_4x3 = np.asarray(xyz_to_cam_4x3, dtype=np.float64)

    # Drop all-zero rows (3-color cameras have a zero 4th row).
    valid = xyz_to_cam_4x3.any(axis=1)
    XYZ_to_CAM = xyz_to_cam_4x3[valid]
    n = XYZ_to_CAM.shape[0]
    if n < 3:
        raise ValueError(
            f"XYZ_to_CAM has only {n} valid rows; need at least 3.")

    # Step 1: RGB(D65) → CAM.
    RGB_to_CAM = XYZ_to_CAM @ working_rgb_to_xyz_d65

    # Step 2: row-normalize so that RGB_to_CAM @ (1,1,1) == (1,...,1).
    # This is the key step we missed when first trying to derive it
    # analytically — it bakes in the daylight WB anchor.
    row_sum = RGB_to_CAM.sum(axis=1, keepdims=True)
    RGB_to_CAM = RGB_to_CAM / row_sum
    daylight_mul = (1.0 / row_sum).flatten()

    # Step 3: pseudoinverse then transpose (per colorspaces.c:2417).
    inv = _pseudoinverse(RGB_to_CAM)
    cam_to_rgb = inv.T  # shape (3, n)

    if n == 4:
        # 4-color sensor (RGBG): fold the second green into G.
        M = cam_to_rgb[:, :3].copy()
        M[:, 1] += cam_to_rgb[:, 3]
        return M, daylight_mul[:3].copy()

    return cam_to_rgb, daylight_mul


def cam_to_prophoto_matrix(xyz_to_cam_4x3: np.ndarray) -> np.ndarray:
    """Convenience wrapper: derive Camera→ProPhoto matrix from rawpy.

    Equivalent to the old two-postprocess + lstsq fit, ~1000× faster.
    """
    M, _ = cam_to_working_matrix(xyz_to_cam_4x3, PRO_TO_XYZ_D65)
    return M
