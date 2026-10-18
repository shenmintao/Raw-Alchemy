"""Colour metadata recovery shared by canonical and neighbour RAW decoders."""
import numpy as np
from loguru import logger

from raw_alchemy.colorspace_matrices import (
    SRGB_TO_XYZ_D65,
    cam_to_working_space_matrix,
    working_rgb_to_xyz_d65,
)


def camera_to_working(raw, raw_path=None):
    """Use cam_xyz, an embedded matrix, or the bundled camera profile.

    DNGs can have a usable embedded profile in ``cmatrix`` while ``cam_xyz``
    is entirely zero (issue #34). A sensor-space identity cannot replace that
    profile. Keep the normal matrix path numerically unchanged.
    """
    try:
        return cam_to_working_space_matrix(raw.rgb_xyz_matrix)
    except ValueError:
        pass

    matrix = np.asarray(getattr(raw, 'color_matrix', []), dtype=np.float64)
    if (matrix.shape == (3, 4) and getattr(raw, 'num_colors', 3) == 3
            and np.isfinite(matrix).all() and not matrix[:, 3].any()
            and np.linalg.matrix_rank(matrix[:, :3]) == 3):
        logger.warning("RAW cam_xyz is unavailable; using LibRaw's embedded colour matrix")
        return np.linalg.solve(
            working_rgb_to_xyz_d65(), SRGB_TO_XYZ_D65 @ matrix[:, :3],
        )
    # HE unpack is provided by LibRaw, but its camera table can lag behind
    # the RawSpeed database shipped by this app (e.g. Nikon Z50 II/Z5 II).
    # Use that same declared profile as the camera's lossless decode path.
    if raw_path is not None and getattr(raw, 'raw_pattern', None) is not None:
        from raw_alchemy.exif import extract_lens_exif
        from raw_alchemy.rawspeed import camera_matrix
        # rawpy does not expose LibRaw's camera make/model. Read the file's
        # EXIF only on this uncommon path; normal decodes need no extra I/O.
        camera, _ = extract_lens_exif(raw_path, None)
        make, model = camera.get('camera_maker'), camera.get('camera_model')
        xyz = camera_matrix(make, model)
        if xyz is not None:
            try:
                matrix = cam_to_working_space_matrix(xyz)
            except ValueError:
                pass
            else:
                logger.warning(f'RAW cam_xyz is unavailable; using the bundled profile for {make} {model}')
                return matrix
    return None


def decode_with_libraw_color(raw):
    """Let LibRaw resolve its profile, then convert linear sRGB to working RGB.

    Pixels are unrotated; the caller applies the saved orientation once.
    LibRaw owns camera white balance on this path, so it must not be applied
    again after the working-space conversion.
    """
    import rawpy
    from raw_alchemy.math_ops import apply_matrix_inplace

    logger.warning("RAW colour matrices are unavailable; using LibRaw's colour processing")
    # Unknown cameras may set LibRaw's raw_color flag, which bypasses its
    # requested output transform. XYZ would then misinterpret sensor RGB as
    # tristimulus values and tint even neutral whites. Use its RGB renderer.
    rgb16 = raw.postprocess(
        gamma=(1, 1), no_auto_bright=True, use_camera_wb=True,
        use_auto_wb=False, output_bps=16, output_color=rawpy.ColorSpace.sRGB,
        user_flip=0, half_size=False, bright=1.0, highlight_mode=2,
    )
    rgb = rgb16.astype(np.float32) / np.float32(65535.0)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("LibRaw colour processing did not return three RGB channels")
    apply_matrix_inplace(rgb, np.linalg.solve(working_rgb_to_xyz_d65(), SRGB_TO_XYZ_D65).astype(np.float32))
    return np.ascontiguousarray(rgb)
