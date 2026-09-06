"""Killable RAW decode boundary on Windows, Linux and macOS.

Only path/mode and small metadata use the pipe; the parent owns pixel storage.
Decoding in the child includes its ONNX lifetime, so it does not spawn again.
"""
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import time

import numpy as np

from raw_alchemy.onnx.isolated_session import _seconds, allocate_shared_memory
from raw_alchemy.pipeline.resources import check_native_memory
from raw_alchemy.pipeline.cancellation import check_cancelled


def _serve_decode(connection, path, mode):
    os.environ['RAWALCHEMY_NATIVE_ISOLATION'] = '0'
    os.environ['RAWALCHEMY_COREML_ISOLATION'] = '0'
    block = None
    try:
        if mode == 'canonical':
            from raw_alchemy.core import _rawpy_decode_to_prophoto
            result = _rawpy_decode_to_prophoto(path)
            metadata = ()
        elif mode == 'preload':
            result, *metadata = decode_cpu(path)
        else:
            raise ValueError('Unknown RAW decoder')
        result = np.ascontiguousarray(result, dtype=np.float32)
        connection.send(('result', result.shape, metadata))
        name = connection.recv()
        block = shared_memory.SharedMemory(name=name)
        np.copyto(np.ndarray(result.shape, dtype=np.float32, buffer=block.buf), result)
        connection.send(('done',))
    except BaseException as exc:
        try:
            connection.send(('error', type(exc).__name__, str(exc)[:2000]))
        except (OSError, EOFError):
            pass
    finally:
        if block is not None:
            block.close()
        connection.close()


def decode_raw(path, mode='canonical', *, worker=_serve_decode):
    context = mp.get_context('spawn')
    parent, child = context.Pipe()
    process = context.Process(target=worker, args=(child, str(path), mode),
                              name='RawAlchemy-Decode', daemon=True)
    block = None
    deadline = time.monotonic() + _seconds('RAWALCHEMY_DECODE_TIMEOUT', 120)

    def receive():
        while True:
            check_native_memory()
            if time.monotonic() >= deadline:
                raise TimeoutError('RAW decoding exceeded its time budget')
            if parent.poll(0.05):
                message = parent.recv()
                if message[0] == 'error':
                    error = MemoryError if message[1] == 'MemoryError' else RuntimeError
                    raise error(f'RAW {message[1]}: {message[2]}')
                return message
            if not process.is_alive():
                raise RuntimeError(f'RAW decoder exited ({process.exitcode})')
    try:
        check_cancelled()
        process.start()
        child.close()
        message = receive()
        if message[0] != 'result':
            raise RuntimeError('Invalid RAW result')
        shape, metadata = message[1:]
        if len(shape) != 3 or shape[2] != 3 or min(shape) <= 0 or shape[0] * shape[1] > 200_000_000:
            raise ValueError('Invalid RAW output dimensions')
        block = allocate_shared_memory(int(np.prod(shape)) * 4)
        parent.send(block.name)
        if receive()[0] != 'done':
            raise RuntimeError('Incomplete RAW output')
        result = np.ndarray(shape, dtype=np.float32, buffer=block.buf).copy()
        return (result, *metadata) if mode == 'preload' else result
    finally:
        child.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(0.5)
            if process.is_alive():
                process.kill()
                process.join(0.5)
            if not process.is_alive():
                process.close()
        parent.close()
        if block is not None:
            block.close()
            block.unlink()


def decode_cpu(path: str):
    """LibRaw neighbour preview; its provenance differs from canonical decode.

    White balance and the working-space transform are shared, but demosaic
    algorithms differ. These pixels must not seed the canonical export cache.
    """
    from raw_alchemy.math_ops import apply_matrix_inplace
    import rawpy
    from raw_alchemy.colorspace_matrices import cam_to_working_space_matrix
    from raw_alchemy.onnx.denoiser import _apply_flip
    from raw_alchemy.exif import extract_lens_exif

    with rawpy.imread(path) as raw:
        wb = np.array(raw.camera_whitebalance, dtype=np.float32)
        flip = raw.sizes.flip
        xyz = np.array(raw.rgb_xyz_matrix, dtype=np.float64)
        # Camera-native demosaic only: unit WB, no auto-bright, linear,
        # unflipped — this app owns white balance, colour and orientation.
        cam = raw.postprocess(
            gamma=(1, 1), no_auto_bright=True,
            user_wb=[1.0, 1.0, 1.0, 1.0], output_bps=16,
            output_color=rawpy.ColorSpace.raw,
            user_flip=0, half_size=False, highlight_mode=2,
        )
    rgb = cam.astype(np.float32) / 65535.0
    del cam
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    elif rgb.shape[2] > 3:
        rgb = np.ascontiguousarray(rgb[:, :, :3])

    g = wb[1] if wb[1] > 0 else 1.0
    rgb[:, :, 0] *= wb[0] / g
    rgb[:, :, 2] *= wb[2] / g
    # cam->working matrix, per-pixel (M @ [r,g,b]); numpy keeps it GPU-free.
    m = cam_to_working_space_matrix(xyz).astype(np.float32)
    apply_matrix_inplace(rgb, m)
    rgb = np.ascontiguousarray(_apply_flip(rgb, flip))
    np.clip(rgb, 0.0, 1.0, out=rgb)
    exif_data, exif_metadata = extract_lens_exif(path, None)
    return rgb, exif_data, exif_metadata
