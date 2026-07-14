import os
import concurrent.futures
from raw_alchemy import config, core

from raw_alchemy.config import SUPPORTED_RAW_EXTENSIONS


def _discover_raw_files(directory: str) -> list[str]:
    """Scan a batch directory once and return deterministic RAW filenames."""
    suffixes = frozenset(ext.lower() for ext in SUPPORTED_RAW_EXTENSIONS)
    with os.scandir(directory) as entries:
        files = [
            entry.name
            for entry in entries
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in suffixes
        ]
    return sorted(files, key=str.casefold)


def _effective_batch_jobs(requested: int, file_count: int) -> int:
    """Bound process count by workload, available RAM and GPU ownership."""
    jobs = max(1, min(int(requested or 1), max(1, int(file_count))))
    try:
        import psutil

        available_mb = int(psutil.virtual_memory().available // (1024 * 1024))
        # Keep one GiB for the OS/UI and budget a conservative native-frame
        # working set per process.
        memory_jobs = max(
            1,
            (max(0, available_mb - 1024)
             // max(1, int(config.BATCH_MEMORY_PER_JOB_MB))),
        )
        jobs = min(jobs, memory_jobs)
    except Exception:
        pass

    try:
        import onnxruntime as ort

        providers = set(ort.get_available_providers())
        if providers.intersection({
            'CUDAExecutionProvider', 'DmlExecutionProvider',
            'ROCMExecutionProvider', 'CoreMLExecutionProvider',
        }):
            jobs = min(jobs, max(1, int(config.GPU_BATCH_MAX_JOBS)))
    except Exception:
        pass
    return max(1, jobs)

def process_path(
    input_path,
    output_path,
    log_space,
    lut_path,
    exposure,
    lens_correct,
    custom_db_path,
    metering_mode,
    jobs,
    logger_func,
    output_format: str = 'tif',
    # New Params
    wb_temp: float = 0.0,
    wb_tint: float = 0.0,
    saturation: float = 1.0,
    contrast: float = 1.0,
    highlight: float = 0.0,
    shadow: float = 0.0,
    # Geometry
    rotation: int = 0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    perspective_corners: tuple | None = None,
    crop: tuple = (0.0, 0.0, 1.0, 1.0),
    # Denoising
    denoise_enabled: bool = False,
    denoise_strength: float = 0.25,
    # Sharpening
    sharpen_strength: float = 0.0,
):
    """
    Orchestrates the processing of a single file or a directory of files.
    Updated to support GUI Progress Bar signaling.
    """
    hdr_output = output_format in {'hdr-heif', 'hdr.heif'}
    normalized_output_format = 'hdr.heif' if hdr_output else output_format
    
    # --- Helper Functions ---
    def log_message(msg):
        """发送普通文本日志"""
        if hasattr(logger_func, 'put'):
            # 如果是 GUI 队列，这里其实可以包一层结构，也可以发纯文本
            # 这里的 logger_func 就是 mp_queue
            logger_func.put(msg) 
        else:
            logger_func(msg)

    def send_signal(data):
        """发送控制信号 (字典)，用于进度条控制"""
        if hasattr(logger_func, 'put'):
            logger_func.put(data)

    output_ext = f".{normalized_output_format}"

    # ============================
    #      Batch Processing
    # ============================
    if os.path.isdir(input_path):
        if not os.path.isdir(output_path):
            error_msg = "For batch processing, the output path must be a directory."
            log_message(f"❌ Error: {error_msg}")
            raise ValueError(error_msg)

        raw_files = _discover_raw_files(input_path)

        if not raw_files:
            log_message("⚠️ No supported RAW files found in the input directory.")
            raise ValueError("No RAW files found.")

        # 【关键修改 1】发送总文件数信号，通知 GUI 初始化进度条
        count = len(raw_files)
        log_message(f"🔍 Found {count} RAW files for parallel processing.")
        send_signal({'total_files': count}) 

        effective_jobs = _effective_batch_jobs(jobs, count)
        if effective_jobs != max(1, int(jobs or 1)):
            log_message(
                f"ℹ️ Batch concurrency limited to {effective_jobs} "
                "to protect RAM/VRAM."
            )

        with concurrent.futures.ProcessPoolExecutor(max_workers=effective_jobs) as executor:
            futures = {
                executor.submit(
                    core.process_image,
                    raw_path=os.path.join(input_path, filename),
                    output_path=os.path.join(output_path, f"{os.path.splitext(filename)[0]}{output_ext}"),
                    log_space=log_space,
                    lut_path=lut_path,
                    exposure=exposure,
                    lens_correct=lens_correct,
                    custom_db_path=custom_db_path,
                    metering_mode=metering_mode,
                    log_queue=logger_func if hasattr(logger_func, 'put') else None,
                    wb_temp=wb_temp,
                    wb_tint=wb_tint,
                    saturation=saturation,
                    contrast=contrast,
                    highlight=highlight,
                    shadow=shadow,
                    rotation=rotation,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    perspective_corners=perspective_corners,
                    crop=crop,
                    denoise_enabled=denoise_enabled,
                    denoise_strength=denoise_strength,
                    sharpen_strength=sharpen_strength,
                    hdr_output=hdr_output,
                ): filename for filename in raw_files
            }
            
            for future in concurrent.futures.as_completed(futures):
                filename = futures[future]
                try:
                    future.result()  # Check for exceptions
                except Exception as exc:
                    log_msg = f"❌ Generated an exception: {exc}"
                    if hasattr(logger_func, 'put'):
                        logger_func.put({'id': filename, 'msg': log_msg})
                    else:
                        log_message(f"[{filename}] {log_msg}")
                finally:
                    # 【关键修改 2】无论成功还是失败，都发送完成信号，让进度条往前走
                    send_signal({'status': 'done'})
        
        log_message("\n🎉 Batch processing complete.")

    # ============================
    #    Single File Processing
    # ============================
    else:
        final_output_path = output_path
        if os.path.isdir(output_path):
            base_name = os.path.basename(input_path)
            file_name, _ = os.path.splitext(base_name)
            final_output_path = os.path.join(output_path, f"{file_name}{output_ext}")
        
        # 单文件也可以看作是 total=1 的批处理，这样进度条能直接满
        send_signal({'total_files': 1})
        
        log_message("⚙️ Processing single file...")
        try:
            core.process_image(
                raw_path=input_path,
                output_path=final_output_path,
                log_space=log_space,
                lut_path=lut_path,
                exposure=exposure,
                lens_correct=lens_correct,
                custom_db_path=custom_db_path,
                metering_mode=metering_mode,
                log_queue=logger_func if hasattr(logger_func, 'put') else None,
                wb_temp=wb_temp,
                wb_tint=wb_tint,
                saturation=saturation,
                contrast=contrast,
                highlight=highlight,
                shadow=shadow,
                rotation=rotation,
                flip_horizontal=flip_horizontal,
                flip_vertical=flip_vertical,
                perspective_corners=perspective_corners,
                crop=crop,
                denoise_enabled=denoise_enabled,
                denoise_strength=denoise_strength,
                sharpen_strength=sharpen_strength,
                hdr_output=hdr_output,
            )
        finally:
            # 发送完成信号
            send_signal({'status': 'done'})
            
        log_message("\n🎉 Single file processing complete.")
