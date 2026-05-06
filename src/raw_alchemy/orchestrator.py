import os
import concurrent.futures
from raw_alchemy import core

from raw_alchemy.config import SUPPORTED_RAW_EXTENSIONS

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
    saturation: float = 1.25,
    contrast: float = 1.1,
    highlight: float = 0.0,
    shadow: float = 0.0,
    # Geometry
    rotation: int = 0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    crop: tuple = (0.0, 0.0, 1.0, 1.0),
    # Denoising
    denoise_enabled: bool = False,
    # Sharpening
    sharpen_strength: float = 0.0,
):
    """
    Orchestrates the processing of a single file or a directory of files.
    Updated to support GUI Progress Bar signaling.
    """
    
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

    output_ext = f".{output_format}"

    # ============================
    #      Batch Processing
    # ============================
    if os.path.isdir(input_path):
        if not os.path.isdir(output_path):
            error_msg = "For batch processing, the output path must be a directory."
            log_message(f"❌ Error: {error_msg}")
            raise ValueError(error_msg)

        raw_files = []
        for ext in SUPPORTED_RAW_EXTENSIONS:
            raw_files.extend([f for f in os.listdir(input_path) if f.lower().endswith(ext)])

        if not raw_files:
            log_message("⚠️ No supported RAW files found in the input directory.")
            raise ValueError("No RAW files found.")

        # 【关键修改 1】发送总文件数信号，通知 GUI 初始化进度条
        count = len(raw_files)
        log_message(f"🔍 Found {count} RAW files for parallel processing.")
        send_signal({'total_files': count}) 
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
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
                    denoise_enabled=denoise_enabled,
                    sharpen_strength=sharpen_strength,
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
                crop=crop,
                denoise_enabled=denoise_enabled,
                sharpen_strength=sharpen_strength,
            )
        finally:
            # 发送完成信号
            send_signal({'status': 'done'})
            
        log_message("\n🎉 Single file processing complete.")