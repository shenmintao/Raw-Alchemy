"""
统一的日志处理模块
使用 loguru 提供一致的日志接口，支持日志文件输出
"""
from typing import Optional, Any
from loguru import logger
import sys
import os
import tempfile
from pathlib import Path


# 获取日志文件路径
def get_log_file_path() -> str:
    """获取日志文件路径"""
    # 在用户目录下创建日志文件夹
    log_dir = os.environ.get('RAW_ALCHEMY_LOG_DIR') or os.path.expanduser('~/.raw_alchemy/logs')
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "raw_alchemy.log")
        with open(log_path, "a", encoding="utf-8"):
            pass
    except OSError:
        log_dir = os.path.join(tempfile.gettempdir(), 'raw_alchemy', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "raw_alchemy.log")
    return log_path


# 配置全局 loguru logger
logger.remove()  # 移除默认处理器

# 添加控制台输出（仅在有stderr时，如非GUI模式）
if sys.stderr is not None:
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",  # 控制台只显示 INFO 及以上级别，DEBUG 日志只写入文件
        colorize=True
    )

# 添加文件输出（自动轮转，保留最近7天）
logger.add(
    get_log_file_path(),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",
    rotation="1 day",      # 每天轮转
    retention="7 days",    # 保留7天
    compression="zip",     # 压缩旧日志
    encoding="utf-8"
)


class LoguruHandler:
    """
    Loguru 日志处理器包装类
    兼容原有的Logger接口
    """
    
    def __init__(self, log_target: Optional[Any] = None, file_id: Optional[str] = None):
        """
        初始化日志处理器
        
        Args:
            log_target: 日志输出目标（仅支持Queue对象用于多进程）
            file_id: 文件标识符，用于多文件处理时区分日志来源
        """
        self.log_target = log_target
        self.file_id = file_id
    
    def _format_message(self, message: str) -> str:
        """格式化消息，添加文件 ID 前缀"""
        if self.file_id:
            return f"[{self.file_id}] {message}"
        return message
    
    def _output(self, message: str, level: str = "INFO"):
        """输出日志"""
        formatted_msg = self._format_message(message)
        
        # 如果有Queue目标（多进程模式），同时发送到Queue和loguru
        if hasattr(self.log_target, 'put'):
            self.log_target.put({
                'id': self.file_id,
                'msg': message,
                'level': level
            })
        
        # 统一使用loguru输出（会同时输出到控制台和文件）
        if level == "SUCCESS":
            logger.success(formatted_msg)
        elif level == "ERROR":
            logger.error(formatted_msg)
        elif level == "WARNING":
            logger.warning(formatted_msg)
        elif level == "DEBUG":
            logger.debug(formatted_msg)
        else:
            logger.info(formatted_msg)
    
    def log(self, message: str, level: str = "INFO"):
        """发送日志消息（兼容旧接口）"""
        self._output(message, level.upper())
    
    def info(self, message: str):
        """信息级别日志"""
        self._output(message, "INFO")
    
    def error(self, message: str):
        """错误级别日志"""
        self._output(message, "ERROR")
    
    def success(self, message: str):
        """成功级别日志"""
        self._output(message, "SUCCESS")
    
    def warning(self, message: str):
        """警告级别日志"""
        self._output(message, "WARNING")
    
    def debug(self, message: str):
        """调试级别日志"""
        self._output(message, "DEBUG")


def create_logger(log_target: Optional[Any] = None, file_id: Optional[str] = None) -> LoguruHandler:
    """
    工厂函数：创建日志处理器实例
    
    Args:
        log_target: 日志输出目标（仅Queue对象用于多进程）
        file_id: 文件标识符
    
    Returns:
        LoguruHandler 实例
    """
    return LoguruHandler(log_target, file_id)


# 为了向后兼容，保留 Logger 别名
Logger = LoguruHandler
