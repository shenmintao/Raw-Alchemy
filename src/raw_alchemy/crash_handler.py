"""
崩溃处理器和错误拦截器
用于捕获和记录系统级崩溃
集成到现有的 loguru 日志系统中
"""
import sys
import faulthandler
import platform
import traceback
from loguru import logger

class CrashHandler:
    """
    安全版崩溃处理器
    
    移除了导致死锁的手动信号拦截，改用 faulthandler 直接对接日志文件描述符。
    """
    
    def __init__(self):
        from raw_alchemy.logger import get_log_file_path
        self.log_path = get_log_file_path()
        self.installed = False
        self._log_file_handle = None  # 保持文件句柄引用
        
    def install(self):
        """安装崩溃处理器"""
        if self.installed:
            return
            
        self._log_system_info()
        
        # 1. 设置 Python 级别的异常钩子 (处理逻辑错误)
        sys.excepthook = self._exception_hook
        
        # 2. 启用 faulthandler (处理 C 级别崩溃: SIGSEGV, SIGABRT)
        # 关键点：我们打开一个独立的文件句柄给 C 层面使用，绕过 Python logging 锁
        try:
            self._log_file_handle = open(self.log_path, "a", encoding="utf-8")
            
            # 写入分隔符
            self._log_file_handle.write(f"\n{'='*20} FAULTHANDLER ENABLED {'='*20}\n")
            self._log_file_handle.flush()
            
            # 启用 faulthandler，崩溃时它会直接往文件里写堆栈
            faulthandler.enable(file=self._log_file_handle, all_threads=True)
            
            logger.info(f"✅ Crash handler installed safely. Path: {self.log_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to enable faulthandler: {e}")
        
        self.installed = True
    
    def _exception_hook(self, exc_type, exc_value, exc_traceback):
        """Python 层面的异常捕获"""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.critical("="*80)
        logger.critical("❌ UNHANDLED PYTHON EXCEPTION")
        logger.critical("="*80)
        logger.critical(f"Type: {exc_type.__name__}")
        logger.critical(f"Value: {exc_value}")
        logger.critical("Traceback:")
        for line in traceback.format_exception(exc_type, exc_value, exc_traceback):
            logger.critical(line.rstrip())
        logger.critical("="*80)
        
        # 尝试显示 GUI 弹窗 (如果可用)
        self._show_crash_dialog(exc_type.__name__, str(exc_value))
        
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    def _log_system_info(self):
        """记录系统环境"""
        logger.info(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
        logger.info(f"Python: {sys.version}")

    def _show_crash_dialog(self, error_type, error_msg):
        """尝试显示崩溃对话框"""
        try:
            from raw_alchemy.ui.crash_dialog import show_crash_dialog

            show_crash_dialog(self.log_path, error_type, error_msg)
        except Exception as e:
            # 崩溃对话框显示失败（例如 GUI 未初始化），记录警告后继续
            logger.warning(f"⚠️ Failed to show crash dialog: {e}")

# 全局崩溃处理器实例
_crash_handler = None


def install_crash_handler():
    """安装全局崩溃处理器"""
    global _crash_handler
    if _crash_handler is None:
        _crash_handler = CrashHandler()
        _crash_handler.install()
    return _crash_handler


def get_crash_handler():
    """获取崩溃处理器实例"""
    global _crash_handler
    return _crash_handler
