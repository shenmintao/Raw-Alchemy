"""
崩溃处理器和错误拦截器
用于捕获和记录系统级崩溃
集成到现有的 loguru 日志系统中
"""
import os
import sys
import faulthandler
import platform
import threading
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
        # faulthandler keeps its file open for the whole session. It gets its
        # own file: holding loguru's file open would make the daily rotation
        # fail on Windows (an open file cannot be renamed).
        self.fault_log_path = os.path.join(
            os.path.dirname(self.log_path), "raw_alchemy_fault.log"
        )
        self.installed = False
        self._log_file_handle = None  # 保持文件句柄引用
        self._dialog_shown = False

    def install(self):
        """安装崩溃处理器"""
        if self.installed:
            return

        self._log_system_info()

        # 1. 设置 Python 级别的异常钩子 (处理逻辑错误)
        sys.excepthook = self._exception_hook
        threading.excepthook = self._thread_exception_hook

        # 2. 启用 faulthandler (处理 C 级别崩溃: SIGSEGV, SIGABRT)
        # 关键点：我们打开一个独立的文件句柄给 C 层面使用，绕过 Python logging 锁
        try:
            self._log_file_handle = open(self.fault_log_path, "a", encoding="utf-8")

            # 写入分隔符
            self._log_file_handle.write(f"\n{'='*20} FAULTHANDLER ENABLED {'='*20}\n")
            self._log_file_handle.flush()

            # 启用 faulthandler，崩溃时它会直接往文件里写堆栈
            faulthandler.enable(file=self._log_file_handle, all_threads=True)

            logger.info(
                f"Crash handler installed. Log: {self.log_path}; "
                f"native faults: {self.fault_log_path}"
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to enable faulthandler: {e}")

        self.installed = True

    def _exception_hook(self, exc_type, exc_value, exc_traceback):
        """Python 层面的异常捕获"""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        self._log_exception("UNHANDLED PYTHON EXCEPTION", exc_type, exc_value, exc_traceback)

        # 尝试显示 GUI 弹窗 (如果可用)
        self._show_crash_dialog(exc_type.__name__, str(exc_value))

        # A windowed (frozen) build has no stderr to print to.
        if sys.stderr is not None:
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

    def _thread_exception_hook(self, args):
        """Exceptions escaping Python threads: log only (no GUI off-thread)."""
        if args.exc_type is SystemExit:
            return
        thread_name = args.thread.name if args.thread is not None else "?"
        self._log_exception(
            f"UNHANDLED EXCEPTION IN THREAD {thread_name}",
            args.exc_type, args.exc_value, args.exc_traceback,
        )

    @staticmethod
    def _log_exception(title, exc_type, exc_value, exc_traceback):
        logger.critical("="*80)
        logger.critical(f"❌ {title}")
        logger.critical("="*80)
        logger.critical(f"Type: {exc_type.__name__}")
        logger.critical(f"Value: {exc_value}")
        logger.critical("Traceback:")
        for line in traceback.format_exception(exc_type, exc_value, exc_traceback):
            logger.critical(line.rstrip())
        logger.critical("="*80)

    def _log_system_info(self):
        """记录系统环境"""
        logger.info(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
        logger.info(f"Python: {sys.version}")

    def _show_crash_dialog(self, error_type, error_msg):
        """尝试显示崩溃对话框"""
        # Qt slot exceptions do not end the app; one dialog per session is
        # enough (a failing paint handler must not stack modal dialogs), and
        # widgets may only be created on the GUI thread.
        if self._dialog_shown:
            return
        if threading.current_thread() is not threading.main_thread():
            return  # QApplication lives on the main thread
        try:
            from raw_alchemy.ui.crash_dialog import show_crash_dialog

            self._dialog_shown = True
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
