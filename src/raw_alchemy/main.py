import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from raw_alchemy import i18n
from raw_alchemy.ui.main_window import MainWindow

def main():
    # --- Fix Numba Cache for Frozen Apps ---
    if getattr(sys, 'frozen', False):
        cache_dir = os.path.expanduser('~/.raw_alchemy/numba_cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        os.environ['NUMBA_CACHE_DIR'] = cache_dir

    # 解决部分Windows环境下缩放问题
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"
    
    app = QApplication(sys.argv)
    
    # Load i18n
    i18n.init_i18n()
    
    # Start JIT Warmup in background
    import threading
    from raw_alchemy import math_ops
    warmup_thread = threading.Thread(target=math_ops.warmup, daemon=True)
    warmup_thread.start()
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
