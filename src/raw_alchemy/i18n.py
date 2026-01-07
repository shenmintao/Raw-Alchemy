
import os
import json

class Translator:
    def __init__(self):
        self.config_file = os.path.expanduser('~/.raw_alchemy_config.json')
        self.current_lang = self._load_language()
        self.translations = {
            'en': {
                'library': 'Library',
                'open_folder': 'Open Folder',
                'no_image_selected': 'No Image Selected',
                'hold_to_compare': 'Hold to Compare',
                'export_current': 'Export Current',
                'export_all_marked': 'Export All Marked',
                'histogram': 'Histogram',
                'exposure': 'Exposure',
                'auto_exposure': 'Auto Exposure',
                'metering_mode': 'Metering Mode',
                'exposure_ev': 'Exposure EV',
                'color_management': 'Color Management',
                'log_space': 'Log Space',
                'lut': 'LUT',
                'lens_correction': 'Lens Correction',
                'enable_lens_correction': 'Enable Lens Correction',
                'custom_lensfun_db': 'Custom Lensfun Database',
                'adjustments': 'Adjustments',
                'reset_all': 'Reset All',
                'temp': 'Temp',
                'tint': 'Tint',
                'saturation': 'Saturation',
                'contrast': 'Contrast',
                'highlights': 'Highlights',
                'shadows': 'Shadows',
                'editor': 'Editor',
                'select_folder': 'Select Folder',
                'select_lut_folder': 'Select LUT Folder',
                'select_lensfun_db': 'Select Lensfun Database XML',
                'db_loaded': 'Database Loaded',
                'db_load_failed': 'Database Load Failed',
                'db_cleared': 'Database Cleared',
                'unmarked': 'Unmarked',
                'marked': 'Marked',
                'delete_image': 'Delete Image',
                'confirm_delete': '''Are you sure you want to delete {filename}?

This will move the file to the recycle bin.''',
                'export_image': 'Export Image',
                'select_export_format': 'Select Export Format',
                'select_export_folder': 'Select Export Folder',
                'batch_export': 'Batch Export',
                'export_success': 'Export Success',
                'export_failed': 'Export Failed',
                'no_files_marked': 'No Files Marked',
                'please_mark_files': 'Please mark files using the tag button first.',
                'saved_to': 'Saved to {path}',
                'all_exported': 'All marked files exported successfully.',
                'using_custom_db': 'Using custom database: {name}',
                'using_default_db': 'Using default Lensfun database',
                'failed_to_load_db': 'Failed to load database: {error}',
                'error': 'Error',
                'send2trash_error': 'send2trash module not installed. Cannot delete file safely.',
                'delete_failed': 'Delete Failed',
                'compare_showing_original': 'Showing Original',
                'compare_loading': 'Original image is still loading, please wait...',
                'optional_db_path': 'Optional: Path to custom XML database',
                'matrix': 'Matrix',
                'average': 'Average',
                'center_weighted': 'Center-Weighted',
                'highlight_safe': 'Highlight-Safe',
                'hybrid': 'Hybrid',
                'none': 'None',
                'language': 'Language',
                'settings': 'Settings',
                'english': 'English',
                'chinese': 'Chinese',
                'restart_required': 'Restart Required',
                'restart_message': 'Please restart the application to apply language changes.',
                'loading': 'Loading...',
            },
            'zh': {
                'library': '图库',
                'open_folder': '打开文件夹',
                'no_image_selected': '未选择图片',
                'hold_to_compare': '按住对比',
                'export_current': '导出当前',
                'export_all_marked': '导出所有标记',
                'histogram': '直方图',
                'exposure': '曝光',
                'auto_exposure': '自动曝光',
                'metering_mode': '测光模式',
                'exposure_ev': '曝光补偿 EV',
                'color_management': '色彩管理',
                'log_space': 'Log 空间',
                'lut': 'LUT',
                'lens_correction': '镜头校正',
                'enable_lens_correction': '启用镜头校正',
                'custom_lensfun_db': '自定义 Lensfun 数据库',
                'adjustments': '调整',
                'reset_all': '重置所有',
                'temp': '色温',
                'tint': '色调',
                'saturation': '饱和度',
                'contrast': '对比度',
                'highlights': '高光',
                'shadows': '阴影',
                'editor': '编辑器',
                'select_folder': '选择文件夹',
                'select_lut_folder': '选择 LUT 文件夹',
                'select_lensfun_db': '选择 Lensfun 数据库 XML',
                'db_loaded': '数据库已加载',
                'db_load_failed': '数据库加载失败',
                'db_cleared': '数据库已清除',
                'unmarked': '取消标记',
                'marked': '已标记',
                'delete_image': '删除图片',
                'confirm_delete': '''确定要删除 {filename} 吗？

文件将被移至回收站。''',
                'export_image': '导出图片',
                'select_export_format': '选择导出格式',
                'select_export_folder': '选择导出文件夹',
                'batch_export': '批量导出',
                'export_success': '导出成功',
                'export_failed': '导出失败',
                'no_files_marked': '未标记文件',
                'please_mark_files': '请先使用标签按钮标记文件。',
                'saved_to': '已保存至 {path}',
                'all_exported': '所有标记文件已成功导出。',
                'using_custom_db': '使用自定义数据库: {name}',
                'using_default_db': '使用默认 Lensfun 数据库',
                'failed_to_load_db': '加载数据库失败: {error}',
                'error': '错误',
                'send2trash_error': '未安装 send2trash 模块，无法安全删除文件。',
                'delete_failed': '删除失败',
                'compare_showing_original': '显示原图',
                'compare_loading': '原图正在加载中，请稍候...',
                'optional_db_path': '可选：自定义 XML 数据库路径',
                'matrix': '矩阵测光',
                'average': '平均测光',
                'center_weighted': '中央重点',
                'highlight_safe': '高光保护',
                'hybrid': '混合测光',
                'none': '无',
                'language': '语言',
                'settings': '设置',
                'english': '英语',
                'chinese': '中文',
                'restart_required': '需要重启',
                'restart_message': '请重启应用程序以应用语言更改。',
                'loading': '加载中...',
            }
        }

    def get(self, key, **kwargs):
        text = self.translations.get(self.current_lang, {}).get(key, key)
        if kwargs:
            return text.format(**kwargs)
        return text

    def set_language(self, lang):
        if lang in self.translations:
            self.current_lang = lang
            self._save_language(lang)
    
    def _load_language(self):
        """Load language preference from config file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('language', 'en')
        except Exception as e:
            print(f"Failed to load language config: {e}")
        return 'en'
    
    def _save_language(self, lang):
        """Save language preference to config file"""
        try:
            config = {}
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            
            config['language'] = lang
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save language config: {e}")

_translator = Translator()

def tr(key, **kwargs):
    return _translator.get(key, **kwargs)

def set_language(lang):
    _translator.set_language(lang)

def get_current_language():
    return _translator.current_lang
