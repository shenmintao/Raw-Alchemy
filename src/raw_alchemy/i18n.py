
import os
import json
from raw_alchemy.utils import resource_path
from loguru import logger

def _config_file_path():
    explicit_file = os.environ.get('RAW_ALCHEMY_CONFIG_FILE')
    if explicit_file:
        return os.path.abspath(os.path.expanduser(explicit_file))

    config_dir = os.environ.get('RAW_ALCHEMY_CONFIG_DIR')
    if not config_dir:
        config_dir = '~/.raw_alchemy'
    return os.path.join(os.path.abspath(os.path.expanduser(config_dir)), 'config.json')

class Translator:
    def __init__(self):
        self.config_file = _config_file_path()
        self.current_lang = self._load_language()
        self.translations = {}
        self._load_translations()

    def _load_translations(self):
        """Load translations from JSON files"""
        # Use resource_path to handle both dev and PyInstaller environments
        locales_dir = resource_path('locales')
        
        # Load each language file
        for lang_code in ['en', 'zh']:
            lang_file = os.path.join(locales_dir, f'{lang_code}.json')
            try:
                if os.path.exists(lang_file):
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        self.translations[lang_code] = json.load(f)
                else:
                    logger.warning(f"Warning: Translation file not found: {lang_file}")
                    self.translations[lang_code] = {}
            except Exception as e:
                logger.error(f"Error loading translation file {lang_file}: {e}")
                self.translations[lang_code] = {}

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
            logger.error(f"Failed to load language config: {e}")
        return 'en'
    
    def _save_language(self, lang):
        """Save language preference to config file"""
        try:
            config = {}
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            
            config['language'] = lang

            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save language config: {e}")
    
    def load_app_settings(self):
        """Load application settings from config file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('app_settings', {})
        except Exception as e:
            logger.error(f"Failed to load app settings: {e}")
        return {}
    
    def save_app_settings(self, settings):
        """Save application settings to config file"""
        try:
            config = {}
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            
            config['app_settings'] = settings

            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save app settings: {e}")

_translator = Translator()

def tr(key, **kwargs):
    return _translator.get(key, **kwargs)

def set_language(lang):
    _translator.set_language(lang)

def get_current_language():
    return _translator.current_lang

def load_app_settings():
    """Load application settings"""
    return _translator.load_app_settings()

def save_app_settings(settings):
    """Save application settings"""
    _translator.save_app_settings(settings)

def init_i18n():
    """Initialize i18n (create config dir if needed)"""
    config_dir = os.path.dirname(_translator.config_file)
    os.makedirs(config_dir, exist_ok=True)
