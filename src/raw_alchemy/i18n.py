
import os
import json

class Translator:
    def __init__(self):
        self.config_file = os.path.expanduser('~/.raw_alchemy_config.json')
        self.current_lang = self._load_language()
        self.translations = {}
        self._load_translations()

    def _load_translations(self):
        """Load translations from JSON files"""
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        locales_dir = os.path.join(current_dir, 'locales')
        
        # Load each language file
        for lang_code in ['en', 'zh']:
            lang_file = os.path.join(locales_dir, f'{lang_code}.json')
            try:
                if os.path.exists(lang_file):
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        self.translations[lang_code] = json.load(f)
                else:
                    print(f"Warning: Translation file not found: {lang_file}")
                    self.translations[lang_code] = {}
            except Exception as e:
                print(f"Error loading translation file {lang_file}: {e}")
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
