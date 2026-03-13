import json
import os
import shutil

from src.core.config import Config

CONFIG_FILE = Config.USER_SETTINGS_PATH
LEGACY_CONFIG_FILE = os.path.join(Config.PROJECT_ROOT, "user_settings.json")


def _ensure_config_dir():
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)


def _migrate_legacy_settings():
    if os.path.abspath(LEGACY_CONFIG_FILE) == os.path.abspath(CONFIG_FILE):
        return
    if os.path.exists(CONFIG_FILE):
        return
    if os.path.exists(LEGACY_CONFIG_FILE):
        _ensure_config_dir()
        shutil.copy2(LEGACY_CONFIG_FILE, CONFIG_FILE)

def load_settings():
    _migrate_legacy_settings()
    if not os.path.exists(CONFIG_FILE):
        _ensure_config_dir()
        return {}
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_setting(key, value):
    _ensure_config_dir()
    settings = load_settings()
    settings[key] = value
    with open(CONFIG_FILE, "w") as f:
        json.dump(settings, f, indent=4)

def get_setting(key, default=None):
    settings = load_settings()
    return settings.get(key, default)
