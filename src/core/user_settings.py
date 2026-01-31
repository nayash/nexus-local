import json
import os

CONFIG_FILE = "user_settings.json"

def load_settings():
    if not os.path.exists(CONFIG_FILE):
        print(f'Config file {CONFIG_FILE} does not exist.')
        return {}
    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
            print(f'loaded settings: {data}')
            return data
    except Exception:
        return {}

def save_setting(key, value):
    settings = load_settings()
    settings[key] = value
    with open(CONFIG_FILE, "w") as f:
        json.dump(settings, f, indent=4)

def get_setting(key, default=None):
    settings = load_settings()
    print(f'settings for key {key}: {settings.get(key, default)}')
    return settings.get(key, default)