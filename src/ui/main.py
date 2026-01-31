import flet as ft
from styles import ColorPalette
from app_layout import AppLayout
from startup_view import StartupView
from src.core.startup import StartupResult
from src.core.user_settings import get_setting, save_setting

def main(page: ft.Page):
    # Page Configuration
    page.title = "Nexus Local"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = ColorPalette.BG_PRIMARY
    page.padding = 0
    page.window_min_width = 800
    page.window_min_height = 600

    if page.data is None:
        page.data = {}

    def on_startup_success(result: StartupResult):
        # Transition to main app
        page.clean()
        app = AppLayout(page)
        # Store search flag if needed in app state
        print(f'Web search enabled: {result.web_search_enabled}')
        # page.client_storage.set("web_search_enabled", result.web_search_enabled)
        page.data["web_search_enabled"] = result.web_search_enabled
        page.data["model_name"] = get_setting("model_name", "llama3.1")
        page.add(app)

    # Initialize with StartupView
    page.add(StartupView(page, on_success=on_startup_success, is_local=False))

if __name__ == "__main__":
    ft.run(main)
