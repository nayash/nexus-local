import asyncio
import flet as ft
from styles import ColorPalette
from app_layout import AppLayout
from startup_view import StartupView
from src.core.startup import StartupResult
from src.core.user_settings import get_setting, save_setting

async def main(page: ft.Page):
    # Page Configuration
    page.title = "Nexus Local"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = ColorPalette.BG_PRIMARY
    page.padding = 0
    
    # Set window size defaults
    page.window_min_width = 800
    page.window_min_height = 600
    
    # Initialize data if missing
    if page.data is None:
        page.data = {}

    def on_startup_success(result: StartupResult):
        # Transition to main app
        page.clean()
        
        # Save state
        print(f'Web search enabled: {result.web_search_enabled}')
        page.data["web_search_enabled"] = result.web_search_enabled
        page.data["model_name"] = get_setting("model_name", "llama3.1")
        
        app = AppLayout(page)
        # Ensure the main app expands
        app.expand = True 
        
        page.add(app)
        page.update() # Sync update inside callback

    # Initialize StartupView
    print(f'startupView adding')
    
    start_view = StartupView(page, on_success=on_startup_success, is_local=True)
    start_view.expand = True 
    
    page.add(start_view)
    print(f'startupView added')

    # FIX: Await sleep (async), but run update synchronously
    await asyncio.sleep(0.1) 
    page.update()  # <--- REMOVED 'await' here

if __name__ == "__main__":
    ft.app(target=main)