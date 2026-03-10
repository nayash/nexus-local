import asyncio
import flet as ft
from src.ui.styles import ColorPalette
from src.ui.app_layout import AppLayout
from src.ui.startup_view import StartupView
from src.core.startup import StartupResult
from src.core.user_settings import get_setting

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
        page.data["feature_readiness"] = result.feature_readiness or {}
        
        # Initialize Notification Infrastructure
        from src.ui.components.notification_box import NotificationOverlay
        from src.ui.managers.notification_manager import NotificationManager
        
        notification_overlay = NotificationOverlay(page)
        NotificationManager.set_overlay(notification_overlay)
        
        app = AppLayout(page)
        app.expand = True 
        
        # Use Stack to layer notifications on TOP of the app
        main_stack = ft.Stack(
            controls=[
                app,
                notification_overlay # Top layer
            ],
            expand=True
        )
        
        page.add(main_stack)
        page.update()

    # Initialize StartupView
    print(f'startupView adding')
    
    start_view = StartupView(page, on_success=on_startup_success, is_dev=False)
    start_view.expand = True 
    
    page.add(start_view)
    print(f'startupView added')

    # FIX: Await sleep (async), but run update synchronously
    await asyncio.sleep(0.1) 
    page.update()  # <--- REMOVED 'await' here


def run_app():
    ft.app(target=main)

if __name__ == "__main__":
    run_app()
