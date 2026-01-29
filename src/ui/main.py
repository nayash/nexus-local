import flet as ft
from styles import ColorPalette
from app_layout import AppLayout

def main(page: ft.Page):
    # Page Configuration
    page.title = "Nexus Local"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = ColorPalette.BG_PRIMARY
    page.padding = 0
    page.window_min_width = 800
    page.window_min_height = 600

    # Initialize Layout
    app = AppLayout(page)
    
    page.add(app)

if __name__ == "__main__":
    ft.run(main)
