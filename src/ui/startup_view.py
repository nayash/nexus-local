import threading
import flet as ft
from typing import Callable
from styles import ColorPalette, TextStyles
from src.core.startup import StartupManager, StartupResult

class StartupView(ft.Container):
    """
    Self-contained Startup UI component.
    Handles system checks and reports progress.
    """
    def __init__(self, page: ft.Page, on_success: Callable[[StartupResult], None]):
        super().__init__(expand=True)
        self._main_page = page
        self.on_success = on_success
        self.manager = StartupManager()

        # UI Components
        self.status_text = ft.Text("Initializing...", style=TextStyles.body_normal())
        self.progress_bar = ft.ProgressBar(width=400, color=ColorPalette.ACCENT, bgcolor=ColorPalette.BG_SECONDARY, value=0)
        
        self.error_msg = ft.Text(color=ColorPalette.TEXT_SECONDARY, text_align=ft.TextAlign.CENTER)
        self.error_container = ft.Column(
            visible=False,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Text("Startup Failed", color=ColorPalette.ERROR, size=20, weight=ft.FontWeight.BOLD),
                self.error_msg,
                ft.ElevatedButton(
                    "Retry", 
                    on_click=lambda _: self.start_checks(), 
                    bgcolor=ColorPalette.ACCENT, 
                    color=ColorPalette.TEXT_PRIMARY
                )
            ]
        )

        self.content = ft.Column(
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Icon(ft.Icons.ROCKET_LAUNCH, size=80, color=ColorPalette.ACCENT),
                ft.Text("Nexus Local", style=TextStyles.header_large()),
                ft.Container(height=20),
                self.status_text,
                ft.Container(height=10),
                self.progress_bar,
                ft.Container(height=20),
                self.error_container
            ]
        )

    def did_mount(self):
        """Automatically start checks when component is added to page."""
        self.start_checks()

    def on_startup_update(self, message: str, progress: float, is_error: bool):
        self.status_text.value = message
        self.progress_bar.value = progress
        if is_error:
            self.status_text.color = ColorPalette.ERROR
        self._main_page.update()

    def start_checks(self):
        # Reset UI
        self.error_container.visible = False
        self.status_text.visible = True
        self.progress_bar.visible = True
        self.status_text.color = ColorPalette.TEXT_PRIMARY
        self._main_page.update()
        
        # Run in thread
        threading.Thread(target=self.run_startup_logic, daemon=True).start()

    def run_startup_logic(self):
        result = self.manager.run_checks(self.on_startup_update)
        
        if result.success:
            self.on_success(result)
        else:
            # Show error
            self.status_text.visible = False
            self.progress_bar.visible = False
            self.error_container.visible = True
            self.error_msg.value = result.error_message
            self._main_page.update()
