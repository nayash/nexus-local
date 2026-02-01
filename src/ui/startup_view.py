import asyncio
import flet as ft
from typing import Callable
from styles import ColorPalette, TextStyles
from src.core.startup import StartupManager, StartupResult

class StartupView(ft.Container):
    """
    Self-contained Startup UI component.
    Handles system checks and reports progress.
    """
    def __init__(self, page: ft.Page, on_success: Callable[[StartupResult], None], is_dev: bool = False):
        super().__init__(expand=True)
        self._main_page = page
        self.on_success = on_success
        self.manager = StartupManager(is_dev=is_dev)

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
                    # Fix: invoke async method correctly from lambda
                    on_click=lambda _: asyncio.create_task(self.start_checks()), 
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
        # FIX: Use asyncio.create_task instead of threading.Thread
        asyncio.create_task(self.start_checks())

    def on_startup_update(self, message: str, progress: float, is_error: bool):
        # This runs in the worker thread, but simple property updates + update() 
        # are generally thread-safe in Flet.
        self.status_text.value = message
        self.progress_bar.value = progress
        if is_error:
            self.status_text.color = ColorPalette.ERROR
        self._main_page.update()

    async def start_checks(self):
        """Async wrapper that offloads blocking work to a thread but keeps control flow on main loop."""
        # Reset UI
        self.error_container.visible = False
        self.status_text.visible = True
        self.progress_bar.visible = True
        self.status_text.color = ColorPalette.TEXT_PRIMARY
        self._main_page.update()
        
        # FIX: Run the blocking manager checks in a thread, but AWAIT it.
        # This returns control to the main thread immediately after the background work is done.
        result = await asyncio.to_thread(self.manager.run_checks, self.on_startup_update)
        
        if result.success:
            print("✅ Startup complete, transitioning to main app...")
            # FIX: This now runs on the MAIN thread, so page.clean() in the callback works instantly.
            self.on_success(result)
        else:
            print(f"❌ Startup failed: {result.error_message}")
            self.status_text.visible = False
            self.progress_bar.visible = False
            self.error_container.visible = True
            self.error_msg.value = result.error_message
            self._main_page.update()