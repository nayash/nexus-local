
from typing import Optional
from src.ui.components.notification_box import NotificationOverlay

class NotificationManager:
    _instance = None
    _overlay: Optional[NotificationOverlay] = None

    @classmethod
    def set_overlay(cls, overlay: NotificationOverlay):
        cls._overlay = overlay

    @classmethod
    def show(cls, message: str, type: str = "info", duration: int = 4):
        if cls._overlay:
            # We must ensure we are on the loop? 
            # Flet operations from background threads without page.update are risky, 
            # but our overlay.add_notification logic calls page.update which is thread-safe in Flet-Async roughly.
            # Best to trust the overlay's internal logic.
            cls._overlay.add_notification(message, type, duration)
        else:
            print(f"[Notifications Not Init] {type.upper()}: {message}")

    @classmethod
    def success(cls, message: str):
        cls.show(message, "success")

    @classmethod
    def error(cls, message: str):
        cls.show(message, "error")

    @classmethod
    def warning(cls, message: str):
        cls.show(message, "warning")

    @classmethod
    def info(cls, message: str):
        cls.show(message, "info")
