import flet as ft
from components.sidebar import Sidebar
from components.chat_view import ChatView
from components.settings_view import SettingsView
from styles import ColorPalette

class AppLayout(ft.Row):
    def __init__(self, page: ft.Page):
        super().__init__()
        self.app_page = page
        self.expand = True
        self.spacing = 0
        
        self.sidebar = Sidebar(
            page, 
            on_settings_click=self.show_settings, 
            on_new_chat_click=self.handle_new_chat,
            on_chat_selected=self.handle_chat_history_click
        )
        
        # Pass callback to refresh sidebar when chat title changes
        self.chat_view = ChatView(on_update=lambda: self.sidebar.refresh_chats())
        self.settings_view = SettingsView(on_back_click=self.show_chat)

        # Initially show Chat View
        self.active_view = self.chat_view

        self.controls = [
            self.sidebar,
            ft.VerticalDivider(width=1, color=ColorPalette.BORDER),
            self.active_view
        ]

    def show_settings(self, e=None):
        self.controls[-1] = self.settings_view
        self.update()

    def show_chat(self, e=None):
        self.active_view = self.chat_view # Ensure correct reference
        self.controls[-1] = self.chat_view
        self.update()

    def handle_new_chat(self):
        """Resets the chat view for a fresh conversation."""
        self.chat_view.start_new_chat()
        self.show_chat()
    
    def handle_chat_history_click(self, chat_id):
        """Loads a previous chat."""
        self.chat_view.load_chat(chat_id)
        self.show_chat()
