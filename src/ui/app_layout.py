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
            on_new_chat_click=self.show_chat
        )
        self.chat_view = ChatView()
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
        self.controls[-1] = self.chat_view
        self.update()
