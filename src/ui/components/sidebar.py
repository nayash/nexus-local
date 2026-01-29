import flet as ft
from styles import ColorPalette, TextStyles

class Sidebar(ft.Container):
    def __init__(self, page: ft.Page, on_settings_click, on_new_chat_click):
        super().__init__()
        self.app_page = page
        self.on_settings_click = on_settings_click
        self.on_new_chat_click = on_new_chat_click
        
        self.width = 250
        self.bgcolor = ColorPalette.BG_SECONDARY
        self.padding = 10
        self.content = self.build_content()

    def build_content(self):
        return ft.Column(
            controls=[
                ft.Container(
                    content=ft.Text("Nexus Local", style=TextStyles.header_large()),
                    padding=ft.Padding(0, 0, 0, 20)
                ),
                self.create_new_chat_button(),
                ft.Divider(color=ColorPalette.BORDER),
                ft.Text("Recent Chats", style=TextStyles.label_small()),
                ft.Container(
                    content=ft.Column(
                        controls=[
                            self.create_history_item("Project Phoenix Research"),
                            self.create_history_item("React Migration Plan"),
                            self.create_history_item("Q4 Financials"),
                        ],
                        spacing=5
                    ),
                    expand=True  # Fill available vertical space
                ),
                ft.Divider(color=ColorPalette.BORDER),
                self.create_action_button("Clear History", ft.Icons.DELETE_OUTLINE),
                self.create_action_button("Settings", ft.Icons.SETTINGS, on_click=self.on_settings_click),
            ],
            expand=True
        )

    def create_new_chat_button(self):
        return ft.Container(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.ADD, color=ColorPalette.TEXT_PRIMARY),
                    ft.Text("New Chat", color=ColorPalette.TEXT_PRIMARY)
                ],
                alignment=ft.MainAxisAlignment.START,
            ),
            padding=10,
            border_radius=8,
            bgcolor=ColorPalette.ACCENT,
            on_click=lambda _: self.on_new_chat_click(),
            ink=True,
        )

    def create_history_item(self, title):
        return ft.Container(
            content=ft.Text(title, color=ColorPalette.TEXT_SECONDARY, size=13, no_wrap=True),
            padding=10,
            border_radius=5,
            on_hover=lambda e: self.toggle_hover(e),
            ink=True
        )

    def create_action_button(self, text, icon, on_click=None):
        return ft.Container(
            content=ft.Row(
                [
                    ft.Icon(icon, size=18, color=ColorPalette.TEXT_SECONDARY),
                    ft.Text(text, color=ColorPalette.TEXT_SECONDARY, size=14)
                ],
                spacing=10
            ),
            padding=10,
            on_click=on_click,
            ink=True
        )

    def toggle_hover(self, e):
        e.control.bgcolor = ColorPalette.BORDER if e.data == "true" else None
        e.control.update()
