import flet as ft
from styles import ColorPalette, TextStyles
from src.core.database import ChatRepository

class Sidebar(ft.Container):
    def __init__(self, page: ft.Page, on_settings_click, on_new_chat_click, on_chat_selected):
        super().__init__()
        self.app_page = page
        self.on_settings_click = on_settings_click
        self.on_new_chat_click = on_new_chat_click
        self.on_chat_selected = on_chat_selected
        self.repo = ChatRepository()
        self.active_chat_id = None
        
        self.width = 250
        self.bgcolor = ColorPalette.BG_SECONDARY
        self.padding = 10
        self.content = self.build_content()

    def build_content(self):
        self.chat_list = ft.Column(spacing=5, scroll=ft.ScrollMode.AUTO)
        self.refresh_chats()
        
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
                    content=self.chat_list,
                    expand=True  # Fill available vertical space
                ),
                ft.Divider(color=ColorPalette.BORDER),
                self.create_action_button("Clear History", ft.Icons.DELETE_OUTLINE, on_click=self.handle_clear_history),
                self.create_action_button("Settings", ft.Icons.SETTINGS, on_click=self.on_settings_click),
            ],
            expand=True
        )

    def refresh_chats(self, active_chat_id=None):
        if active_chat_id is not None:
            self.active_chat_id = active_chat_id
            
        self.chat_list.controls.clear()
        chats = self.repo.get_recent_chats()
        for chat in chats:
            self.chat_list.controls.append(
                self.create_history_item(chat["title"], chat["id"])
            )
        try:
             self.chat_list.update()
        except:
             pass

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

    def create_history_item(self, title, chat_id):
        is_active = chat_id == self.active_chat_id
        return ft.Container(
            content=ft.Row(
                controls=[
                    ft.Container(
                        content=ft.Text(
                            title, 
                            color=ColorPalette.TEXT_PRIMARY if is_active else ColorPalette.TEXT_SECONDARY, 
                            size=13, 
                            weight=ft.FontWeight.BOLD if is_active else ft.FontWeight.NORMAL,
                            no_wrap=True, 
                            overflow=ft.TextOverflow.ELLIPSIS
                        ),
                        expand=True,
                        on_click=lambda _: self.on_chat_selected(chat_id),
                    ),
                    ft.IconButton(
                        icon=ft.Icons.DELETE_OUTLINE,
                        icon_color=ColorPalette.TEXT_SECONDARY,
                        icon_size=14,
                        tooltip="Delete",
                        on_click=lambda _: self.delete_individual_chat(chat_id)
                    )
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                spacing=0
            ),
            padding=ft.padding.only(left=10, right=0, top=0, bottom=0),
            border_radius=5,
            bgcolor=ColorPalette.ACCENT_SURFACE if is_active else None,
            on_hover=lambda e: self.toggle_hover(e),
            data=chat_id
        )

    def delete_individual_chat(self, chat_id):
        try:
            self.repo.delete_chat(chat_id)
            self.refresh_chats()
            self.on_new_chat_click()
            # Notification called
            from src.ui.managers.notification_manager import NotificationManager
            NotificationManager.success("Chat deleted")
        except Exception as ex:
            print(f"Error deleting chat: {ex}")

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
    
    def handle_clear_history(self, e):
        """Clears all chat history and resets the UI."""
        try:
            self.repo.clear_all_chats()
            self.refresh_chats()
            self.on_new_chat_click()
            
            # Show confirmation
            from src.ui.managers.notification_manager import NotificationManager
            NotificationManager.success("History cleared successfully")
        except Exception as ex:
            print(f"Error clearing history: {ex}")
            NotificationManager.error(f"Error clearing history: {ex}")

    def toggle_hover(self, e):
        e.control.bgcolor = ColorPalette.BORDER if e.data == "true" else None
        e.control.update()
