import flet as ft
from components.sidebar import Sidebar
from components.chat_view import ChatView
from components.settings_view import SettingsView
from components.file_viewer_view import FileViewerView
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
        # Also pass reference to show_file_viewer method
        self.chat_view = ChatView(
            on_update=lambda cid: self.sidebar.refresh_chats(cid), 
            page=page,
            on_view_file=self.show_file_viewer
        )
        self.settings_view = SettingsView(on_back_click=self.show_chat)
        self.file_viewer_view = FileViewerView(on_back_click=self.show_chat)

        # Initially show Chat View
        self.active_view = self.chat_view

        self.controls = [
            self.sidebar,
            ft.GestureDetector(
                content=ft.Container(
                    width=10, 
                    bgcolor=ft.Colors.TRANSPARENT,
                    content=ft.Container(
                        bgcolor=ColorPalette.BORDER,
                        width=1,
                    ),
                    alignment=ft.Alignment(0, 0)
                ),
                on_pan_update=self.handle_resize_sidebar,
                mouse_cursor=ft.MouseCursor.RESIZE_LEFT_RIGHT,
                drag_interval=10 # Throttle updates slightly
            ),
            self.active_view
        ]

    def handle_resize_sidebar(self, e: ft.DragUpdateEvent):
        """Resizes the sidebar within 10% to 30% of page width."""
        if not self.page:
            return
            

            
        # Use global_position.x to determine the new width directly
        # This is more accurate than delta accumulation which can lag
        new_width = e.global_position.x
        
        # Constraints
        page_width = self.page.width
        min_width = page_width * 0.10
        max_width = page_width * 0.30
        
        # Apply constraints
        if new_width < min_width:
            new_width = min_width
        elif new_width > max_width:
            new_width = max_width
            
        self.sidebar.width = new_width
        self.sidebar.update()


    def show_settings(self, e=None):
        self.controls[-1] = self.settings_view
        self.update()

    def show_chat(self, e=None):
        self.active_view = self.chat_view # Ensure correct reference
        self.controls[-1] = self.chat_view
        self.update()
    
    def show_file_viewer(self, file_path):
        """Switch to file viewer and load the file"""
        self.file_viewer_view.load_file(file_path)
        self.controls[-1] = self.file_viewer_view
        self.update()

    def handle_new_chat(self):
        """Resets the chat view for a fresh conversation."""
        self.chat_view.start_new_chat()
        self.sidebar.refresh_chats(None)
        self.show_chat()
    
    def handle_chat_history_click(self, chat_id):
        """Loads a previous chat."""
        self.chat_view.load_chat(chat_id)
        self.sidebar.refresh_chats(chat_id)
        self.show_chat()
