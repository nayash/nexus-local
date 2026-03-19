import flet as ft
from src.ui.components.sidebar import Sidebar
from src.ui.components.chat_view import ChatView
from src.ui.components.settings_view import SettingsView
from src.ui.components.file_viewer_view import FileViewerView
from src.ui.styles import ColorPalette

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
        self.chat_view.set_chat_width_provider(self.get_chat_view_width)
        self.chat_view.refresh_message_widths()
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
                drag_interval=40 # Throttle updates slightly
            ),
            self.active_view
        ]

        if hasattr(self.app_page, "on_resize"):
            self.app_page.on_resize = self.handle_page_resize
        elif hasattr(self.app_page, "on_resized"):
            self.app_page.on_resized = self.handle_page_resize

    def get_chat_view_width(self):
        page_width = getattr(self.app_page, "width", 0) or 0
        sidebar_width = getattr(self.sidebar, "width", 250) or 250
        splitter_width = 10
        chat_width = page_width - sidebar_width - splitter_width
        return max(int(chat_width), 320)

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
        self.chat_view.refresh_message_widths()

    def handle_page_resize(self, e):
        if not self.app_page:
            return

        page_width = self.app_page.width
        min_width = page_width * 0.10
        max_width = page_width * 0.30

        if self.sidebar.width < min_width:
            self.sidebar.width = min_width
        elif self.sidebar.width > max_width:
            self.sidebar.width = max_width

        self.chat_view.refresh_message_widths()
        self.update()


    def show_settings(self, e=None):
        self.controls[-1] = self.settings_view
        self.update()

    def show_chat(self, e=None):
        self.active_view = self.chat_view # Ensure correct reference
        self.controls[-1] = self.chat_view
        self.chat_view.on_show()
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
