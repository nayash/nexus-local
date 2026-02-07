import flet as ft
from styles import ColorPalette
import os

class FileViewerView(ft.Container):
    def __init__(self, on_back_click):
        super().__init__()
        self.on_back_click = on_back_click
        self.current_file_path = None
        
        self.expand = True
        self.bgcolor = ColorPalette.BG_PRIMARY
        self.padding = 20
        
        # Header with back button and file name
        self.file_title = ft.Text("", size=18, weight=ft.FontWeight.BOLD, color=ColorPalette.TEXT_PRIMARY)
        
        self.header = ft.Row(
            [
                ft.IconButton(
                    icon=ft.Icons.ARROW_BACK,
                    on_click=on_back_click,
                    icon_color=ColorPalette.TEXT_PRIMARY
                ),
                self.file_title,
            ],
            spacing=10
        )
        
        # Content area
        self.content_markdown = ft.Markdown(
            "No file loaded",
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            expand=True
        )
        
        self.content_container = ft.Container(
            content=ft.Column(
                [self.content_markdown],
                scroll=ft.ScrollMode.AUTO,
                expand=True
            ),
            bgcolor=ColorPalette.BG_SECONDARY,
            border_radius=10,
            padding=20,
            expand=True
        )
        
        self.content = ft.Column(
            [
                self.header,
                self.content_container
            ],
            spacing=15,
            expand=True
        )
    
    def load_file(self, file_path):
        """Load and display a file"""
        self.current_file_path = file_path
        
        if not os.path.exists(file_path):
            self.file_title.value = "Error"
            self.content_markdown.value = f"**File not found:** `{file_path}`"
            # Don't call self.update() here - control not on page yet
            return
        
        try:
            filename = os.path.basename(file_path)
            self.file_title.value = f"📄 {filename}"
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Truncate if too large
            if len(content) > 100000:
                content = content[:100000] + "\n\n... [File too large, truncated] ..."
            
            # Display in code block to preserve formatting
            self.content_markdown.value = f"```\n{content}\n```"
            # Don't call self.update() here - AppLayout will handle it after adding to page
            
        except Exception as ex:
            self.file_title.value = "Error Reading File"
            self.content_markdown.value = f"**Error:** `{str(ex)}`"
            # Don't call self.update() here either
