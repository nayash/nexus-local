import flet as ft
from styles import ColorPalette, TextStyles

class SettingsView(ft.Container):
    def __init__(self, on_back_click):
        super().__init__()
        self.on_back_click = on_back_click
        self.expand = True
        self.bgcolor = ColorPalette.BG_PRIMARY
        self.padding = 40
        self.content = self.build_content()

    def build_content(self):
        return ft.Column(
            controls=[
                ft.Row(
                    [
                        ft.IconButton(
                            icon=ft.Icons.ARROW_BACK,
                            icon_color=ColorPalette.TEXT_PRIMARY,
                            on_click=self.on_back_click
                        ),
                        ft.Text("Settings", style=TextStyles.header_large()),
                    ],
                    spacing=20
                ),
                ft.Divider(color=ColorPalette.BORDER, height=40),
                
                # File Ingestion Section
                ft.Text("Knowledge Base", style=TextStyles.body_normal(), weight=ft.FontWeight.BOLD),
                ft.Text("Manage your local files for RAG ingestion.", style=TextStyles.label_small()),
                ft.Container(height=20),
                
                self.create_setting_card(
                    "Ingestion Directory", 
                    "Choose a folder to index for local search.", 
                    ft.FilledButton("Browse Folder", bgcolor=ColorPalette.ACCENT, color=ColorPalette.TEXT_PRIMARY)
                ),

                ft.Container(height=20),
                
                # Database Management
                ft.Text("Database", style=TextStyles.body_normal(), weight=ft.FontWeight.BOLD),
                ft.Container(height=10),
                 self.create_setting_card(
                    "Clear Vector Database", 
                    "Permanently remove all indexed documents.", 
                    ft.FilledButton("Clear All Data", bgcolor=ColorPalette.ERROR, color=ColorPalette.TEXT_PRIMARY)
                ),
            ],
            scroll=ft.ScrollMode.AUTO
        )

    def create_setting_card(self, title, subtitle, control):
        return ft.Container(
            content=ft.Row(
                [
                    ft.Column(
                        [
                            ft.Text(title, color=ColorPalette.TEXT_PRIMARY, size=16),
                            ft.Text(subtitle, color=ColorPalette.TEXT_SECONDARY, size=12),
                        ],
                        expand=True
                    ),
                    control
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN
            ),
            padding=20,
            border=ft.Border.all(1, ColorPalette.BORDER),
            border_radius=10,
            bgcolor=ColorPalette.BG_SECONDARY
        )
