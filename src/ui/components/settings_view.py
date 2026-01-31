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
        # We need a reference to the snackbar for status updates
        self.status_snack = ft.SnackBar(content=ft.Text(""), show_close_icon=True)
        
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
                    "Ingest Directory", 
                    "Select a folder to create a new vector table.", 
                    ft.FilledButton(
                        "Browse Folder", 
                        bgcolor=ColorPalette.ACCENT, 
                        color=ColorPalette.TEXT_PRIMARY,
                        on_click=self.handle_browse_folder
                    )
                ),
                
                 ft.Container(height=10),

                 self.create_setting_card(
                    "Ingest File", 
                    "Select a single file to add to 'documents' table.", 
                    ft.FilledButton(
                        "Browse File", 
                        bgcolor=ColorPalette.ACCENT, 
                        color=ColorPalette.TEXT_PRIMARY,
                        on_click=self.handle_browse_file
                    )
                ),
                
                ft.Container(height=20),
                
                # Database Management
                ft.Text("Database", style=TextStyles.body_normal(), weight=ft.FontWeight.BOLD),
                ft.Container(height=10),
                 self.create_setting_card(
                    "Clear Vector Database", 
                    "Permanently remove all indexed documents.", 
                    ft.FilledButton("Clear All Data", bgcolor=ColorPalette.ERROR, color=ColorPalette.TEXT_PRIMARY,
                        on_click=self.clear_database # TODO: Implement this
                    )
                ),
            ],
            scroll=ft.ScrollMode.AUTO
        )

    def did_mount(self):
        # Determine if we can add to overlay
        if self.page:
            self.page.overlay.append(self.status_snack)
            self.page.update()

    async def handle_browse_folder(self, e):
        path = await ft.FilePicker().get_directory_path()
        if path:
            self.run_ingestion(path)

    async def handle_browse_file(self, e):
        files = await ft.FilePicker().pick_files(allow_multiple=False)
        if files:
            file_path = files[0].path
            self.run_ingestion(file_path)

    def run_ingestion(self, path):
        # Show specific loading message
        self.show_snack(f"Starting ingestion for: {path}...", color="blue")
        
        # Run in background to avoid freezing UI
        # In a real app, use threading or async task properly.
        # Flet often handles async handlers.
        import threading
        from src.rag.ingestion import ingest_path
        
        def task():
            try:
                success, msg, _ = ingest_path(path)
                color = "green" if success else "red"
                self.show_snack(msg, color)
            except Exception as ex:
                self.show_snack(f"Error: {str(ex)}", "red")
        
        threading.Thread(target=task, daemon=True).start()

    def show_snack(self, message, color):
        self.status_snack.content = ft.Text(message, color="white")
        self.status_snack.bgcolor = color if color != "blue" else ColorPalette.ACCENT
        self.status_snack.open = True
        self.status_snack.update()

    def clear_database(self, e):
        # Placeholder for clearing DB logic
        self.show_snack("Clear Database not implemented yet (Safety check needed).", "red")

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
