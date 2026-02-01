import flet as ft
from styles import ColorPalette, TextStyles
import asyncio
from src.ui.managers.notification_manager import NotificationManager

class SettingsView(ft.Container):
    def __init__(self, on_back_click):
        super().__init__()
        self.on_back_click = on_back_click
        self.expand = True
        self.bgcolor = ColorPalette.BG_PRIMARY
        self.padding = 40
        self.content = self.build_content()

    def build_content(self):
        
        # Confirmation Dialog for clearing DB
        self.clear_confirm_dialog = ft.AlertDialog(
            title=ft.Text("Confirm Deletion"),
            content=ft.Text("Are you sure you want to permanently delete ALL indexed data in the vector database? This action cannot be undone."),
            actions=[
                ft.TextButton("Cancel", on_click=self.close_dialog),
                ft.ElevatedButton("Delete All", bgcolor=ColorPalette.ERROR, color=ft.Colors.WHITE, on_click=self.confirm_clear_database),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        
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
            self.page.overlay.append(self.clear_confirm_dialog)
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

    async def run_ingestion(self, path):
        # Show specific loading message
        NotificationManager.info(f"Starting ingestion for: {path}...")
        
        # Run in background to avoid freezing UI
        # In a real app, use threading or async task properly.
        # Flet often handles async handlers.
        import threading
        from src.rag.ingestion import ingest_path
        
        async def task():
            try:
                success, msg, _ = ingest_path(path)
                if success:
                    NotificationManager.success(msg)
                else:
                    NotificationManager.error(msg)
            except Exception as ex:
                NotificationManager.error(f"Error: {str(ex)}")
        
        threading.Thread(target=task, daemon=True).start()
    


    async def clear_database(self, e):
        """Triggers the safety confirmation prompt."""
        print("Clear database called")
        self.clear_confirm_dialog.open = True
        self.page.update()

    def close_dialog(self, e):
        self.clear_confirm_dialog.open = False
        self.page.update()

    async def confirm_clear_database(self, e):
        """Executes the actual deletion after confirmation."""
        # 1. Close dialog
        self.clear_confirm_dialog.open = False
        self.page.update()
        
        # 2. Show loading snack
        # 2. Show loading snack
        NotificationManager.info("Clearing Vector Database...")
        
        try:
            from src.rag.storage import clear_all_tables
            # Use threading to keep UI snappy if it takes a while
            try:
                print('calling clear_all_data asyncio')
                await asyncio.to_thread(clear_all_tables)
                print('clear_all_data asyncio completed')
                
                NotificationManager.success("Vector database cleared successfully!")
            except Exception as ex:
                print(f"Error clearing DB: {ex}")
                NotificationManager.error(f"Failed to clear database: {str(ex)}")
            
        except ImportError as ex:
            NotificationManager.error("Critical Error: Storage module not found.")

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
