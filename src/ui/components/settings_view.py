import flet as ft
from styles import ColorPalette, TextStyles
import asyncio
from src.ui.managers.notification_manager import NotificationManager

from src.core.watcher_manager import WatcherManager

class SettingsView(ft.Container):
    def __init__(self, on_back_click):
        super().__init__()
        self.on_back_click = on_back_click
        self.expand = True
        self.bgcolor = ColorPalette.BG_PRIMARY
        self.padding = 40
        self.watcher_manager = WatcherManager() # Initialize manager
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
        
        # Watched Paths List
        self.watched_paths_column = ft.Column(spacing=10)
        # self.refresh_watched_paths()  <-- Moved to did_mount

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
                
                # --- Watched Folders Section ---
                ft.Text("Watched Folders", style=TextStyles.body_normal(), weight=ft.FontWeight.BOLD),
                ft.Text("Folders monitored for auto-organization and ingestion.", style=TextStyles.label_small()),
                ft.Container(height=10),
                
                self.create_setting_card(
                    "Add Watched Folder",
                    "Select a folder to watch, organize, and ingest.",
                    ft.FilledButton(
                        "Add Folder",
                        bgcolor=ColorPalette.ACCENT,
                        color=ColorPalette.TEXT_PRIMARY,
                        on_click=self.handle_add_watched_folder
                    )
                ),
                
                ft.Container(height=20),
                ft.Text("Active Watches", style=TextStyles.body_normal(), weight=ft.FontWeight.BOLD),
                self.watched_paths_column,
                
                ft.Divider(color=ColorPalette.BORDER, height=40),

                # File Ingestion Section
                ft.Text("Manual Ingestion", style=TextStyles.body_normal(), weight=ft.FontWeight.BOLD),
                ft.Text("Manually ingest files or folders.", style=TextStyles.label_small()),
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
            self.refresh_watched_paths() # Load data when mounted
            self.page.update()

    def refresh_watched_paths(self):
        """Refreshes the list of watched paths."""
        self.watched_paths_column.controls.clear()
        paths = self.watcher_manager.get_watched_paths()
        
        if not paths:
             self.watched_paths_column.controls.append(
                 ft.Text("No folders currently watched.", style=TextStyles.label_small(), italic=True)
             )
        else:
            for p in paths:
                self.watched_paths_column.controls.append(
                    self.create_watched_path_item(p)
                )
        
        if self.page:
            self.watched_paths_column.update()

    def create_watched_path_item(self, path_data):
        return ft.Container(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.Icons.FOLDER_OPEN, color=ColorPalette.ACCENT),
                    ft.Column(
                        [
                            ft.Text(path_data['path'], style=TextStyles.body_normal(), weight=ft.FontWeight.BOLD),
                            ft.Text(f"Table: {path_data['table_name']}", style=TextStyles.label_small())
                        ],
                        expand=True
                    ),
                    ft.IconButton(
                        ft.Icons.DELETE_OUTLINE, 
                        icon_color=ColorPalette.ERROR,
                        tooltip="Stop Watching",
                        on_click=lambda e: self.handle_stop_watching(path_data['id'])
                    )
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN
            ),
            padding=10,
            border=ft.Border.all(1, ColorPalette.BORDER),
            border_radius=8,
            bgcolor=ColorPalette.BG_SECONDARY
        )

    async def handle_add_watched_folder(self, e):
        path = await ft.FilePicker().get_directory_path()
        if path:
            NotificationManager.info(f"Initializing watcher for: {path}...")
            
            # Run in background to avoid freezing UI logic (though manager calls service which is async/threaded? 
            # Manager uses synchronous ingestion calls currently. So we MUST run in thread.)
            
            def _background_init():
                return self.watcher_manager.initialize_path(path)

            try:
                success, msg = await asyncio.to_thread(_background_init)
                if success:
                    NotificationManager.success(msg)
                    self.refresh_watched_paths()
                else:
                    NotificationManager.error(msg)
            except Exception as ex:
                NotificationManager.error(f"Error: {str(ex)}")

    async def handle_stop_watching(self, path_id):
        success, msg = self.watcher_manager.stop_watching(path_id)
        if success:
             NotificationManager.success(msg)
             self.refresh_watched_paths()
        else:
             NotificationManager.error(msg)

    async def handle_browse_folder(self, e):
        path = await ft.FilePicker().get_directory_path()
        if path:
            await self.run_ingestion(path)

    async def handle_browse_file(self, e):
        files = await ft.FilePicker().pick_files(allow_multiple=False)
        if files:
            file_path = files[0].path
            await self.run_ingestion(file_path)

    async def run_ingestion(self, path):
        # Show specific loading message
        NotificationManager.info(f"Starting ingestion for: {path}...")
        
        # Run blocking ingestion in background thread
        from src.rag.ingestion import ingest_path
        import asyncio
        
        try:
            # Use asyncio.to_thread to run blocking function in background
            success, msg, _ = await asyncio.to_thread(ingest_path, path, "parent")
            if success:
                NotificationManager.success(msg)
            else:
                NotificationManager.error(msg)
        except Exception as ex:
            NotificationManager.error(f"Error: {str(ex)}")
    

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
