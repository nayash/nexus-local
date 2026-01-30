import flet as ft
from styles import ColorPalette, TextStyles
from src.core.config import Config

class ChatView(ft.Container):
    def __init__(self):
        super().__init__()
        self.expand = True
        self.bgcolor = ColorPalette.BG_PRIMARY
        self.padding = 0  # Padding handling inside
        self.content = self.build_content()

    def build_content(self):
        self.chat_history = ft.ListView(
            expand=True,
            spacing=20,
            padding=20,
            auto_scroll=True
        )
        
        # Add some dummy messages for UI visualization
        self.add_message("Hello! I am Nexus. How can I help you today?", is_user=False)
        self.add_message("Can you summarize the 'Project Phoenix' PDF?", is_user=True)
        self.add_message("Certainly. Parsing local files... The 'Project Phoenix' document outlines the Q3 migration strategy...", is_user=False)

        self.input_field = ft.TextField(
            hint_text="Ask anything...",
            hint_style=ft.TextStyle(color=ColorPalette.TEXT_SECONDARY),
            border_color=ColorPalette.BORDER,
            bgcolor=ColorPalette.BG_SECONDARY,
            color=ColorPalette.TEXT_PRIMARY,
            multiline=True,
            min_lines=1,
            max_lines=5,
            expand=True,
            border_radius=20,
            content_padding=15
        )

        model_dropdown = ft.Dropdown(
            options=[ft.dropdown.Option(model) for model in Config.SUPPORTED_MODELS],
            value=Config.SUPPORTED_MODELS[0] if Config.SUPPORTED_MODELS else None,
            width=150,
            text_style=ft.TextStyle(color=ColorPalette.TEXT_PRIMARY, size=12),
            bgcolor=ColorPalette.BG_SECONDARY,
            border_color=ColorPalette.BORDER,
            border_radius=10,
        )

        return ft.Column(
            controls=[
                self.chat_history,
                ft.Container(
                    content=ft.Column([
                        ft.Row(
                            controls=[
                                ft.IconButton(ft.Icons.ATTACH_FILE, icon_color=ColorPalette.TEXT_SECONDARY, tooltip="Attach File"),
                                self.input_field,
                                ft.IconButton(ft.Icons.SEND_ROUNDED, icon_color=ColorPalette.ACCENT, tooltip="Send"),
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                        ),
                        ft.Row(
                            controls=[
                                ft.Container(expand=True), # Spacer
                                model_dropdown
                            ],
                            alignment=ft.MainAxisAlignment.END
                        )
                    ]),
                    padding=20,
                    bgcolor=ColorPalette.BG_PRIMARY,
                    border=ft.Border(top=ft.BorderSide(1, ColorPalette.BORDER))
                )
            ],
            expand=True,
            spacing=0
        )

    def add_message(self, text, is_user):
        alignment = ft.MainAxisAlignment.END if is_user else ft.MainAxisAlignment.START
        bg_color = ColorPalette.ACCENT if is_user else ColorPalette.BG_SECONDARY
        text_color = ColorPalette.TEXT_PRIMARY if is_user else ColorPalette.TEXT_PRIMARY # Both white for now
        
        avatar = ft.CircleAvatar(
            content=ft.Text("U" if is_user else "N"),
            bgcolor=ColorPalette.BORDER,
            radius=16
        )

        message_bubble = ft.Container(
            content=ft.Text(text, color=text_color, size=14),
            bgcolor=bg_color,
            padding=15,
            border_radius=ft.BorderRadius(
                top_left=15, top_right=15, 
                bottom_left=0 if is_user else 15, 
                bottom_right=15 if is_user else 0
            ),
            width=None, # Auto width
        )

        row_controls = [message_bubble]
        if not is_user:
            row_controls.insert(0, avatar)
        
        self.chat_history.controls.append(
            ft.Row(
                controls=row_controls,
                alignment=alignment,
                vertical_alignment=ft.CrossAxisAlignment.START,
            )
        )
