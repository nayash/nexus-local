
import flet as ft
import asyncio
from src.ui.styles import ColorPalette, TextStyles

class NotificationCard(ft.Container):
    """
    A single toast notification card.
    """
    def __init__(self, message: str, color: str, icon: str, on_dismiss=None):
        super().__init__()
        self.on_dismiss = on_dismiss
        self.margin = ft.margin.only(bottom=10)
        self.padding = ft.padding.all(15)
        self.border_radius = 8
        self.bgcolor = ColorPalette.BG_SECONDARY
        self.border = ft.border.only(left=ft.BorderSide(5, color))
        self.shadow = ft.BoxShadow(
            spread_radius=1,
            blur_radius=10,
            color=ft.Colors.with_opacity(0.3, "#000000"),
            offset=ft.Offset(0, 4)
        )
        self.width = 350
        # self.offset = ft.transform.Offset(0, -0.5) # Removed due to Flet version conflict
        self.animate_offset = None # ft.Animation(300, ft.AnimationCurve.EASE_OUT_CUBIC)
        self.opacity = 0
        self.animate_opacity = ft.Animation(300, ft.AnimationCurve.EASE_IN)
        
        self.content = ft.Row(
            controls=[
                ft.Icon(icon, color=color, size=20),
                ft.Text(message, color=ColorPalette.TEXT_PRIMARY, size=14, selectable=True, width=260),
                ft.IconButton(
                    icon=ft.Icons.CLOSE, 
                    icon_size=14, 
                    icon_color=ColorPalette.TEXT_SECONDARY,
                    on_click=lambda e: self.dismiss()
                )
            ],
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10
        )

    def did_mount(self):
        # Trigger entrance animation
        # self.offset = ft.transform.Offset(0, 0)
        self.opacity = 1
        self.update()
        
    def dismiss(self):
        if self.on_dismiss:
            self.on_dismiss(self)

class NotificationOverlay(ft.Container):
    """
    Overlay container to hold notification cards.
    Place this at the top of a Stack in the main page.
    """
    def __init__(self, page: ft.Page):
        super().__init__()
        self.main_page = page
        self.cards_column = ft.Column(
            alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=ft.CrossAxisAlignment.END,
            spacing=0,
        )
        # Position: Top Right
        self.padding = 20
        self.content = self.cards_column
        self.right = 0
        self.top = 0
        # Ensure clicks pass through empty areas
        # self.ignore_interactions = True # Flet doesn't have this on Container easily, but Stack order helps.
        # Actually, Container doesn't block clicks if transparent, but content does.
    
    def add_notification(self, message: str, type: str = "info", duration: int = 4):
        color = ColorPalette.ACCENT
        icon = ft.Icons.INFO_OUTLINE
        
        if type == "success":
            color = ColorPalette.SUCCESS # Need to ensure SUCCESS is in palette or use hardcoded
            icon = ft.Icons.CHECK_CIRCLE_OUTLINE
            if not hasattr(ColorPalette, "SUCCESS"): color = "green"
        elif type == "error":
            color = ColorPalette.ERROR
            icon = ft.Icons.ERROR_OUTLINE
        elif type == "warning":
            color = "orange"
            icon = ft.Icons.WARNING_AMBER_ROUNDED
            
        card = NotificationCard(message, color, icon, on_dismiss=self.remove_card)
        self.cards_column.controls.insert(0, card) # Add to top
        self.main_page.update()
        
        # Auto-dismiss
        if duration > 0:
            asyncio.create_task(self._auto_dismiss(card, duration))
            
    async def _auto_dismiss(self, card, duration):
        await asyncio.sleep(duration)
        self.remove_card(card)
        
    def remove_card(self, card):
        if card in self.cards_column.controls:
            # Animate exit?
            # card.opacity = 0
            # card.update()
            # await asyncio.sleep(0.3)
            try:
                self.cards_column.controls.remove(card)
                self.main_page.update()
            except:
                pass 
