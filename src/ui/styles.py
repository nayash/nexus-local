import flet as ft

class ColorPalette:
    # Premium Dark Mode Palette
    BG_PRIMARY = "#0F1117"  # Very dark blue-ish gray
    BG_SECONDARY = "#1E2029" # Slightly lighter sidebars/containers
    ACCENT = "#3B82F6"      # Premium Blue
    ACCENT_HOVER = "#2563EB"
    TEXT_PRIMARY = "#FFFFFF"
    TEXT_SECONDARY = "#9CA3AF"
    BORDER = "#2D3748"
    ERROR = "#EF4444"
    SUCCESS = "#10B981"
    ACCENT_SURFACE = "#333C4D" # More distinct highlight for active items

class TextStyles:
    @staticmethod
    def header_large():
        return ft.TextStyle(
            size=24,
            weight=ft.FontWeight.BOLD,
            color=ColorPalette.TEXT_PRIMARY,
            font_family="Inter"
        )
    
    @staticmethod
    def body_normal():
        return ft.TextStyle(
            size=14,
            color=ColorPalette.TEXT_PRIMARY,
            font_family="Roboto"
        )

    @staticmethod
    def label_small():
        return ft.TextStyle(
            size=12,
            color=ColorPalette.TEXT_SECONDARY,
            font_family="Roboto"
        )
