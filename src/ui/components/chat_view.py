import asyncio
import flet as ft
from styles import ColorPalette, TextStyles
from src.core.config import Config
from src.core.utils import run_ollama_pull
from src.core.user_settings import save_setting, get_setting
from src.ui.agent_interface import run_agent_stream
from src.core.database import ChatRepository
import threading

class ChatView(ft.Container):
    def __init__(self, on_update=None):
        super().__init__()
        self.repo = ChatRepository()
        from src.ui.managers.notification_manager import NotificationManager
        self.current_chat_id = None
        self.on_update = on_update
        
        self.expand = True
        self.bgcolor = ColorPalette.BG_PRIMARY
        self.padding = 0  # Padding handling inside
        self.content = self.build_content()
        self.is_processing = False

    def build_content(self):
        self.chat_history = ft.ListView(
            expand=True,
            spacing=20,
            padding=20,
            auto_scroll=True
        )
        
        # Add some dummy messages if history is empty
        # self.add_message("Hello! I am Nexus. How can I help you today?", is_user=False)
        self.add_message("Nexus is ready. Ask me anything.", is_user=False)

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
            content_padding=15,
            shift_enter=True, # Enter to submit, Shift+Enter for new line
            on_submit=self.trigger_send # Bind Enter key
        )
        print(f'model_name to be made default: {get_setting("model_name", Config.SUPPORTED_MODELS[0] if Config.SUPPORTED_MODELS else None)}')
        model_dropdown = ft.Dropdown(
            options=[ft.dropdown.Option(model) for model in Config.SUPPORTED_MODELS],
            value=get_setting("model_name", Config.SUPPORTED_MODELS[0] if Config.SUPPORTED_MODELS else None),
            width=150,
            text_style=ft.TextStyle(color=ColorPalette.TEXT_PRIMARY, size=12),
            bgcolor=ColorPalette.BG_SECONDARY,
            border_color=ColorPalette.BORDER,
            border_radius=10,
            on_select=self.on_model_change
        )
        
        self.focused_file_indicator = ft.Container(visible=False, height=0)

        return ft.Column(
            controls=[
                self.chat_history,
                # Focused File Indicator
                self.focused_file_indicator,
                ft.Container(
                    content=ft.Column([
                        ft.Row(
                            controls=[
                                ft.IconButton(
                                    ft.Icons.ATTACH_FILE, 
                                    icon_color=ColorPalette.TEXT_SECONDARY, 
                                    tooltip="Attach File (Focus Mode)",
                                    on_click=self.handle_attach_file
                                ),
                                self.input_field,
                                ft.IconButton(
                                    ft.Icons.SEND_ROUNDED, 
                                    icon_color=ColorPalette.ACCENT, 
                                    tooltip="Send",
                                    on_click=self.trigger_send
                                ),
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
    

    def start_new_chat(self):
        self.current_chat_id = None
        self.chat_history.controls.clear()
        self.add_message("Nexus is ready. Ask me anything.", is_user=False)
        self.chat_history.update()

    def load_chat(self, chat_id):
        self.current_chat_id = chat_id
        self.chat_history.controls.clear()
        
        try:
            messages = self.repo.get_chat_history(chat_id)
            if not messages:
                self.add_message("Start of conversation.", is_user=False)
            
            for msg in messages:
                self.add_message(msg["content"], is_user=(msg["role"] == "user"))
        except Exception as e:
            NotificationManager.error(f"Failed to load chat: {e}")
            
        self.chat_history.update()

    def _generate_title_async(self, chat_id, query):
        """
        Background task to generate a short title for the chat using the LLM.
        """
        try:
            print(f"Background: Generating title for chat {chat_id} with query '{query}'")
            from src.agents.nodes import get_cached_llm
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Use a fast model if available, or just the current one
            model_name = get_setting("model_name", "llama3.1")
            print(f"Background: Using model {model_name} for title creation")
            llm = get_cached_llm(model_name, with_tools=False)
            
            prompt = [
                SystemMessage(content="You are a helpful assistant. Generate a short, concise title (max 4 words) for the following query. Do not use quotes."),
                HumanMessage(content=query)
            ]
            
            response = llm.invoke(prompt)
            title = response.content.strip().replace('"', '')
            print(f"Background: Generated title '{title}'")
            
            # Update DB
            self.repo.update_chat_title(chat_id, title)
            print(f"Chat {chat_id} renamed to: {title}")
            
            if self.on_update:
                print("Background: Triggering UI update callback...")
                self.on_update()
            
        except Exception as e:
            print(f"Error generating title: {e}")
    
    async def handle_attach_file(self, e):
        try:
            files = await ft.FilePicker().pick_files(allow_multiple=False)
            if files:
                file_path = files[0].path
                # Show loading notification
                NotificationManager.info(f"Focusing on {files[0].name}...")
                
                # Ingest quietly
                from src.rag.ingestion import ingest_file
                
                success, msg, _ = ingest_file(file_path, table_name="documents")
                
                if success:
                    self.page.data["focused_file"] = file_path
                    self.update_focus_ui(file_path)
                    NotificationManager.success("Focused! Agent will only search this file.")
                else:
                    NotificationManager.error(f"Failed to ingest: {msg}")
        except Exception as ex:
             NotificationManager.error(f"Error in attach: {ex}")
             print(f"Error in attach: {ex}")

    def clear_focus(self, e):
        self.page.data["focused_file"] = None
        self.update_focus_ui(None)
        NotificationManager.info("Focus cleared. Searching all knowledge.")

    def update_focus_ui(self, file_path):
        if file_path:
             import os
             filename = os.path.basename(file_path)
             self.focused_file_indicator.content = ft.Row(
                 [
                     ft.Icon(ft.Icons.ATTACH_FILE, size=16, color=ColorPalette.ACCENT),
                     ft.Text(f"Focused: {filename}", color=ColorPalette.ACCENT, size=12, weight=ft.FontWeight.BOLD),
                     ft.IconButton(ft.Icons.CLOSE, scale=0.5, icon_color=ColorPalette.TEXT_SECONDARY, on_click=self.clear_focus, tooltip="Clear Focus")
                 ],
                 alignment=ft.MainAxisAlignment.CENTER
             )
             self.focused_file_indicator.visible = True
             self.focused_file_indicator.height = 30
             self.focused_file_indicator.bgcolor = ColorPalette.BG_SECONDARY
             self.focused_file_indicator.border = ft.border.all(1, ColorPalette.ACCENT)
        else:
             self.focused_file_indicator.visible = False
             self.focused_file_indicator.height = 0
        
        self.focused_file_indicator.update()
    
    async def trigger_send(self, e):
        """Wrapper to handle async send event"""
        if self.is_processing:
            return
        await self.send_message()

    async def send_message(self):
        query = self.input_field.value
        if not query or not query.strip():
            return

        self.is_processing = True
        self.input_field.value = ""
        self.input_field.update()
        
        # --- Persistence Start ---
        is_new_chat = False
        if not self.current_chat_id:
            print("Creating new chat session...")
            self.current_chat_id = self.repo.create_chat(title="New Chat")
            is_new_chat = True
            print(f"New chat created with ID: {self.current_chat_id}")
        
        # Save User Message
        self.repo.add_message(self.current_chat_id, "user", query)
        print(f"Saved user message to DB for chat {self.current_chat_id}")
        
        # Trigger renaming if new chat
        if is_new_chat:
            print("Starting background title generation thread...")
            threading.Thread(target=self._generate_title_async, args=(self.current_chat_id, query), daemon=True).start()
        # --- Persistence End ---

        # 1. Show User Message
        self.add_message(query, is_user=True)
        
        # 2. Show "Thinking" Placeholder
        thinking_text = "Thinking..."
        bot_message_control = self.add_message(thinking_text, is_user=False)
        
        # 3. Stream Response
        full_response = ""
        try:
            # Prepare context (include focused_file if available in page.data)
            context = {
                "focused_file": self.page.data.get("focused_file")
            }
            
            async for chunk in run_agent_stream(query, [], context):
                full_response += chunk
                # Re-parse and update the entire content of the bubble
                bot_message_control.content = self._parse_message_content(full_response)
                bot_message_control.update()
            
            # --- Persistence of Bot Response ---
            self.repo.add_message(self.current_chat_id, "assistant", full_response)
                
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(f"Error details: {ex}")
            bot_message_control.content = ft.Text(f"Error: {str(ex)}", color=ColorPalette.ERROR)
            bot_message_control.update()
        
        self.is_processing = False

    def _parse_message_content(self, text):
        """
        Parses message text to handle <think>...</think> tags.
        Returns a Column of controls (Markdown + ExpansionTile).
        
        Updated for Streaming: Handles open <think> tags gracefully.
        """
        import re
        
        controls = []
        
        try:
            # Type Safety Check
            if not isinstance(text, str):
                text = str(text)

            # 1. Check for <think> block
            # We only handle ONE think block for now (standard for R1/DeepSeek)
            # Regex to find the start of the block
            start_match = re.search(r'<think>', text)
            
            if start_match:
                # A. Content BEFORE the think block
                pre_text = text[:start_match.start()].strip()
                if pre_text:
                    controls.append(
                        ft.Markdown(
                            pre_text,
                            selectable=True,
                            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                            on_tap_link=lambda e: self.page.launch_url(e.data),
                        )
                    )
                
                # B. The Think Block
                # Check if it is closed
                remainder = text[start_match.end():]
                end_match = re.search(r'</think>', remainder)
                
                if end_match:
                    # Closed thought
                    thought_text = remainder[:end_match.start()].strip()
                    post_text = remainder[end_match.end():].strip()
                    
                    if thought_text:
                        controls.append(
                            ft.Container(
                                content=ft.ExpansionTile(
                                    title=ft.Text("Thought Process", size=12, italic=True, color=ColorPalette.TEXT_SECONDARY),
                                    controls=[
                                        ft.Markdown(
                                            thought_text,
                                            selectable=True,
                                            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                                            code_theme="atom-one-dark"
                                        )
                                    ],
                                    # initially_expanded=False,
                                    bgcolor=ft.Colors.TRANSPARENT,
                                    collapsed_bgcolor=ft.Colors.TRANSPARENT,
                                    text_color=ColorPalette.TEXT_SECONDARY,
                                    controls_padding=ft.padding.only(left=10, bottom=10),
                                ),
                                border=ft.border.all(1, ColorPalette.BORDER),
                                border_radius=10,
                                margin=ft.margin.only(top=5, bottom=5)
                            )
                        )
                    
                    # C. Content AFTER the think block
                    if post_text:
                         controls.append(
                            ft.Markdown(
                                post_text,
                                selectable=True,
                                extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                                on_tap_link=lambda e: self.page.launch_url(e.data),
                            )
                        )
                else:
                    # Open thought (Streaming in progress)
                    thought_text = remainder.strip()
                    if thought_text:
                         controls.append(
                            ft.Container(
                                content=ft.ExpansionTile(
                                    title=ft.Text("Thinking...", size=12, italic=True, color=ColorPalette.ACCENT),
                                    controls=[
                                        ft.Markdown(
                                            thought_text + " █", # Cursor effect
                                            selectable=True,
                                            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                                            code_theme="atom-one-dark"
                                        )
                                    ],
                                    # initially_expanded=True, # Auto-expand while thinking
                                    bgcolor=ft.Colors.TRANSPARENT,
                                    collapsed_bgcolor=ft.Colors.TRANSPARENT,
                                    text_color=ColorPalette.ACCENT,
                                    controls_padding=ft.padding.only(left=10, bottom=10),
                                ),
                                border=ft.border.all(1, ColorPalette.ACCENT), 
                                border_radius=10,
                                margin=ft.margin.only(top=5, bottom=5)
                            )
                        )
            
            else:
                # No think block, just regular markdown
                controls.append(
                    ft.Markdown(
                        text,
                        selectable=True,
                        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                        on_tap_link=lambda e: self.page.launch_url(e.data),
                    )
                )
        except Exception as e:
            print(f"Error parsing message content: {e}")
            # Fallback to simple Text control
            controls = [ft.Text(str(text), color=ColorPalette.TEXT_PRIMARY)]
            
        return ft.Column(controls=controls, spacing=5, tight=True)

    async def on_model_change(self, e):
        """
        Called when the user selects a different model.
        Checks if the model exists locally; if not, pulls it.
        """
        print("Model change event triggered")
        model_name = e.control.value
        if not model_name:
            return
 
        # 1. Notify start
        from src.ui.managers.notification_manager import NotificationManager
        NotificationManager.info(f"Checking/Downloading status of {model_name}...")
 
        # 4. Define the callback to update the Text control directly
        async def progress_callback(msg: str):
            print(f'Progress: {msg}')
            # Optional: Show interval updates if long running, but avoid spamming toast
            if "pulling" in msg and "%" in msg:
                 pass # Too noisy
            else:
                 pass # NotificationManager.info(msg) # Still too noisy maybe
 
        # 5. Run the pull logic
        try:
            # Ensure run_ollama_pull is imported and is async
            result_msg = await run_ollama_pull(model_name, progress_callback)
            print(f'Result of model selection: {result_msg}')
            
            if "ready" in result_msg:
                self.page.data["model_name"] = model_name
                save_setting("model_name", model_name)
                NotificationManager.success(f"Model {model_name} ready.")
            else:
                NotificationManager.info(result_msg)
                
        except Exception as ex:
            NotificationManager.error(f"Error selecting model: {str(ex)}")

    def add_message(self, text, is_user):
        alignment = ft.MainAxisAlignment.END if is_user else ft.MainAxisAlignment.START
        bg_color = ColorPalette.ACCENT if is_user else ColorPalette.BG_SECONDARY
        
        avatar = ft.CircleAvatar(
            content=ft.Text("U" if is_user else "N"),
            bgcolor=ColorPalette.BORDER,
            radius=16
        )
 
        # Bubble content
        if is_user:
            # User messages usually don't have <think> blocks, just use simple text/markdown
            content_control = ft.Text(text, color=ColorPalette.TEXT_PRIMARY, size=14)
        else:
            # Bot messages use the parser
            content_control = self._parse_message_content(text)

        message_bubble = ft.Container(
            content=content_control,
            bgcolor=bg_color,
            padding=15,
            border_radius=ft.BorderRadius(
                top_left=15, top_right=15, 
                bottom_left=0 if is_user else 15, 
                bottom_right=15 if is_user else 0
            ),
            width=600
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
        try:
            self.chat_history.update()
        except RuntimeError:
            pass
        return message_bubble  # Return container to allow updating content later
