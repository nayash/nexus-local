import asyncio
import os
import base64
import re
import flet as ft
from src.ui.styles import ColorPalette, TextStyles
from src.core.config import Config
from src.core.utils import run_ollama_pull
from src.core.user_settings import save_setting, get_setting
from src.ui.agent_interface import run_agent_stream
from src.core.database import ChatRepository
from src.ui.managers.notification_manager import NotificationManager
import threading


def _feature_ready(page: ft.Page, key: str) -> bool:
    readiness = (page.data or {}).get("feature_readiness") or {}
    if key not in readiness:
        return True
    check = readiness.get(key) or {}
    return bool(check.get("ready", False))

class ChatView(ft.Container):
    def __init__(self, on_update=None, page=None, on_view_file=None):
        super().__init__()
        self.app_page = page  # Store page reference (can't use self.page, it's a Flet property)
        self.on_view_file = on_view_file  # Callback to switch to file viewer
        self.repo = ChatRepository()
        from src.ui.managers.notification_manager import NotificationManager
        self.notification_manager = NotificationManager
        self.current_chat_id = None
        self.on_update = on_update
        
        self.expand = True
        self.bgcolor = ColorPalette.BG_PRIMARY
        self.padding = 0  # Padding handling inside
        self.content = self.build_content()
        self.is_processing = False
        self._processing_chat_id = None
        self._streaming_bot_message_control = None
        self._streaming_response_buffer = ""

    @staticmethod
    def _strip_think_content(text: str) -> str:
        """
        Remove reasoning-tag markup robustly for display/storage.
        - Removes all well-formed <think>...</think> blocks.
        - If <think> is unclosed, drops everything after it.
        - If stray </think> appears, drops only that tag.
        """
        cleaned = text or ""
        if not cleaned:
            return ""

        lower = cleaned.lower()
        out_parts = []
        cursor = 0
        close_tag = "</think>"

        while cursor < len(cleaned):
            open_idx = lower.find("<think", cursor)
            close_idx = lower.find(close_tag, cursor)

            if open_idx == -1 and close_idx == -1:
                out_parts.append(cleaned[cursor:])
                break

            if close_idx != -1 and (open_idx == -1 or close_idx < open_idx):
                out_parts.append(cleaned[cursor:close_idx])
                cursor = close_idx + len(close_tag)
                continue

            out_parts.append(cleaned[cursor:open_idx])
            open_end = lower.find(">", open_idx)
            if open_end == -1:
                break

            close_after_open = lower.find(close_tag, open_end + 1)
            if close_after_open == -1:
                break

            cursor = close_after_open + len(close_tag)

        cleaned = "".join(out_parts)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _clean_assistant_text_for_storage(text: str) -> str:
        """Persist only final user-visible answer text (no reasoning tags)."""
        return ChatView._strip_think_content(text)

    def build_content(self):
        # File Viewer Dialog
        self.file_viewer_dialog = ft.AlertDialog(
            title=ft.Text("File Content"),
            content=ft.Column(
                [
                    ft.Markdown(
                        "Loading...", 
                        selectable=True, 
                        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB
                    )
                ],
                scroll=ft.ScrollMode.AUTO,
                height=400,
                width=600
            ),
            actions=[
                ft.TextButton("Close", on_click=self.close_file_viewer)
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            modal=True,  # Ensure dialog shows as modal overlay
        )
        # self.app_page.dialog = self.file_viewer_dialog # Can't set page.dialog here, need page first.

        self.chat_history = ft.ListView(
            expand=True,
            spacing=20,
            padding=20,
            auto_scroll=True
        )
        
        # Add some dummy messages if history is empty
        # self.add_message("Hello! I am Nexus. How can I help you today?", is_user=False)
        self.add_message("Nexus is ready. Ask me anything. I keep the answers useful and the jokes light.", is_user=False)

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
        print("=== START_NEW_CHAT called ===")
        self.clear_focus(None) # Clear any existing file focus
        self.current_chat_id = None
        print(f"current_chat_id set to: {self.current_chat_id}")
        self.chat_history.controls.clear()
        self.add_message("Nexus is ready. Ask me anything. I keep the answers useful and the jokes light.", is_user=False)
        self._safe_update_chat_history()

    def load_chat(self, chat_id):
        self.current_chat_id = chat_id
        
        try:
            chat = self.repo.get_chat(chat_id)
            focused_file = chat.get("focused_file") if chat else None
            self.app_page.data["focused_file"] = focused_file
            self.update_focus_ui(focused_file)
        except Exception as e:
            NotificationManager.error(f"Failed to load chat: {e}")

        self._render_chat_history(chat_id)

    def on_show(self):
        """
        Called when ChatView becomes visible again.
        Re-sync with DB after background completion if needed.
        """
        if self.current_chat_id:
            self._render_chat_history(self.current_chat_id)

    def _refresh_current_chat_from_db(self):
        """Reload current chat messages without changing focused-file state."""
        if not self.current_chat_id:
            return

        self._render_chat_history(self.current_chat_id)

    def _render_chat_history(self, chat_id: str):
        """Render a chat from DB and append the in-flight placeholder when relevant."""
        self.chat_history.controls.clear()

        try:
            messages = self.repo.get_chat_history(chat_id)
            if not messages:
                self.add_message("Start of conversation.", is_user=False)
            for msg in messages:
                self.add_message(msg["content"], is_user=(msg["role"] == "user"))

            if self.is_processing and self._processing_chat_id == chat_id:
                thinking_text = self._streaming_response_buffer or "Thinking..."
                self._streaming_bot_message_control = self.add_message(thinking_text, is_user=False)
            else:
                self._streaming_bot_message_control = None
        except Exception as e:
            NotificationManager.error(f"Failed to refresh chat: {e}")

        self._safe_update_chat_history()

    def _safe_update_chat_history(self):
        """
        Update chat history only when attached.
        Detached controls can raise runtime errors during navigation.
        """
        try:
            if self.chat_history.page is not None:
                self.chat_history.update()
        except RuntimeError:
            pass

    def _generate_title_async(self, chat_id, query):
        """
        Background task to generate a short title for the chat using the LLM.
        """
        import re  # Import re module for regex operations
        try:
            print(f"Background: Generating title for chat {chat_id} with query '{query}'")
            try:
                from src.agents.nodes import get_cached_llm
                from langchain_core.messages import SystemMessage, HumanMessage
            except ImportError as ie:
                print(f"Background: ERROR importing LLM/LangChain: {ie}")
                return

            print("Background: Imports successful. Getting LLM...")
            
            # Use a fast model if available, or just the current one
            model_name = get_setting("model_name", "llama3.1")
            print(f"Background: Using model {model_name} for title creation")
            llm = get_cached_llm(model_name, with_tools=False)
            print("Background: LLM acquired. Invoking...")
            
            prompt = [
                SystemMessage(content="You are a title generator. Generate ONLY a very short title (3-5 words maximum) that summarizes what the user is asking about. DO NOT answer the question, DO NOT explain anything, ONLY output the title. No quotes, no newlines, just the title."),
                HumanMessage(content=f"Generate a short title for this user query: {query}")
            ]
            
            response = llm.invoke(prompt)
            print(f"Background: Raw title response: '{response.content}'")
            
            try:
                # Post-process: strip newlines, extra spaces, and truncate
                content = response.content or ""
                title_line = content.strip().split('\n')[0]
                if not title_line:
                    title_line = "New Chat"
                
                title = re.sub(r'\s+', ' ', title_line)             # Normalize whitespace
                title = title.replace('"', '').replace("'", "") # Remove quotes
                
                if len(title) > 40:
                    title = title[:37] + "..."
            except Exception as e:
                print(f"Background: Error processing title '{response.content}': {e}")
                title = "New Chat"
                
            print(f"Background: Final title '{title}'")
            
            # Update DB
            self.repo.update_chat_title(chat_id, title)
            print(f"Chat {chat_id} renamed to: {title}")
            
            if self.on_update:
                print("Background: Triggering UI update callback...")
                self.on_update(chat_id)
            
        except Exception as e:
            print(f"Error generating title: {e}")
    
    def _is_file_already_indexed(self, file_path: str) -> bool:
        """Check if this file path is already present in the multimodal registry."""
        try:
            from src.rag.ingestion_multimodal import is_source_indexed_multimodal

            return is_source_indexed_multimodal(file_path)
        except Exception as ex:
            print(f"Error checking if file indexed: {ex}")
            return False  # On error, assume not indexed so we ingest safely

    async def handle_attach_file(self, e):
        if not _feature_ready(self.app_page, "multimodal"):
            NotificationManager.error(
                "Multimodal ingestion is not ready. Run `nexus-local setup --download-onnx --check-multimodal` and retry."
            )
            return
        try:
            files = await ft.FilePicker().pick_files(allow_multiple=False)
            if files:
                file_path = files[0].path
                
                # Skip ingestion if this exact file is already in the vector DB
                if self._is_file_already_indexed(file_path):
                    print(f"File already indexed, skipping ingestion: {file_path}")
                    self.app_page.data["focused_file"] = file_path
                    if self.current_chat_id:
                        self.repo.update_chat_focused_file(self.current_chat_id, file_path)
                    self.update_focus_ui(file_path)
                    NotificationManager.success(f"Focused on {files[0].name}.")
                    return

                NotificationManager.info(f"Indexing and focusing on {files[0].name}...")
                
                # Ingest quietly
                from src.rag.ingestion import ingest_file
                
                success, msg, _ = ingest_file(file_path, strategy="multimodal")
                
                if success:
                    self.app_page.data["focused_file"] = file_path
                    if self.current_chat_id:
                        self.repo.update_chat_focused_file(self.current_chat_id, file_path)
                    self.update_focus_ui(file_path)
                    NotificationManager.success("Focused! Agent will only search this file.")
                else:
                    NotificationManager.error(f"Failed to ingest: {msg}")
        except Exception as ex:
             NotificationManager.error(f"Error in attach: {ex}")
             print(f"Error in attach: {ex}")

    def clear_focus(self, e):
        self.app_page.data["focused_file"] = None
        if self.current_chat_id:
            self.repo.update_chat_focused_file(self.current_chat_id, None)
        self.update_focus_ui(None)
        # NotificationManager.info("Focus cleared. Searching all knowledge.")

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
        self._streaming_response_buffer = ""
        self._streaming_bot_message_control = None
        self.input_field.value = ""
        if self.input_field.page is not None:
            self.input_field.update()
        
        # --- Persistence Start ---
        print(f"\n=== SEND_MESSAGE: current_chat_id at start: {self.current_chat_id} ===")
        is_new_chat = False
        if not self.current_chat_id:
            print(">>> Creating new chat session...")
            self.current_chat_id = self.repo.create_chat(
                title="New Chat",
                focused_file=self.app_page.data.get("focused_file")
            )
            is_new_chat = True
            print(f">>> New chat created with ID: {self.current_chat_id}")
            print(f">>> is_new_chat: {is_new_chat}")
            if self.on_update:
                print(">>> Calling on_update callback")
                self.on_update(self.current_chat_id)
        
        # Save User Message
        self.repo.add_message(self.current_chat_id, "user", query)
        print(f"Saved user message to DB for chat {self.current_chat_id}")
        
        # Trigger renaming on first message of a new chat
        print(f"\n>>> Checking renaming trigger: is_new_chat={is_new_chat}")
        if is_new_chat:
            print(f">>> YES - Starting title generation thread for chat {self.current_chat_id}")
            try:
                thread = threading.Thread(
                    target=self._generate_title_async, 
                    args=(self.current_chat_id, query), 
                    daemon=True
                )
                thread.start()
                print(f">>> Thread started successfully: {thread}")
            except Exception as e:
                print(f">>> ERROR starting title generation thread: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f">>> NO - Skipping renaming (not a new chat)")
        # --- Persistence End ---

        origin_chat_id = self.current_chat_id
        self._processing_chat_id = origin_chat_id

        # 1. Show User Message
        self.add_message(query, is_user=True)
        
        # 2. Show "Thinking" Placeholder
        thinking_text = "Thinking..."
        bot_message_control = self.add_message(thinking_text, is_user=False)
        self._streaming_bot_message_control = bot_message_control
        
        # 3. Stream Response
        full_response = ""
        try:
            # Prepare context (include focused_file if available in page.data)
            print(f'chat_view: focused_file: {self.app_page.data.get("focused_file")}')
            previous_reasoning = self.repo.get_last_assistant_reasoning(origin_chat_id)
            context = {
                "focused_file": self.app_page.data.get("focused_file"),
                "last_assistant_reasoning": previous_reasoning or "",
            }
            
            # Fetch limited history for context (Sliding Window: last 20 messages)
            history = self.repo.get_chat_history(origin_chat_id, limit=20)
            # Pass history[:-1] because the current 'query' is already the last item in history
            # and run_agent_stream appends 'query' to the history it receives.
            async for chunk in run_agent_stream(query, history[:-1], context):
                full_response += chunk
                self._streaming_response_buffer = full_response
                if self.current_chat_id == origin_chat_id:
                    # Re-parse and update the entire content of the bubble only
                    # when the originating chat is currently visible.
                    target_control = self._streaming_bot_message_control or bot_message_control
                    target_control.content = self._parse_message_content(full_response)
                    # Update via parent (chat_history) instead of the bubble directly,
                    # because the bubble may not have a page reference yet if
                    # the initial chat_history.update() in add_message was silently caught.
                    self._safe_update_chat_history()
            
            # --- Persistence of Bot Response ---
            cleaned_response = self._clean_assistant_text_for_storage(full_response)
            self.repo.add_message(
                origin_chat_id,
                "assistant",
                cleaned_response or full_response,
                reasoning_content=(context.get("last_turn_reasoning") or ""),
            )
                
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(f"Error details: {ex}")
            if self.current_chat_id == origin_chat_id:
                target_control = self._streaming_bot_message_control or bot_message_control
                target_control.content = ft.Text(f"Error: {str(ex)}", color=ColorPalette.ERROR)
                self._safe_update_chat_history()
        
        self.is_processing = False
        self._processing_chat_id = None
        self._streaming_bot_message_control = None
        self._streaming_response_buffer = ""

    def _parse_message_content(self, text):
        """
        Parses message text to handle <think>...</think> tags.
        Returns a Column of controls (Markdown + ExpansionTile).
        
        Updated for Streaming: Handles open <think> tags gracefully.
        """
        import re

        def normalize_think_tags(raw_text: str) -> str:
            normalized = raw_text.replace("&lt;think&gt;", "<think>")
            normalized = normalized.replace("&lt;/think&gt;", "</think>")
            return normalized

        def append_rich_content(target_controls, chunk_text):
            plot_pattern = re.compile(r'<nexus-plot(?: mime="([^"]+)")?>(.*?)</nexus-plot>', re.DOTALL)
            open_plot_pattern = re.compile(r'<nexus-plot(?: mime="[^"]*")?>', re.DOTALL)
            cursor = 0

            def append_markdown(markdown_text):
                markdown_text = markdown_text.strip()
                if not markdown_text:
                    return

                target_controls.append(
                    ft.Markdown(
                        markdown_text,
                        selectable=True,
                        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                        on_tap_link=self.handle_link_click,
                    )
                )

            for match in plot_pattern.finditer(chunk_text):
                append_markdown(chunk_text[cursor:match.start()])

                mime_type = match.group(1) or "image/png"
                image_b64 = (match.group(2) or "").strip()
                if mime_type == "image/png" and image_b64:
                    try:
                        target_controls.append(
                            ft.Container(
                                content=ft.Image(
                                    src=base64.b64decode(image_b64),
                                    fit=ft.BoxFit.CONTAIN,
                                    border_radius=10,
                                    width=560,
                                ),
                                margin=ft.margin.only(top=5, bottom=5),
                            )
                        )
                    except Exception as image_error:
                        print(f"Error rendering nexus plot: {image_error}")
                        target_controls.append(
                            ft.Text(
                                "[Plot image could not be rendered.]",
                                color=ColorPalette.TEXT_SECONDARY,
                                italic=True,
                            )
                        )

                cursor = match.end()

            trailing_part = chunk_text[cursor:]
            open_plot_match = open_plot_pattern.search(trailing_part)
            if open_plot_match:
                append_markdown(trailing_part[:open_plot_match.start()])
            else:
                append_markdown(trailing_part)

        controls = []
        
        try:
            # Type Safety Check
            if not isinstance(text, str):
                text = str(text)
            text = normalize_think_tags(text)

            # Guardrail: malformed or repeated think tags should never leak into
            # visible answer text.
            think_open_count = len(re.findall(r"<think\b[^>]*>", text, flags=re.IGNORECASE))
            think_close_count = len(re.findall(r"</think>", text, flags=re.IGNORECASE))
            if think_open_count > 1 or think_close_count > 1 or (think_open_count != think_close_count):
                sanitized = self._strip_think_content(text)
                append_rich_content(controls, sanitized)
                return ft.Column(controls=controls, spacing=5, tight=True)

            # Strip wrapping code fences that some models accidentally add
            # e.g. ```\nThe answer is...\n``` → The answer is...
            stripped = text.strip()
            if stripped.startswith("```") and stripped.endswith("```") and stripped.count("```") == 2:
                # Remove the opening fence (and optional language tag) and the closing fence
                inner = stripped[3:]  # remove opening ```
                newline_pos = inner.find("\n")
                if newline_pos != -1:
                    inner = inner[newline_pos + 1:]  # skip optional language tag line
                if inner.endswith("```"):
                    inner = inner[:-3]
                text = inner.strip()

            # 1. Check for <think> block
            # We only handle ONE think block for now (standard for R1/DeepSeek)
            # Regex to find the start of the block
            start_match = re.search(r'<think\b[^>]*>', text, flags=re.IGNORECASE)
            
            if start_match:
                # A. Content BEFORE the think block
                pre_text = text[:start_match.start()].strip()
                if pre_text:
                    append_rich_content(controls, pre_text)
                
                # B. The Think Block
                # Check if it is closed
                remainder = text[start_match.end():]
                end_match = re.search(r'</think>', remainder, flags=re.IGNORECASE)
                
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
                         append_rich_content(controls, post_text)
                else:
                    # Broken/unclosed think block: keep only pre-think content.
                    # Never render in-progress reasoning text in the main answer.
                    pass
            
            else:
                # No think block, just regular markdown
                append_rich_content(controls, text)
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

        if not _feature_ready(self.app_page, "ollama"):
            NotificationManager.error(
                "Ollama is not ready. Run `nexus-local setup --install-ollama --start-ollama --pull-models`."
            )
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
                self.app_page.data["model_name"] = model_name
                save_setting("model_name", model_name)
                NotificationManager.success(f"Model {model_name} ready.")
            else:
                NotificationManager.info(result_msg)
                
        except Exception as ex:
            self.notification_manager.error(f"Error selecting model: {str(ex)}")

    def handle_link_click(self, e):
        print(f"\n=== HANDLE_LINK_CLICK called ===")
        print(f"Raw URL from event: '{e.data}'")
        from urllib.parse import unquote
        import webbrowser
        url = e.data
        if url.startswith("http"):
            print(f"Web URL detected, launching: {url}")
            webbrowser.open(url)
        else:
            # Local file path - decode URL encoding
            file_path = unquote(url)
            print(f"Local file detected, decoded path: '{file_path}'")
            self.show_file_viewer(file_path)

    def show_file_viewer(self, file_path):
        """Show file content using the full-page viewer"""
        print(f"\n=== SHOW_FILE_VIEWER called with: '{file_path}' ===")
        
        if not self.on_view_file:
            print("ERROR: on_view_file callback is None! Cannot show file.")
            return
        
        print(f"Calling on_view_file callback...")
        self.on_view_file(file_path)
        print("File viewer should be visible now!")

    def close_file_viewer(self, e):
        self.file_viewer_dialog.open = False
        self.app_page.update()

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
        self._safe_update_chat_history()
        return message_bubble  # Return container to allow updating content later
