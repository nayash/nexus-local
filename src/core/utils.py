import asyncio
import logging
import json
from typing import Callable, Optional

logger = logging.getLogger(__name__)

async def run_ollama_pull(model_name: str, progress_callback: Optional[Callable[[str], None]] = None) -> str:
    """
    Executes 'ollama pull {model_name}' asynchronously.
    Streams output to the progress_callback if provided.
    """
    # 1. Check if model exists first
    check_command = ["ollama", "show", model_name]
    process_check = await asyncio.create_subprocess_exec(
        *check_command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await process_check.wait()
    
    if process_check.returncode == 0:
        logger.info(f"✅ Model {model_name} already exists. Skipping download.")
        return f"✅ Model {model_name} is ready (Cached)."

    command = ["ollama", "pull", model_name]
    
    try:
        logger.info(f"⬇️ Starting pull for model: {model_name}")
        
        if progress_callback:
            await progress_callback(f"Starting download for {model_name}...")

        # Create subprocess asynchronously
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Determine which stream to read. Ollama often prints progress to stderr.
        # We'll read both concurrently.
        async def read_stream(stream, is_stderr=False):
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded_line = line.decode().strip()
                if decoded_line:
                    logger.info(f"Ollama: {decoded_line}")
                    if progress_callback:
                        # Simple heuristic to clean up progress bars
                        clean_msg = decoded_line.replace("\u001b", "").replace("[?25l", "").replace("[?25h", "")
                        await progress_callback(clean_msg)

        await asyncio.gather(
            read_stream(process.stdout),
            read_stream(process.stderr, is_stderr=True)
        )

        return_code = await process.wait()
        
        if return_code != 0:
            error_msg = f"Ollama pull failed with code {return_code}"
            logger.error(error_msg)
            return f"❌ Failed to download {model_name}."
            
        logger.info(f"✅ Successfully downloaded {model_name}")
        return f"✅ Model {model_name} is ready."
        
    except Exception as e:
        logger.error(f"Execution error: {e}")
        return f"❌ System error during pull: {str(e)}"