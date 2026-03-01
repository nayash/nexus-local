import os
import sys
from urllib.parse import urljoin

import requests

from src.core.config import Config


FILES = (
    "text_encoder.onnx",
    "vision_encoder.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


def download_file(base_url: str, filename: str, target_dir: str):
    url = urljoin(base_url.rstrip("/") + "/", filename)
    response = requests.get(url, timeout=60)
    if response.status_code == 404:
        return False
    response.raise_for_status()
    with open(os.path.join(target_dir, filename), "wb") as handle:
        handle.write(response.content)
    return True


def main():
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = os.getenv("MULTIMODAL_EMBED_MODEL_URL", "")

    if not base_url:
        print("Provide a base URL as an argument or set MULTIMODAL_EMBED_MODEL_URL.")
        return 1

    target_dir = Config.MULTIMODAL_MODEL_DIR
    os.makedirs(target_dir, exist_ok=True)
    downloaded = 0
    for filename in FILES:
        try:
            if download_file(base_url, filename, target_dir):
                print(f"Downloaded {filename}")
                downloaded += 1
        except Exception as exc:
            print(f"Failed to download {filename}: {exc}")

    print(f"Completed. Downloaded {downloaded} file(s) into {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
