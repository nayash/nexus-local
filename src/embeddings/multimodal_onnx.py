import os
from typing import Iterable, List, Optional

import numpy as np

from src.core.config import Config

_EMBEDDER_SINGLETON = None
_EMBEDDER_ERROR: Optional[Exception] = None


def _resolve_existing_path(base_dir: str, candidates: Iterable[str]) -> str:
    for candidate in candidates:
        path = os.path.join(base_dir, candidate)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of the expected files were found in {base_dir}: {list(candidates)}")


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return array / norms


class MultimodalOnnxEmbedder:
    def __init__(self, model_dir: str, device: str = "cuda", providers: Optional[list] = None):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime-gpu is not installed") from exc

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is not installed") from exc

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Multimodal model directory not found: {model_dir}")

        self.model_dir = model_dir
        self.device = device
        self.ort = ort
        self.providers = providers or list(Config.ORT_PROVIDERS)
        self.active_providers = list(self.providers)

        tokenizer_dir = model_dir
        tokenizer_subdir = os.path.join(model_dir, "tokenizer")
        if os.path.isdir(tokenizer_subdir):
            tokenizer_dir = tokenizer_subdir

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)

        text_model_path = _resolve_existing_path(
            model_dir,
            ("text_encoder.onnx", "text_model.onnx", "model_text.onnx"),
        )
        vision_model_path = _resolve_existing_path(
            model_dir,
            ("vision_encoder.onnx", "image_encoder.onnx", "vision_model.onnx", "model_vision.onnx"),
        )

        self.text_session = ort.InferenceSession(text_model_path, providers=self.providers)
        self.vision_session = ort.InferenceSession(vision_model_path, providers=self.providers)

        if self.device.lower() == "cuda":
            desired = "CUDAExecutionProvider"
            text_providers = set(self.text_session.get_providers())
            vision_providers = set(self.vision_session.get_providers())
            if desired not in text_providers or desired not in vision_providers:
                cpu_providers = ["CPUExecutionProvider"]
                print("⚠️ CUDAExecutionProvider is unavailable for multimodal embeddings. Falling back to CPUExecutionProvider.")
                self.text_session = ort.InferenceSession(text_model_path, providers=cpu_providers)
                self.vision_session = ort.InferenceSession(vision_model_path, providers=cpu_providers)
                self.device = "cpu"
                self.active_providers = cpu_providers

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="np",
        )
        outputs = self.text_session.run(
            None,
            self._prepare_text_inputs(encoded),
        )
        pooled = self._pool_text_outputs(outputs, encoded.get("attention_mask"))
        normalized = _l2_normalize(pooled.astype(np.float32))
        return normalized.tolist()

    def embed_image(self, pil_image) -> List[float]:
        return self.embed_images([pil_image])[0]

    def embed_images(self, images: List) -> List[List[float]]:
        if not images:
            return []

        batch = np.stack([self._preprocess_image(image) for image in images], axis=0).astype(np.float32)
        outputs = self.vision_session.run(
            None,
            self._prepare_vision_inputs(batch),
        )
        pooled = self._pool_vision_outputs(outputs)
        normalized = _l2_normalize(pooled.astype(np.float32))
        return normalized.tolist()

    def _prepare_text_inputs(self, encoded) -> dict:
        feed = {}
        batch_size, seq_len = encoded["input_ids"].shape
        for input_meta in self.text_session.get_inputs():
            name = input_meta.name
            dtype = np.int64 if "int64" in input_meta.type else np.int32
            lowered = name.lower()

            if name in encoded:
                feed[name] = np.asarray(encoded[name], dtype=dtype)
            elif lowered == "attention_mask":
                feed[name] = np.asarray(encoded.get("attention_mask"), dtype=dtype)
            elif lowered == "token_type_ids":
                feed[name] = np.zeros((batch_size, seq_len), dtype=dtype)
            elif lowered == "position_ids":
                feed[name] = np.broadcast_to(np.arange(seq_len, dtype=dtype), (batch_size, seq_len))
            else:
                raise KeyError(f"Unsupported text input tensor: {name}")
        return feed

    def _prepare_vision_inputs(self, batch: np.ndarray) -> dict:
        feed = {}
        for input_meta in self.vision_session.get_inputs():
            name = input_meta.name
            if "pixel" in name.lower() or input_meta.type.startswith("tensor(float"):
                feed[name] = batch
            else:
                raise KeyError(f"Unsupported vision input tensor: {name}")
        return feed

    def _pool_text_outputs(self, outputs: List[np.ndarray], attention_mask) -> np.ndarray:
        if not outputs:
            raise RuntimeError("Text encoder returned no outputs")

        output = next((item for item in outputs if isinstance(item, np.ndarray)), None)
        if output is None:
            raise RuntimeError("Text encoder produced no ndarray outputs")

        if output.ndim == 2:
            return output

        if output.ndim == 3:
            if attention_mask is not None:
                indices = np.maximum(attention_mask.sum(axis=1) - 1, 0)
                batch_indices = np.arange(output.shape[0])
                return output[batch_indices, indices]
            return output[:, 0, :]

        return output.reshape(output.shape[0], -1)

    def _pool_vision_outputs(self, outputs: List[np.ndarray]) -> np.ndarray:
        if not outputs:
            raise RuntimeError("Vision encoder returned no outputs")

        output = next((item for item in outputs if isinstance(item, np.ndarray)), None)
        if output is None:
            raise RuntimeError("Vision encoder produced no ndarray outputs")

        if output.ndim == 2:
            return output
        if output.ndim == 3:
            return output[:, 0, :]
        return output.reshape(output.shape[0], -1)

    def _preprocess_image(self, image) -> np.ndarray:
        from PIL import Image

        if not isinstance(image, Image.Image):
            raise TypeError("embed_image expects a PIL.Image.Image instance")

        image = image.convert("RGB")
        target = 224
        width, height = image.size
        scale = target / min(width, height)
        resized = image.resize((max(int(round(width * scale)), target), max(int(round(height * scale)), target)))

        left = max((resized.width - target) // 2, 0)
        top = max((resized.height - target) // 2, 0)
        cropped = resized.crop((left, top, left + target, top + target))

        array = np.asarray(cropped).astype(np.float32) / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        array = (array - mean) / std
        array = np.transpose(array, (2, 0, 1))
        return array


def get_multimodal_embedder(force_refresh: bool = False) -> Optional[MultimodalOnnxEmbedder]:
    global _EMBEDDER_SINGLETON, _EMBEDDER_ERROR

    if force_refresh:
        _EMBEDDER_SINGLETON = None
        _EMBEDDER_ERROR = None

    if not Config.MULTIMODAL_EMBEDDINGS_ENABLED:
        return None

    if _EMBEDDER_SINGLETON is not None:
        return _EMBEDDER_SINGLETON

    if _EMBEDDER_ERROR is not None:
        return None

    try:
        _EMBEDDER_SINGLETON = MultimodalOnnxEmbedder(
            model_dir=Config.MULTIMODAL_MODEL_DIR,
            device=Config.EMBEDDING_DEVICE,
            providers=list(Config.ORT_PROVIDERS),
        )
        return _EMBEDDER_SINGLETON
    except Exception as exc:
        _EMBEDDER_ERROR = exc
        print(f"⚠️ Multimodal embedder unavailable: {exc}")
        return None
