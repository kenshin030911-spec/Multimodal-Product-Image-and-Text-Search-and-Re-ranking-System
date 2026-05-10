"""Transformers CLIP/FashionCLIP runtime helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageEncodingError(ValueError):
    """Raised when a single image cannot be decoded for real encoding."""

    def __init__(self, image_path: Path, reason: str) -> None:
        super().__init__(reason)
        self.image_path = image_path
        self.reason = reason


class TransformersCLIPRuntime:
    """Shared CLIPModel/CLIPProcessor runtime for image and text encoders."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        normalize_embeddings: bool = True,
        model_revision: str | None = None,
    ) -> None:
        torch_module, clip_model_cls, clip_processor_cls = _load_transformers_deps()
        self.torch = torch_module
        self.model_name = model_name
        self.requested_device = device
        self.device = resolve_device(device, torch_module)
        self.normalize_embeddings = normalize_embeddings
        self.model_revision = model_revision

        try:
            self.processor = clip_processor_cls.from_pretrained(
                model_name,
                revision=model_revision,
            )
            self.model = clip_model_cls.from_pretrained(
                model_name,
                revision=model_revision,
            )
        except Exception as exc:
            raise RuntimeError(f"无法加载 Transformers CLIP 模型 {model_name}: {exc}") from exc

        self.model.to(self.device)
        self.model.eval()
        self.torch_dtype = _infer_torch_dtype(self.model)
        self.embedding_dim = _infer_embedding_dim(self.model)

    def encode_images(self, image_paths: Sequence[Path]) -> np.ndarray:
        """Encode RGB images with CLIPModel.get_image_features."""
        if not image_paths:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        images = [_load_rgb_image(image_path) for image_path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = _inputs_to_device(inputs, self.device)

        with self.torch.inference_mode():
            features = self.model.get_image_features(**inputs)
        return _to_numpy_features(features, self.normalize_embeddings)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts with CLIPModel.get_text_features and tokenizer truncation."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        inputs = self.processor(
            text=list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = _inputs_to_device(inputs, self.device)

        with self.torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        return _to_numpy_features(features, self.normalize_embeddings)


def resolve_device(device: str, torch_module: Any) -> str:
    """Resolve auto/cpu/cuda into the concrete torch device string."""
    normalized = device.strip().lower()
    if normalized == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        if not torch_module.cuda.is_available():
            raise ValueError(
                "device=cuda 但当前 torch.cuda.is_available() 为 False。"
                "请安装 CUDA 版 PyTorch，或改用 --device cpu。"
            )
        return "cuda"
    raise ValueError("device 只支持 auto、cpu 或 cuda。")


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows while avoiding division by zero."""
    if vectors.size == 0:
        return vectors.astype(np.float32, copy=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    return (vectors / safe_norms).astype(np.float32, copy=False)


def _load_transformers_deps() -> tuple[Any, Any, Any]:
    """Import heavy dependencies only when a real encoder is explicitly used."""
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError(
            "真实 encoder 需要安装 torch、transformers、safetensors。"
            "请先运行 pip install -r requirements.txt；如果 torch 安装失败，"
            "请使用 PyTorch 官方安装选择器获取适合当前 Windows CPU/CUDA 环境的命令。"
        ) from exc
    return torch, CLIPModel, CLIPProcessor


def _load_rgb_image(image_path: Path) -> Image.Image:
    """Open an image with Pillow and force RGB decode for CLIP."""
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB").copy()
    except (FileNotFoundError, OSError, UnidentifiedImageError, ValueError) as exc:
        raise ImageEncodingError(image_path, f"image_decode_failed: {exc.__class__.__name__}") from exc


def _inputs_to_device(inputs: Any, device: str) -> Any:
    """Move processor tensor outputs to the requested torch device."""
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _to_numpy_features(features: Any, normalize_embeddings: bool) -> np.ndarray:
    """Detach torch features and return float32 numpy arrays."""
    tensor = _extract_feature_tensor(features)
    vectors = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    if normalize_embeddings:
        return l2_normalize(vectors)
    return vectors


def _extract_feature_tensor(features: Any) -> Any:
    """Extract an embedding tensor from Tensor or Transformers ModelOutput."""
    if hasattr(features, "detach"):
        return features

    pooler_output = getattr(features, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output

    image_embeds = getattr(features, "image_embeds", None)
    if image_embeds is not None:
        return image_embeds

    text_embeds = getattr(features, "text_embeds", None)
    if text_embeds is not None:
        return text_embeds

    last_hidden_state = getattr(features, "last_hidden_state", None)
    if last_hidden_state is not None:
        return last_hidden_state[:, 0, :]

    try:
        first_item = features[0]
    except (KeyError, IndexError, TypeError):
        first_item = None
    if first_item is not None and hasattr(first_item, "detach"):
        return first_item

    raise ValueError("无法从 Transformers 输出中提取 embedding tensor。")


def _infer_torch_dtype(model: Any) -> str:
    """Read the first model parameter dtype for metadata."""
    try:
        return str(next(model.parameters()).dtype)
    except StopIteration:
        return "unknown"


def _infer_embedding_dim(model: Any) -> int:
    """Infer CLIP projection dimension from config or projection layers."""
    projection_dim = getattr(model.config, "projection_dim", None)
    if projection_dim:
        return int(projection_dim)

    visual_projection = getattr(model, "visual_projection", None)
    out_features = getattr(visual_projection, "out_features", None)
    if out_features:
        return int(out_features)

    text_projection = getattr(model, "text_projection", None)
    out_features = getattr(text_projection, "out_features", None)
    if out_features:
        return int(out_features)

    raise ValueError("无法推断 CLIP embedding_dim。")
