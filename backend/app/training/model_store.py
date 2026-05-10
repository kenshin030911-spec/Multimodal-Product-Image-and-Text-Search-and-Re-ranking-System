"""Persistence helpers for trained lightweight reranker models."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from joblib import dump, load


def save_trained_reranker(
    model: Any,
    model_path: Path,
    meta: dict[str, Any],
    meta_path: Path,
    overwrite: bool = False,
) -> None:
    """Save a trained model and metadata, rejecting accidental overwrites."""
    save_model_artifacts(
        model=model,
        model_path=model_path,
        meta=meta,
        meta_path=meta_path,
        overwrite=overwrite,
    )


def load_trained_reranker(
    model_path: Path,
    meta_path: Path,
    expected_feature_names: Sequence[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load a trained reranker and validate feature names when requested."""
    return load_model_artifacts(
        model_path=model_path,
        meta_path=meta_path,
        expected_feature_names=expected_feature_names,
        artifact_name="trained reranker",
    )


def save_model_artifacts(
    model: Any,
    model_path: Path,
    meta: dict[str, Any],
    meta_path: Path,
    overwrite: bool = False,
) -> None:
    """Save a model object and metadata, rejecting accidental overwrites."""
    _ensure_can_write(model_path=model_path, meta_path=meta_path, overwrite=overwrite)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)
        file.write("\n")


def load_model_artifacts(
    model_path: Path,
    meta_path: Path,
    expected_feature_names: Sequence[str] | None = None,
    artifact_name: str = "model",
) -> tuple[Any, dict[str, Any]]:
    """Load a model object and validate feature names when requested."""
    if not model_path.is_file():
        raise FileNotFoundError(f"{artifact_name} model not found: {model_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"{artifact_name} meta not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as file:
        meta = json.load(file)

    if expected_feature_names is not None:
        expected = list(expected_feature_names)
        actual = list(meta.get("feature_names") or [])
        if actual != expected:
            raise ValueError(
                f"{artifact_name} feature_names mismatch: "
                f"expected {expected}, got {actual}",
            )

    return load(model_path), meta


def _ensure_can_write(model_path: Path, meta_path: Path, overwrite: bool) -> None:
    """Check model/meta output paths before writing."""
    if overwrite:
        return
    existing = [path for path in (model_path, meta_path) if path.exists()]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            "model output already exists; use --overwrite to replace: "
            f"{names}",
        )
