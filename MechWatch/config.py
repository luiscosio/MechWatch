from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class WatchdogConfig:
    """Shared configuration for calibration, runtime, and dashboard."""

    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    dataset_name: str = "L1Fthrasir/Facts-true-false"
    dataset_config: Optional[str] = None
    dataset_split: str = "train"
    sample_size: int = 400
    layer_index: Optional[int] = None  # defaults to mid-layer if None
    seed: int = 42
    max_new_tokens: int = 128
    temperature: float = 0.1
    top_p: float = 0.9
    threshold: float = 0.15
    device: str = "cuda"
    dtype: str = "float16"
    vector_path: Path = field(default_factory=lambda: Path("artifacts/deception_vector.pt"))
    stats_path: Path = field(default_factory=lambda: Path("artifacts/deception_stats.json"))

    def resolved_layer(self, n_layers: int) -> int:
        """Return the configured layer index, defaulting to the middle layer."""
        if self.layer_index is not None:
            if not (0 <= self.layer_index < n_layers):
                raise ValueError(f"layer_index {self.layer_index} outside [0, {n_layers})")
            return self.layer_index
        return n_layers // 2

    def torch_dtype(self) -> torch.dtype:
        """Return torch dtype from string identifier."""
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        key = self.dtype.lower() if isinstance(self.dtype, str) else str(self.dtype)
        if key not in mapping:
            raise ValueError(f"Unsupported dtype '{self.dtype}'. Choose from {sorted(mapping)}.")
        return mapping[key]


def load_config() -> WatchdogConfig:
    """Load configuration, allowing optional overrides via env vars."""

    defaults = WatchdogConfig()
    cfg = WatchdogConfig(
        model_name=os.getenv("WATCHDOG_MODEL", defaults.model_name),
        dataset_name=os.getenv("WATCHDOG_DATASET", defaults.dataset_name),
        sample_size=int(os.getenv("WATCHDOG_SAMPLE_SIZE", defaults.sample_size)),
        layer_index=(
            int(os.getenv("WATCHDOG_LAYER"))
            if os.getenv("WATCHDOG_LAYER") is not None
            else defaults.layer_index
        ),
        dataset_config=os.getenv("WATCHDOG_DATASET_CONFIG", defaults.dataset_config),
        dataset_split=os.getenv("WATCHDOG_DATASET_SPLIT", defaults.dataset_split),
        max_new_tokens=int(os.getenv("WATCHDOG_MAX_TOKENS", defaults.max_new_tokens)),
        temperature=float(os.getenv("WATCHDOG_TEMPERATURE", defaults.temperature)),
        top_p=float(os.getenv("WATCHDOG_TOP_P", defaults.top_p)),
        threshold=float(os.getenv("WATCHDOG_THRESHOLD", defaults.threshold)),
        device=os.getenv("WATCHDOG_DEVICE", defaults.device),
        dtype=os.getenv("WATCHDOG_DTYPE", defaults.dtype),
        vector_path=Path(os.getenv("WATCHDOG_VECTOR_PATH", defaults.vector_path)),
        stats_path=Path(os.getenv("WATCHDOG_STATS_PATH", defaults.stats_path)),
        seed=int(os.getenv("WATCHDOG_SEED", defaults.seed)),
    )
    cfg.vector_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.stats_path.parent.mkdir(parents=True, exist_ok=True)
    return cfg

