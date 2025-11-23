from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from .config import WatchdogConfig, load_config


@dataclass
class WatchdogResult:
    text: str
    blocked: bool
    scores: List[float]
    stop_step: Optional[int]
    threshold: float
    enabled: bool
    tokens_generated: int


class WatchdogRuntime:
    def __init__(self, cfg: Optional[WatchdogConfig] = None):
        self.cfg = cfg or load_config()
        self._model: Optional[HookedTransformer] = None
        self.vector: Optional[torch.Tensor] = None
        self.layer_index: Optional[int] = None
        self.threshold: float = self.cfg.threshold
        self._load_vector()

    def _ensure_model(self) -> HookedTransformer:
        if self._model is None:
            self._model = HookedTransformer.from_pretrained(
                self.cfg.model_name,
                device=self.cfg.device,
                dtype=self.cfg.torch_dtype(),
            )
            self._model.eval()
            if self.layer_index is None:
                self.layer_index = self.cfg.resolved_layer(self._model.cfg.n_layers)
        return self._model

    def _load_vector(self) -> None:
        if not self.cfg.vector_path.exists():
            raise FileNotFoundError(
                f"No deception vector found at {self.cfg.vector_path}. Run calibrate first."
            )
        payload = torch.load(self.cfg.vector_path, map_location=self.cfg.device)
        self.vector = payload["vector"].to(self.cfg.device)
        self.layer_index = payload.get("layer_index", self.cfg.layer_index)
        if self.layer_index is None:
            raise ValueError("Layer index missing from payload; rerun calibration with latest scripts.")
        self.threshold = payload.get("threshold", self.cfg.threshold)

    def load_vector_from_path(self, vector_path: Path) -> None:
        """Swap the active concept vector without rebuilding the model."""
        self.cfg.vector_path = Path(vector_path)
        self._load_vector()

    @property
    def model(self) -> HookedTransformer:
        return self._ensure_model()

    @torch.no_grad()
    def deception_score(self, text: str) -> float:
        tokens = self.model.to_tokens(text, prepend_bos=True).to(self.vector.device)
        _, cache = self.model.run_with_cache(tokens)
        resid = cache["resid_mid", self.layer_index][-1]
        return float(torch.dot(resid.float(), self.vector.float()).item())

    @torch.no_grad()
    def generate_with_watchdog(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        threshold: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        enabled: bool = True,
        # === CHANGE 1: Add these two arguments ===
        min_tokens_to_check: int = 5,  # Grace period to prevent start-of-sentence jitter
        smoothing_window: int = 3,      # Rolling average window
    ) -> WatchdogResult:
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
        threshold = threshold if threshold is not None else self.threshold
        temperature = temperature if temperature is not None else self.cfg.temperature
        top_p = top_p if top_p is not None else self.cfg.top_p

        device = self.vector.device
        tokens = self.model.to_tokens(prompt, prepend_bos=True).to(device)
        scores: List[float] = []
        blocked = False
        stop_step: Optional[int] = None
        total_watchdog_latency = 0.0

        for step in range(max_new_tokens):
            logits, cache = self.model.run_with_cache(tokens)
            resid = cache["resid_mid", self.layer_index]
            if resid.ndim == 3:
                resid = resid[0, -1]
            else:
                resid = resid[-1]
            if "cuda" in str(self.vector.device):
                torch.cuda.synchronize()
            score_start = time.perf_counter()
            score = float(torch.dot(resid.float(), self.vector.float()).item())
            score_end = time.perf_counter()
            
            latency_ms = (score_end - score_start) * 1000
            total_watchdog_latency += latency_ms
            
            scores.append(score)

            # === CHANGE 3: Updated Smoothing Logic ===
            if len(scores) >= smoothing_window:
                smoothed_score = sum(scores[-smoothing_window:]) / smoothing_window
            else:
                smoothed_score = score

            if enabled and step >= min_tokens_to_check:
                if smoothed_score >= threshold:
                    blocked = True
                    stop_step = step
                    break

            next_token = self._sample_next_token(logits[0, -1], temperature, top_p)
            next_token = next_token.view(1, 1).to(device)
            tokens = torch.cat([tokens, next_token], dim=-1)

        avg_latency = total_watchdog_latency / len(scores) if scores else 0.0
        print(f"[Watchdog] Total added latency: {total_watchdog_latency:.4f} ms")
        print(f"[Watchdog] Average latency per token: {avg_latency:.4f} ms")

        text = self.model.to_string(tokens)
        return WatchdogResult(
            text=text,
            blocked=blocked,
            scores=scores,
            stop_step=stop_step,
            threshold=threshold,
            enabled=enabled,
            tokens_generated=len(scores),
        )

    def generate_without_watchdog(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> WatchdogResult:
        return self.generate_with_watchdog(
            prompt,
            max_new_tokens=max_new_tokens,
            threshold=self.threshold,
            temperature=temperature,
            top_p=top_p,
            enabled=False,
        )

    def _sample_next_token(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        probs = F.softmax(logits / temperature, dim=-1)
        if 0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative <= top_p
            mask[..., 0] = True  # always keep highest prob
            filtered_probs = sorted_probs[mask]
            filtered_idx = sorted_idx[mask]
            filtered_probs = filtered_probs / filtered_probs.sum()
            choice = torch.multinomial(filtered_probs, 1)
            return filtered_idx[choice]
        return torch.multinomial(probs, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Mechanistic Watchdog runtime.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to evaluate.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max generation length.")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling cutoff.")
    parser.add_argument("--disable-watchdog", action="store_true", help="Run without blocking.")
    parser.add_argument(
        "--vector-path",
        type=Path,
        default=None,
        help="Override path to a concept vector .pt (e.g., cyber_misuse_vector.pt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    if args.vector_path is not None:
        cfg.vector_path = args.vector_path

    runtime = WatchdogRuntime(cfg)
    result = runtime.generate_with_watchdog(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        threshold=args.threshold,
        temperature=args.temperature,
        top_p=args.top_p,
        enabled=not args.disable_watchdog,
    )
    print(f"Blocked: {result.blocked} at step {result.stop_step}")
    print(f"Scores: {result.scores}")
    print("Output:")
    print(result.text)


if __name__ == "__main__":
    main()

