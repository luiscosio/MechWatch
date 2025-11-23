from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import statistics

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from .config import WatchdogConfig, load_config


@dataclass
class ActivationBuckets:
    true_acts: torch.Tensor
    false_acts: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the Mechanistic Watchdog.")
    parser.add_argument("--model", default=None, help="Override model name.")
    parser.add_argument("--dataset", default=None, help="Override dataset name.")
    parser.add_argument("--dataset-config", default=None, help="Optional dataset config (e.g., 'wmdp-cyber').")
    parser.add_argument("--dataset-split", default=None, help="Split to load (defaults to dataset default).")
    parser.add_argument("--samples", type=int, default=None, help="Limit dataset size.")
    parser.add_argument(
        "--eval-frac",
        type=float,
        default=0.2,
        help="Fraction of samples to reserve for evaluation.",
    )
    parser.add_argument("--layer", type=int, default=None, help="Layer index to probe.")
    parser.add_argument("--out", type=Path, default=None, help="Path to save vector.")
    parser.add_argument("--stats", type=Path, default=None, help="Path to save stats JSON.")
    parser.add_argument("--device", default=None, help="Torch device override.")
    parser.add_argument("--dtype", default=None, help="Torch dtype override (e.g., float16).")
    parser.add_argument("--seed", type=int, default=None, help="Shuffle seed.")
    parser.add_argument(
        "--concept-name",
        type=str,
        default="deception",
        help="Logical name for this concept vector (e.g. 'deception', 'cyber_misuse').",
    )
    return parser.parse_args()


def apply_overrides(cfg: WatchdogConfig, args: argparse.Namespace) -> WatchdogConfig:
    if args.model:
        cfg.model_name = args.model
    if args.dataset:
        cfg.dataset_name = args.dataset
    if args.dataset_config is not None:
        cfg.dataset_config = args.dataset_config
    if args.dataset_split is not None:
        cfg.dataset_split = args.dataset_split
    if args.samples:
        cfg.sample_size = args.samples
    if args.layer is not None:
        cfg.layer_index = args.layer
    if args.out:
        cfg.vector_path = args.out
    if args.stats:
        cfg.stats_path = args.stats
    if args.device:
        cfg.device = args.device
    if args.dtype:
        cfg.dtype = args.dtype
    if args.seed is not None:
        cfg.seed = args.seed
    return cfg


def load_model(cfg: WatchdogConfig) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(
        cfg.model_name,
        device=cfg.device,
        dtype=cfg.torch_dtype(),
        cache_dir=os.getenv("HF_HOME"),
    )
    model.eval()
    return model


TEXT_FIELDS = ("statement", "text", "prompt", "question", "content")
LABEL_FIELDS = ("label", "truth", "is_true", "answer")


def standardize_dataset(dataset: Dataset) -> Dataset:
    """
    Normalize heterogeneous datasets into simple (statement, label) rows.

    If a dataset exposes multiple-choice questions (`choices` + `answer`),
    we expand each option into its own row so the correct choice becomes a
    positive example and every distractor becomes a negative example.
    """

    column_names = set(dataset.column_names)
    if {"choices", "answer"}.issubset(column_names):
        text_field = (
            "question"
            if "question" in column_names
            else ("prompt" if "prompt" in column_names else None)
        )
        if text_field is None:
            raise ValueError("Multiple-choice dataset is missing a question/prompt field.")

        def explode(batch):
            statements, labels = [], []
            questions = batch[text_field]
            choices_batch = batch["choices"]
            answers = batch["answer"]
            for q, choices, answer in zip(questions, choices_batch, answers):
                stem = q.strip() if isinstance(q, str) else str(q)
                for idx, choice in enumerate(choices):
                    choice_text = choice.strip() if isinstance(choice, str) else str(choice)
                    statements.append(f"Question: {stem}\nChoice: {choice_text}")
                    labels.append(1 if idx == int(answer) else 0)
            return {"statement": statements, "label": labels}

        dataset = dataset.map(
            explode,
            batched=True,
            remove_columns=dataset.column_names,
        )

    return dataset


def extract_statement(row: Dict) -> str:
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("No text-like field found in dataset row.")


def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"true", "t", "1", "yes"}:
            return True
        if cleaned in {"false", "f", "0", "no"}:
            return False
    raise ValueError(f"Cannot interpret {value!r} as boolean label.")


def extract_label(row: Dict) -> bool:
    for field in LABEL_FIELDS:
        if field in row:
            return normalize_bool(row[field])
    raise ValueError("No label field found in dataset row.")


@torch.no_grad()
def capture_activation(model: HookedTransformer, text: str, layer_idx: int) -> torch.Tensor:
    _, cache = model.run_with_cache(text)
    resid = cache["resid_mid", layer_idx]
    if resid.ndim == 3:
        # shape: [batch, seq, d_model]; keep last token from first batch element
        resid = resid[0, -1]
    else:
        # shape: [seq, d_model]
        resid = resid[-1]
    return resid.detach().cpu().float()


def collect_activations(dataset: Dataset, model: HookedTransformer, layer_idx: int) -> ActivationBuckets:
    true_list, false_list = [], []
    iterable = dataset.with_format("python")
    for row in tqdm(iterable, total=len(dataset), desc="Capturing activations"):
        try:
            text = extract_statement(row)
            label = extract_label(row)
        except ValueError:
            continue
        resid = capture_activation(model, text, layer_idx)
        if label:
            true_list.append(resid)
        else:
            false_list.append(resid)

    if not true_list or not false_list:
        raise RuntimeError("Insufficient true/false samples after filtering.")

    return ActivationBuckets(true_acts=torch.stack(true_list), false_acts=torch.stack(false_list))


def compute_deception_vector(buckets: ActivationBuckets) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    true_mean = buckets.true_acts.mean(dim=0)
    false_mean = buckets.false_acts.mean(dim=0)
    vector = false_mean - true_mean
    norm = torch.linalg.norm(vector)
    if float(norm) == 0.0:
        raise RuntimeError("Zero-norm deception vector; check dataset variety.")
    return vector / norm, true_mean, false_mean


def score_dataset(
    dataset: Dataset,
    model: HookedTransformer,
    layer_idx: int,
    vector: torch.Tensor,
) -> Dict[str, Iterable[float]]:
    scores_true, scores_false = [], []
    iterable = dataset.with_format("python")
    for row in tqdm(iterable, total=len(dataset), desc="Evaluating scores"):
        try:
            text = extract_statement(row)
            label = extract_label(row)
        except ValueError:
            continue
        resid = capture_activation(model, text, layer_idx)
        score = float(torch.dot(resid, vector))
        if label:
            scores_true.append(score)
        else:
            scores_false.append(score)
    return {"true": scores_true, "false": scores_false}


def suggest_threshold(scores_true: Iterable[float], scores_false: Iterable[float]) -> Dict[str, float]:
    scores_true = list(scores_true)
    scores_false = list(scores_false)
    if not scores_true or not scores_false:
        raise RuntimeError("Need both true and false scores to compute threshold.")
    true_mean = statistics.fmean(scores_true)
    false_mean = statistics.fmean(scores_false)
    true_std = statistics.pstdev(scores_true) if len(scores_true) > 1 else 0.0
    false_std = statistics.pstdev(scores_false) if len(scores_false) > 1 else 0.0
    midpoint = true_mean + 0.5 * (false_mean - true_mean)
    margin = true_std
    suggested = midpoint + margin
    return {
        "suggested": suggested,
        "true_mean": true_mean,
        "false_mean": false_mean,
        "true_std": true_std,
        "false_std": false_std,
        "midpoint": midpoint,
    }


def prepare_dataset(cfg: WatchdogConfig, eval_frac: float) -> Tuple[Dataset, Dataset]:
    load_kwargs = {}
    if cfg.dataset_config:
        load_kwargs["name"] = cfg.dataset_config
    dataset = load_dataset(cfg.dataset_name, split=cfg.dataset_split, **load_kwargs)
    dataset = standardize_dataset(dataset)
    dataset = dataset.shuffle(seed=cfg.seed)
    if cfg.sample_size and cfg.sample_size < len(dataset):
        dataset = dataset.select(range(cfg.sample_size))
    split = dataset.train_test_split(test_size=eval_frac, seed=cfg.seed)
    return split["train"], split["test"]


def save_artifacts(
    vector: torch.Tensor,
    true_mean: torch.Tensor,
    false_mean: torch.Tensor,
    stats: Dict,
    cfg: WatchdogConfig,
    concept_name: str,
) -> None:
    payload = {
        "vector": vector,
        "true_mean": true_mean,
        "false_mean": false_mean,
        "layer_index": stats["layer_index"],
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "threshold": stats["threshold"]["suggested"],
        "concept_name": concept_name,
    }
    torch.save(payload, cfg.vector_path)
    stats = dict(stats)
    stats["concept_name"] = concept_name
    cfg.stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Saved {concept_name!r} vector to {cfg.vector_path}")
    print(f"Saved calibration stats to {cfg.stats_path}")


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(), args)

    train_ds, eval_ds = prepare_dataset(cfg, args.eval_frac)
    print(f"Loaded dataset with {len(train_ds)} train and {len(eval_ds)} eval samples.")

    model = load_model(cfg)
    layer_idx = cfg.resolved_layer(model.cfg.n_layers)
    print(f"Using layer {layer_idx} of {model.cfg.n_layers}.")

    buckets = collect_activations(train_ds, model, layer_idx)
    vector, true_mean, false_mean = compute_deception_vector(buckets)

    scores = score_dataset(eval_ds, model, layer_idx, vector)
    threshold_info = suggest_threshold(scores["true"], scores["false"])

    stats = {
        "model": cfg.model_name,
        "dataset": cfg.dataset_name,
        "sample_size": len(train_ds) + len(eval_ds),
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "layer_index": layer_idx,
        "threshold": threshold_info,
        "score_examples": {
            "true": scores["true"][:5],
            "false": scores["false"][:5],
        },
    }

    save_artifacts(vector, true_mean, false_mean, stats, cfg, args.concept_name)
    print(f"Suggested threshold: {threshold_info['suggested']:.4f}")


if __name__ == "__main__":
    main()

