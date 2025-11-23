"""Builds a bio-safe vs bio-misuse contrastive dataset for Watchdog calibration.

This utility loads two biology-focused subsets from the WMDP corpora—one
hazardous ("forget") and one benign ("retain")—and constructs a balanced,
contrastive dataset that isolates harmful intent within biological content.
The resulting JSONL file can be consumed by `MechWatch.calibrate`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

from datasets import (
    Dataset,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)

# ---------------------------------------------------------------------------
# Configurable constants (tweak as needed)

HARMFUL_DATASET_NAME = "cais/wmdp-bio-forget-corpus"
HARMFUL_CONFIG_NAME = None  # Dedicated dataset, so no config needed

SAFE_DATASET_NAME = "cais/wmdp-corpora"
SAFE_CONFIG_NAME = "bio-retain-corpus"  # TODO: adjust to actual config name

DATASET_SPLIT = "train"
MAX_SAMPLES_PER_CLASS = 200
OUTPUT_PATH = "artifacts/bio_safe_misuse.jsonl"

TEXT_COLUMN_CANDIDATES = ("statement", "text", "prompt", "question", "content")
SHUFFLE_SEED = 42


def _detect_text_column(dataset: Dataset, *, candidates: Iterable[str]) -> str:
    """Return the first column name that exists in the dataset."""
    for candidate in candidates:
        if candidate in dataset.column_names:
            return candidate
    available = ", ".join(dataset.column_names)
    raise ValueError(
        "Failed to locate a text column. Checked candidates "
        f"{candidates}, available columns: {available}"
    )


def _prepare_subset(
    dataset_name: str,
    config_name: str | None,
    split: str,
    label: int,
    max_samples: int,
) -> Dataset:
    """Load, normalize, shuffle, and downsample a dataset subset."""
    _verify_config_available(dataset_name, config_name)
    load_kwargs = {"split": split}
    if config_name is not None:
        load_kwargs["name"] = config_name
    raw_ds = load_dataset(dataset_name, **load_kwargs)
    if not isinstance(raw_ds, Dataset):
        raise TypeError(
            f"Expected `datasets.Dataset`, got {type(raw_ds).__name__} "
            f"for {dataset_name}:{config_name} ({split})"
        )

    text_column = _detect_text_column(raw_ds, candidates=TEXT_COLUMN_CANDIDATES)

    def _has_text(example: dict) -> bool:
        value = example.get(text_column, "")
        text = str(value).strip()
        return bool(text)

    filtered = raw_ds.filter(_has_text)

    def _format_example(example: dict) -> dict:
        text = str(example[text_column]).strip()
        return {"statement": text, "label": label}

    formatted = filtered.map(
        _format_example,
        remove_columns=[col for col in filtered.column_names],
    )

    shuffled = formatted.shuffle(seed=SHUFFLE_SEED)
    sample_count = min(len(shuffled), max_samples)
    if sample_count < len(shuffled):
        shuffled = shuffled.select(range(sample_count))

    return shuffled


def _verify_config_available(dataset_name: str, config_name: str) -> None:
    """Ensure the requested config exists, raising a helpful error otherwise."""
    if config_name is None:
        return
    available = get_dataset_config_names(dataset_name)
    if config_name not in available:
        formatted = ", ".join(available) or "<none>"
        raise ValueError(
            f"Config '{config_name}' not found for dataset '{dataset_name}'. "
            f"Available configs: {formatted}. "
            "If this config is gated (e.g., WMDP bio-forget), request access via "
            "the CAIS form and ensure you have run `huggingface-cli login`."
        )


def build_bio_safe_misuse_dataset() -> Tuple[Dataset, int, int]:
    """Construct the combined contrastive dataset."""
    harmful_ds = _prepare_subset(
        HARMFUL_DATASET_NAME,
        HARMFUL_CONFIG_NAME,
        DATASET_SPLIT,
        label=1,
        max_samples=MAX_SAMPLES_PER_CLASS,
    )
    safe_ds = _prepare_subset(
        SAFE_DATASET_NAME,
        SAFE_CONFIG_NAME,
        DATASET_SPLIT,
        label=0,
        max_samples=MAX_SAMPLES_PER_CLASS,
    )

    combined = concatenate_datasets([harmful_ds, safe_ds]).shuffle(seed=SHUFFLE_SEED)
    return combined, len(harmful_ds), len(safe_ds)


def save_dataset(dataset: Dataset, output_path: str) -> Path:
    """Persist the dataset to JSONL and return the absolute path."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(str(output), lines=True, orient="records")
    return output.resolve()


def main() -> None:
    dataset, harmful_count, safe_count = build_bio_safe_misuse_dataset()
    output_path = save_dataset(dataset, OUTPUT_PATH)
    total = len(dataset)

    print(f"Harmful samples: {harmful_count}")
    print(f"Safe samples: {safe_count}")
    print(f"Total samples: {total}")
    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()

