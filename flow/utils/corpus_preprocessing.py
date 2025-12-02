#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   corpus_preprocessing.py
@Time    :   2025/08/17 15:14:52
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Preprocess text data
"""

import logging
from pathlib import Path
from typing import Optional, Union

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from flow.config import DatasetConfig


def _select_split(
    dataset: Union[Dataset, DatasetDict], split: Optional[str]
) -> Dataset:
    """Return the correct split from a DatasetDict or the dataset itself."""
    if isinstance(dataset, DatasetDict):
        if split:
            if split not in dataset:
                raise ValueError(
                    f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}"
                )
            return dataset[split]
        if "train" in dataset:
            return dataset["train"]
        # Fall back to the first available split to keep behavior predictable
        first_split = next(iter(dataset.values()))
        return first_split
    return dataset


def _load_local_dataset(dataset_path: Path, split: Optional[str]) -> Dataset:
    """Load dataset from a local directory with legacy fallback paths."""
    try:
        dataset = load_from_disk(str(dataset_path))
    except FileNotFoundError:
        dataset_path = dataset_path / "aggregated" / "flat_corpus"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
        dataset = load_from_disk(str(dataset_path))

    dataset = _select_split(dataset, split)
    logging.info(f"Dataset loaded with {len(dataset)} samples from {dataset_path}")
    return dataset


def load_corpus_dataset(
    dataset_ref: Union[Path, str, DatasetConfig], split: Optional[str] = None
) -> Dataset:
    """
    Load a corpus dataset from Hugging Face Hub or local disk.

    Args:
        dataset_ref: DatasetConfig, Hugging Face dataset id, or local path.
        split: Optional split name when loading from Hub or DatasetDict.
    """
    try:
        if isinstance(dataset_ref, DatasetConfig):
            target_split = dataset_ref.resolve_split(split)
            if dataset_ref.use_hub():
                dataset = load_dataset(dataset_ref.hub_id, split=target_split)
                logging.info(
                    f"Dataset loaded with {len(dataset)} samples from {dataset_ref.hub_id} (split: {target_split})"
                )
                return dataset

            dataset_path = dataset_ref.local_path_for_split(split)
            return _load_local_dataset(dataset_path, target_split)

        # Fallback: treat Path/str as either local path or dataset id
        target_split = split
        if isinstance(dataset_ref, Path):
            if not dataset_ref.exists():
                raise FileNotFoundError(f"Dataset path not found: {dataset_ref}")
            return _load_local_dataset(dataset_ref, target_split)

        dataset_path = Path(str(dataset_ref))
        if dataset_path.exists():
            return _load_local_dataset(dataset_path, target_split)

        # Assume Hugging Face dataset id when path does not exist
        target_split = target_split or "train"
        dataset = load_dataset(str(dataset_ref), split=target_split)
        logging.info(
            f"Dataset loaded with {len(dataset)} samples from {dataset_ref} (split: {target_split})"
        )
        return dataset
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise
