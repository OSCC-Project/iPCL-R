#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   launch_evaluation.py
@Time    :   2025/08/01 11:16:14
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Command-line launcher for Stage 3 evaluation using Accelerate framework,
             handles model inference, metric calculation, and comprehensive routing evaluation
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Union

import torch
from accelerate import Accelerator, PartialState
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    GenerationConfig,
    PreTrainedTokenizerFast,
    T5GemmaForConditionalGeneration,
)

from flow.config import FlowConfig
from flow.evaluation import EvaluationPipeline
from flow.tokenization import TokenizationPipeline, UnifiedTokenizer
from flow.utils import load_corpus_dataset, setup_logging


def load_components(
    config: FlowConfig,
) -> Union[Dataset, PreTrainedTokenizerFast, T5GemmaForConditionalGeneration]:
    """Load evaluation dataset, tokenizer, and model from config"""

    # Dataset
    dataset_config = config.dataset
    target_split = dataset_config.validation_split
    dataset = load_corpus_dataset(dataset_config, split=target_split)

    tokenization_pipeline = TokenizationPipeline(config)
    dataset = tokenization_pipeline.preprocess_corpus(dataset)
    # Set remove_columns=False to keep original columns for evaluation ('source_design')
    dataset = tokenization_pipeline.build_token_dataset(dataset, remove_columns=False)

    dataset_source = (
        dataset_config.hub_id
        if dataset_config.use_hub()
        else dataset_config.local_path_for_split(target_split)
    )
    logging.info(f"Dataset loaded with {len(dataset)} samples from {dataset_source}")

    # Tokenizer & Model
    unified_tokenizer = UnifiedTokenizer.from_pretrained(
        config.tokenization.paths.tokenizer_save_dir
    )
    tokenizer = unified_tokenizer.tokenizer
    logging.info(
        f"Tokenizer loaded from {config.tokenization.paths.tokenizer_save_dir}"
    )

    try:
        model = T5GemmaForConditionalGeneration.from_pretrained(
            config.training.paths.model_save_dir
        ).eval()
        logging.info(f"Model loaded from {config.training.paths.model_save_dir}")
    except Exception as e:
        checkpoints = list(
            Path(config.training.paths.model_save_dir).glob("checkpoint-*")
        )
        if checkpoints:
            last_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[-1]))
            logging.info(f"Loading model from last checkpoint: {last_checkpoint}")
            model = T5GemmaForConditionalGeneration.from_pretrained(
                last_checkpoint
            ).eval()
        else:
            logging.error(
                f"Failed to load model from {config.training.paths.model_save_dir}"
            )
            raise e

    return dataset, tokenizer, model


def run_evaluation(config: FlowConfig):
    """Run evaluation"""
    evaluation_config = config.evaluation
    training_paths_config = config.training.paths
    training_hyperparameters_config = config.training.hyperparameters
    tokenization_paths_config = config.tokenization.paths
    evaluation_paths_config = evaluation_config.paths
    dataset_config = config.dataset
    performance_config = evaluation_config.performance
    generation_config = evaluation_config.generation
    logging.info("🚀 Initializing evaluation pipeline")
    dataset_source = (
        dataset_config.hub_id
        if dataset_config.use_hub()
        else dataset_config.local_path_for_split(dataset_config.validation_split)
    )
    logging.info(
        f"   Dataset: {dataset_source} (split: {dataset_config.validation_split})"
    )
    logging.info(f"   Tokenizer: {tokenization_paths_config.tokenizer_save_dir}")
    logging.info(f"   Model: {training_paths_config.model_save_dir}")
    logging.info(f"   Output: {evaluation_paths_config.output_dir}")

    accelerator = Accelerator()
    # Load dataset, tokenizer, and model
    with PartialState().main_process_first():
        dataset, tokenizer, model = load_components(config)
        logging.info("✅ Components loaded successfully")

    def collect_fn(batch):
        """Custom collate function to handle variable-length sequences"""
        source_tokens = [item["source_tokens"] for item in batch]
        encs = tokenizer(
            source_tokens,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            padding="max_length",
            max_length=training_hyperparameters_config.max_src_len,
        )
        return encs

    dataloader = DataLoader(
        dataset,
        batch_size=performance_config.batch_size,
        num_workers=performance_config.dataloader_num_workers,
        pin_memory=performance_config.dataloader_pin_memory,
        collate_fn=collect_fn,
        shuffle=False,
    )

    model, dataloader = accelerator.prepare(model, dataloader)

    # Configure generation parameters
    model_generation_config = GenerationConfig(
        max_new_tokens=generation_config.max_new_tokens,
        num_beams=generation_config.num_beams,
        do_sample=generation_config.do_sample,
        temperature=generation_config.temperature
        if generation_config.do_sample
        else None,
        top_p=generation_config.top_p if generation_config.do_sample else None,
        top_k=generation_config.top_k if generation_config.do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
    )

    logging.info("🎯 Generation parameters:")
    logging.info(f"   Max new tokens: {generation_config.max_new_tokens}")
    logging.info(f"   Beam search: {generation_config.num_beams}")
    logging.info(f"   Do sample: {generation_config.do_sample}")
    if generation_config.do_sample:
        logging.info(f"   Temperature: {generation_config.temperature}")
        logging.info(f"   Top-p: {generation_config.top_p}")
        logging.info(f"   Top-k: {generation_config.top_k}")
    logging.info(f"   Batch size: {performance_config.batch_size}")

    # Run inference
    logging.info("⚡ Running inference...")
    inference_start_time = time.time()

    predictions = []
    for batch in tqdm(
        dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process
    ):
        with torch.no_grad():
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            outputs = accelerator.unwrap_model(model).generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=model_generation_config,
            )

            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            gathered_preds = accelerator.gather_for_metrics(preds)

            if accelerator.is_main_process:
                predictions.extend(gathered_preds)

    inference_time = time.time() - inference_start_time
    logging.info(f"✅ Inference completed in {inference_time:.2f}s")
    logging.info(f"   Throughput: {len(dataset) / inference_time:.2f} samples/sec")

    # Use existing metrics system for evaluation
    logging.info("📊 Calculating evaluation metrics...")

    if accelerator.is_main_process:
        # Sort predictions by original indices to ensure correct order
        if len(predictions) != len(dataset):
            logging.error(
                f"Number of predictions ({len(predictions)}) does not match dataset size ({len(dataset)}). Truncating."
            )
            raise ValueError("Inconsistent prediction and dataset sizes")

        dataset = dataset.add_column("predictions", predictions)

        # Evaluation pipeline
        evaluation_pipeline = EvaluationPipeline(config)
        dataset = evaluation_pipeline.calculate_metrics(dataset)

        # Save DEF inference metadata for EDA tool evaluation
        evaluation_pipeline.save_def_inference_metadata(dataset)
        evaluation_pipeline.save_def_inference_metadata_txt(dataset)

        # Save results
        logging.info("💾 Saving evaluation results...")
        dataset.save_to_disk(evaluation_paths_config.output_dir)


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Launch evaluation")
    parser.add_argument(
        "--flow-config",
        type=str,
        required=True,
        help="Path to flow configuration JSON file",
    )

    args = parser.parse_args()

    # Load flow configuration
    try:
        flow_config = FlowConfig.from_config_file(Path(args.flow_config))
    except Exception as e:
        logging.info(f"❌ Error loading config from {args.flow_config}: {e}")
        sys.exit(1)

    logging.info("🔍 Starting evaluation")
    logging.info(f"   Config: {args.flow_config}")

    try:
        # Run evaluation
        run_evaluation(flow_config)

        logging.info("✅ Evaluation completed successfully!")
        logging.info(f"   Results saved to: {flow_config.evaluation.paths.output_dir}")

    except Exception as e:
        logging.info(f"❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
