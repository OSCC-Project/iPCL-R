#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   training.py
@Time    :   2025/08/17 15:06:32
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Training
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pyarrow.compute as pc
import torch
from accelerate import PartialState
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from flow import FlowConfig
from flow.utils import load_corpus_dataset, setup_logging


def create_flow_config(output_dir: Path) -> FlowConfig:
    flow_config = FlowConfig()
    with PartialState().main_process_first():
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created output directory: {output_dir}")

    flow_config.replace_path_prefixes(output_dir)

    # Tokenization
    flow_config.tokenization.workflow.tokenizer_algorithm = "None"

    # Training
    flow_config.training.hyperparameters.batch_size_per_device = 1
    flow_config.training.hyperparameters.gradient_accumulation_steps = 4
    flow_config.training.hyperparameters.num_train_epochs = 1
    flow_config.training.hyperparameters.learning_rate = 1e-5
    flow_config.training.hyperparameters.warmup_ratio = 0.001
    flow_config.training.hyperparameters.weight_decay = 0.01
    flow_config.training.hyperparameters.optimizer_type = "adamw_8bit"
    flow_config.training.hyperparameters.scheduler_type = "linear"
    flow_config.training.hyperparameters.logging_steps = 1
    flow_config.training.performance.num_workers = 48

    # Evaluation
    flow_config.evaluation.performance.batch_size = 1
    flow_config.evaluation.generation.num_beams = 1
    flow_config.evaluation.generation.max_new_tokens = 1024
    flow_config.evaluation.performance.num_workers = 48

    with PartialState().main_process_first():
        save_path = output_dir / "flow_config.json"
        flow_config.save_to_file(save_path)
        logging.info(f"FlowConfig saved to {save_path}")
    return flow_config


def build_sft_dataset(dataset_dir: Path, remove_columns: bool = True) -> Dataset:
    with PartialState().main_process_first():
        dataset = load_corpus_dataset(dataset_dir)

    def convert_to_sft_sample(batch: Dict[str, Any]) -> Dict[str, Any]:
        instruction = "Generate the 'Routing' solution based on the provided 'Driver' and 'Loads'."
        batch_driver = batch["driver"]
        batch_loads = batch["loads"]
        batch_tree_seq = batch["tree_seq"]

        # build corpus
        prompts = []
        completions = []
        for driver, loads, tree_seq in zip(batch_driver, batch_loads, batch_tree_seq):
            loads = ";".join(loads) if isinstance(loads, list) else loads
            tree_seq = "".join(tree_seq) if isinstance(tree_seq, list) else tree_seq

            prompt = f"{instruction}\nDriver: {driver}\nLoads: {loads}"
            # for save context length
            prompt = prompt.replace(", ", ",")

            completion = tree_seq
            completion = completion.replace(", ", ",")
            completion = completion.replace("[BRANCH]", ">")
            completion = completion.replace("[END]", "<")

            prompts.append(prompt)
            completions.append(completion)

        return {"prompt": prompts, "completion": completions}

    with PartialState().main_process_first():
        sft_dataset = dataset.map(
            convert_to_sft_sample,
            remove_columns=dataset.column_names if remove_columns else None,
            batched=True,
            num_proc=48,
            desc="Converting to SFT format",
        )

    logging.info(f"Converted dataset to SFT format with {len(sft_dataset)} samples.")
    logging.info(f"Sample SFT data: {sft_dataset[0]}")

    return sft_dataset


def stats_token_length(
    model: str, flow_config: FlowConfig, train_dataset: Dataset, eval_dataset: Dataset
):
    with PartialState().main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(model)

    def tokenized(batch: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize the input text using the provided tokenizer"""
        batch_tokens = []
        for prompt, completion in zip(batch["prompt"], batch["completion"]):
            text = prompt + " " + completion
            text_tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
            batch_tokens.append(text_tokens)
        return {"tokens": batch_tokens}

    def stats(dataset: Dataset, name: str):
        """Calculate and log token length statistics for a dataset"""
        with PartialState().main_process_first():
            dataset = dataset.map(
                tokenized,
                batched=True,
                num_proc=flow_config.training.performance.num_workers,
                desc=f"Tokenizing {name} dataset",
            )
        table = dataset.data.table
        lengths = pc.list_value_length(table["tokens"])
        avg_length = pc.mean(lengths).as_py()
        max_length = pc.max(lengths).as_py()
        min_length = pc.min(lengths).as_py()
        quantile_length = pc.quantile(lengths, [0.85, 0.9, 0.95])
        logging.info(
            f"Token length statistics for {name} dataset: "
            f"Avg: {avg_length}, Max: {max_length}, Min: {min_length}, "
            f"Quantile Stats (<85%): {quantile_length[0].as_py()}, (<90%): {quantile_length[1].as_py()}, (<95%): {quantile_length[2].as_py()}"
        )

    stats(train_dataset, "Training")
    stats(eval_dataset, "Evaluation")


def create_sft_config(flow_config: FlowConfig) -> SFTConfig:
    paths_config = flow_config.training.paths
    hyperparameters_config = flow_config.training.hyperparameters
    performance_config = flow_config.training.performance

    generation_config = flow_config.evaluation.generation
    eval_performance_config = flow_config.evaluation.performance

    return SFTConfig(
        output_dir=paths_config.model_save_dir,
        overwrite_output_dir=True,
        # Training hyperparameters
        per_device_train_batch_size=hyperparameters_config.batch_size_per_device,
        # num_train_epochs=hyperparameters_config.num_train_epochs,
        # warmup_ratio=hyperparameters_config.warmup_ratio,
        max_steps=10000,
        warmup_steps=100,
        learning_rate=hyperparameters_config.learning_rate,
        max_grad_norm=hyperparameters_config.max_grad_norm,
        gradient_accumulation_steps=hyperparameters_config.gradient_accumulation_steps,
        max_length=generation_config.max_new_tokens,
        optim=hyperparameters_config.optimizer_type,
        lr_scheduler_type=hyperparameters_config.scheduler_type,
        weight_decay=hyperparameters_config.weight_decay,
        seed=hyperparameters_config.seed,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        # Evaluation and saving
        per_device_eval_batch_size=eval_performance_config.batch_size,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Logging
        logging_dir=os.path.join(
            paths_config.logging_dir,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        ),
        logging_strategy=hyperparameters_config.logging_strategy,
        logging_steps=hyperparameters_config.logging_steps,
        # Data loading
        dataset_num_proc=performance_config.num_workers,
        dataloader_num_workers=performance_config.dataloader_num_workers,
        dataloader_pin_memory=performance_config.dataloader_pin_memory,
        # Mixed precision and optimization
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"],
    )


def create_sft_trainer(args: argparse.Namespace) -> SFTTrainer:
    flow_config = create_flow_config(args.output_dir)

    logging.info("Building SFT dataset...")
    with PartialState().main_process_first():
        sft_train_dataset = build_sft_dataset(args.train_dataset_dir)
        sft_eval_dataset = build_sft_dataset(args.eval_dataset_dir)

    logging.info("Statistics token length...")
    with PartialState().main_process_first():
        stats_token_length(
            args.pretrained_model, flow_config, sft_train_dataset, sft_eval_dataset
        )

    sft_config = create_sft_config(flow_config)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    callbacks = []
    if flow_config.training.hyperparameters.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=flow_config.training.hyperparameters.early_stopping_patience
            )
        )
    return SFTTrainer(
        model=args.pretrained_model,
        train_dataset=sft_train_dataset,
        eval_dataset=sft_eval_dataset,
        args=sft_config,
        peft_config=lora_config,
        callbacks=callbacks,
    )


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="SFT LLM for Routing Generation")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/local_data1/liweiguo/experiments/sft_llm"),
    )

    parser.add_argument(
        "--train-dataset-dir",
        type=Path,
        default=Path("/mnt/local_data1/liweiguo/dataset/experiments/train"),
    )

    parser.add_argument(
        "--eval-dataset-dir",
        type=Path,
        default=Path("/mnt/local_data1/liweiguo/dataset/experiments/val"),
    )

    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
    )

    trainer = create_sft_trainer(parser.parse_args())
    trainer.train()


if __name__ == "__main__":
    main()
