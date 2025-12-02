#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   init_env.py
@Time    :   2025/09/26 00:23:23
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Generates the initial environment for model size comparison experiments.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

from flow import FlowConfig
from flow.utils import setup_logging


def create_alias(tokenizer_algorithm: str, vocab_size: int) -> str:
    """Create a unique alias for the tokenizer algorithm and vocabulary size"""
    if tokenizer_algorithm == "DecimalWordLevel":
        # No alias needed for DecimalWordLevel
        return tokenizer_algorithm

    alias_vocab_size = (
        f"{vocab_size // 1000}K" if vocab_size >= 1000 else str(vocab_size)
    )
    return f"{tokenizer_algorithm}-{alias_vocab_size}"


def create_model_config(
    tokenizer_algorithms: List[str],
    vocab_sizes: List[int],
) -> List[Dict[str, Any]]:
    """Create model configurations for different sizes"""
    # Small, Medium, Large
    model_configs = []
    algorithms_set = set()
    for tokenizer_algorithm in tokenizer_algorithms:
        for vocab_size in vocab_sizes:
            alias = create_alias(tokenizer_algorithm, vocab_size)
            if alias in algorithms_set:
                # Avoid duplicate configurations for 'DecimalWordLevel'
                continue
            else:
                algorithms_set.add(alias)
            config = [
                {
                    "alias": f"Small-{alias}",
                    "tokenizer_algorithm": tokenizer_algorithm,
                    "vocab_size": vocab_size,
                    "max_sequence_length": 512,
                    "hidden_size": 256,
                    "intermediate_size": 1024,
                    "num_hidden_layers": 4,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 64,
                    "max_position_embeddings": 256,
                    "sliding_window": 128,
                    "dropout_rate": 0.1,
                    "num_train_epochs": 100,
                    "batch_size_per_device": 256,
                    "max_src_len": 128,
                    "max_tgt_len": 256,
                    "max_new_tokens": 512,
                    "evaluation_batch_size": 32,
                },
                {
                    "alias": f"Medium-{alias}",
                    "tokenizer_algorithm": tokenizer_algorithm,
                    "vocab_size": vocab_size,
                    "max_sequence_length": 512,
                    "hidden_size": 512,
                    "intermediate_size": 2048,
                    "num_hidden_layers": 6,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 4,
                    "head_dim": 64,
                    "max_position_embeddings": 256,
                    "sliding_window": 128,
                    "dropout_rate": 0.1,
                    "num_train_epochs": 100,
                    "batch_size_per_device": 128,
                    "max_src_len": 128,
                    "max_tgt_len": 256,
                    "max_new_tokens": 512,
                    "evaluation_batch_size": 32,
                },
                {
                    "alias": f"Large-{alias}",
                    "tokenizer_algorithm": tokenizer_algorithm,
                    "vocab_size": vocab_size,
                    "max_sequence_length": 512,
                    "hidden_size": 1024,
                    "intermediate_size": 4096,
                    "num_hidden_layers": 12,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "head_dim": 64,
                    "max_position_embeddings": 256,
                    "sliding_window": 128,
                    "dropout_rate": 0.1,
                    "num_train_epochs": 100,
                    "batch_size_per_device": 32,
                    "max_src_len": 128,
                    "max_tgt_len": 256,
                    "max_new_tokens": 512,
                    "evaluation_batch_size": 32,
                },
            ]
            model_configs.extend(config)
    return model_configs


def create_work_space(
    work_dir: Path,
    data_synthesis_dir: Path,
    evaluation_dataset_dir: Path,
    dataset_source: str,
    dataset_id: str,
    tokenizer_algorithms: List[str],
    vocab_sizes: List[int],
) -> Dict[str, Path]:
    """Create the workspace and FlowConfig for the tokenizer comparison experiment"""
    model_configs = create_model_config(tokenizer_algorithms, vocab_sizes)
    for config in model_configs:
        alias = config["alias"]
        flow_config = FlowConfig()
        sub_work_dir = work_dir / alias
        if not sub_work_dir.exists():
            sub_work_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created sub-work directory: {sub_work_dir}")
        flow_config.replace_path_prefixes(sub_work_dir)

        # Dataset settings
        flow_config.dataset.train_split = "train"
        flow_config.dataset.validation_split = "validation"
        if dataset_source == "hub":
            flow_config.dataset.source = "hub"
            flow_config.dataset.hub_id = dataset_id
        else:
            flow_config.dataset.source = "local"
            flow_config.dataset.train_local_dir = str(data_synthesis_dir)
            flow_config.dataset.eval_local_dir = str(evaluation_dataset_dir)

        # Tokenization settings
        flow_config.tokenization.workflow.tokenizer_algorithm = config[
            "tokenizer_algorithm"
        ]
        flow_config.tokenization.workflow.target_vocab_size = config["vocab_size"]
        flow_config.tokenization.workflow.max_sequence_length = config[
            "max_sequence_length"
        ]

        # Training settings
        flow_config.training.model.hidden_size = config["hidden_size"]
        flow_config.training.model.intermediate_size = config["intermediate_size"]
        flow_config.training.model.num_hidden_layers = config["num_hidden_layers"]
        flow_config.training.model.num_attention_heads = config["num_attention_heads"]
        flow_config.training.model.num_key_value_heads = config["num_key_value_heads"]
        flow_config.training.model.head_dim = config["head_dim"]
        flow_config.training.model.max_position_embeddings = config[
            "max_position_embeddings"
        ]
        flow_config.training.model.sliding_window = config["sliding_window"]
        flow_config.training.model.dropout_rate = config["dropout_rate"]
        flow_config.training.hyperparameters.num_train_epochs = config[
            "num_train_epochs"
        ]
        flow_config.training.hyperparameters.batch_size_per_device = config[
            "batch_size_per_device"
        ]
        flow_config.training.hyperparameters.max_src_len = config["max_src_len"]
        flow_config.training.hyperparameters.max_tgt_len = config["max_tgt_len"]

        # Evaluation settings
        flow_config.evaluation.generation.max_new_tokens = config["max_new_tokens"]
        flow_config.evaluation.performance.batch_size = config["evaluation_batch_size"]

        flow_config_save_path = sub_work_dir / "flow_config.json"
        flow_config.save_to_file(flow_config_save_path)
        logging.info(f"Saved FlowConfig for '{alias}' to '{flow_config_save_path}'")
    logging.info(f"Created {len(model_configs)} unique configurations in {work_dir}")

    return model_configs


def create_script(work_dir: Path, model_configs: List[Dict[str, Any]]) -> Path:
    """Create a script to run the tokenizer comparison experiment"""
    script_path = work_dir / "run_model_size_comparison.sh"
    with open(script_path, "w") as script_file:
        script_file.write("#!/bin/bash\n\n")
        for config in model_configs:
            alias = config["alias"]
            config_path = work_dir / alias / "flow_config.json"
            script_file.write(f"# Running tokenizer comparison for {alias}\n")
            script_file.write(
                f"echo 'Running tokenizer comparison for {alias} with config {config_path}'\n"
            )
            script_file.write(
                f"python -m flow.launch_tokenization --flow-config {config_path}\n"
            )
            script_file.write(
                f"accelerate launch -m flow.launch_training --flow-config {config_path}\n"
            )
            # script_file.write(
            #     f"accelerate launch -m flow.launch_evaluation --flow-config {config_path}\n"
            # )
            script_file.write(
                f"accelerate launch -m --config_file /home/liweiguo/.cache/huggingface/accelerate/fast_evaluation.yaml flow.launch_evaluation --flow-config {config_path}\n"
            )
    script_path.chmod(script_path.stat().st_mode | 0o111)
    logging.info(f"Created script: {script_path}")

    return script_path


def main():
    setup_logging()
    """Main entry point for pipeline initialization"""
    parser = argparse.ArgumentParser(
        description="Initialize flow pipeline configuration files"
    )

    # Configuration generation
    parser.add_argument(
        "--work-dir",
        type=Path,
        metavar="WORK_DIR",
        default=Path(
            "/mnt/local_data1/liweiguo/experiments/model_size_comparison/work_dir"
        ),
        help="Create the corresponding directory and configuration in WORK_DIR based on the experiment setup",
    )

    parser.add_argument(
        "--data-synthesis-dir",
        type=Path,
        default=Path("/mnt/local_data1/liweiguo/dataset/experiments/train"),
        help="Directory containing the data synthesis files",
    )

    parser.add_argument(
        "--evaluation-dataset-dir",
        type=Path,
        default=Path("/mnt/local_data1/liweiguo/dataset/experiments/val"),
        help="Directory containing the evaluation dataset files",
    )

    parser.add_argument(
        "--dataset-source",
        type=str,
        choices=["hub", "local"],
        default="hub",
        help="Choose to load dataset from Hugging Face Hub or local paths.",
    )

    parser.add_argument(
        "--dataset-id",
        type=str,
        default="AiEDA/iPCL-R",
        help="Hugging Face dataset identifier when using hub source.",
    )

    parser.add_argument(
        "--tokenizer-algorithm",
        type=str,
        nargs="+",
        default=[
            "DecimalWordLevel",
            "Seg-BPE",
            "Concat-BPE",
            "Seg-BBPE",
            "Concat-BBPE",
        ],
        help="List of tokenizer algorithms to compare",
    )

    parser.add_argument(
        "--target-vocab-size",
        type=int,
        nargs="+",
        default=[1000, 4000, 16000],
        help="List of target vocabulary sizes for tokenization",
    )

    args = parser.parse_args()

    work_dir = args.work_dir
    if not work_dir.exists():
        work_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"🔧 Initializing FlowConfig in: {work_dir}")

    data_synthesis_dir = args.data_synthesis_dir
    if args.dataset_source == "local" and not data_synthesis_dir.exists():
        logging.error(f"Data synthesis directory {data_synthesis_dir} does not exist.")

    evaluation_dataset_dir = args.evaluation_dataset_dir
    if args.dataset_source == "local" and not evaluation_dataset_dir.exists():
        logging.error(
            f"Evaluation dataset directory {evaluation_dataset_dir} does not exist."
        )

    model_configs = create_work_space(
        work_dir=work_dir,
        data_synthesis_dir=data_synthesis_dir,
        evaluation_dataset_dir=evaluation_dataset_dir,
        dataset_source=args.dataset_source,
        dataset_id=args.dataset_id,
        tokenizer_algorithms=args.tokenizer_algorithm,
        vocab_sizes=args.target_vocab_size,
    )

    script_path = create_script(work_dir=work_dir, model_configs=model_configs)

    logging.info(f"✅ Initialization complete. Script created at: {script_path}")


if __name__ == "__main__":
    main()
