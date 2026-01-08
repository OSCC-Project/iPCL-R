#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   training.py
@Time    :   2025/10/15 11:42:11
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Launch GRPO fine-tuning on routing models with wirelength-driven rewards.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

from accelerate import PartialState
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import PreTrainedTokenizerFast, T5GemmaForConditionalGeneration
from trl import GRPOConfig

from experiments.grpo_ft.grpo_t5gemma_trainer import GRPOEncoderDecoderTrainer
from experiments.grpo_ft.rewards import create_reward
from flow import FlowConfig
from flow.tokenization import UnifiedTokenizer
from flow.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GRPO fine-tuning with wirelength-aware rewards",
    )
    parser.add_argument(
        "--flow-config-path",
        type=Path,
        default=Path(
            "/mnt/local_data1/liweiguo/experiments/model_size_comparison/work_dir/Large-DecimalWordLevel/flow_config.json"
        ),
        help="Path to the flow configuration file for the Large-DecimalWordLevel experiment.",
    )
    parser.add_argument(
        "--ft-config-path",
        type=Path,
        default=Path("./experiments/grpo_ft/flow_config.json"),
        help="Path to the flow configuration file for the GRPO fine-tuning experiment.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum number of GRPO optimization steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Learning rate for GRPO updates.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=128,
        help="Per-device batch size for GRPO rollouts.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps for GRPO updates.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.02,
        help="KL penalty coefficient for GRPO (controls divergence from reference).",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging interval for training metrics.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Checkpoint saving interval.",
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default="gated_timing_composite",
        choices=[
            "wirelength",
            "adaptive_wl_via",
            "connectivity",
            "graceful",
            "elmore_delay",
            "gated_wl_composite",
            "gated_timing_composite",
            "weighted_timing_composite",
        ],
        help=(
            "Reward type to use. Options: wirelength, adaptive_wl_via, connectivity, "
            "graceful, elmore_delay, gated_wl_composite (wirelength/via improvement gated "
            "by connectivity & graceful), gated_timing_composite (Elmore delay improvement "
            "gated by connectivity & graceful), weighted_timing_composite (weighted sum of "
            "Elmore delay, connectivity, and graceful)."
        ),
    )
    parser.add_argument(
        "--via-weight",
        type=float,
        default=1.0,
        help="Weight applied to via-count improvement when computing adaptive reward.",
    )
    parser.add_argument(
        "--composite-weights",
        type=str,
        default="0.8,0.1,0.1",
        help="Comma-separated weights for composite reward: improvement,connectivity,graceful (e.g., '0.8,0.1,0.1')",
    )
    parser.add_argument(
        "--connectivity-penalty",
        type=float,
        default=-1.0,
        help="Penalty when connectivity check fails.",
    )
    parser.add_argument(
        "--graceful-penalty",
        type=float,
        default=-1.0,
        help="Penalty when graceful check fails.",
    )
    parser.add_argument(
        "--wirelength-failure-penalty",
        type=float,
        default=-0.2,
        help="Penalty when wirelength parsing fails (for WirelengthReward).",
    )
    parser.add_argument(
        "--elmore-unit-resistance",
        type=float,
        default=1.0,
        help="Unit resistance per Manhattan step for Elmore delay (normalized ohms).",
    )
    parser.add_argument(
        "--elmore-unit-capacitance",
        type=float,
        default=1.0,
        help="Unit capacitance per Manhattan step for Elmore delay (normalized farads).",
    )
    parser.add_argument(
        "--elmore-load-capacitance",
        type=float,
        default=1.0,
        help="Load capacitance placed on each sink when computing Elmore delay (normalized farads).",
    )
    parser.add_argument(
        "--elmore-failure-penalty",
        type=float,
        default=-0.2,
        help="Penalty when Elmore delay parsing fails (for ElmoreDelayReward).",
    )
    parser.add_argument(
        "--elmore-improvement-clip",
        type=float,
        default=0.5,
        help="Clamp for normalized Elmore delay improvement.",
    )
    parser.add_argument(
        "--elmore-improvement-scale",
        type=float,
        default=5.0,
        help="Scale factor applied to normalized Elmore delay improvement.",
    )
    parser.add_argument(
        "--connectivity-mode",
        type=str,
        default="binary",
        choices=["binary", "ratio"],
        help="Connectivity scoring mode: 'binary' for pass/fail, 'ratio' for partial credit.",
    )
    parser.add_argument(
        "--graceful-mode",
        type=str,
        default="binary",
        choices=["binary", "ratio"],
        help="Graceful scoring mode: 'binary' for pass/fail, 'ratio' for partial credit.",
    )
    parser.add_argument(
        "--log-reward-components",
        dest="log_reward_components",
        action="store_true",
        default=True,
        help="Log per-component reward statistics (adaptive/connectivity/graceful/mask) to TensorBoard (default: on).",
    )
    parser.add_argument(
        "--no-log-reward-components",
        dest="log_reward_components",
        action="store_false",
        help="Disable logging of per-component reward statistics.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on the number of training samples for debugging.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=None,
        help="Override for maximum prompt length (defaults to flow config).",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=128,
        help="Maximum completion length for generation (overrides flow config).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping value applied to GRPO updates.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 training (disabled by default).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable BF16 training if supported by hardware.",
    )
    return parser.parse_args()


def load_flow_config(flow_config_path: Path) -> FlowConfig:
    if not flow_config_path.exists():
        raise FileNotFoundError(f"Flow config not found: {flow_config_path}")
    return FlowConfig.from_config_file(flow_config_path)


def _ensure_dataset(dataset: Union[DatasetDict, Dataset]) -> Dataset:
    if isinstance(dataset, DatasetDict):
        return dataset["train"]
    return dataset


def prepare_grpo_dataset(
    config: FlowConfig,
    max_completion_length: Optional[int] = None,
    max_train_samples: Optional[int] = None,
) -> Dataset:
    token_dataset_dir = Path(config.tokenization.paths.token_dataset_dir)
    if not token_dataset_dir.exists():
        raise FileNotFoundError(f"Token dataset not found: {token_dataset_dir}")

    logging.info("Loading token dataset from %s", token_dataset_dir)
    dataset = load_from_disk(str(token_dataset_dir))
    dataset = _ensure_dataset(dataset)

    if max_completion_length is not None:
        original_count = len(dataset)

        def _within_length(example):
            tokens = example.get("target_tokens")
            if tokens is None:
                return False
            if isinstance(tokens, list):
                return len(tokens) <= max_completion_length
            return len(str(tokens).split()) <= max_completion_length

        dataset = dataset.filter(_within_length)
        filtered_count = len(dataset)
        logging.info(
            "Filtered dataset by max completion length %d: %d -> %d samples",
            max_completion_length,
            original_count,
            filtered_count,
        )

    hyper = config.training.hyperparameters
    split_ratio = hyper.train_split_ratio
    seed = hyper.seed

    if 0.0 < split_ratio < 1.0:
        dataset_dict = dataset.train_test_split(
            test_size=1.0 - split_ratio,
            seed=seed,
        )
        train_dataset = dataset_dict["train"]
    else:
        train_dataset = dataset

    if max_train_samples is not None:
        train_dataset = train_dataset.shuffle(seed=seed).select(
            range(max_train_samples)
        )
        logging.info("Using a subset of %d samples for training", len(train_dataset))

    def build_prompt_completion(batch):
        prompts: List[str] = []
        completions: List[str] = []
        for tokens in batch["source_tokens"]:
            if isinstance(tokens, list):
                prompts.append(" ".join(tokens))
            else:
                prompts.append(str(tokens))
        for tokens in batch["target_tokens"]:
            if isinstance(tokens, list):
                completions.append(" ".join(tokens))
            else:
                completions.append(str(tokens))
        return {"prompt": prompts, "completion": completions}

    train_dataset = train_dataset.map(
        build_prompt_completion,
        batched=True,
        desc="Building GRPO prompts",
    )

    keep_columns = {
        "prompt",
        "completion",
        "source_tokens",
        "target_tokens",
        "relative_tree_seq",
        "relative_loads",  # Required by ConnectivityReward and GracefulReward
        "driver",
        "loads",
        "net_name",
    }
    removable = [
        column for column in train_dataset.column_names if column not in keep_columns
    ]
    if removable:
        train_dataset = train_dataset.remove_columns(removable)

    logging.info("Prepared GRPO dataset with %d samples", len(train_dataset))
    return train_dataset


def load_unified_tokenizer(config: FlowConfig) -> UnifiedTokenizer:
    tokenizer_dir = Path(config.tokenization.paths.tokenizer_save_dir)
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")
    unified_tokenizer = UnifiedTokenizer.from_pretrained(tokenizer_dir)
    if unified_tokenizer.tokenizer is None:
        raise RuntimeError("Unified tokenizer failed to load underlying tokenizer.")
    logging.info("Unified tokenizer loaded from %s", tokenizer_dir)
    return unified_tokenizer


def load_model(
    config: FlowConfig,
) -> PeftModel:
    model_dir = Path(config.training.paths.model_save_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    try:
        model = T5GemmaForConditionalGeneration.from_pretrained(str(model_dir))
        logging.info("Loaded model with value head from %s", model_dir)
        return get_peft_model(model, lora_config)
    except Exception as exc:
        checkpoints = sorted(
            [path for path in model_dir.glob("checkpoint-*") if path.is_dir()],
            key=lambda path: int(path.name.split("-")[-1]),
        )
        if not checkpoints:
            raise RuntimeError(f"Failed to load model from {model_dir}: {exc}") from exc
        last_checkpoint = checkpoints[-1]
        logging.info(
            "Falling back to last checkpoint at %s for value head initialization",
            last_checkpoint,
        )
        model = T5GemmaForConditionalGeneration.from_pretrained(str(last_checkpoint))
        return get_peft_model(model, lora_config)


def build_training_config(
    args: argparse.Namespace,
    tokenizer: PreTrainedTokenizerFast,
    flow_config: FlowConfig,
) -> Tuple[GRPOConfig, Path]:
    hyper = flow_config.training.hyperparameters
    paths = flow_config.training.paths
    generation_config = flow_config.evaluation.generation
    prompt_length = args.max_prompt_length or hyper.max_src_len
    completion_length = args.max_completion_length or hyper.max_tgt_len

    output_dir = Path(paths.model_save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if generation_config.max_new_tokens:
        max_new_tokens = min(generation_config.max_new_tokens, completion_length)
    else:
        max_new_tokens = completion_length

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        beta=args.beta,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=hyper.seed,
        max_prompt_length=prompt_length,
        max_completion_length=completion_length,
        fp16=args.fp16,
        bf16=args.bf16,
        generation_kwargs={
            "max_new_tokens": max_new_tokens,
            "num_beams": generation_config.num_beams,
            "do_sample": generation_config.do_sample,
            "temperature": generation_config.temperature
            if generation_config.do_sample
            else None,
            "top_p": generation_config.top_p if generation_config.do_sample else None,
            "top_k": generation_config.top_k if generation_config.do_sample else None,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "decoder_start_token_id": tokenizer.bos_token_id,
        },
    )

    logging.info(
        "GRPOConfig prepared: lr=%.2e, batch=%d, steps=%d, prompt_len=%d, completion_len=%d",
        args.learning_rate,
        args.per_device_train_batch_size,
        args.max_steps,
        prompt_length,
        completion_length,
    )

    training_args.log_reward_components = args.log_reward_components
    training_args.max_grad_norm = args.max_grad_norm

    return training_args, output_dir


def main() -> None:
    args = parse_args()
    args.log_level = args.log_level.upper()
    setup_logging(level=args.log_level)

    if args.fp16 and args.bf16:
        raise ValueError("Specify at most one of --fp16 or --bf16.")

    base_flow_config = load_flow_config(args.flow_config_path)
    ft_flow_config = load_flow_config(args.ft_config_path)

    tokenizer_wrapper = load_unified_tokenizer(base_flow_config)

    composite_weights = tuple(
        float(value.strip())
        for value in args.composite_weights.split(",")
        if value.strip()
    )
    if len(composite_weights) != 3:
        raise ValueError(
            "composite-weights must have three values: improvement,connectivity,graceful."
        )

    # Create reward function based on type
    reward_type = args.reward_type

    if reward_type == "gated_wl_composite":
        reward = create_reward(
            reward_type,
            tokenizer_wrapper,
            composite_weights=composite_weights,
            adaptive_kwargs={
                "via_weight": args.via_weight,
            },
            connectivity_kwargs={
                "penalty": args.connectivity_penalty,
                "use_continuous": args.connectivity_mode == "ratio",
            },
            graceful_kwargs={
                "penalty": args.graceful_penalty,
                "use_continuous": args.graceful_mode == "ratio",
            },
            wirelength_kwargs={
                "failure_penalty": args.wirelength_failure_penalty,
            },
        )
        logging.info(
            "Created gated wirelength composite reward (connectivity_mode=%s, wirelength_failure_penalty=%.2f)",
            args.connectivity_mode,
            args.wirelength_failure_penalty,
        )
    elif reward_type == "gated_timing_composite":
        reward = create_reward(
            reward_type,
            tokenizer_wrapper,
            elmore_kwargs={
                "unit_resistance": args.elmore_unit_resistance,
                "unit_capacitance": args.elmore_unit_capacitance,
                "load_capacitance": args.elmore_load_capacitance,
                "failure_penalty": args.elmore_failure_penalty,
                "improvement_clip": args.elmore_improvement_clip,
                "improvement_scale": args.elmore_improvement_scale,
            },
            connectivity_kwargs={
                "penalty": args.connectivity_penalty,
                "use_continuous": args.connectivity_mode == "ratio",
            },
            graceful_kwargs={
                "penalty": args.graceful_penalty,
                "use_continuous": args.graceful_mode == "ratio",
            },
        )
        logging.info(
            "Created gated timing composite reward (connectivity_mode=%s, graceful_mode=%s, elmore_unit_resistance=%.2f, elmore_unit_capacitance=%.2f, elmore_load_capacitance=%.2f)",
            args.connectivity_mode,
            args.graceful_mode,
            args.elmore_unit_resistance,
            args.elmore_unit_capacitance,
            args.elmore_load_capacitance,
        )
    elif reward_type == "weighted_timing_composite":
        reward = create_reward(
            reward_type,
            tokenizer_wrapper,
            composite_weights=composite_weights,
            elmore_kwargs={
                "unit_resistance": args.elmore_unit_resistance,
                "unit_capacitance": args.elmore_unit_capacitance,
                "load_capacitance": args.elmore_load_capacitance,
                "failure_penalty": args.elmore_failure_penalty,
                "improvement_clip": args.elmore_improvement_clip,
                "improvement_scale": args.elmore_improvement_scale,
            },
            connectivity_kwargs={
                "penalty": args.connectivity_penalty,
                "use_continuous": args.connectivity_mode == "ratio",
            },
            graceful_kwargs={
                "penalty": args.graceful_penalty,
                "use_continuous": args.graceful_mode == "ratio",
            },
        )
        logging.info(
            "Created weighted timing composite reward (connectivity_mode=%s, graceful_mode=%s, composite_weights=%s, elmore_unit_resistance=%.2f, elmore_unit_capacitance=%.2f, elmore_load_capacitance=%.2f)",
            args.connectivity_mode,
            args.graceful_mode,
            composite_weights,
            args.elmore_unit_resistance,
            args.elmore_unit_capacitance,
            args.elmore_load_capacitance,
        )
    else:
        reward_kwargs: dict = {}
        if reward_type == "wirelength":
            reward_kwargs["failure_penalty"] = args.wirelength_failure_penalty
        elif reward_type == "connectivity":
            reward_kwargs["penalty"] = args.connectivity_penalty
            reward_kwargs["use_continuous"] = args.connectivity_mode == "ratio"
        elif reward_type == "graceful":
            reward_kwargs["penalty"] = args.graceful_penalty
            reward_kwargs["use_continuous"] = args.graceful_mode == "ratio"
        elif reward_type == "adaptive_wl_via":
            reward_kwargs["via_weight"] = args.via_weight
            reward_kwargs["failure_penalty"] = args.wirelength_failure_penalty
        elif reward_type == "elmore_delay":
            reward_kwargs = {
                "unit_resistance": args.elmore_unit_resistance,
                "unit_capacitance": args.elmore_unit_capacitance,
                "load_capacitance": args.elmore_load_capacitance,
                "failure_penalty": args.elmore_failure_penalty,
                "improvement_clip": args.elmore_improvement_clip,
                "improvement_scale": args.elmore_improvement_scale,
            }

        reward = create_reward(
            reward_type,
            tokenizer_wrapper,
            **reward_kwargs,
        )
        logging.info("Created %s reward", reward_type)

    with PartialState().main_process_first():
        train_dataset = prepare_grpo_dataset(
            base_flow_config,
            max_completion_length=args.max_completion_length,
            max_train_samples=args.max_train_samples,
        )

    model = load_model(base_flow_config)

    tokenizer = tokenizer_wrapper.tokenizer

    training_args, training_output_dir = build_training_config(
        args,
        tokenizer,
        ft_flow_config,
    )

    trainer = GRPOEncoderDecoderTrainer(
        model=model,
        reward_funcs=reward,
        args=training_args,
        train_dataset=train_dataset,
    )

    logging.info(
        "Starting GRPO fine-tuning. Outputs will be saved to %s", training_output_dir
    )
    trainer.train()
    logging.info("GRPO fine-tuning finished.")


if __name__ == "__main__":
    main()
