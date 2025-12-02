#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   pipeline.py
@Time    :   2025/08/01 11:15:24
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Model training pipeline with HuggingFace Transformers,
             Accelerate distributed training, custom model architecture configuration,
             and comprehensive training loop management
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from accelerate import Accelerator, PartialState
from datasets import Dataset, load_from_disk
from lion_pytorch import Lion
from torch.optim import AdamW
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5GemmaConfig,
    T5GemmaForConditionalGeneration,
    T5GemmaModuleConfig,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    set_seed,
)
from transformers.optimization import Adafactor, AdafactorSchedule

from flow.config import FlowConfig


class TrainingPipeline:
    """Model training pipeline for net routing generation (Stage 3)"""

    def __init__(self, accelerator: Accelerator, flow_config: FlowConfig):
        self.accelerator = accelerator
        self.flow_config = flow_config
        self.tokenization_paths_config = flow_config.tokenization.paths
        self.training_config = flow_config.training
        self.paths_config = self.training_config.paths
        self.model_config = self.training_config.model
        self.hyperparameters_config = self.training_config.hyperparameters
        self.performance_config = self.training_config.performance

        self.model = None
        self.tokenizer = None
        self.trainer = None

        set_seed(self.hyperparameters_config.seed)

    def run_training(self) -> Dict[str, Any]:
        """Run the complete training pipeline (Stage main function)"""
        start_time = time.time()
        logging.info("Starting model training pipeline")

        # Load tokenizer
        self.tokenizer = self._load_tokenizer()

        # Split dataset into train/validation
        train_dataset, eval_dataset = self._load_or_create_dataset()

        # Initialize model
        self.model = self._initialize_T5Gemma_model(self.tokenizer)

        # Setup training arguments
        training_args = self._setup_training_arguments()

        # Initialize trainer
        self.trainer = self._initialize_trainer(
            self.model, self.tokenizer, train_dataset, eval_dataset, training_args
        )

        # Train the model
        training_results = self._train_model(self.trainer)

        total_time = time.time() - start_time
        logging.info(f"Training pipeline completed in {total_time:.2f}s")

        return {
            "training_results": training_results,
            "model_path": self.paths_config.model_save_dir,
        }

    def _load_tokenizer(self) -> PreTrainedTokenizerFast:
        """Load the tokenizer from the configured path"""
        tokenizer_path = self.tokenization_paths_config.tokenizer_save_dir
        logging.info(f"Loading tokenizer from {tokenizer_path}")

        with PartialState().main_process_first():
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
            logging.info(
                f"Tokenizer loaded from {tokenizer_path} (vocab size: {len(tokenizer.get_vocab())})"
            )

        return tokenizer

    def _load_or_create_dataset(self) -> Tuple[Dataset, Dataset]:
        """Split dataset into train and validation sets"""
        split_dataset_dir = Path(self.paths_config.split_dataset_dir)
        if split_dataset_dir.exists():
            try:
                logging.info(f"Loading pre-split dataset from {split_dataset_dir}")

                with PartialState().main_process_first():
                    split_dataset = load_from_disk(split_dataset_dir)
                train_dataset = split_dataset["train"]
                eval_dataset = split_dataset["test"]
                logging.info(
                    f"Loaded split dataset: {len(train_dataset)} train, {len(eval_dataset)} eval"
                )
                return train_dataset, eval_dataset
            except Exception as e:
                logging.warning(
                    f"Failed to load split dataset: {e}. Re-creating dataset."
                )

        token_dataset_path = self.tokenization_paths_config.token_dataset_dir
        logging.info(f"Loading token dataset from {token_dataset_path}")

        with PartialState().main_process_first():
            dataset = load_from_disk(token_dataset_path)
            logging.info(
                f"Splitting dataset (ratio: {self.hyperparameters_config.train_split_ratio})"
            )

        # Truncate dataset
        def tokenize_sample(batch):
            source_encs = self.tokenizer(
                batch["source_tokens"],
                truncation=True,
                max_length=self.hyperparameters_config.max_src_len,
                add_special_tokens=False,
            )
            labels = self.tokenizer(
                text_target=batch["target_tokens"],
                truncation=True,
                max_length=self.hyperparameters_config.max_tgt_len,
                add_special_tokens=False,
            )

            # Convert pad_token to -100 for labels
            labels["input_ids"] = [
                (tok if tok != self.tokenizer.pad_token_id else -100)
                for tok in labels["input_ids"]
            ]

            batch["input_ids"] = source_encs["input_ids"]
            batch["attention_mask"] = source_encs["attention_mask"]
            batch["labels"] = labels["input_ids"]

            return batch

        with PartialState().main_process_first():
            tokenized_dataset = dataset.map(
                tokenize_sample,
                batched=True,
                num_proc=self.performance_config.num_workers,
                remove_columns=dataset.column_names,
                desc="Tokenizing dataset",
            )

        # Create train/eval split
        shuffled_dataset = tokenized_dataset.shuffle(
            seed=self.hyperparameters_config.seed
        )
        split_dataset = shuffled_dataset.train_test_split(
            test_size=1.0 - self.hyperparameters_config.train_split_ratio,
            seed=self.hyperparameters_config.seed,
        )

        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        logging.info(
            f"Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval"
        )

        # Save split datasets if configured
        if self.accelerator.is_main_process and self.paths_config.split_dataset_dir:
            split_dataset.save_to_disk(self.paths_config.split_dataset_dir)
            logging.info(
                f"Saved split dataset to {self.paths_config.split_dataset_dir}"
            )

        self.accelerator.wait_for_everyone()

        return train_dataset, eval_dataset

    def _initialize_T5Gemma_model(
        self, tokenizer: PreTrainedTokenizerFast
    ) -> T5GemmaForConditionalGeneration:
        """Initialize model with configuration matching the original version"""
        logging.info("Initializing model for net routing generation")
        logging.info("vocab size: %s", len(tokenizer.get_vocab()))

        encoder_config = T5GemmaModuleConfig(
            hidden_size=self.model_config.hidden_size,
            intermediate_size=self.model_config.intermediate_size,
            num_hidden_layers=self.model_config.num_hidden_layers,
            num_attention_heads=self.model_config.num_attention_heads,
            num_key_value_heads=self.model_config.num_key_value_heads,
            head_dim=self.model_config.head_dim,
            max_position_embeddings=self.model_config.max_position_embeddings,
            sliding_window=self.model_config.sliding_window,
            tie_word_embeddings=True,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=tokenizer.bos_token_id,
            attn_implementation="eager",
            use_cache=True,
        )

        decoder_config = T5GemmaModuleConfig(
            hidden_size=self.model_config.hidden_size,
            intermediate_size=self.model_config.intermediate_size,
            num_hidden_layers=self.model_config.num_hidden_layers,
            num_attention_heads=self.model_config.num_attention_heads,
            num_key_value_heads=self.model_config.num_key_value_heads,
            head_dim=self.model_config.head_dim,
            max_position_embeddings=self.model_config.max_position_embeddings,
            sliding_window=self.model_config.sliding_window,
            tie_word_embeddings=True,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=tokenizer.bos_token_id,
            attn_implementation="eager",
            use_cache=True,
        )

        # Create model configuration matching the original version exactly
        config = T5GemmaConfig(
            encoder=encoder_config,
            decoder=decoder_config,
            hidden_size=self.model_config.hidden_size,
            query_pre_attn_scalar=self.model_config.hidden_size,
            is_encoder_decoder=True,
            dropout_rate=self.model_config.dropout_rate,
            tie_word_embeddings=True,
            vocab_size=len(tokenizer.get_vocab()),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=tokenizer.bos_token_id,
            attn_implementation="eager",
            use_cache=True,
        )

        # Create model
        model = T5GemmaForConditionalGeneration(config)

        # Log model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        unit, scale = ("B", 1e9) if total_params >= 1e9 else ("M", 1e6)

        logging.info(
            f"Model Parameters | Total: {total_params / scale:.2f}{unit} | "
            f"Trainable: {trainable_params / scale:.2f}{unit}"
        )

        return model

    def _setup_training_arguments(self) -> Seq2SeqTrainingArguments:
        """Setup training arguments for the model"""
        args = Seq2SeqTrainingArguments(
            output_dir=self.paths_config.model_save_dir,
            overwrite_output_dir=True,
            # Training hyperparameters
            num_train_epochs=self.hyperparameters_config.num_train_epochs,
            per_device_train_batch_size=self.hyperparameters_config.batch_size_per_device,
            per_device_eval_batch_size=self.hyperparameters_config.batch_size_per_device,
            ddp_find_unused_parameters=False,
            max_grad_norm=self.hyperparameters_config.max_grad_norm,
            gradient_accumulation_steps=1,
            # Evaluation and saving
            eval_strategy=self.hyperparameters_config.eval_strategy,
            save_strategy=self.hyperparameters_config.save_strategy,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Logging
            logging_dir=os.path.join(
                self.paths_config.logging_dir,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            ),
            logging_strategy=self.hyperparameters_config.logging_strategy,
            logging_steps=self.hyperparameters_config.logging_steps,
            # Data loading
            dataloader_num_workers=self.performance_config.dataloader_num_workers,
            dataloader_pin_memory=self.performance_config.dataloader_pin_memory,
            # Mixed precision and optimization
            fp16=torch.cuda.is_available(),
            report_to=["tensorboard"],
            # Reproducibility
            seed=self.hyperparameters_config.seed,
        )

        logging.info("Training arguments configured.")
        return args

    def _get_optimizer_and_scheduler(
        self, model, train_dataset_size: int
    ) -> Tuple[Any, Any]:
        """Get optimizer and scheduler based on configuration"""

        logging.info(f"Optimizer: {self.hyperparameters_config.optimizer_type}")
        logging.info(f"Scheduler: {self.hyperparameters_config.scheduler_type}")
        # Setup optimizer
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        global_batch_size = (
            self.hyperparameters_config.batch_size_per_device * world_size
        )
        steps_per_epoch = train_dataset_size // global_batch_size
        num_training_steps = (
            steps_per_epoch * self.hyperparameters_config.num_train_epochs
        )
        if self.hyperparameters_config.optimizer_type == "adamw":
            optimizer = AdamW(
                model.parameters(),
                lr=self.hyperparameters_config.learning_rate,
                weight_decay=self.hyperparameters_config.weight_decay,
            )
        elif self.hyperparameters_config.optimizer_type == "adafactor":
            optimizer = Adafactor(
                model.parameters(),
                relative_step=True,
                scale_parameter=True,
                warmup_init=True,
            )
        elif self.hyperparameters_config.optimizer_type == "lion":
            optimizer = Lion(
                [
                    {
                        "params": [p for p in model.parameters() if p.requires_grad],
                        "weight_decay": self.hyperparameters_config.weight_decay,
                    }
                ],
                lr=self.hyperparameters_config.learning_rate,
            )
        else:
            logging.warning(
                f"Optimizer {self.hyperparameters_config.optimizer_type} not available, falling back to AdamW"
            )
            optimizer = AdamW(
                model.parameters(),
                lr=self.hyperparameters_config.learning_rate,
                weight_decay=self.hyperparameters_config.weight_decay,
            )

        # Setup scheduler
        if self.hyperparameters_config.scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(
                    num_training_steps * self.hyperparameters_config.warmup_ratio
                ),
                num_training_steps=num_training_steps,
            )
        elif self.hyperparameters_config.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(
                    num_training_steps * self.hyperparameters_config.warmup_ratio
                ),
                num_training_steps=num_training_steps,
            )
        elif (
            self.hyperparameters_config.scheduler_type == "adafactor"
            and self.hyperparameters_config.optimizer_type == "adafactor"
        ):
            scheduler = AdafactorSchedule(optimizer)
        else:
            scheduler = None

        return optimizer, scheduler

    def _initialize_trainer(
        self,
        model,
        tokenizer: PreTrainedTokenizerFast,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        training_args: Seq2SeqTrainingArguments,
    ) -> Seq2SeqTrainer:
        """Initialize the Seq2Seq trainer (following original approach)"""
        # Callbacks
        callbacks = []
        if self.hyperparameters_config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.hyperparameters_config.early_stopping_patience
                )
            )

        # Initialize trainer
        optimizer, scheduler = self._get_optimizer_and_scheduler(
            model, len(train_dataset)
        )

        # Initialize data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer, model, pad_to_multiple_of=8, padding=True
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizers=(optimizer, scheduler),
            data_collator=data_collator,
            callbacks=callbacks,
        )

        logging.info("Seq2Seq trainer initialized")
        return trainer

    def _train_model(self, trainer: Seq2SeqTrainer) -> Dict[str, Any]:
        """Train the model"""
        logging.info("Starting model training")

        # Resume from checkpoint if specified
        if self.performance_config.resume_from_checkpoint:
            logging.info("Resuming training from checkpoint...")

        # Train the model
        train_result = trainer.train(
            resume_from_checkpoint=self.performance_config.resume_from_checkpoint
        )

        # Save training metrics
        metrics = train_result.metrics
        logging.info(f"Training completed. Final metrics: {metrics}")

        return {
            "train_result": train_result,
            "metrics": metrics,
            "log_history": trainer.state.log_history,
        }
