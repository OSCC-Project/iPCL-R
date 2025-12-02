# Flow Module

The flow module implements a comprehensive 3-stage machine learning pipeline for routing pattern generation: tokenization � training � evaluation. It provides end-to-end infrastructure for transforming chip routing data into trained transformer models.

## Overview

This module orchestrates the complete machine learning workflow, from raw routing sequences to trained models capable of generating new routing patterns. It integrates domain-specific tokenization, transformer training, and routing-specific evaluation metrics.

## Pipeline Architecture

### Stage 1: Tokenization
**Transform routing sequences into machine learning-ready token representations**

- **UnifiedTokenizer**: Supports 5 algorithms with domain-specific optimizations
  - DecimalWordLevel: Coordinate-based tokenization for spatial reasoning
  - Seg-BPE/Concat-BPE: Byte-pair encoding variants with segmentation strategies
  - Seg-BBPE/Concat-BBPE: Balanced byte-pair encoding for improved coverage
- **Direction Encoding**: Converts 3D coordinates to directional tokens (R/L/U/D/T/B)
- **Tree Structure**: Handles routing trees with PUSH/POP and BRANCH/END tokens
- **Special Tokens**: BOS/EOS, PAD, DRIVER/LOAD for routing semantics

### Stage 2: Training  
**Train sequence-to-sequence transformer models for routing generation**

- **Architecture**: T5-Gemma encoder-decoder transformer with domain adaptations
- **Distributed Training**: HuggingFace Accelerate integration for multi-GPU setups
- **Optimization**: Multiple optimizers (AdamW, Adafactor, Lion) with advanced scheduling
- **Monitoring**: TensorBoard integration with comprehensive logging and checkpointing

### Stage 3: Evaluation
**Comprehensive assessment using NLP and routing-specific metrics**

- **Multi-metric Framework**: ROUGE, BLEU, exact match for sequence similarity
- **Routing Metrics**: RED (Routing Edit Distance), coordinate accuracy validation
- **Structure Validation**: Tree integrity checking, connectivity analysis
- **EDA Integration**: DEF format output for industry tool verification

## Core Components

### FlowConfig
**Centralized configuration management for all pipeline stages**

- JSON-based configuration with inheritance and templating
- Path management and prefix replacement for different environments
- Hyperparameter organization by stage and component
- Validation and default value handling

#### Configuration Reference (flow/config.py)
Use `dataset` to choose between Hugging Face Hub or local folders. Legacy path fields are removed; paths live in the dataset section.

**DatasetConfig**

| Field              | Type             | Default                                        | Meaning                                                                  |
| ------------------ | ---------------- | ---------------------------------------------- | ------------------------------------------------------------------------ |
| `source`           | `hub` \| `local` | `hub`                                          | Dataset source; use `hub` for Hugging Face, `local` for on-disk folders. |
| `hub_id`           | str              | `AiEDA/iPCL-R`                                 | HF dataset id.                                                           |
| `train_split`      | str              | `train`                                        | Split name for training/tokenization.                                    |
| `validation_split` | str              | `validation`                                   | Split name for evaluation.                                               |
| `train_local_dir`  | str              | `/path/to/data_synthesis`                      | Local directory for train split (when `source=local`).                   |
| `eval_local_dir`   | str              | `/path/to/stage_evaluation/evaluation_dataset` | Local directory for eval split (when `source=local`).                    |

**TokenizationStageConfig**

| Section     | Field                    | Default                                            | Meaning                                                                                       |
| ----------- | ------------------------ | -------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| paths       | `token_dataset_dir`      | `/path/to/stage_tokenization/token_dataset`        | Output HF dataset path.                                                                       |
|             | `tokenizer_save_dir`     | `/path/to/stage_tokenization/tokenizer`            | Trained tokenizer save dir.                                                                   |
|             | `output_metadata_path`   | `/path/to/stage_tokenization/output_metadata.json` | Metadata JSON path.                                                                           |
| workflow    | `tokenizer_algorithm`    | `DecimalWordLevel`                                 | Tokenizer algorithm (`DecimalWordLevel`, `Seg-BPE`, `Concat-BPE`, `Seg-BBPE`, `Concat-BBPE`). |
|             | `target_vocab_size`      | `0`                                                | Vocab size target (0 keeps default).                                                          |
|             | `max_sequence_length`    | `1024`                                             | Max seq length for tokenizer training.                                                        |
|             | `save_metadata`          | `True`                                             | Whether to emit metadata JSON.                                                                |
| performance | `num_workers`            | `16`                                               | Map/process workers.                                                                          |
|             | `batch_size`             | `1000`                                             | Batch size for preprocessing.                                                                 |
| advanced    | `overlap_info_require`   | `False`                                            | Require overlap info.                                                                         |
|             | `overlap_top_k`          | `3`                                                | Top-k overlap items.                                                                          |
|             | `connected_info_require` | `False`                                            | Require connectivity info.                                                                    |
|             | `connected_top_k`        | `3`                                                | Top-k connectivity items.                                                                     |
|             | `use_coord_sorted_input` | `True`                                             | Sort coordinates before processing.                                                           |
| root        | `log_level`              | `INFO`                                             | Logging level for tokenization.                                                               |

**TrainingStageConfig**

| Section         | Field                         | Default                                 | Meaning                       |
| --------------- | ----------------------------- | --------------------------------------- | ----------------------------- |
| paths           | `split_dataset_dir`           | `/path/to/stage_training/split_dataset` | Cached train/val splits.      |
|                 | `model_save_dir`              | `/path/to/stage_training/model`         | Model checkpoint dir.         |
|                 | `logging_dir`                 | `/path/to/stage_training/logs`          | Training logs.                |
| model           | `hidden_size`                 | `256`                                   | Model hidden size.            |
|                 | `intermediate_size`           | `1024`                                  | FFN size.                     |
|                 | `num_hidden_layers`           | `4`                                     | Encoder/decoder layers.       |
|                 | `num_attention_heads`         | `4`                                     | Attention heads.              |
|                 | `num_key_value_heads`         | `2`                                     | KV heads.                     |
|                 | `head_dim`                    | `64`                                    | Per-head dimension.           |
|                 | `max_position_embeddings`     | `512`                                   | Positional embedding length.  |
|                 | `sliding_window`              | `256`                                   | Sliding window span.          |
|                 | `dropout_rate`                | `0.1`                                   | Dropout probability.          |
| hyperparameters | `max_src_len`                 | `512`                                   | Max encoder length.           |
|                 | `max_tgt_len`                 | `512`                                   | Max decoder length.           |
|                 | `train_split_ratio`           | `0.9`                                   | Train/test split ratio.       |
|                 | `num_train_epochs`            | `10`                                    | Epochs.                       |
|                 | `batch_size_per_device`       | `64`                                    | Per-device batch size.        |
|                 | `gradient_accumulation_steps` | `1`                                     | Grad accumulation steps.      |
|                 | `learning_rate`               | `1e-4`                                  | Base LR.                      |
|                 | `weight_decay`                | `0.005`                                 | Weight decay.                 |
|                 | `warmup_ratio`                | `0.05`                                  | LR warmup ratio.              |
|                 | `max_grad_norm`               | `1.0`                                   | Gradient clipping norm.       |
|                 | `optimizer_type`              | `adafactor`                             | Optimizer choice.             |
|                 | `scheduler_type`              | `adafactor`                             | Scheduler choice.             |
|                 | `eval_strategy`               | `epoch`                                 | Evaluation cadence.           |
|                 | `save_strategy`               | `epoch`                                 | Checkpoint cadence.           |
|                 | `early_stopping_patience`     | `3`                                     | Early stop patience (epochs). |
|                 | `logging_strategy`            | `steps`                                 | Logging cadence type.         |
|                 | `logging_steps`               | `100`                                   | Logging interval.             |
|                 | `seed`                        | `42`                                    | Random seed.                  |
| performance     | `num_workers`                 | `16`                                    | Preprocessing workers.        |
|                 | `batch_size`                  | `1000`                                  | Preprocessing batch size.     |
|                 | `dataloader_num_workers`      | `16`                                    | DataLoader workers.           |
|                 | `dataloader_pin_memory`       | `True`                                  | Pin memory flag.              |
|                 | `resume_from_checkpoint`      | `False`                                 | Resume flag.                  |
| root            | `log_level`                   | `INFO`                                  | Logging level for training.   |

**EvaluationStageConfig**

| Section     | Field                         | Default                             | Meaning                       |
| ----------- | ----------------------------- | ----------------------------------- | ----------------------------- |
| paths       | `output_dir`                  | `/path/to/stage_evaluation`         | Evaluation output dir.        |
|             | `metrics_dir`                 | `/path/to/stage_evaluation/metrics` | Metrics export dir.           |
|             | `plots_dir`                   | `/path/to/stage_evaluation/plots`   | Plot export dir.              |
|             | `logging_dir`                 | `/path/to/stage_evaluation/logs`    | Eval logs.                    |
| generation  | `max_new_tokens`              | `1024`                              | Max generation length.        |
|             | `num_beams`                   | `4`                                 | Beam search width.            |
|             | `do_sample`                   | `False`                             | Enable sampling.              |
|             | `temperature`                 | `0.9`                               | Sampling temperature.         |
|             | `top_p`                       | `1.0`                               | Nucleus sampling p.           |
|             | `top_k`                       | `50`                                | Top-k sampling k.             |
|             | `repetition_penalty`          | `1.0`                               | Repetition penalty.           |
|             | `length_penalty`              | `1.0`                               | Beam length penalty.          |
|             | `early_stopping`              | `True`                              | Stop beams early.             |
| metrics     | `calculate_rouge`             | `True`                              | Compute ROUGE.                |
|             | `calculate_bleu`              | `True`                              | Compute BLEU.                 |
|             | `calculate_exact_match`       | `True`                              | Exact-match metric.           |
|             | `calculate_domain_metrics`    | `True`                              | Domain metrics flag.          |
|             | `use_coordinate_parsing`      | `True`                              | Enable coord parsing.         |
|             | `use_tree_structure_analysis` | `True`                              | Enable tree analysis.         |
|             | `use_routing_metrics`         | `True`                              | Enable routing metrics.       |
| performance | `num_workers`                 | `16`                                | Eval preprocessing workers.   |
|             | `batch_size`                  | `64`                                | Eval batch size.              |
|             | `dataloader_num_workers`      | `16`                                | Eval DataLoader workers.      |
|             | `dataloader_pin_memory`       | `True`                              | Pin memory flag.              |
| output      | `save_predictions`            | `True`                              | Save predictions.             |
|             | `save_metrics`                | `True`                              | Save metrics.                 |
|             | `num_demo_examples`           | `5`                                 | Demo sample count.            |
| root        | `log_level`                   | `INFO`                              | Logging level for evaluation. |

### Launchers
**Command-line entry points for each pipeline stage**

- `launch_tokenization.py`: Stage 1 execution with progress reporting
- `launch_training.py`: Stage 2 with distributed training support
- `launch_evaluation.py`: Stage 3 with accelerated inference

### Pipelines
**Core processing logic for each stage**

- **TokenizationPipeline**: Corpus preprocessing, tokenizer training, dataset conversion
- **TrainingPipeline**: Model initialization, training loop, checkpoint management
- **EvaluationPipeline**: Inference, metrics calculation, result aggregation

## Usage

### Pipeline Configuration
```bash
# Generate initial configuration
python -m flow.pipeline_init --create-flow-config config.json

# Customize paths and hyperparameters
vim config.json
```

### Sequential Execution
```bash
# Stage 1: Tokenization
python -m flow.launch_tokenization --flow-config config.json

# Stage 2: Training (distributed)
accelerate launch -m flow.launch_training --flow-config config.json

# Stage 3: Evaluation (distributed)
accelerate launch -m flow.launch_evaluation --flow-config config.json
```

## Key Features

### Domain-Specific Optimizations
- Spatial reasoning through coordinate-based tokenization
- Routing tree structure preservation with specialized tokens
- Direction-aware encoding for 3D chip layouts

### Production-Ready Infrastructure  
- Fault-tolerant training with automatic checkpointing
- Memory-efficient data loading with configurable workers
- Comprehensive logging and monitoring integration

### Evaluation Rigor
- Multi-level validation: syntactic, semantic, and structural
- Industry-standard EDA tool compatibility
- Statistical significance testing for model comparisons

## Integration

The flow module serves as the central orchestrator:
- **Input**: ML-ready datasets from the data synthesis module
- **Validation**: Experimental analysis from the experiments module  
- **Output**: Trained models and evaluation results for downstream applications

Results include trained tokenizers, transformer models, and comprehensive evaluation reports.
