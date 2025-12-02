#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   config.py
@Time    :   2025/08/01 11:16:09
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   Unified configuration system
"""

import json
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

# =============================================================================
#                              DATASET CONFIGURATION
# =============================================================================


@dataclass
class DatasetConfig:
    """Dataset configuration supporting Hugging Face Hub and local paths"""

    source: str = "hub"  # Options: "hub" (Hugging Face) or "local"
    hub_id: str = "AiEDA/iPCL-R"
    train_split: str = "train"
    validation_split: str = "validation"
    train_local_dir: str = "/path/to/data_synthesis"
    eval_local_dir: str = "/path/to/stage_evaluation/evaluation_dataset"

    def resolve_split(self, split: Optional[str]) -> str:
        """Return the target split, falling back to defaults."""
        if split:
            return split
        return self.train_split

    def local_path_for_split(self, split: Optional[str]) -> Path:
        """Return the local directory to use for the requested split."""
        target_split = self.resolve_split(split)
        if target_split == self.validation_split:
            return Path(self.eval_local_dir)
        return Path(self.train_local_dir)

    def use_hub(self) -> bool:
        """Whether to load from the Hugging Face Hub."""
        return self.source.lower() == "hub"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DatasetConfig":
        """Create DatasetConfig from dictionary with backward compatibility."""
        return cls(
            source=config_dict.get("source", "hub"),
            hub_id=config_dict.get("hub_id", "AiEDA/iPCL-R"),
            train_split=config_dict.get("train_split", "train"),
            validation_split=config_dict.get("validation_split", "validation"),
            train_local_dir=config_dict.get(
                "train_local_dir",
                "/path/to/data_synthesis",
            ),
            eval_local_dir=config_dict.get(
                "eval_local_dir",
                "/path/to/stage_evaluation/evaluation_dataset",
            ),
        )


# =============================================================================
#                           TOKENIZATION CONFIGURATION
# =============================================================================


class TokenizationAlgorithm(Enum):
    """Tokenization methods used in experiments"""

    NONE = "None"  # No tokenization
    DECIMAL_WORD_LEVEL = "DecimalWordLevel"  # Traditional word-based tokenization
    SEG_BPE = "Seg-BPE"  # Segmented Byte Pair Encoding
    CONCAT_BPE = "Concat-BPE"  # Concatenated Byte Pair Encoding
    SEG_BBPE = "Seg-BBPE"  # Segmented Byte-level BPE
    CONCAT_BBPE = "Concat-BBPE"  # Concatenated Byte-level BPE


@dataclass
class TokenizationPaths:
    """Tokenization paths configuration"""

    token_dataset_dir: str = "/path/to/stage_tokenization/token_dataset"
    tokenizer_save_dir: str = "/path/to/stage_tokenization/tokenizer"
    output_metadata_path: str = "/path/to/stage_tokenization/output_metadata.json"


@dataclass
class TokenizationWorkflow:
    """Tokenization workflow configuration"""

    tokenizer_algorithm: str = "DecimalWordLevel"
    target_vocab_size: int = 0
    max_sequence_length: int = 1024
    save_metadata: bool = True


@dataclass
class TokenizationPerformance:
    """Tokenization performance configuration"""

    num_workers: int = 16
    batch_size: int = 1000


@dataclass
class TokenizationAdvanced:
    """Tokenization advanced configuration"""

    overlap_info_require: bool = False
    overlap_top_k: int = 3
    connected_info_require: bool = False
    connected_top_k: int = 3
    use_coord_sorted_input: bool = True


@dataclass
class TokenizationStageConfig:
    """Complete tokenization stage configuration"""

    paths: TokenizationPaths = field(default_factory=TokenizationPaths)
    workflow: TokenizationWorkflow = field(default_factory=TokenizationWorkflow)
    performance: TokenizationPerformance = field(
        default_factory=TokenizationPerformance
    )
    advanced: TokenizationAdvanced = field(default_factory=TokenizationAdvanced)
    log_level: str = (
        "INFO"  # Add log_level attribute for UnifiedTokenizer compatibility
    )

    def _dataclass_to_dict(self, obj: Any) -> Any:
        """Recursively convert dataclass objects to dictionaries"""
        if is_dataclass(obj):
            return {
                field.name: self._dataclass_to_dict(getattr(obj, field.name))
                for field in obj.__dataclass_fields__.values()
            }
        elif isinstance(obj, (list, tuple)):
            return [self._dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._dataclass_to_dict(value) for key, value in obj.items()}
        else:
            return obj

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary using recursive approach"""
        config_dict = self._dataclass_to_dict(self)
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TokenizationStageConfig":
        """Create TokenizationStageConfig from dictionary with automatic nested structure support"""
        try:
            # Extract nested sections
            paths_dict = config_dict.get("paths", {})
            workflow_dict = config_dict.get("workflow", {})
            performance_dict = config_dict.get("performance", {})
            advanced_dict = config_dict.get("advanced", {})

            # Create nested config objects
            paths = TokenizationPaths(**paths_dict)
            workflow = TokenizationWorkflow(**workflow_dict)
            performance = TokenizationPerformance(**performance_dict)
            advanced = TokenizationAdvanced(**advanced_dict)

            # Create instance with nested configs
            return cls(
                paths=paths,
                workflow=workflow,
                performance=performance,
                advanced=advanced,
                log_level=config_dict.get("log_level", "INFO"),
            )
        except Exception as e:
            raise ValueError(
                f"Failed to create TokenizationStageConfig from dictionary: {e}"
            )


# =============================================================================
#                             TRAINING CONFIGURATION
# =============================================================================


@dataclass
class TrainingPaths:
    """Training paths configuration"""

    split_dataset_dir: str = "/path/to/stage_training/split_dataset"
    model_save_dir: str = "/path/to/stage_training/model"
    logging_dir: str = "/path/to/stage_training/logs"


@dataclass
class TrainingModel:
    """Training model configuration"""

    hidden_size: int = 256
    intermediate_size: int = 1024
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 64
    # suggest: num_attention_heads * head_dim = hidden_size
    max_position_embeddings: int = 512
    sliding_window: int = 256
    dropout_rate: float = 0.1


@dataclass
class TrainingHyperparameters:
    """Training hyperparameters configuration"""

    max_src_len: int = 512
    max_tgt_len: int = 512
    train_split_ratio: float = 0.9
    num_train_epochs: int = 10
    batch_size_per_device: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.0001
    weight_decay: float = 0.005
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    optimizer_type: str = "adafactor"
    scheduler_type: str = "adafactor"
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    early_stopping_patience: int = 3
    logging_strategy: str = "steps"
    logging_steps: int = 100
    seed: int = 42


@dataclass
class TrainingPerformance:
    """Training performance configuration"""

    num_workers: int = 16
    batch_size: int = 1000
    dataloader_num_workers: int = 16
    dataloader_pin_memory: bool = True
    resume_from_checkpoint: bool = False


@dataclass
class TrainingStageConfig:
    """Complete training stage configuration"""

    paths: TrainingPaths = field(default_factory=TrainingPaths)
    model: TrainingModel = field(default_factory=TrainingModel)
    hyperparameters: TrainingHyperparameters = field(
        default_factory=TrainingHyperparameters
    )
    performance: TrainingPerformance = field(default_factory=TrainingPerformance)
    log_level: str = "INFO"

    def _dataclass_to_dict(self, obj: Any) -> Any:
        """Recursively convert dataclass objects to dictionaries"""
        if is_dataclass(obj):
            return {
                field.name: self._dataclass_to_dict(getattr(obj, field.name))
                for field in obj.__dataclass_fields__.values()
            }
        elif isinstance(obj, (list, tuple)):
            return [self._dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._dataclass_to_dict(value) for key, value in obj.items()}
        else:
            return obj

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary using recursive approach"""
        config_dict = self._dataclass_to_dict(self)
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingStageConfig":
        """Create TrainingStageConfig from dictionary with automatic nested structure support"""
        try:
            # Extract nested sections
            paths_dict = config_dict.get("paths", {})
            model_dict = config_dict.get("model", {})
            hyperparameters_dict = config_dict.get("hyperparameters", {})
            performance_dict = config_dict.get("performance", {})

            # Create nested config objects
            paths = TrainingPaths(**paths_dict)
            model = TrainingModel(**model_dict)
            hyperparameters = TrainingHyperparameters(**hyperparameters_dict)
            performance = TrainingPerformance(**performance_dict)

            # Create instance with nested configs
            return cls(
                paths=paths,
                model=model,
                hyperparameters=hyperparameters,
                performance=performance,
                log_level=config_dict.get("log_level", "INFO"),
            )
        except Exception as e:
            raise ValueError(
                f"Failed to create TrainingStageConfig from dictionary: {e}"
            )


# =============================================================================
#                           EVALUATION CONFIGURATION
# =============================================================================


@dataclass
class EvaluationPaths:
    """Evaluation paths configuration"""

    output_dir: str = "/path/to/stage_evaluation"
    metrics_dir: str = "/path/to/stage_evaluation/metrics"
    plots_dir: str = "/path/to/stage_evaluation/plots"
    logging_dir: str = "/path/to/stage_evaluation/logs"


@dataclass
class EvaluationGeneration:
    """Evaluation generation configuration"""

    max_new_tokens: int = 1024
    num_beams: int = 4
    do_sample: bool = False
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = True


@dataclass
class EvaluationMetrics:
    """Evaluation metrics configuration"""

    calculate_rouge: bool = True
    calculate_bleu: bool = True
    calculate_exact_match: bool = True
    calculate_domain_metrics: bool = True
    use_coordinate_parsing: bool = True
    use_tree_structure_analysis: bool = True
    use_routing_metrics: bool = True


@dataclass
class EvaluationPerformance:
    """Evaluation performance configuration"""

    num_workers: int = 16
    batch_size: int = 64
    dataloader_num_workers: int = 16
    dataloader_pin_memory: bool = True


@dataclass
class EvaluationOutput:
    """Evaluation output configuration"""

    save_predictions: bool = True
    save_metrics: bool = True
    num_demo_examples: int = 5


@dataclass
class EvaluationStageConfig:
    """Complete evaluation stage configuration"""

    paths: EvaluationPaths = field(default_factory=EvaluationPaths)
    generation: EvaluationGeneration = field(default_factory=EvaluationGeneration)
    metrics: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    performance: EvaluationPerformance = field(default_factory=EvaluationPerformance)
    output: EvaluationOutput = field(default_factory=EvaluationOutput)
    log_level: str = "INFO"

    def _dataclass_to_dict(self, obj: Any) -> Any:
        """Recursively convert dataclass objects to dictionaries"""
        if is_dataclass(obj):
            return {
                field.name: self._dataclass_to_dict(getattr(obj, field.name))
                for field in obj.__dataclass_fields__.values()
            }
        elif isinstance(obj, (list, tuple)):
            return [self._dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._dataclass_to_dict(value) for key, value in obj.items()}
        else:
            return obj

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary using recursive approach"""
        config_dict = self._dataclass_to_dict(self)
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EvaluationStageConfig":
        """Create EvaluationStageConfig from dictionary with automatic nested structure support"""
        try:
            # Extract nested sections
            paths_dict = config_dict.get("paths", {})
            generation_dict = config_dict.get("generation", {})
            metrics_dict = config_dict.get("metrics", {})
            performance_dict = config_dict.get("performance", {})
            output_dict = config_dict.get("output", {})

            # Create nested config objects
            paths = EvaluationPaths(**paths_dict)
            generation = EvaluationGeneration(**generation_dict)
            metrics = EvaluationMetrics(**metrics_dict)
            performance = EvaluationPerformance(**performance_dict)
            output = EvaluationOutput(**output_dict)

            # Create instance with nested configs
            return cls(
                paths=paths,
                generation=generation,
                metrics=metrics,
                performance=performance,
                output=output,
                log_level=config_dict.get("log_level", "INFO"),
            )
        except Exception as e:
            raise ValueError(
                f"Failed to create EvaluationStageConfig from dictionary: {e}"
            )


# =============================================================================
#                             MAIN FLOW CONFIGURATION
# =============================================================================


@dataclass
class FlowConfig:
    """Simplified flow configuration containing only stage configs"""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tokenization: TokenizationStageConfig = field(
        default_factory=TokenizationStageConfig
    )
    training: TrainingStageConfig = field(default_factory=TrainingStageConfig)
    evaluation: EvaluationStageConfig = field(default_factory=EvaluationStageConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FlowConfig":
        """Create FlowConfig from dictionary with simplified structure"""
        # Extract stage sections directly
        dataset_data = config_dict.get("dataset") or {}
        tokenization_data = config_dict.get("tokenization", {})
        training_data = config_dict.get("training", {})
        evaluation_data = config_dict.get("evaluation", {})

        # Build dataset config
        dataset_config = DatasetConfig.from_dict(dataset_data)

        # Create stage configs using their from_dict methods
        tokenization_config = TokenizationStageConfig.from_dict(tokenization_data)
        training_config = TrainingStageConfig.from_dict(training_data)
        evaluation_config = EvaluationStageConfig.from_dict(evaluation_data)

        config = cls(
            dataset=dataset_config,
            tokenization=tokenization_config,
            training=training_config,
            evaluation=evaluation_config,
        )
        return config

    def _dataclass_to_dict(self, obj: Any) -> Any:
        """Recursively convert dataclass objects to dictionaries"""
        if is_dataclass(obj):
            return {
                field.name: self._dataclass_to_dict(getattr(obj, field.name))
                for field in obj.__dataclass_fields__.values()
            }
        elif isinstance(obj, (list, tuple)):
            return [self._dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._dataclass_to_dict(value) for key, value in obj.items()}
        else:
            return obj

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary with simplified structure"""
        base_dict = self._dataclass_to_dict(self)

        # Add metadata fields to maintain JSON structure compatibility
        return {
            "_description": "Simplified Flow Pipeline Configuration",
            "_version": "2.0",
            "_structure": "consolidated",
            "_note": "All stage configurations consolidated into single file",
            **base_dict,
        }

    @classmethod
    def from_config_file(cls, config_path: Path) -> "FlowConfig":
        """Create FlowConfig from JSON config file"""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def save_to_file(self, output_path: Path):
        """Save FlowConfig to JSON file with Path serialization support"""
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4, default=str)

    def create_flow_config(
        self, output_path: Path, project_output_dir: Path = Path("./")
    ):
        """Create flow configuration file with proper path prefix replacement"""
        # Replace default path prefixes with project_output_dir in all stage configs
        self.replace_path_prefixes(project_output_dir)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the configuration
        self.save_to_file(output_path)
        return output_path

    def replace_path_prefixes(self, project_output_dir: Path):
        """Replace /path/to/ prefixes with project_output_dir in all path fields"""
        project_dir = project_output_dir.resolve().as_posix()

        # Replace dataset local paths
        for field_name in ("train_local_dir", "eval_local_dir"):
            current_path = getattr(self.dataset, field_name)
            if current_path and str(current_path).startswith("/path/to/"):
                new_path = str(current_path).replace("/path/to", project_dir)
                setattr(self.dataset, field_name, new_path)

        # Replace tokenization paths
        tokenization_paths = self.tokenization.paths
        for field_name in tokenization_paths.__dataclass_fields__:
            current_path = getattr(tokenization_paths, field_name)
            if current_path.startswith("/path/to/"):
                new_path = current_path.replace("/path/to", project_dir)
                setattr(tokenization_paths, field_name, new_path)

        # Replace training paths
        training_paths = self.training.paths
        for field_name in training_paths.__dataclass_fields__:
            current_path = getattr(training_paths, field_name)
            if current_path.startswith("/path/to/"):
                new_path = current_path.replace("/path/to", project_dir)
                setattr(training_paths, field_name, new_path)

        # Replace evaluation paths
        evaluation_paths = self.evaluation.paths
        for field_name in evaluation_paths.__dataclass_fields__:
            current_path = getattr(evaluation_paths, field_name)
            if current_path.startswith("/path/to/"):
                new_path = current_path.replace("/path/to", project_dir)
                setattr(evaluation_paths, field_name, new_path)
