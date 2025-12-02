# Experiments Module

The experiments module contains validation and optimization studies for evaluating different aspects of the iPCL-R pipeline, from tokenization algorithms to model architectures and fine-tuning approaches.

## Overview

This module provides comprehensive experimental analysis tools for validating design decisions, comparing methodologies, and optimizing the routing pattern generation pipeline. Each experiment focuses on a specific aspect of the system with rigorous statistical analysis and visualization.

## Research Areas

### Tokenization Comparison
**Statistical analysis of tokenization algorithms across vocabulary sizes**

- **Algorithms Tested**: DecimalWordLevel, Seg-BPE, Concat-BPE, Seg-BBPE, Concat-BBPE
- **Vocabulary Sizes**: 1000, 4000, 16000 tokens
- **Analysis Features**: Symbol frequency ranking plots, statistical significance testing
- **Output**: Comprehensive comparison reports with consistent color schemes and confidence intervals

### LLM Fine-tuning (SFT)
**Supervised fine-tuning experiments with large language models**

- **Model Support**: Qwen with LoRA configuration
- **Training Features**: Multi-GPU distributed training, early stopping, mixed precision
- **Optimization**: AdamW 8-bit optimizer, linear scheduling, gradient accumulation
- **Evaluation**: Token length statistics, comprehensive loss tracking

### Symbol Analysis  
**Domain-specific vs human language tokenization comparison**

- **Tokenizer Types**: Human language vs domain-specific routing tokenizers
- **Analysis Methods**: Token overlap analysis, vocabulary utilization studies
- **Research Pipeline**: Dataset processing, tokenizer comparison, comprehensive reporting
- **Insights**: Key findings on domain adaptation and vocabulary efficiency

### Model Architecture Studies
**Parameter scaling and performance analysis**

- **Model Variants**: Small/medium/large parameter configurations
- **Performance Metrics**: Training efficiency, memory usage, inference speed
- **Architecture**: T5-Gemma encoder-decoder variants
- **Scaling Laws**: Parameter count vs performance relationship analysis

## Key Methodologies

### Multi-stage Filtering
- Coordinate validation for routing sequences
- Tree structure integrity checking
- Statistical significance testing across experiments

### Comprehensive Evaluation
- NLP metrics (ROUGE, BLEU) combined with routing-specific metrics (RED)
- Visualization with scientific plotting standards
- Statistical confidence intervals and error analysis

### Distributed Computing
- HuggingFace Accelerate integration for multi-GPU training
- Efficient data loading with configurable worker processes
- Memory optimization for large-scale experiments

## Usage

### Tokenizer Comparison Analysis
```bash
python -m experiments.tokenizer_comparison.init_env --work-dir /path/to/work_dir

source /path/to/work_dir/run_tokenizer_comparison.sh
```

### Model Size Comparison
```bash
python -m experiments.model_size_comparison.init_env --work-dir /path/to/work_dir

source /path/to/work_dir/run_model_size_comparison.sh
```

### Feature Ablation
```bash
TBD...
```

### LLM Fine-tuning Training
```bash
accelerate launch -m experiments.sft_llm.training

# Use 'vllm' for faster inference
python -m experiments.sft_llm.evaluation
```

### Symbol Analysis Pipeline
```bash
python -m experiments.symbol_analysis.main
```

### Demo Visualization
```bash
# Routing Serialization
python -m experiments.demo.serialization

# Tree Visualization (mp4 format)
python -m experiments.demo.treeization
