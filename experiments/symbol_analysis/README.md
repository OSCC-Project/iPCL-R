# Symbol Analysis Module

A comprehensive analysis framework for validating the superiority of domain-specific tokenization over traditional human language tokenization in routing generation tasks.

## Overview

This module provides empirical evidence supporting the hypothesis that **coordinate-aware, domain-specific tokenization is superior to human language symbolic systems** for routing pattern generation. The analysis focuses on compression efficiency, semantic coherence, and pattern alignment.

## Module Components

### 1. DatasetProcessor (`dataset_processor.py`)
- Loads HuggingFace datasets using same logic as existing demos
- Extracts `tree_seq` corpus for human language tokenization (raw sequences)
- **Applies serialization preprocessing** (same as `experiments/demo/serialization.py`) to generate `target_tokens`
- Uses `UnifiedTokenizer` to convert raw data into coordinate-aware directional tokens
- Provides dataset statistics and analysis

### 2. TokenizerComparator (`tokenizer_comparator.py`)
- Loads human language tokenizer (default: `google/t5gemma-2b-2b-ul2`)
- Loads domain-specific tokenizer (default: DecimalWordLevel tokenizer)
- Performs comprehensive vocabulary usage analysis
- Generates statistical comparisons and efficiency metrics

### 3. ResearchAnalyzer (`research_analyzer.py`)
- **Compression Analysis**: Compares compression ratios and information density
- **Semantic Coherence**: Analyzes routing-relevance of generated tokens
- **Pattern Efficiency**: Evaluates coordinate pattern coverage and efficiency
- **Research Insights**: Generates quantitative evidence and qualitative observations

### 4. Main Pipeline (`main.py`)
- CLI interface with configurable parameters
- Orchestrates complete analysis workflow

## Usage

### Basic Usage
```bash
# Run with default parameters
python -m experiments.symbol_analysis.main

# Results saved to: /mnt/local_data1/liweiguo/experiments/symbol_analysis
```

### Custom Configuration
```bash
python -m experiments.symbol_analysis.main \
  --dataset-dir /path/to/your/dataset \
  --output-dir /path/to/output \
  --human-tokenizer google/t5-base \
  --domain-specific-tokenizer /path/to/domain/tokenizer \
  --human-model /path/to/local/human/model \
  --domain-model /path/to/local/domain/model \
  --log-level INFO
```

### CLI Parameters

| Parameter                     | Type   | Default                                                 | Description                              |
| ----------------------------- | ------ | ------------------------------------------------------- | ---------------------------------------- |
| `--output-dir`                | Path   | `/mnt/local_data1/liweiguo/experiments/symbol_analysis` | Output directory for results             |
| `--dataset-dir`               | Path   | `/mnt/local_data1/liweiguo/dataset/experiments/val`     | Dataset directory path                   |
| `--human-tokenizer`           | String | `google/t5gemma-2b-2b-ul2`                              | Human language tokenizer model name      |
| `--domain-specific-tokenizer` | Path   | DecimalWordLevel tokenizer path                         | Domain-specific tokenizer path           |
| `--log-level`                 | Choice | `INFO`                                                  | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `--human-model`               | Path/String | `None`                                            | Local path to human model for embedding visualization |
| `--domain-model`              | Path   | `None`                                                  | Local path to domain model for embedding visualization |

Note: Embedding plots only load models from local directories (no remote download).
